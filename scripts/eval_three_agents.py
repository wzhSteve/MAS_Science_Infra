#!/usr/bin/env python3
"""Three-user-project evaluator: agentflow / epc-aw-main / single-ppo (and
any future user_space project, selected by --projects).

Per project x dataset, this script:
  1. Serves the local Qwen3-4B via vLLM (hosted child process, always shut
     down at exit) unless --endpoint points at an already-running
     OpenAI-compatible server; a fast probe fails the run early instead of
     hanging (no stub endpoint allowed to spin).
  2. Drives each question through the exact live MAS path used by the UI:
     ExecutionService -> run_compiled_episode -> user_gateway windows
     (planner/executor/verifier of the user project). No parallel worker
     code is touched.
  3. Scores with the canonical rl.rewards.outcome functions (QA F1 for
     hotpot + bamboogle via the non-gsm8k branch; acc = F1 >= 0.5).
  4. Reuses the four attribution metrics from the last RL training log
     analysis: parse-error rate, search pollution rate (cache sentinels /
     JSON error blobs leaking into answers), context-window-exceeded rate,
     per-question wall time. Data comes from the collected agent message
     trace (tool_invoke/tool_result/error kinds), same fields the RL
     collector emits.
  5. Persists artifacts under artifacts/three_agents_eval/round<k>/:
     summary.json + detail__{project}__{dataset}.jsonl per-question rows,
     eval.log console copy.

Usage:
  # exited dev loop: one project, 2 questions per dataset (true vLLM)
  python3 scripts/eval_three_agents.py --smoke --round 0

  # full round: default projects x {hotpotqa, bamboogle} x 40 questions
  python3 scripts/eval_three_agents.py --round 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mas"))

os.environ.setdefault("TIR_MAX_MODEL_LEN", "5120")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

log = logging.getLogger("eval_three_agents")

DEFAULT_PROJECTS = ["agentflow", "epc-aw-main", "single-ppo"]
DEFAULT_DATASETS = ["hotpotqa", "bamboogle"]
MODEL_PATH_DEFAULT = str(ROOT / "LLM" / "Qwen3-4B")
MODEL_NAME = "Qwen3-4B"
DATASET_PATHS = {
    "hotpotqa": ROOT / "data" / "hotpotqa.json",
    "bamboogle": ROOT / "data" / "bamboogle.json",
}
SEARCH_CACHE_PATH = ROOT / "data" / "search_cache.json"


def _vllm_server_command_prefix() -> List[str]:
    """Prefer the repo .venv vllm CLI, then PATH vllm, then python -m.

    The training stack lives in .venv (system python has no vllm).
    """
    venv_vllm = ROOT / ".venv" / "bin" / "vllm"
    if venv_vllm.is_file():
        return [str(venv_vllm), "serve"]
    from shutil import which

    exe = which("vllm")
    if exe:
        return [exe, "serve"]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]


# --------------------------------------------------------------- attribution
# markers reused verbatim from the RL training log analysis drives.

POLLUTION_MARKERS = (
    "BadRequestError",
    "ContextWindowExceeded",
    "{'error'",
    '{"error"',
)
# any tool output / answer text starting with (or containing a line starting
# with) one of these means the search layer was unavailable for that call.
SEARCH_UNAVAILABLE_PREFIXES = (
    "search_unavailable",
    "fetch_unavailable",
)
# JSON error blobs that leak into the final answer channel take several
# shapes all carrying a recognizable error signature.
PARSE_ERROR_SIGNATURES = (
    "Error parsing LLM response",
    "verifier error",
    "JSONDecodeError",
    "Expecting property name enclosed in double quotes",
    "Expecting value: line 1 column 1",
)
CONTEXT_WINDOW_MARKER = "ContextWindowExceeded"


# --------------------------------------------------------------------------
# pure helpers -- unit-tested inline by --self-test
# --------------------------------------------------------------------------

def detect_pollution(prediction: Optional[str]) -> bool:
    """True when the final answer channel carries backend error blobs."""
    if not prediction:
        return False
    text = str(prediction)
    return any(marker in text for marker in POLLUTION_MARKERS)


def classify_tool_call(
    payload: Dict[str, Any],
    direction: str,
) -> Optional[Dict[str, Any]]:
    """Normalize one trace entry (tool_invoke / tool_result) to a metric row.

    direction 'invoke' -> call attempted; 'result' -> call answered.
    Returns None when the message is not tool-related.
    """
    if not isinstance(payload, dict):
        return None
    tool = (
        payload.get("tool")
        or payload.get("tool_name")
        or (payload.get("args") or {}).get("tool_name")
        if isinstance(payload.get("args"), dict)
        else payload.get("tool_name")
    )
    tool = str(tool or "").strip()
    output = str(payload.get("output") or payload.get("content") or "")
    errored = payload.get("ok") is False
    hit = bool(
        output
        and not errored
        and not output.lstrip().lower().startswith(SEARCH_UNAVAILABLE_PREFIXES)
        and not detect_pollution(output)
    )
    row: Dict[str, Any] = {
        "tool": tool,
        "direction": direction,
        "ok": bool(payload.get("ok")) if payload.get("ok") is not None else not errored,
    }
    if direction == "result":
        row["hit"] = hit
        row["output_head"] = output[:120]
    else:
        row["hit"] = False  # invoke rows never count toward the hit denominator
    return row


def iter_tool_calls(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    """Collect normalized tool invoke/result rows from the episode trace."""
    out: List[Dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        kind = str(msg.get("kind") or "")
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        if kind == "tool_invoke":
            row = classify_tool_call(payload, "invoke")
        elif kind == "tool_result":
            row = classify_tool_call(payload, "result")
        else:
            continue
        if row is not None:
            row["turn"] = msg.get("turn", 0)
            out.append(row)
    return out


def count_parse_errors(messages: Sequence[Any], final_answer: Optional[str]) -> int:
    """Parse/verifier errors visible in this episode's window chatter."""
    n = 0
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        kind = str(msg.get("kind") or "")
        payload = msg.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        text = json.dumps(
            payload if kind == "error" else {k: payload.get(k) for k in ("error", "llm_error", "analysis")},
            ensure_ascii=False,
            default=str,
        )
        if any(sig in text for sig in PARSE_ERROR_SIGNATURES):
            n += 1
            continue
        # window-level verifier/JSON errors marked via kind=error payloads
        if kind == "error" and payload.get("error"):
            n += 1
        # llm_error recorded inside window traces (user_gateway.detail lines)
        trace = payload.get("trace")
        if isinstance(trace, dict) and trace.get("llm_error"):
            n += 1
    if final_answer and detect_pollution(final_answer):
        n += 1
    return n


def count_context_exceeded(messages: Sequence[Any], final_answer: Optional[str]) -> int:
    """Occurrences of the 400/ContextWindowExceeded 400-over-limit failure."""
    blob = json.dumps(list(messages or []), ensure_ascii=False, default=str)
    n = blob.count(CONTEXT_WINDOW_MARKER)
    if final_answer:
        n += str(final_answer).count(CONTEXT_WINDOW_MARKER)
    return n


def aggregate_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-question rows into the project x dataset summary."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "acc": 0.0,
            "mean_f1": 0.0,
            "parse_error_rate": 0.0,
            "search_call_total": 0,
            "search_hit_rate": 0.0,
            "pollution_rate": 0.0,
            "context_exceeded_rate": 0.0,
            "mean_wall_s": 0.0,
            "error_rate": 0.0,
        }
    acc = sum(1 for r in rows if r.get("correct")) / n
    f1s = [float(r.get("f1") or 0.0) for r in rows]
    parse_err = sum(int(r.get("n_parse_errors") or 0) for r in rows)
    search_calls = [c for r in rows for c in (r.get("tool_calls") or [])]  # flattened
    n_results = [c for c in search_calls if c.get("direction") == "result"]

    hits = sum(1 for c in n_results if c.get("hit"))
    n_polluted = sum(1 for r in rows if r.get("polluted"))
    n_ctx = sum(int(r.get("n_context_exceeded") or 0) for r in rows)
    n_errored = sum(1 for r in rows if r.get("error"))
    walls = [float(r.get("wall_s") or 0.0) for r in rows]
    return {
        "n": n,
        "acc": round(acc, 4),
        "mean_f1": round(sum(f1s) / len(f1s), 4),
        "parse_error_rate": round(parse_err / n, 4),
        "search_call_total": len(search_calls),
        "search_result_total": len(n_results),
        "search_hit_rate": round(hits / len(n_results), 4) if n_results else 0.0,
        "pollution_rate": round(n_polluted / n, 4),
        "context_exceeded_rate": round(n_ctx / n, 4),
        "mean_wall_s": round(sum(walls) / len(walls), 2),
        "error_rate": round(n_errored / n, 4),
    }


def summarize_details(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    return aggregate_metrics(details)


def self_test() -> int:
    """Inline smoke of the pure logic (no GPU, no server, no user projects)."""
    import tempfile

    failures: List[str] = []

    def check(name: str, cond: bool) -> None:
        status = "ok" if cond else "FAIL"
        print(f"  [{status}] {name}")
        if not cond:
            failures.append(name)

    # --- pollution --------------------------------------------------------
    check(
        "pollution detects BadRequestError",
        detect_pollution("BadRequestError: 400 from http://127.0.0.1:8000/v1"),
    )
    check(
        "pollution detects ContextWindowExceeded",
        detect_pollution("<answer>ContextWindowExceeded ...</answer>"),
    )
    check(
        "pollution detects {'error' dict blob",
        detect_pollution("{'error': {'message': 'too long'}}"),
    )
    check("pollution clean answer negative", not detect_pollution("Prussian"))
    check("pollution empty is negative", not detect_pollution(None))

    # --- tool call classification ------------------------------------------
    hit_row = classify_tool_call(
        {"tool": "Wikipedia_Search_Tool", "ok": True, "output": "wikipedia results for: a\n1. Title"},
        "result",
    )
    check("tool hit classified", bool(hit_row and hit_row["hit"]))
    unavail_row = classify_tool_call(
        {"tool": "Wikipedia_Search_Tool", "ok": True, "output": "search_unavailable: blocked"},
        "result",
    )
    check("search_unavailable is a miss", bool(unavail_row and not unavail_row["hit"]))
    empty_row = classify_tool_call(
        {"tool": "Wikipedia_Search_Tool", "ok": True, "output": ""},
        "result",
    )
    check("empty output is a miss", bool(empty_row and not empty_row["hit"]))

    # --- trace iteration ----------------------------------------------------
    msgs = [
        {"kind": "tool_invoke", "turn": 1, "payload": {"tool_name": "Wikipedia_Search_Tool", "args": {}}},
        {"kind": "tool_result", "turn": 1, "payload": {"tool": "Wikipedia_Search_Tool", "ok": True, "output": "wikipedia results for: x"}},
        {"kind": "error", "turn": 2, "payload": {"error": "Error parsing LLM response: invalid JSON"}},
        {"kind": "verify", "turn": 2, "payload": {"reason": "fine"}},
    ]
    calls = iter_tool_calls(msgs)
    check("iter_tool_calls counts 2", len(calls) == 2)
    check("iter_tool_calls hit ratio", sum(1 for c in calls if c.get("hit")) == 1)
    check(
        "parse errors counted 1",
        count_parse_errors(msgs, None) == 1,
    )
    check(
        "context exceeded counted",
        count_context_exceeded(msgs, "ContextWindowExceeded once") == 1,
    )

    # --- aggregation ---------------------------------------------------------
    rows = [
        {
            "correct": True,
            "f1": 1.0,
            "n_parse_errors": 0,
            "tool_calls": [
                {"tool": "w", "direction": "invoke", "hit": False},
                {"tool": "w", "direction": "result", "hit": True},
            ],
            "polluted": False,
            "n_context_exceeded": 0,
            "error": None,
            "wall_s": 10.0,
        },
        {
            "correct": False,
            "f1": 0.0,
            "n_parse_errors": 1,
            "tool_calls": [{"tool": "w", "direction": "result", "hit": False}],
            "polluted": True,
            "n_context_exceeded": 1,
            "error": "episode failed",
            "wall_s": 30.0,
        },
    ]
    agg = aggregate_metrics(rows)
    check("agg n=2", agg["n"] == 2)
    check("agg acc=0.5", abs(agg["acc"] - 0.5) < 1e-9)
    check("agg mean_f1=0.5", abs(agg["mean_f1"] - 0.5) < 1e-9)
    check("agg parse_error_rate=0.5", abs(agg["parse_error_rate"] - 0.5) < 1e-9)
    check("agg search_call_total=3", agg["search_call_total"] == 3)
    check("agg search_result_total=2", agg["search_result_total"] == 2)
    check("agg search_hit_rate=1/2", abs(agg["search_hit_rate"] - 0.5) < 1e-9)
    check("agg pollution_rate=0.5", abs(agg["pollution_rate"] - 0.5) < 1e-9)
    check("agg context_exceeded_rate=0.5", abs(agg["context_exceeded_rate"] - 0.5) < 1e-9)
    check("agg mean_wall_s=20", abs(agg["mean_wall_s"] - 20.0) < 1e-9)
    check("agg error_rate=0.5", abs(agg["error_rate"] - 0.5) < 1e-9)
    empty = aggregate_metrics([])
    check("agg empty n=0", empty["n"] == 0)

    # --- JSONL round trip ----------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "detail__x__y.jsonl"
        rows_disk = [{"q": 1, "text": "当"}]  # ensure_ascii exercise
        write_jsonl(path, rows_disk)
        back = read_jsonl(path)
        check("jsonl round trip", back == rows_disk)

    if failures:
        print(f"self-test: {len(failures)} failure(s): {failures}")
        return 1
    print("self-test: all checks passed")
    return 0


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------

def load_dataset(name: str, n: int) -> List[Dict[str, Any]]:
    path = DATASET_PATHS[name]
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("data") or []
    tasks: List[Dict[str, Any]] = []
    for row in rows[:n]:
        answers = row.get("answer")
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a) for a in (answers or []) if str(a).strip()]
        tasks.append(
            {
                "id": str(row.get("pid") if str(row.get("pid") or "") not in ("", "None") else row.get("idx")),
                "question": str(row.get("question") or ""),
                "answer": answers[0] if answers else "",
                "answers": answers,
                "source": "hotpot",  # QA F1 branch in rl.rewards.accuracy
            }
        )
    return tasks


# --------------------------------------------------------------------------
# vLLM hosting
# --------------------------------------------------------------------------

@dataclass
class _ServedLLM:
    proc: Optional[subprocess.Popen] = None
    endpoint: str = ""
    model: str = MODEL_NAME
    started_by_us: bool = False
    port: int = 8000
    pgid: Optional[int] = None


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def probe_endpoint(endpoint: str, model: str, timeout_s: float = 5.0) -> Tuple[bool, str]:
    """Fast /v1/models probe so a dead endpoint fails the run instead of each
    question hanging on connect timeouts. Returns (ok, message)."""
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        return False, f"openai package unavailable: {exc}"
    try:
        client = OpenAI(base_url=endpoint, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=timeout_s)
        models = client.models.list()
        names = [str(getattr(m, "id", "") or "") for m in (models.data or [])]
        if names and model not in names:
            return False, f"endpoint up but model {model!r} not in served list {names[:5]}"
        return True, f"probe ok: {len(names)} model(s)"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _signal_pg(pgid: int, sig: int) -> None:
    """Send ``sig`` to the whole process group; ignore races."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def stop_served(svc: _ServedLLM) -> None:
    """Terminate the hosted vLLM process *group*; never raise.

    vLLM's engine cores are forked children. With ``start_new_session=True``
    the wrapper is the session/group leader; killing only the wrapper leaves
    orphans holding the GPU. Always ``killpg(SIGTERM -> wait -> SIGKILL)``.
    """
    proc = svc.proc
    pgid = svc.pgid
    svc.proc = None
    if proc is None:
        return
    if pgid is None:
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None
    svc.pgid = None
    if proc.poll() is not None and pgid is None:
        return
    log.info("stopping hosted vLLM (pid=%s pgid=%s)", proc.pid, pgid)
    if pgid is not None:
        _signal_pg(pgid, signal.SIGTERM)
    else:
        try:
            proc.send_signal(signal.SIGTERM)
        except Exception:
            pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log.warning("vLLM did not exit in 30s; SIGKILL process group")
        if pgid is not None:
            _signal_pg(pgid, signal.SIGKILL)
        else:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass
    # last resort: any orphan vllm serve on that port
    try:
        subprocess.run(
            ["pkill", "-f", f"vllm serve.*--port {svc.port}"],
            capture_output=True,
            timeout=10,
        )
    except Exception:
        pass


def ensure_llm(args: argparse.Namespace) -> _ServedLLM:
    """Probe --endpoint; else launch a hosted vLLM serve for Qwen3-4B."""
    if args.endpoint:
        ok, msg = probe_endpoint(args.endpoint, args.model)
        if not ok:
            log.error("endpoint probe failed: %s --- %s", msg, _ENDPOINT_HINT)
            raise SystemExit(3)
        log.info("using provided endpoint %s (%s)", args.endpoint, msg)
        return _ServedLLM(endpoint=args.endpoint.rstrip("/"), model=args.model)

    port = args.port
    if _port_open(port):
        endpoint = f"http://127.0.0.1:{port}/v1"
        ok, msg = probe_endpoint(endpoint, args.model)
        if ok:
            log.info("reusing vLLM already on :%d (%s)", port, msg)
            return _ServedLLM(endpoint=endpoint, model=args.model, port=port)
        if not args.serve_if_busy:
            log.error(
                "port %d busy but not a usable %s endpoint (%s); pass --endpoint or free the port",
                port, args.model, msg,
            )
            raise SystemExit(3)
        log.warning("port %d busy, not our model; trying next ports", port)
        while _port_open(port):
            port += 1
            if port > 8100:
                log.error("no free port found in 8000..8100")
                raise SystemExit(3)

    model_path = args.model_path
    if not Path(model_path, "config.json").is_file():
        log.error(
            "model weights not found at %s --- set --model-path to Qwen3-4B weights "
            "or pass --endpoint to an OpenAI-compatible server", model_path,
        )
        raise SystemExit(3)

    endpoint = f"http://127.0.0.1:{port}/v1"
    cmd = [
        *_vllm_server_command_prefix(),
        model_path,
        "--served-model-name", args.model,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--trust-remote-code",
        "--max-model-len", str(args.max_model_len),
        "--gpu-memory-utilization", str(args.gpu_mem_frac),
    ]
    log.info("starting hosted vLLM: %s", " ".join(cmd[:8]) + " ...")
    proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
        start_new_session=True,  # new session/group so killpg reaps engine cores
    )
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        pgid = None
    log.info("hosted vLLM launched pid=%s pgid=%s", proc.pid, pgid)
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
    # wait for readiness: port + /v1/models
    deadline = time.time() + args.serve_timeout
    last_err = ""
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            try:
                out = proc.stdout.read() if proc.stdout else b""
                tail = out.decode("utf-8", "replace")[-2000:]
            except Exception:
                pass
            log.error("vLLM serve exited early (code=%s). Tail:\n%s", proc.returncode, tail)
            raise SystemExit(4)
        if _port_open(port):
            ok, last_err = probe_endpoint(endpoint, args.model, timeout_s=3.0)
            if ok:
                log.info("hosted vLLM ready on %s (pid=%s pgid=%s)", endpoint, proc.pid, pgid)
                return _ServedLLM(
                    proc=proc,
                    endpoint=endpoint,
                    model=args.model,
                    started_by_us=True,
                    port=port,
                    pgid=pgid,
                )
        time.sleep(3.0)
    stop_served(_ServedLLM(proc=proc, port=port, pgid=pgid))
    log.error(
        "vLLM not ready after %ss (last probe: %s). GPU may be busy --- see nvidia-smi; "
        "or pass --endpoint to reuse a running server.", args.serve_timeout, last_err or "no probe",
    )
    raise SystemExit(4)


_ENDPOINT_HINT = (
    "start scripts/serve_local_llm.sh or pass --endpoint http://127.0.0.1:8000/v1"
)


# --------------------------------------------------------------------------
# workflow generation for one user project
# --------------------------------------------------------------------------

def _executor_agent_id(project_id: str) -> str:
    """Executor node id from the project's own manifest, with fallback.

    agentflow/epc-aw-main declare ``u_<pid>__executor``; single-ppo declares
    ``u_single-ppo__solver``. The generated workflow must use the project's
    own id so the user entry module routes tool_invoke windows correctly.
    """
    fallback = f"u_{project_id}__executor"
    try:
        from workflow.user_gateway.registry import load_manifest

        agent_ids = list(load_manifest(project_id).get("agent_ids") or [])
    except Exception:
        return fallback
    for aid in agent_ids:
        aid = str(aid)
        if aid.startswith("u_") and aid not in ("planner", "verifier"):
            return aid
    return fallback


def workflow_for_project(project_id: str) -> Optional[Dict[str, Any]]:
    """Build the centralized 3-window workflow bound to one user project.

    Mirrors experiments/agentflow_4algo/workflow.yaml but parameterized by
    project id, so any user_space project (agentflow / epc-aw-main /
    single-ppo / future ones) plugs in via its manifest -- nothing here
    imports user code.
    """
    executor_id = _executor_agent_id(project_id)
    return {
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "hub": {
            "role": "planner",
            "skills": [],
            "max_feedback_hops": 6,
            "system_prompt": (
                "You are the planner window. Decompose the question and pick the "
                "next step. Output JSON with keys next/args/sub_goal/done. "
                f"next must be {executor_id} while more work is needed."
            ),
        },
        "tools": [],
        "llm": {"kind": "local", "model": MODEL_NAME, "base_url": "http://127.0.0.1:8000/v1"},
        "memory": {"agent": "messages", "system": "none"},
        "archive": {"window": "post_first_tool"},
        "agents": [
            {
                "id": "planner",
                "kind": "planner",
                "role": "planner",
                "label": f"{project_id}_planner",
                "trainable": True,
                "system_prompt": (
                    f"Wrap {project_id} planner window. Output JSON next/args/sub_goal/done. "
                    f"next is the executor window ({executor_id})."
                ),
                "tools": [],
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id},
            },
            {
                "id": executor_id,
                "kind": "blank",
                "role": "executor",
                "label": f"{project_id}_executor",
                "trainable": False,
                "system_prompt": f"Wrap {project_id} executor window (tool command + execute).",
                "profile": {
                    "backend": "user_space",
                    "user_project": project_id,
                    "llm_required": False,
                },
                "meta": {"origin": "user", "user_project": project_id},
            },
            {
                "id": "verifier",
                "kind": "verifier",
                "role": "verifier",
                "label": f"{project_id}_verifier",
                "trainable": False,
                "system_prompt": f"Wrap {project_id} verifier window.",
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id},
            },
        ],
        "edges": [
            {"from": "planner", "to": executor_id, "kind": "message"},
            {"from": executor_id, "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
        "routers": [],
    }


def project_exists(project_id: str) -> bool:
    try:
        from workflow.user_gateway.registry import load_manifest

        load_manifest(project_id)
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# episode execution reusing the mas data layer
# --------------------------------------------------------------------------

def _force_prompts_lite_false(ctx_mod: Any) -> None:
    """In-process prompts_lite=false; does not touch contracts on disk."""
    os.environ["SCIENCE_EPC_AW_PROMPTS_LITE"] = "0"
    if hasattr(ctx_mod, "_prompts_lite_enabled"):
        ctx_mod._prompts_lite_enabled = lambda: False  # type: ignore[assignment]
    orig_load = getattr(ctx_mod, "_load_runtime", None)
    if callable(orig_load) and not getattr(orig_load, "_eval_lite_forced_off", False):

        def _wrapped_load(orig=orig_load):  # type: ignore[no-untyped-def]
            cfg = orig()
            if isinstance(cfg, dict):
                cfg = dict(cfg)
                cfg["prompts_lite"] = False
            return cfg

        _wrapped_load._eval_lite_forced_off = True  # type: ignore[attr-defined]
        ctx_mod._load_runtime = _wrapped_load  # type: ignore[assignment]
    # Undo agentflow module-level open shims if already installed.
    for mod_name in (
        "agentflow.models.planner",
        "agentflow.models.executor",
        "agentflow.models.diagnoser",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        shim = getattr(mod, "open", None)
        orig = getattr(shim, "_agentflow_lite_orig", None) if shim is not None else None
        if orig is not None:
            mod.open = orig  # type: ignore[attr-defined]
        elif shim is not None and getattr(shim, "_agentflow_lite", False):
            try:
                delattr(mod, "open")
            except Exception:
                pass
    # Undo epc-aw builtins.open lite marker if present.
    try:
        import builtins

        marker = getattr(ctx_mod, "_LITE_MARKER", "_science_prompt_lite")
        opened = builtins.open
        if getattr(opened, marker, False):
            orig_b = getattr(opened, "_orig_open", None) or getattr(opened, "__wrapped__", None)
            # epc-aw stores orig in closure; fall back to clearing by re-bind if recorded
            stored = getattr(ctx_mod, "_eval_builtins_open_orig", None)
            if stored is not None:
                builtins.open = stored  # type: ignore[assignment]
    except Exception:
        pass


def _preflight_prompts_lite(project_id: str) -> None:
    """Install lite shims; on failure force prompts_lite=false in-process.

    Never writes user_space contracts. epc-aw also honors
    SCIENCE_EPC_AW_PROMPTS_LITE=0 after a failed install.
    """
    try:
        from workflow.user_gateway.loader import load_entry_module

        load_entry_module(project_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("lite preflight: could not load %s entry (%s)", project_id, exc)
        return
    ctx_mod = sys.modules.get("context")
    if ctx_mod is None:
        return
    install = getattr(ctx_mod, "_install_lite_prompts", None)
    if not callable(install):
        return
    try:
        install()
        log.info("prompts_lite preflight ok for %s", project_id)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "lite shim failed for %s (%s); in-process prompts_lite=false (contracts untouched)",
            project_id,
            exc,
        )
        _force_prompts_lite_false(ctx_mod)


def run_one_question(
    task: Dict[str, Any],
    *,
    project_id: str,
    svc: _ServedLLM,
    spec_dir: Path,
    agent_timeout_s: float,
) -> Dict[str, Any]:
    """One question through the live centralized MAS + user windows."""
    import yaml

    from workflow.runtime import ExecutionService, LLMConfig
    from workflow.user_gateway.sandbox import bind_eval_cancel, reset_eval_cancel

    workflow = workflow_for_project(project_id)
    spec_path = spec_dir / f"workflow__{project_id}.yaml"
    spec_path.write_text(
        yaml.safe_dump(workflow, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    llm = LLMConfig(
        endpoint=svc.endpoint,
        model=svc.model,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        temperature=0.0,
    )
    t0 = time.time()
    row: Dict[str, Any] = {
        "idx": task["id"],
        "question": task["question"][:200],
        "gold": task.get("answer") or "",
        "project": project_id,
    }
    error: Optional[str] = None
    traj = None
    cancel = threading.Event()
    timer: Optional[threading.Timer] = None
    cancel_token = bind_eval_cancel(cancel)
    try:
        # Soft-reset the user-space episode context (same call the official
        # UI eval loop makes before every question, rollout_runs.py:517) so
        # consecutive questions never share outline/memory/stop flags.
        ctx_mod = sys.modules.get("context")
        if ctx_mod is not None and hasattr(ctx_mod, "get_episode_context"):
            try:
                ctx_mod.get_episode_context(reset=True)
            except Exception:
                pass
        service = ExecutionService(
            mock=False,
            spec_path=str(spec_path),
            llm=llm,
            agent_llms={},
            archive_root=str(spec_dir / "archives"),
        )
        if agent_timeout_s and agent_timeout_s > 0:
            timer = threading.Timer(float(agent_timeout_s), cancel.set)
            timer.daemon = True
            timer.start()
        with _episode_env(svc):
            traj = service.run(task)
        if cancel.is_set() and traj is not None:
            # Runtime soft-stops with meta error "用户停止"; rename for clarity.
            meta_err = str((traj.meta or {}).get("error") or "")
            if meta_err in ("", "用户停止"):
                error = f"question wall-clock exceeded {agent_timeout_s:.0f}s"
    except Exception as exc:  # noqa: BLE001
        if cancel.is_set():
            error = f"question wall-clock exceeded {agent_timeout_s:.0f}s"
        else:
            error = f"{type(exc).__name__}: {exc}"
    finally:
        if timer is not None:
            timer.cancel()
        reset_eval_cancel(cancel_token)
    wall = time.time() - t0

    final_answer = None
    messages: List[Dict[str, Any]] = []
    if traj is not None:
        final_answer = traj.final_answer
        messages = [m if isinstance(m, dict) else dict(m) for m in (traj.messages or [])]
        if traj.meta.get("error") and not error:
            err_txt = str(traj.meta.get("error"))
            if cancel.is_set() and err_txt == "用户停止":
                error = f"question wall-clock exceeded {agent_timeout_s:.0f}s"
            else:
                error = err_txt

    from rl.rewards.outcome import accuracy, extract_answer_text, f1_score

    pred = final_answer or ""
    extracted = extract_answer_text(pred) if pred else None
    score_text = str(extracted or pred or "")
    gold = str(task.get("answer") or "")
    aliases = [str(a) for a in (task.get("answers") or []) if str(a)]
    f1 = f1_score(score_text, [gold] + aliases) if (gold or aliases) else 0.0
    # accuracy() reuses the exact production branch; source='hotpot' -> QA F1
    prod_acc = accuracy(pred or None, gold, source=str(task.get("source")), aliases=aliases or None)

    tool_calls = iter_tool_calls(messages)
    row.update(
        {
            "final_answer": str(pred)[:500],
            "f1": round(float(f1), 4),
            "correct": bool(f1 >= 0.5),
            "prod_acc": float(prod_acc),
            "n_parse_errors": count_parse_errors(messages, pred),
            "n_context_exceeded": count_context_exceeded(messages, pred),
            "tool_calls": tool_calls[:40],
            "polluted": detect_pollution(pred),
            "error": error,
            "wall_s": round(wall, 2),
        }
    )
    return row


_LLM_ENV_KEYS = (
    "OPENAI_API_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "MODEL",
    "MODEL_Name",
    "MODEL_NAME",
)


class _episode_env:
    """Bind OPENAI_* to the eval server for the duration of one episode.

    Same approach as science_infra.control.rollout_runs._bind_episode_llm.
    """

    def __init__(self, svc: _ServedLLM) -> None:
        self.svc = svc
        self.prev: Dict[str, Optional[str]] = {}

    def __enter__(self) -> None:
        keys = list(_LLM_ENV_KEYS) + [
            "SCIENCE_EPC_AW_MAX_MODEL_LEN",
            "SCIENCE_EPC_AW_MAX_COMPLETION",
            "MAS_MAX_COMPLETION_TOKENS",
        ]
        self.prev = {k: os.environ.get(k) for k in keys}
        endpoint = self.svc.endpoint.rstrip("/")
        os.environ["OPENAI_API_BASE_URL"] = endpoint
        os.environ["OPENAI_API_BASE"] = endpoint
        os.environ["OPENAI_BASE_URL"] = endpoint
        os.environ["OPENAI_API_KEY"] = "EMPTY"
        os.environ["OPENAI_MODEL"] = self.svc.model
        os.environ["MODEL"] = self.svc.model
        os.environ["MODEL_Name"] = self.svc.model
        os.environ["MODEL_NAME"] = self.svc.model
        os.environ.setdefault("SCIENCE_EPC_AW_MAX_MODEL_LEN", "8192")
        os.environ.setdefault("SCIENCE_EPC_AW_MAX_COMPLETION", "2048")
        os.environ.setdefault("MAS_MAX_COMPLETION_TOKENS", "2048")

    def __exit__(self, *exc: Any) -> None:
        for key, value in self.prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", default=",".join(DEFAULT_PROJECTS),
                    help=f"comma list; missing projects are skipped with a warning (default: {','.join(DEFAULT_PROJECTS)})")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help=f"comma list (default: {','.join(DEFAULT_DATASETS)})")
    ap.add_argument("--n-per-dataset", type=int, default=40)
    ap.add_argument("--round", type=int, default=1, help="artifacts land in artifacts/three_agents_eval/round<k>")
    ap.add_argument("--endpoint", default=os.environ.get("OPENAI_BASE_URL", ""),
                    help="OpenAI-compatible chat API base, e.g. http://127.0.0.1:8000/v1. Empty => host local vLLM.")
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT,
                    help="local Qwen3-4B weights dir for hosted vLLM")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gpu-mem-frac", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192, help="vLLM serve max_model_len")
    ap.add_argument("--serve-timeout", type=float, default=900.0,
                    help="seconds to wait for hosted vLLM readiness")
    ap.add_argument("--serve-if-busy", action="store_true",
                    help="when port 8000 is busy with a foreign server, pick the next free port")
    ap.add_argument("--agent-timeout", type=float, default=600.0,
                    help="per-question wall-clock budget (soft-cancel via eval_cancelled; smoke caps at 300s)")
    ap.add_argument("--window-timeout", type=float, default=None,
                    help="SCIENCE_USER_WINDOW_TIMEOUT per user window (default: mas gateway 300s). "
                         "Smoke defaults to min(180, MAS_WINDOW_TIMEOUT_S). "
                         "Raise (e.g. 900) when the local vLLM is slow and windows legitimately need longer.")
    ap.add_argument("--smoke", action="store_true",
                    help="2 questions per dataset on the first existing project only, round dir 'smoke' unless --round given")
    ap.add_argument("--index-offset", type=int, default=0,
                    help="skip the first N questions of each dataset (round 2 halves, etc.)")
    ap.add_argument("--self-test", action="store_true", help="run inline unit checks and exit")
    return ap


def _check_runtime_deps() -> None:
    """Fail fast with an actionable message when the interpreter lacks the
    MAS runtime deps (training stack lives in .venv, not system python)."""
    venv_python = ROOT / ".venv" / "bin" / "python"
    missing: List[str] = []
    for mod in ("langchain_core", "openai", "yaml", "pydantic"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing and venv_python.is_file():
        log.error(
            "interpreter %s lacks %s. Re-run with:\n  %s scripts/eval_three_agents.py ...",
            sys.executable, ", ".join(missing), venv_python,
        )
        raise SystemExit(5)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.endpoint and not _port_open(args.port):
        # early venv hint before we spend minutes loading weights
        try:
            import langchain_core  # noqa: F401
            import openai  # noqa: F401
        except Exception:
            venv = ROOT / ".venv" / "bin" / "python"
            log.warning(
                "note: %s is missing MAS runtime deps; if episodes fail with "
                "ModuleNotFoundError re-run with %s",
                sys.executable, venv,
            )

    round_dir = ROOT / "artifacts" / "three_agents_eval" / (
        "smoke" if (args.smoke and args.round == 0) else f"round{args.round}{'' if not args.smoke else '_smoke'}"
    )
    if not args.smoke:
        round_dir = ROOT / "artifacts" / "three_agents_eval" / f"round{args.round}"

    if args.smoke:
        # smoke default artifact dir: artifacts/three_agents_eval/smoke
        # (a nonzero --round names it round<k>_smoke instead)
        args.n_per_dataset = 2
        # Hard caps so a hung window cannot pin the GPU for hours again.
        # Window ≤180s (MAS_WINDOW_TIMEOUT_S / --window-timeout, then capped).
        # Question wall ≤300s via eval_cancelled soft-stop.
        if args.window_timeout is None:
            env_win = os.environ.get("MAS_WINDOW_TIMEOUT_S", "").strip()
            try:
                args.window_timeout = float(env_win) if env_win else 180.0
            except ValueError:
                args.window_timeout = 180.0
        args.window_timeout = min(float(args.window_timeout), 180.0)
        if float(args.agent_timeout) > 300.0:
            args.agent_timeout = 300.0

    round_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.addHandler(sh)
    fh = logging.FileHandler(round_dir / "eval.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.propagate = False

    # search cache wiring: missing file is fine (tools fall back to live
    # bing/wiki and back-fill the cache), reuse mas/tools contract.
    os.environ["MAS_SEARCH_CACHE"] = str(SEARCH_CACHE_PATH)
    if args.window_timeout:
        os.environ["SCIENCE_USER_WINDOW_TIMEOUT"] = str(int(args.window_timeout))
        log.info("window timeout override: %ss", int(args.window_timeout))
    if args.smoke:
        log.info(
            "smoke budgets: window≤%ss question_wall≤%ss",
            int(args.window_timeout or 180),
            int(args.agent_timeout),
        )
    if SEARCH_CACHE_PATH.is_file():
        with SEARCH_CACHE_PATH.open("r", encoding="utf-8") as fh_:
            n_keys = len(json.load(fh_))
        log.info("search cache: %d keys", n_keys)
    else:
        log.info("search cache file missing -> starts empty (tools tolerate this)")

    projects = [p.strip() for p in str(args.projects).split(",") if p.strip()]
    datasets = [d.strip() for d in str(args.datasets).split(",") if d.strip()]
    for name in datasets:
        if name not in DATASET_PATHS:
            log.error("unknown dataset %r (known: %s)", name, ", ".join(DATASET_PATHS))
            return 2
    unknown_ds = [d for d in datasets if d not in DATASET_PATHS]
    if unknown_ds:
        log.error("unknown dataset(s): %s", ", ".join(unknown_ds))
        return 2

    existing_projects = []
    for pid in projects:
        if project_exists(pid):
            existing_projects.append(pid)
            log.info("project ready: %s", pid)
        else:
            log.warning("project %s not scaffolded yet --- skipping (warn only)", pid)
    if not existing_projects:
        log.error("no runnable project among %s", projects)
        return 2

    if args.smoke:
        if len(projects) <= 1:
            # single-project smoke: keep it minimal (task-book default)
            existing_projects = existing_projects[:1]
        # else: the caller explicitly listed several projects; keep them all
        log.info(
            "smoke mode: %d project(s) %s, 2 questions/dataset",
            len(existing_projects), ",".join(existing_projects),
        )
        args.n_per_dataset = 2

    for pid in existing_projects:
        _preflight_prompts_lite(pid)

    spec_dir = round_dir / "specs"
    spec_dir.mkdir(parents=True, exist_ok=True)

    _check_runtime_deps()
    svc = ensure_llm(args)
    started_by_us = svc.started_by_us
    exit_code = 0
    summary: Dict[str, Any] = {
        "round": args.round,
        "smoke": bool(args.smoke),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": svc.model,
        "endpoint": svc.endpoint,
        "hosted_vllm_by_eval": started_by_us,
        "vllm_pgid": svc.pgid,
        "window_timeout_s": args.window_timeout,
        "agent_timeout_s": args.agent_timeout,
        "n_per_dataset": args.n_per_dataset,
        "projects": {},
        "dataset_paths": {k: str(v) for k, v in DATASET_PATHS.items() if k in datasets},
    }
    try:
        for pid in existing_projects:
            summary["projects"][pid] = {}
            for ds in datasets:
                tasks = load_dataset(ds, args.n_per_dataset + args.index_offset)
                tasks = tasks[args.index_offset:]
                if not tasks:
                    log.warning("%s/%s: no tasks after offset %d", pid, ds, args.index_offset)
                    continue
                rows: List[Dict[str, Any]] = []
                log.info("=== %s x %s: %d questions ===", pid, ds, len(tasks))
                for i, task in enumerate(tasks, 1):
                    log.info("[%s/%s] %d/%d q=%s", pid, ds, i, len(tasks), task["question"][:80])
                    row = run_one_question(
                        task,
                        project_id=pid,
                        svc=svc,
                        spec_dir=spec_dir,
                        agent_timeout_s=args.agent_timeout,
                    )
                    rows.append(row)
                    log.info(
                        "  -> f1=%.3f acc=%s polluted=%s parse_err=%d wall=%.1fs err=%s",
                        row["f1"], row["correct"], row["polluted"],
                        row["n_parse_errors"], row["wall_s"],
                        (row["error"] or "")[:80],
                    )
                detail_path = round_dir / f"detail__{pid}__{ds}.jsonl"
                write_jsonl(detail_path, rows)
                metrics = aggregate_metrics(rows)
                summary["projects"][pid][ds] = metrics
                summary["projects"][pid][ds]["detail_file"] = str(detail_path.relative_to(ROOT))
                log.info("%s x %s metrics: %s", pid, ds, json.dumps(metrics, ensure_ascii=False))
    except KeyboardInterrupt:
        log.warning("interrupted; artifacts so far are kept")
        exit_code = 130
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
        raise
    finally:
        summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out = round_dir / "summary.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        log.info("wrote %s", out)
        if started_by_us:
            stop_served(svc)
            log.info("hosted vLLM shut down")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
