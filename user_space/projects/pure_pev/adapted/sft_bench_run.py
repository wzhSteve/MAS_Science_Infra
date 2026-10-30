"""Run the copied AgentFlow windows on the first 8 APPO valid questions.

Baseline (AGENTFLOW_SFT_PROMPTS unset) uses the official f-strings.
Adapted mode turns on the short prompts and the ``<search>`` / ``<python>``
conversion. Tool calls are capped at 12s. Search retries are forced to one
short attempt so Bright Data cannot sit in backoff.

Writes ``artifacts/sft_bench.json``, merging one model/phase at a time.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
PROJECT = ROOT / "user_space" / "projects" / "pure_pev"
ADAPTED = PROJECT / "adapted"
PARQUET = ROOT / "ref_Rep" / "APPO-main" / "APPO" / "rl_datasets" / "valid.parquet"
REPORT = PROJECT / "artifacts" / "sft_bench.json"

os.environ["BRIGHT_DATA_TIMEOUT"] = "6"
os.environ["BRIGHT_DATA_MAX_RETRIES"] = "1"
os.environ.setdefault("AGENTFLOW_LLM_TIMEOUT", "180")
os.environ.setdefault("AGENTFLOW_MAX_STEPS", "2")
os.environ.setdefault("AGENTFLOW_MAX_TOKENS", "4096")
os.environ.setdefault("AGENTFLOW_MAX_TIME", "240")
os.environ.setdefault("TIR_MAX_MODEL_LEN", "8192")

for p in (str(ROOT), str(ROOT / "mas"), str(ADAPTED)):
    if p not in sys.path:
        sys.path.insert(0, p)

from workflow.env_load import load_repo_dotenv  # noqa: E402

load_repo_dotenv()
os.environ["BRIGHT_DATA_TIMEOUT"] = "6"
os.environ["BRIGHT_DATA_MAX_RETRIES"] = "1"

import context as ctx_mod  # noqa: E402

_EXECUTE_RE = re.compile(r"execution\s*=\s*tool\.execute\s*\(", re.I)
_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)


def _clip(text: Any, limit: int = 3500) -> str:
    raw = str(text if text is not None else "")
    raw = raw.replace("\x00", "")
    if len(raw) <= limit:
        return raw
    head = 700
    return raw[:head] + "\n...[clipped]...\n" + raw[-(limit - head):]


def _load_questions(n: int = 8) -> List[Dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(PARQUET)
    rows: List[Dict[str, Any]] = []
    for i, row in frame.head(n).iterrows():
        extra = row["extra_info"]
        if hasattr(extra, "item"):
            extra = extra.item()
        reward = row["reward_model"]
        if hasattr(reward, "item"):
            reward = reward.item()
        question = ""
        if isinstance(extra, dict):
            question = str(extra.get("question") or "")
        truth = ""
        if isinstance(reward, dict):
            truth = str(reward.get("ground_truth") or "")
        rows.append({"index": int(i), "question": question, "ground_truth": truth})
    return rows


def _verdict(raw: str) -> Dict[str, Any]:
    from sft_prompts import parse_conclusion

    token = parse_conclusion(raw)
    if token:
        return {"kind": "explicit", "conclusion": token}
    return {"kind": "default", "conclusion": "CONTINUE"}


def _json_get(data: Dict[str, Any], key: str) -> str:
    value = data.get(key)
    if isinstance(value, list):
        return "\n".join(str(x) for x in value)
    return str(value or "")


def _step_record(ep: Any, plan: Dict[str, Any], tool: Optional[Dict[str, Any]], verify: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    step = int(ep.step_count)
    data = ep.json_data
    plan_raw = _json_get(data, f"action_predictor_{step}_response")
    exec_raw = _json_get(data, f"tool_commander_{step}_response")
    ver_raw = _json_get(data, f"verifier_{step}_response")
    tool_name = ep.tool_name
    available = list(ep.planner.available_tools)
    planner_ok = bool(ep.context and ep.sub_goal and tool_name in available)
    planner_why = ""
    if not ep.context or not ep.sub_goal or not tool_name:
        planner_why = "missing Context / Sub-Goal / Tool Name"
    elif tool_name not in available:
        planner_why = f"tool not in five: {tool_name}"

    command = ""
    result = ""
    invoked = False
    if tool is not None:
        command = str(tool.get("payload", {}).get("command") or "")
        result = str(tool.get("payload", {}).get("output") or "")
        invoked = "tool was not found" not in command
    command_ok = bool(_EXECUTE_RE.search(command))
    executor_ok = invoked and command_ok
    executor_why = ""
    if not invoked:
        executor_why = "executor not called; planner tool name did not match"
    elif not command_ok:
        executor_why = "no execution = tool.execute(...) after extract"

    verdict = _verdict(ver_raw) if verify is not None else {"kind": "missing", "conclusion": ""}
    verifier_ok = verdict["kind"] == "explicit"
    verifier_why = "" if verifier_ok else f"conclusion kind={verdict['kind']}"

    answer = ""
    ready = False
    if verify is not None:
        answer = str(verify.get("payload", {}).get("answer") or "")
        ready = bool(verify.get("payload", {}).get("ready_to_stop"))
    answer_ok = bool(_ANSWER_RE.search(answer)) and bool(_ANSWER_RE.search(answer).group(1).strip())  # type: ignore[union-attr]
    feeds_next = planner_ok and executor_ok and verifier_ok

    return {
        "step": step,
        "planner": {
            "ok": planner_ok,
            "why": planner_why,
            "context": _clip(ep.context, 500),
            "sub_goal": _clip(ep.sub_goal, 500),
            "tool_name": tool_name,
            "raw": _clip(plan_raw),
            "query_analysis": _clip(ep.query_analysis, 800),
        },
        "executor": {
            "ok": executor_ok,
            "invoked": invoked,
            "why": executor_why,
            "command": _clip(command, 800),
            "result": _clip(result, 800),
            "raw": _clip(exec_raw),
        },
        "verifier": {
            "ok": verifier_ok,
            "kind": verdict["kind"],
            "conclusion": verdict["conclusion"],
            "why": verifier_why,
            "ready_to_stop": ready,
            "raw": _clip(ver_raw),
        },
        "answer": _clip(answer, 800),
        "answer_ok": answer_ok if ready else None,
        "feeds_next": feeds_next,
        "plan_next": (plan.get("payload") or {}).get("next"),
    }


def _failure_samples(steps: List[Dict[str, Any]], limit: int = 2) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    seen = {"planner": 0, "executor": 0, "verifier": 0, "answer": 0}
    for step in steps:
        for role in ("planner", "executor", "verifier"):
            block = step[role]
            if block.get("ok") or seen[role] >= limit:
                continue
            seen[role] += 1
            samples.append({
                "step": step["step"],
                "role": role,
                "why": block.get("why"),
                "raw": block.get("raw"),
                "parsed": {
                    "tool_name": step["planner"].get("tool_name"),
                    "command": step["executor"].get("command"),
                    "conclusion": step["verifier"].get("conclusion"),
                    "kind": step["verifier"].get("kind"),
                },
            })
        if step.get("answer_ok") is False and seen["answer"] < limit:
            seen["answer"] += 1
            samples.append({
                "step": step["step"],
                "role": "answer",
                "why": "stopped without an extractable <answer>",
                "raw": step.get("answer"),
            })
    return samples


def _rates(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = [s for q in questions for s in q.get("steps") or []]
    n_steps = len(steps) or 1
    n_q = len(questions) or 1

    def count(pred) -> int:
        return sum(1 for s in steps if pred(s))

    advanced = sum(1 for q in questions if any(s.get("feeds_next") for s in q.get("steps") or []))
    stopped = [q for q in questions if q.get("stopped")]
    answered = sum(1 for q in questions if q.get("answer_ok"))
    return {
        "steps": len(steps),
        "questions": len(questions),
        "planner_ok": count(lambda s: s["planner"]["ok"]),
        "executor_ok": count(lambda s: s["executor"]["ok"]),
        "executor_invoked": count(lambda s: s["executor"]["invoked"]),
        "verifier_explicit": count(lambda s: s["verifier"]["ok"]),
        "feeds_next_steps": count(lambda s: s.get("feeds_next")),
        "episodes_advanced": advanced,
        "stopped": len(stopped),
        "stopped_with_answer": answered,
        "planner_rate": round(count(lambda s: s["planner"]["ok"]) / n_steps, 3),
        "executor_rate": round(count(lambda s: s["executor"]["ok"]) / n_steps, 3),
        "verifier_rate": round(count(lambda s: s["verifier"]["ok"]) / n_steps, 3),
        "episode_advance_rate": round(advanced / n_q, 3),
        "answer_rate": round(answered / n_q, 3),
    }


def _run_question(ctx: Any, item: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    question = item["question"]
    payload = {"question": question, "source": "appo_valid"}
    steps: List[Dict[str, Any]] = []
    error = ""
    stopped = False
    answer = ""
    t0 = time.time()
    try:
        for _ in range(max_steps):
            plan = ctx.plan_window({"payload": payload})
            if (plan.get("payload") or {}).get("done"):
                answer = str((plan.get("payload") or {}).get("answer") or "")
                stopped = True
                break
            ep = ctx.episode(payload)
            tool = ctx.exec_window({"payload": payload})
            verify = ctx.verify_window({"payload": payload})
            rec = _step_record(ep, plan, tool, verify)
            steps.append(rec)
            if (verify.get("payload") or {}).get("ready_to_stop"):
                stopped = True
                answer = str((verify.get("payload") or {}).get("answer") or "")
                break
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
    answer_ok = False
    if stopped and answer:
        match = _ANSWER_RE.search(answer)
        answer_ok = bool(match and match.group(1).strip())
    direct = ""
    try:
        direct = str(getattr(ctx.episode(payload), "direct_output", "") or "")
    except Exception:
        direct = ""
    return {
        "index": item["index"],
        "question": question,
        "ground_truth": item["ground_truth"],
        "steps": steps,
        "stopped": stopped,
        "answer": _clip(answer, 800),
        "answer_ok": answer_ok,
        "model_emitted_answer_tag": "<answer>" in direct.lower(),
        "direct_output": _clip(direct, 800),
        "elapsed_s": round(time.time() - t0, 1),
        "error": _clip(error, 1500),
        "advanced": any(s.get("feeds_next") for s in steps),
    }


def _merge(phase: str, model_name: str, block: Dict[str, Any], questions: List[Dict[str, Any]]) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    doc: Dict[str, Any] = {}
    if REPORT.is_file():
        try:
            doc = json.loads(REPORT.read_text(encoding="utf-8"))
        except Exception:
            doc = {}
    doc.setdefault("project", "user_space/projects/pure_pev")
    doc.setdefault("upload", "symlink ../agentflow/upload")
    doc.setdefault("questions_source", str(PARQUET))
    doc.setdefault("harness", {
        "tool_timeout_s": 12,
        "bright_data_timeout_s": 6,
        "bright_data_max_retries": 1,
        "memory_result_chars": 800,
        "temperature": 0.0,
    })
    phases = doc.setdefault("phases", {})
    phase_doc = phases.setdefault(phase, {"models": {}})
    # The shared 8-question index belongs to baseline/adapted. Later phases
    # keep their own index so a 20-question run cannot replace it.
    if phase in ("baseline", "adapted") or "questions" not in doc:
        doc["questions"] = [{"index": q["index"], "question": q["question"], "ground_truth": q["ground_truth"]} for q in questions]
    else:
        phase_doc["questions"] = [{"index": q["index"], "question": q["question"], "ground_truth": q["ground_truth"]} for q in questions]
    phase_doc["models"][model_name] = block
    REPORT.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, help="baseline keeps official prompts; any other name turns compressed prompts on")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    if args.phase == "baseline":
        os.environ.pop("AGENTFLOW_SFT_PROMPTS", None)
    else:
        os.environ["AGENTFLOW_SFT_PROMPTS"] = "1"
        # Stale shells may still export 768/1024. Compressed-prompt runs always generate 4096.
        os.environ["AGENTFLOW_MAX_TOKENS"] = "4096"
    os.environ["OPENAI_API_BASE"] = args.base_url
    os.environ["OPENAI_API_BASE_URL"] = args.base_url
    os.environ["MODEL_NAME"] = args.model_name
    os.environ["AGENTFLOW_MAX_STEPS"] = str(args.max_steps)
    os.environ["AGENTFLOW_TOKENIZER"] = os.environ.get("AGENTFLOW_TOKENIZER") or os.environ.get("MODEL_PATH", "")

    original_load = ctx_mod.load_runtime

    def load_runtime() -> Dict[str, Any]:
        cfg = original_load()
        cfg["temperature"] = 0.0
        cfg["memory_result_chars"] = 800
        cfg["max_steps"] = args.max_steps
        return cfg

    ctx_mod.load_runtime = load_runtime  # type: ignore[assignment]

    import mas.tools.search as search

    search.BRIGHT_DATA_TIMEOUT = 6
    search.BRIGHT_DATA_MAX_RETRIES = 1

    questions = _load_questions(args.n)
    ctx = ctx_mod.get_episode_context()
    comps = ctx.components()
    comps.executor.max_time = 12
    print(
        "tools", comps.available_tools,
        "sft", ctx_mod.sft_prompts_enabled(),
        "max_tokens", os.environ.get("AGENTFLOW_MAX_TOKENS"),
        "cap", comps.planner.llm_engine.max_tokens_cap,
        flush=True,
    )
    print("search timeout", search.BRIGHT_DATA_TIMEOUT, "retries", search.BRIGHT_DATA_MAX_RETRIES, "key", bool(search.BRIGHT_DATA_API_KEY), flush=True)

    ran: List[Dict[str, Any]] = []
    for item in questions:
        print(f"Q{item['index']} start {item['question'][:80]}", flush=True)
        rec = _run_question(ctx, item, args.max_steps)
        ran.append(rec)
        flags = [
            f"s{s['step']}:p={int(s['planner']['ok'])}/e={int(s['executor']['ok'])}/v={int(s['verifier']['ok'])}"
            for s in rec["steps"]
        ]
        print(
            f"Q{item['index']} {rec['elapsed_s']}s advanced={rec['advanced']} stop={rec['stopped']} "
            f"answer={rec['answer_ok']} {' '.join(flags)} err={bool(rec['error'])}",
            flush=True,
        )
        rates = _rates(ran)
        _merge(args.phase, args.model_name, {
            "model_name": args.model_name,
            "phase": args.phase,
            "sft_prompts": ctx_mod.sft_prompts_enabled(),
            "max_tokens": int(os.environ.get("AGENTFLOW_MAX_TOKENS") or 4096),
            "max_steps": args.max_steps,
            "parse_rates": rates,
            "failure_samples": _failure_samples([s for q in ran for s in q["steps"]]),
            "questions": ran,
            "partial": len(ran) < len(questions),
        }, questions)

    rates = _rates(ran)
    print("RATES", json.dumps(rates), flush=True)


if __name__ == "__main__":
    main()
