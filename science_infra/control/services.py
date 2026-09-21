"""Control services: LLM health/start, MAS collect, harness diagnose, RL train."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from science_infra.control.events import BUS
from science_infra.control.experiments import (
    VALID_ALGOS,
    artifacts_dir,
    collect_path,
    diagnose_path,
    ensure_experiment,
    exp_dir,
    load_bundle,
    normalize_rl_data_paths,
    workflow_executable,
    write_json,
)
from science_infra.control.paths import tir_agent_root
from science_infra.env import repo_root
from science_infra.control.process_manager import PROCS
from science_infra.control.llm_config import (
    resolve_legacy_llm_config,
    resolve_llm_config,
    resource_llm_config,
)
from science_infra.control.model_resources import resolve_binding
from science_infra.control.readiness import probe_llm


def _ensure_tir_on_path() -> None:
    p = str(tir_agent_root())
    if p not in sys.path:
        sys.path.insert(0, p)


def _llm_env(exp_id: str, llm: Dict[str, Any]) -> Dict[str, str]:
    return resolve_llm_config(exp_id, llm=llm).subprocess_env()


async def llm_health(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    *,
    experiment_id: str = "demo",
    model: Optional[str] = None,
    kind: Optional[str] = None,
) -> Dict[str, Any]:
    config = resolve_llm_config(
        experiment_id,
        base_url=base_url,
        api_key=api_key,
        model=model,
        kind=kind,
    )
    return await probe_llm(config, experiment_id=experiment_id)


def start_local_llm(exp_id: str, *, stop_if_running: bool = True) -> Dict[str, Any]:
    bundle = load_bundle(exp_id)
    resource = resolve_binding(exp_id, "inference")
    llm = resource.data["config"] if resource else bundle["llm"]
    effective = (
        resource_llm_config(resource)
        if resource
        else resolve_legacy_llm_config(exp_id, llm=llm)
    )
    if llm.get("kind") != "local":
        raise ValueError("llm.kind must be 'local' to start vLLM")
    model_path = str(llm.get("model_path") or llm.get("model") or "").strip()
    if not model_path:
        raise ValueError("llm.model_path required for local start")
    port = int(llm.get("port") or 8000)
    gpu_mem = float(llm.get("gpu_memory_utilization") or 0.45)
    vllm = shutil.which("vllm")
    if vllm:
        argv = [
            vllm,
            "serve",
            model_path,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(gpu_mem),
        ]
    else:
        # Fallback: python -m vllm.entrypoints.openai.api_server
        argv = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_path,
            "--port",
            str(port),
            "--gpu-memory-utilization",
            str(gpu_mem),
        ]
    base_url = f"http://127.0.0.1:{port}/v1"
    if resource is not None:
        endpoint = urlsplit(effective.base_url)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in ("127.0.0.1", "localhost")
            or (endpoint.port or 80) != port
            or endpoint.path.rstrip("/") != "/v1"
        ):
            raise ValueError(
                "本地模型资源端点必须与配置端口匹配，例如 "
                "http://127.0.0.1:8000/v1。"
            )
        base_url = effective.base_url
    if resource is None:
        from science_infra.control.experiments import save_section

        llm2 = dict(llm)
        llm2["base_url"] = base_url
        if not llm2.get("model"):
            llm2["model"] = model_path
        save_section(exp_id, "llm", llm2)

    if stop_if_running and PROCS.active("train"):
        raise RuntimeError("train is running; stop train or confirm GPU conflict before starting local LLM")

    mp = PROCS.start(
        kind="llm",
        experiment_id=exp_id,
        argv=argv,
        cwd=tir_agent_root(),
        env=effective.subprocess_env(),
        meta={
            "base_url": base_url,
            "model_path": model_path,
            "port": port,
            "model_binding": effective.public(),
        },
        replace=True,
    )
    BUS.publish(exp_id, "llm_status", {"state": "starting", "run_id": mp.run_id, "base_url": base_url})
    return {"run_id": mp.run_id, "base_url": base_url, "argv": argv}


def stop_local_llm(exp_id: str) -> Dict[str, Any]:
    st = PROCS.stop("llm")
    BUS.publish(exp_id, "llm_status", {"state": "stopped"})
    return st or {"running": False}


def list_gpus() -> Dict[str, Any]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"gpus": [], "count": 0, "error": "nvidia-smi not found"}
    try:
        out = subprocess.check_output(
            [
                smi,
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=8,
        )
    except Exception as e:
        return {"gpus": [], "count": 0, "error": str(e)}
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append(
                {
                    "id": int(parts[0]),
                    "name": parts[1],
                    "mem_total_mb": float(parts[2]),
                    "mem_used_mb": float(parts[3]),
                    "util": float(parts[4]),
                }
            )
        except (TypeError, ValueError):
            continue
    return {"gpus": gpus, "count": len(gpus)}


AGL_ORIGIN = os.environ.get("AGL_METRICS_ORIGIN", "http://127.0.0.1:4747").rstrip("/")

_AGL_SPA_PATHS = frozenset(
    {
        "",
        "metrics",
        "rollouts",
        "science",
        "resources",
        "traces",
        "runners",
        "settings",
    }
)


def agl_dashboard_dir() -> Path:
    return repo_root() / "agent-lightning" / "agentlightning" / "dashboard"


def agl_dashboard_built() -> bool:
    return (agl_dashboard_dir() / "index.html").is_file()


def agl_spa_missing_message() -> str:
    return (
        "AGL Dashboard SPA not built (agent-lightning/agentlightning/dashboard/index.html missing). "
        "Run: cd agent-lightning/dashboard && npm install && npm run build "
        "(or ./run.sh ui / ./run.sh train, which auto-build)."
    )


def is_agl_spa_path(origin_path: str) -> bool:
    """True for dashboard client routes proxied as /agl/<path> → origin /<path>."""
    p = origin_path.lstrip("/")
    if p.startswith("v1/") or p.startswith("assets/"):
        return False
    first = p.split("/", 1)[0] if p else ""
    return first in _AGL_SPA_PATHS


def agl_health() -> Dict[str, Any]:
    """Probe AGL LightningStore (only up while train is running)."""
    url = f"{AGL_ORIGIN}/v1/agl/health"
    dash_ok = agl_dashboard_built()
    try:
        r = httpx.get(url, timeout=2.0)
        out: Dict[str, Any] = {
            "ok": r.status_code < 500,
            "status_code": r.status_code,
            "origin": AGL_ORIGIN,
            "ui_path": "/agl/metrics",
            "dashboard_built": dash_ok,
        }
        if not dash_ok:
            out["dashboard_hint"] = agl_spa_missing_message()
        return out
    except Exception as e:
        out = {
            "ok": False,
            "error": str(e),
            "origin": AGL_ORIGIN,
            "ui_path": "/agl/metrics",
            "dashboard_built": dash_ok,
        }
        if not dash_ok:
            out["dashboard_hint"] = agl_spa_missing_message()
        return out


_AGL_SPA_TOS = (
    "/rollouts",
    "/metrics",
    "/science",
    "/resources",
    "/traces",
    "/runners",
    "/settings",
)


def rewrite_agl_payload(body: bytes, content_type: str) -> bytes:
    """Prefix /assets, /v1 and SPA routes so the dashboard works behind /agl."""
    ct = (content_type or "").lower()
    if not any(x in ct for x in ("text/html", "javascript", "text/css", "application/json")):
        return body
    try:
        text = body.decode("utf-8")
    except Exception:
        return body
    text = text.replace("/agl/assets/", "\x00AGL_ASSETS\x00")
    text = text.replace("/assets/", "/agl/assets/")
    text = text.replace("\x00AGL_ASSETS\x00", "/agl/assets/")
    text = text.replace("/agl/v1/", "\x00AGL_V1\x00")
    text = text.replace("/v1/", "/agl/v1/")
    text = text.replace("/agl/v1/", "\x00AGL_V1\x00")
    text = text.replace("v1/agl/", "/agl/v1/agl/")
    text = text.replace("\x00AGL_V1\x00", "/agl/v1/")
    text = text.replace("http://127.0.0.1:4747", "/agl").replace("http://localhost:4747", "/agl")
    text = text.replace('{path:"/",element:', '{path:"/agl",element:')
    text = text.replace("{path:'/',element:", "{path:'/agl',element:")
    for route in _AGL_SPA_TOS:
        text = text.replace(f'to:"{route}"', f'to:"/agl{route}"')
        text = text.replace(f"to:'{route}'", f"to:'/agl{route}'")
    return text.encode("utf-8")


def mas_palette() -> Dict[str, Any]:
    _ensure_tir_on_path()
    try:
        from workflow.plugins import REGISTRY

        skills = REGISTRY.list_skills()
        roles = REGISTRY.list_roles()
    except Exception:
        skills = ["react_loop", "verifier"]
        roles = ["hub", "verifier"]
    tools = ["web_search", "wikipedia_search", "execute_python"]
    edge_kinds = ["message", "tool_call", "feedback", "route", "sample_barrier"]
    # agent-framework W2: unified agent palette (schema 0.3). The UI renders a
    # single "Agent" drag category from these kind templates; legacy roles/tools
    # fields stay for the transition period.
    agent_templates = [
        {
            "id": "tool_agent",
            "kind": "tool",
            "label": "Tool Agent",
            "hint": "封装工具为 agent，可开 LLM 后端",
        },
        {
            "id": "blank",
            "kind": "blank",
            "label": "空白 Agent",
            "hint": "自定义 profile 多专家",
        },
        {
            "id": "verifier",
            "kind": "verifier",
            "label": "Verifier",
            "hint": "校验上游产出并反馈",
        },
        {
            "id": "planner",
            "kind": "planner",
            "label": "Planner",
            "hint": "任务分解与派发",
        },
        {
            "id": "hub",
            "kind": "hub",
            "label": "Hub",
            "hint": "ReAct 主循环入口",
        },
        {
            "id": "router",
            "kind": "router",
            "label": "Router",
            "hint": "多专家路由：从 candidates 中选择一个 agent",
        },
    ]
    tool_agents = []
    try:
        from tools.tool_agents import TOOL_AGENTS

        for _ta in TOOL_AGENTS.values():
            tool_agents.append(
                {
                    "id": _ta.id,
                    "backend": _ta.backend,
                    "llm_required": bool(_ta.llm_required),
                    "description": str(_ta.description or ""),
                }
            )
    except Exception:
        tool_agents = [
            {"id": t, "backend": "pure", "llm_required": False, "description": ""}
            for t in tools
        ]
    return {
        "skills": skills,
        "roles": roles,
        "tools": tools,
        "agent_templates": agent_templates,
        "tool_agents": tool_agents,
        "edge_kinds": edge_kinds,
        "sampling_modes": ["grpo_n", "arpo", "aepo", "appo", "rae"],
        "gate_types": [
            "entropy_delta",
            "dual_entropy",
            "always",
            "tool_ok",
            "tool_error",
            "verifier_pass",
            "verifier_fail",
            "contradiction",
            "failure_trigger",
        ],
        "templates": [
            {
                "id": "hub_react",
                "label": "Hub ReAct (executable)",
                "workflow": {
                    "schema_version": "0.1.0",
                    "topology": "hub_react",
                    "entry_agent": "hub",
                    "hub": {"role": "orchestrator", "skills": ["react_loop"]},
                    "tools": ["web_search", "wikipedia_search", "execute_python"],
                    "agents": [
                        {
                            "id": "hub",
                            "role": "orchestrator",
                            "skills": ["react_loop"],
                            "tools": ["web_search", "wikipedia_search", "execute_python"],
                            "trainable": True,
                        }
                    ],
                    "edges": [],
                },
            },
            {
                "id": "hub_verify",
                "label": "Hub + Verifier feedback",
                "workflow": {
                    "schema_version": "0.2.0",
                    "topology": "graph",
                    "entry_agent": "hub",
                    "hub": {
                        "role": "orchestrator",
                        "skills": ["react_loop"],
                        "verify": "verifier",
                        "max_feedback_hops": 1,
                    },
                    "tools": ["web_search", "wikipedia_search", "execute_python"],
                    "agents": [
                        {
                            "id": "hub",
                            "role": "orchestrator",
                            "skills": ["react_loop"],
                            "tools": ["web_search", "wikipedia_search", "execute_python"],
                            "trainable": True,
                        },
                        {"id": "verifier", "role": "verifier", "skills": ["verifier"], "trainable": False},
                    ],
                    "edges": [{"from": "verifier", "to": "hub", "kind": "feedback"}],
                },
            },
            {
                "id": "pev_draft",
                "label": "Planner-Executor-Verifier",
                "workflow": {
                    "schema_version": "0.2.0",
                    "topology": "graph",
                    "entry_agent": "planner",
                    "hub": {"role": "orchestrator", "skills": ["react_loop"]},
                    "tools": ["web_search", "execute_python"],
                    "agents": [
                        {
                            "id": "planner",
                            "role": "planner",
                            "skills": ["react_loop"],
                            "system_prompt": "You are a planner. Decompose the task, then hand off.",
                            "trainable": True,
                        },
                        {
                            "id": "executor",
                            "role": "executor",
                            "skills": ["react_loop"],
                            "tools": ["web_search", "execute_python"],
                            "system_prompt": "You are an executor. Use tools to solve the task.",
                            "trainable": True,
                        },
                        {"id": "verifier", "role": "verifier", "skills": ["verifier"], "trainable": False},
                    ],
                    "edges": [
                        {"from": "planner", "to": "executor", "kind": "route"},
                        {"from": "executor", "to": "verifier", "kind": "message"},
                        {"from": "verifier", "to": "planner", "kind": "feedback"},
                        {"from": "executor", "to": "execute_python", "kind": "tool_call"},
                        {"from": "executor", "to": "web_search", "kind": "tool_call"},
                    ],
                },
            },
        ],
    }


def sample_parquet_tasks(
    parquet: str | Path,
    *,
    n: int = 5,
    source: str = "gsm8k",
) -> List[Dict[str, Any]]:
    """Sample first N rows from a tir_agent parquet (same logic as run.sh live-api-data)."""
    import ast

    import pandas as pd

    path = Path(parquet)
    if not path.is_file():
        # Allow paths relative to tir_agent/data or tir_agent root
        cand = tir_agent_root() / parquet
        if cand.is_file():
            path = cand
        else:
            cand2 = tir_agent_root() / "data" / Path(parquet).name
            if cand2.is_file():
                path = cand2
            else:
                raise FileNotFoundError(f"parquet not found: {parquet}")
    df = pd.read_parquet(path)
    src = (source or "").strip()
    if src and src.lower() not in ("", "all", "*"):
        df = df[df["source"].astype(str) == src]
    if len(df) == 0:
        raise ValueError(f"no rows after filter source={source!r} in {path}")
    df = df.head(max(1, int(n)))
    tasks: List[Dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        answers = row.get("answers")
        if isinstance(answers, str):
            try:
                answers = ast.literal_eval(answers)
            except Exception:
                answers = [row.get("answer")]
        tasks.append(
            {
                "id": str(row.get("id") or ""),
                "question": str(row.get("question") or ""),
                "answer": str(row.get("answer") or ""),
                "answers": answers if isinstance(answers, list) else [str(row.get("answer") or "")],
                "source": str(row.get("source") or "gsm8k"),
            }
        )
    return tasks


DEMO_COLLECT_TASKS: List[Dict[str, Any]] = [
    {"id": "demo-1", "question": "What is 1+1?", "answer": "2", "source": "gsm8k", "_mock_answer": "2"},
    {"id": "demo-2", "question": "What is 2+2?", "answer": "4", "source": "gsm8k", "_mock_answer": "4"},
    {"id": "demo-3", "question": "What is 3+3?", "answer": "6", "source": "gsm8k", "_mock_answer": "6"},
    {"id": "demo-4", "question": "What is 5+7?", "answer": "12", "source": "gsm8k", "_mock_answer": "12"},
]


def _demo_collect_tasks(n: int) -> List[Dict[str, Any]]:
    """UI `n` = number of demo questions (one trajectory each), not GRPO group size."""
    want = max(1, int(n))
    out: List[Dict[str, Any]] = []
    for i in range(want):
        row = dict(DEMO_COLLECT_TASKS[i % len(DEMO_COLLECT_TASKS)])
        if i >= len(DEMO_COLLECT_TASKS):
            row["id"] = f"{row['id']}-{i + 1}"
        out.append(row)
    return out


def run_collect(
    exp_id: str,
    *,
    mock: bool = True,
    n: int = 1,
    tasks: Optional[List[Dict[str, Any]]] = None,
    algo: str = "grpo",
    parquet: Optional[str] = None,
    data_n: Optional[int] = None,
    source: str = "gsm8k",
    sequential: bool = False,
) -> Dict[str, Any]:
    ensure_experiment(exp_id)
    bundle = load_bundle(exp_id)
    wf = bundle["workflow"]
    exe = workflow_executable(wf)
    if not exe.get("ok"):
        raise RuntimeError(exe.get("reason") or "workflow not executable")

    _ensure_tir_on_path()
    from workflow import Collector, batch_to_train_signal
    from workflow.contracts import TrajectoryBatch

    spec_path = str(exp_dir(exp_id) / "workflow.yaml")
    effective = resolve_llm_config(exp_id, llm=bundle["llm"])

    if tasks is not None:
        task_list = list(tasks)
        data_meta: Dict[str, Any] = {"mode": "explicit_tasks"}
    elif parquet:
        task_list = sample_parquet_tasks(parquet, n=int(data_n or 5), source=source)
        data_meta = {
            "mode": "parquet",
            "parquet": str(parquet),
            "data_n": len(task_list),
            "source": source,
        }
        write_json(artifacts_dir(exp_id) / "tasks_sampled.json", task_list)
    else:
        task_list = _demo_collect_tasks(n)
        data_meta = {"mode": "demo", "n_tasks": len(task_list)}

    collector = Collector(
        mock=mock,
        endpoint=effective.base_url or None,
        model=effective.model or None,
        api_key=effective.api_key,
        n=1,
        spec_path=spec_path,
        archive_root=str(exp_dir(exp_id) / "artifacts" / "archives"),
    )
    BUS.publish(
        exp_id,
        "collect_progress",
        {"state": "running", "n_tasks": len(task_list), "mock": mock, **data_meta},
    )

    # Sequential collect (like run.sh live-api-data) for clearer progress on long runs
    if sequential or (parquet and not mock):
        trajs = []
        for i, task in enumerate(task_list):
            BUS.publish(
                exp_id,
                "collect_progress",
                {
                    "state": "running",
                    "index": i,
                    "total": len(task_list),
                    "task_id": task.get("id"),
                },
            )
            traj = collector.collect_one(dict(task))
            trajs.append(traj)
        vals = [float(t.final_reward) for t in trajs if t.final_reward is not None]
        mean = sum(vals) / len(vals) if vals else 0.0
        batch = TrajectoryBatch(
            trajectories=trajs,
            meta={
                "n_trajectories": len(trajs),
                "mean_reward": mean,
                "source": "control-ui",
                **data_meta,
            },
        )
    else:
        batch = collector.collect(task_list)
        batch.meta = dict(batch.meta or {})
        batch.meta.update(data_meta)

    signal = batch_to_train_signal(batch, algo=algo or str(bundle["rl"].get("algo") or "grpo"))
    payload = {
        "batch": batch.model_dump(mode="json"),
        "train_signal": {
            "advantage": signal.advantage.model_dump(mode="json"),
            "loss": signal.loss.model_dump(mode="json"),
            "meta": signal.meta,
        },
    }
    out = collect_path(exp_id)
    write_json(out, payload)
    rows = []
    for t in batch.trajectories:
        kinds = [e.kind.value if hasattr(e.kind, "value") else str(e.kind) for e in t.events]
        meta = dict(t.meta or {})
        rows.append(
            {
                "id": (t.task or {}).get("id"),
                "answer": t.final_answer,
                "reward": t.final_reward,
                "tool": "tool_call" in kinds,
                "group_id": meta.get("group_id") or (t.task or {}).get("id"),
                "sample_index": meta.get("sample_index"),
                "is_branch": bool(meta.get("is_branch") or t.branch_parent_id),
                "branch_parent_id": t.branch_parent_id,
                "sampling_mode": meta.get("sampling_mode"),
            }
        )
    BUS.publish(
        exp_id,
        "collect_progress",
        {
            "state": "done",
            "path": str(out),
            "n": batch.meta.get("n_trajectories"),
            "mean_reward": batch.meta.get("mean_reward"),
            "rows": rows,
        },
    )
    return {
        "path": str(out),
        "n": batch.meta.get("n_trajectories"),
        "mean_reward": batch.meta.get("mean_reward"),
        "train_signal": payload["train_signal"],
        "rows": rows,
        "tasks": [{"id": t.get("id"), "answer": t.get("answer")} for t in task_list],
    }


def run_diagnose(exp_id: str, *, metrics_path: Optional[str] = None) -> Dict[str, Any]:
    ensure_experiment(exp_id)
    bundle = load_bundle(exp_id)
    plugins = list(bundle["harness"].get("plugins") or [])
    cpath = collect_path(exp_id)
    if not cpath.is_file():
        raise FileNotFoundError(f"no collect.json for {exp_id}; run collect first")
    raw = json.loads(cpath.read_text(encoding="utf-8"))
    _ensure_tir_on_path()
    from workflow.contracts import TrajectoryBatch
    from workflow.harness import HARNESS

    if isinstance(raw, dict) and "batch" in raw:
        batch = TrajectoryBatch.model_validate(raw["batch"])
    else:
        batch = TrajectoryBatch.model_validate(raw)
    hyps = HARNESS.diagnose({"batch": batch, "metrics_path": metrics_path}, names=plugins or None)
    rows = [h.model_dump(mode="json") for h in hyps]
    out = diagnose_path(exp_id)
    write_json(out, rows)
    BUS.publish(exp_id, "diagnose_done", {"n": len(rows), "path": str(out)})
    return {"path": str(out), "hypotheses": rows, "n": len(rows)}


def _offline_step_rewards(limit: int = 200) -> List[Dict[str, Any]]:
    """Step-level reward series from the last train run's metrics.jsonl.

    Path resolution order:
    1. ``AGL_METRICS_JSONL`` env (if the service itself was started with it).
    2. Scan train run stdout logs (newest first) for an ``AGL_METRICS_JSONL=``
       line — the training glue prints it at startup.
    3. Glob ``mas/checkpoints/AgentLightning/*/metrics.jsonl`` (newest first).
    """
    import glob as _glob
    import re as _re

    candidates: List[str] = []
    env_path = os.environ.get("AGL_METRICS_JSONL", "").strip()
    if env_path:
        candidates.append(env_path)
    pat = _re.compile(r"^AGL_METRICS_JSONL=(\S+)", _re.MULTILINE)
    for row in PROCS.list_runs():
        if row.get("kind") != "train":
            continue
        lp = row.get("log_path") or ""
        try:
            text = Path(lp).read_text(encoding="utf-8", errors="replace")[:200_000]
        except OSError:
            continue
        m = pat.search(text)
        if m:
            candidates.append(m.group(1))
    repo = Path(__file__).resolve().parents[2]
    candidates.extend(sorted(_glob.glob(str(repo / "mas" / "checkpoints" / "AgentLightning" / "*" / "metrics.jsonl")), key=lambda p: os.path.getmtime(p), reverse=True))
    seen = set()
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in seen or not os.path.isfile(cand):
            seen.add(cand)
            continue
        seen.add(cand)
        out: List[Dict[str, Any]] = []
        try:
            with open(cand, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    val = rec.get("training/reward")
                    if val is None:
                        continue
                    try:
                        step_idx = int(rec.get("training/global_step") or len(out) + 1)
                        # Metrics from restarted runs may repeat global_step
                        # (e.g. 1,2,1 after a crash); keep the series monotonic
                        # so the chart x-axis stays well-formed.
                        if out and step_idx <= out[-1]["index"]:
                            step_idx = out[-1]["index"] + 1
                        out.append(
                            {
                                "index": step_idx,
                                "reward": float(val),
                            }
                        )
                    except (TypeError, ValueError):
                        continue
        except OSError:
            continue
        if out:
            return out[-limit:]
    return []


def fetch_agl_training_metrics() -> Dict[str, Any]:
    """Same payload as AGL dashboard GET /v1/agl/metrics (only while store is up)."""
    try:
        r = httpx.get(
            f"{AGL_ORIGIN}/v1/agl/metrics",
            params={"rollout_limit": 200, "step_limit": 200},
            timeout=3.0,
        )
        if r.status_code >= 400:
            return {}
        data = r.json()
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _agl_reward_series(agl: Dict[str, Any]) -> Dict[str, Any]:
    train_rewards: List[Dict[str, Any]] = []
    for i, row in enumerate(agl.get("rollout_rewards") or []):
        if not isinstance(row, dict):
            continue
        rew = row.get("final_reward")
        if rew is None:
            continue
        try:
            train_rewards.append(
                {
                    "index": len(train_rewards) + 1,
                    "id": str(row.get("rollout_id") or i),
                    "reward": float(rew),
                    "status": row.get("status"),
                    "mode": row.get("mode"),
                }
            )
        except (TypeError, ValueError):
            continue
    step_rewards: List[Dict[str, Any]] = []
    for i, step in enumerate(agl.get("training_steps") or []):
        if not isinstance(step, dict):
            continue
        val = step.get("critic/rewards/mean")
        if val is None:
            val = step.get("training/reward")
        if val is None:
            continue
        try:
            step_rewards.append(
                {
                    "index": int(step.get("training/global_step") or i + 1),
                    "reward": float(val),
                }
            )
        except (TypeError, ValueError):
            continue
    offline = False
    if not train_rewards and not step_rewards:
        # AGL store is down (training stopped). Fall back to the metrics.jsonl
        # the training process sinks to disk (path is printed in run stdout
        # as AGL_METRICS_JSONL=...). Without this, the Monitor reward charts
        # go blank whenever training is not actively running.
        offline_series = _offline_step_rewards()
        if offline_series:
            step_rewards = offline_series
            offline = True
    return {
        "train_rewards": train_rewards,
        "step_rewards": step_rewards,
        "agl_online": bool(agl),
        "agl_source": "offline_metrics_jsonl" if offline else ("agl_store" if agl else None),
    }


def monitor_model(exp_id: str) -> Dict[str, Any]:
    from science_infra.ui.dashboard import build_dashboard_model

    cpath = collect_path(exp_id)
    dpath = diagnose_path(exp_id)
    agl_series = _agl_reward_series(fetch_agl_training_metrics())
    if not cpath.is_file():
        model: Dict[str, Any] = {
            "empty": True,
            "n": 0,
            "rewards": [],
            "trajectories": [],
            "hypotheses": [],
            "mean_reward": None,
        }
    else:
        raw = json.loads(cpath.read_text(encoding="utf-8"))
        _ensure_tir_on_path()
        from workflow.contracts import TrajectoryBatch

        train_signal = None
        if isinstance(raw, dict) and "batch" in raw:
            batch = TrajectoryBatch.model_validate(raw["batch"])
            train_signal = raw.get("train_signal")
        else:
            batch = TrajectoryBatch.model_validate(raw)
        hyps: List[Any] = []
        if dpath.is_file():
            from workflow.harness import Hypothesis

            for row in json.loads(dpath.read_text(encoding="utf-8")):
                hyps.append(Hypothesis.model_validate(row))
        else:
            try:
                from workflow.harness import HARNESS

                hyps = HARNESS.diagnose({"batch": batch})
            except Exception:
                hyps = []
        model = build_dashboard_model(batch, hyps, train_signal=train_signal)
    bundle = load_bundle(exp_id)
    model["agl_metrics_url"] = bundle["meta"].get("agl_metrics_url")
    model["experiment_id"] = exp_id
    model.update(agl_series)
    if agl_series.get("train_rewards") or agl_series.get("step_rewards"):
        model["empty"] = False
    return model


def _sync_workflow_sampling_into_rl(rl: Dict[str, Any], workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Map MASSpec.sampling onto rl.yaml so train_tir_agent sees tir_algo / rollout.n / tir.*."""
    sampling = workflow.get("sampling") if isinstance(workflow, dict) else None
    if not sampling:
        return rl
    _ensure_tir_on_path()
    root = str(tir_agent_root().parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from rl.hooks.overlay import apply_sample_policy

    synced = apply_sample_policy(rl, sampling)
    tir_algo = str((synced.get("algorithm") or {}).get("tir_algo") or synced.get("algo") or "grpo").lower()
    synced["algo"] = tir_algo
    n = (synced.get("actor_rollout_ref") or {}).get("rollout", {}).get("n")
    if n is not None:
        synced["rollout_per_gpu"] = int(n)
    return synced


def start_train(
    exp_id: str,
    *,
    stop_llm: bool = True,
    confirm_gpu: bool = False,
) -> Dict[str, Any]:
    from science_infra.control.experiments import save_section
    from science_infra.control.training import build_training_plan, persisted_rl

    plan = build_training_plan(exp_id)
    if not plan.ready:
        raise RuntimeError(
            "训练启动检查未通过："
            + "；".join(issue["message"] for issue in plan.blocking_issues)
        )
    bundle = plan.bundle
    rl = plan.rl
    ids = plan.gpu_ids
    saved_rl = persisted_rl(plan)
    save_section(
        exp_id,
        "rl",
        {key: value for key, value in saved_rl.items() if not str(key).startswith("_")},
    )
    algo = plan.algorithm
    if algo not in VALID_ALGOS:
        raise ValueError(f"algo must be one of {VALID_ALGOS}")
    profile = plan.profile
    if profile not in ("fast", "a800", "a800_2gpu"):
        raise ValueError("profile must be fast|a800|a800_2gpu")

    if PROCS.active("llm"):
        if stop_llm or confirm_gpu:
            PROCS.stop("llm")
            BUS.publish(exp_id, "llm_status", {"state": "stopped_for_train"})
        else:
            raise RuntimeError("local LLM running; pass stop_llm=true or confirm_gpu=true")

    rl_yaml = exp_dir(exp_id) / "rl.yaml"
    train_script = tir_agent_root() / "train_tir_agent.py"
    argv = [
        sys.executable,
        str(train_script),
        profile,
        "--algo",
        algo,
        "--rl-yaml",
        str(rl_yaml),
    ]
    if rl.get("n_runners") is not None:
        argv.extend(["--n-runners", str(int(rl["n_runners"]))])
    model_path = plan.source.model_path
    if model_path:
        argv.extend(["--model", str(model_path)])

    _ensure_tir_on_path()
    try:
        from workflow.spec import MASSpec
        from workflow.compiler import trainable_agents

        spec = MASSpec.model_validate(bundle["workflow"])
        names = trainable_agents(spec)
        if names:
            argv.extend(["--active-agent", names[0]])
    except Exception:
        pass

    env = _llm_env(exp_id, bundle["llm"])
    root = str(tir_agent_root().parent)
    env["PYTHONPATH"] = os.pathsep.join(
        [root, str(tir_agent_root()), env.get("PYTHONPATH", "")]
    )
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in ids)
    env["VLLM_USE_V1"] = env.get("VLLM_USE_V1") or "1"
    note = rl.get("_profile_downgraded")
    mp = PROCS.start(
        kind="train",
        experiment_id=exp_id,
        argv=argv,
        cwd=tir_agent_root(),
        env=env,
        meta={
            "profile": profile,
            "algo": algo,
            "rl_yaml": str(rl_yaml),
            "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
            "n_gpus": len(ids),
            "downgrade": note,
            "training_source": plan.source.public(),
            "preflight_revision": plan.revision,
        },
        replace=True,
    )
    return {
        "run_id": mp.run_id,
        "argv": argv,
        "log_path": str(mp.log_path),
        "cuda_visible_devices": env["CUDA_VISIBLE_DEVICES"],
        "n_gpus": len(ids),
        "profile": profile,
        "note": note,
    }


def stop_train(exp_id: str) -> Dict[str, Any]:
    st = PROCS.stop("train")
    BUS.publish(exp_id, "train_stopped", {})
    return st or {"running": False}
