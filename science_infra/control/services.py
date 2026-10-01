"""Control services: LLM health/start, MAS collect, harness diagnose, RL train."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit
from uuid import uuid4

import psutil

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
from science_infra.control.model_resources import get_resource, resolve_binding
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


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pids_listening(port: int) -> List[int]:
    hexport = f"{port:04X}".lower()
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="utf-8") as handle:
                next(handle, None)
                for line in handle:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    local = parts[1].rsplit(":", 1)
                    if len(local) != 2 or local[1].lower() != hexport:
                        continue
                    if parts[3] != "0A":
                        continue
                    inodes.add(parts[9])
        except OSError:
            continue
    if not inodes:
        return []
    found: set[int] = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            for link in fd_dir.iterdir():
                try:
                    target = os.readlink(link)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    found.add(int(entry.name))
                    break
        except OSError:
            continue
    return sorted(found)


def _kill_process_tree(pid: int, *, timeout: float = 20.0) -> None:
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = proc.children(recursive=True)
    for item in [proc, *children]:
        try:
            item.terminate()
        except psutil.Error:
            continue
    _, alive = psutil.wait_procs([proc, *children], timeout=timeout)
    for item in alive:
        try:
            item.kill()
        except psutil.Error:
            continue


def _gpu_vllm_pids() -> List[int]:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        out = subprocess.check_output(
            [smi, "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
            timeout=8,
        )
    except Exception:
        return []
    pids: List[int] = []
    for line in out.splitlines():
        raw = line.strip().split(",")[0].strip()
        if not raw.isdigit():
            continue
        pid = int(raw)
        try:
            proc = psutil.Process(pid)
        except psutil.NoSuchProcess:
            continue
        blob_parts: List[str] = []
        current: Optional[psutil.Process] = proc
        for _ in range(8):
            if current is None:
                break
            try:
                blob_parts.append(" ".join(current.cmdline()).lower())
                current = current.parent()
            except psutil.Error:
                break
        blob = " ".join(blob_parts)
        if "science-infra" in blob or "ray::" in blob:
            continue
        if (
            "vllm" in blob
            or "openai.api_server" in blob
            or "served-model-name" in blob
            or "multiprocessing.spawn" in blob
        ):
            pids.append(pid)
    return pids


def _free_local_llm_port(port: int) -> List[int]:
    """Stop managed vLLM and any leftover listener/GPU workers on this port."""
    killed: List[int] = []
    try:
        PROCS.stop("llm", timeout=20.0, reason="replaced")
    except Exception:
        pass
    for pid in _pids_listening(port):
        _kill_process_tree(pid)
        killed.append(pid)
    for pid in _gpu_vllm_pids():
        _kill_process_tree(pid)
        killed.append(pid)
    deadline = time.time() + 45
    while time.time() < deadline:
        if not _port_open("127.0.0.1", port) and not _gpu_vllm_pids():
            break
        time.sleep(0.4)
    return sorted(set(killed))


def _openai_models(base_url: str, timeout: float = 3.0) -> Optional[List[str]]:
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return None
    ids: List[str] = []
    for item in payload.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def _reuse_local_llm(
    exp_id: str,
    *,
    llm: Dict[str, Any],
    resource: Any,
    base_url: str,
    model_path: str,
    models: List[str],
) -> Dict[str, Any]:
    served = str(llm.get("model") or Path(model_path).name)
    if resource is None:
        from science_infra.control.experiments import save_section

        llm2 = dict(llm)
        llm2["kind"] = "local"
        llm2["base_url"] = base_url
        if not llm2.get("model"):
            llm2["model"] = served
        save_section(exp_id, "llm", llm2)
    BUS.publish(
        exp_id,
        "llm_status",
        {"state": "ready", "run_id": "reused", "base_url": base_url, "reused": True, "models": models},
    )
    return {
        "run_id": "reused",
        "base_url": base_url,
        "reused": True,
        "models": models,
        "argv": [],
    }


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
    served_name = str(llm.get("model") or Path(model_path).name)
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

    managed = PROCS.active("llm")
    existing = _openai_models(base_url)
    same_model = bool(existing) and (
        served_name in existing or any(served_name in name for name in existing)
    )
    if managed is not None and same_model:
        return _reuse_local_llm(
            exp_id, llm=llm, resource=resource, base_url=base_url,
            model_path=model_path, models=existing or [served_name],
        )

    _free_local_llm_port(port)
    if _port_open("127.0.0.1", port):
        raise RuntimeError(f"端口 {port} 仍被占用，无法为所选本地模型启动 vLLM。")

    extra = [
        "--served-model-name",
        served_name,
        "--trust-remote-code",
        "--host",
        "0.0.0.0",
        "--max-model-len",
        "8192",
    ]
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
            *extra,
        ]
    else:
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
            *extra,
        ]
    if resource is None:
        from science_infra.control.experiments import save_section

        llm2 = dict(llm)
        llm2["base_url"] = base_url
        if not llm2.get("model"):
            llm2["model"] = served_name
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
    return {"run_id": mp.run_id, "base_url": base_url, "argv": argv, "reused": False}


def stop_local_llm(exp_id: str) -> Dict[str, Any]:
    bundle = load_bundle(exp_id)
    resource = resolve_binding(exp_id, "inference")
    llm = resource.data["config"] if resource else bundle["llm"]
    port = int(llm.get("port") or 8000)
    killed = _free_local_llm_port(port)
    BUS.publish(exp_id, "llm_status", {"state": "stopped", "killed": killed})
    return {"running": False, "killed": killed}


async def list_llm_options(
    exp_id: str,
    *,
    kind: str = "local",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    port: Optional[int] = None,
) -> Dict[str, Any]:
    """List models the UI can pick: local weights under LLM/, or remote /v1/models."""
    kind = (kind or "local").strip().lower()
    if kind == "local":
        from science_infra.control.training import discover_local_models

        listen_port = int(port or 8000)
        serving_url = f"http://127.0.0.1:{listen_port}/v1"
        served = _openai_models(serving_url) or []
        served_set = {str(name) for name in served}
        items: List[Dict[str, Any]] = []
        for candidate in discover_local_models():
            name = str(candidate.get("name") or "")
            items.append(
                {
                    "id": str(candidate.get("path") or name),
                    "name": name,
                    "source": "local",
                    "path": candidate.get("path"),
                    "model_type": candidate.get("model_type"),
                    "architectures": list(candidate.get("architectures") or []),
                    "served": name in served_set or any(name and name in s for s in served_set),
                }
            )
        local_root = str(repo_root() / "LLM")
        return {
            "kind": "local",
            "local_root": local_root,
            "items": items,
            "serving": {"base_url": serving_url, "models": served} if served else None,
            "message": None if items else f"在 {local_root} 下没有发现带 config.json 的本地模型。",
        }

    probe = await probe_llm(
        resolve_legacy_llm_config(
            exp_id,
            llm={"kind": "api", "base_url": base_url or "", "model": ""},
            base_url=base_url,
            api_key=api_key,
            kind="api",
            model="",
        ),
        experiment_id=exp_id,
    )
    items = [
        {"id": str(mid), "name": str(mid), "source": "api", "served": False}
        for mid in probe.get("models") or []
    ]
    return {
        "kind": "api",
        "items": items,
        "probe": probe,
        "serving": None,
        "message": probe.get("message"),
    }


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
        skills = ["verifier"]
        roles = ["planner", "verifier"]
    tools = ["wikipedia_search", "google_search", "web_search", "python_coder", "think"]
    edge_kinds = ["message", "tool_call", "feedback", "route", "sample_barrier"]
    # agent-framework W2: unified agent palette (schema 0.3). The UI renders a
    # single "Agent" drag category from these kind templates; legacy roles/tools
    # fields stay for the transition period.
    agent_templates = [
        {
            "id": "planner",
            "kind": "planner",
            "label": "Planner",
            "hint": "任务分解，输出 sub_goal 与下游 tool/子任务",
        },
        {
            "id": "verifier",
            "kind": "verifier",
            "label": "Verifier",
            "hint": "校验 tool-agent 产出并反馈",
        },
        {
            "id": "blank",
            "kind": "blank",
            "label": "自定义 Agent",
            "hint": "JSON Schema 自描述的空白 agent",
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
            {"id": t, "backend": "llm", "llm_required": True, "description": ""}
            for t in tools
        ]
    user_projects = _mas_palette_user_projects()
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
        "templates": _mas_palette_templates(),
        "user_projects": user_projects,
    }


def _mas_palette_templates() -> List[Dict[str, Any]]:
    """Load GraphPalette templates from ``mas/specs/templates/*.yaml``."""
    _ensure_tir_on_path()
    from workflow.templates import list_palette_templates

    return list_palette_templates()


def _mas_palette_user_projects() -> List[Dict[str, Any]]:
    try:
        from science_infra.control.user_projects import palette_user_projects

        return palette_user_projects()
    except Exception:
        return []


def sample_parquet_tasks(
    parquet: str | Path,
    *,
    n: int = 5,
    source: str = "gsm8k",
) -> List[Dict[str, Any]]:
    """Sample the first N rows from a parquet or JSON task file."""
    from science_infra.control.task_files import load_task_rows

    rows = load_task_rows(parquet)
    src = (source or "").strip()
    if src and src.lower() not in ("", "all", "*"):
        rows = [row for row in rows if str(row.get("source") or "") == src]
    if not rows:
        raise ValueError(f"no rows after filter source={source!r} in {parquet}")
    return rows[: max(1, int(n))]


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
    from workflow.runtime import LLMConfig

    spec_path = str(exp_dir(exp_id) / "workflow.yaml")
    effective = resolve_llm_config(exp_id, llm=bundle["llm"])
    agent_llms: dict[str, LLMConfig] = {}
    for agent in wf.get("agents") or []:
        if not isinstance(agent, dict):
            continue
        agent_id = str(agent.get("id") or "").strip()
        model_ref = str(agent.get("model") or "inherit").strip()
        if not agent_id or not model_ref or model_ref == "inherit":
            continue
        resource = get_resource(model_ref)
        if resource.data["type"] == "inference":
            config = resource_llm_config(resource)
            agent_llms[agent_id] = LLMConfig(
                endpoint=config.base_url,
                model=config.model,
                api_key=config.api_key,
            )
        else:
            model_path = str(resource.data["config"]["model_path"])
            agent_llms[agent_id] = LLMConfig(
                endpoint="http://127.0.0.1:8000/v1",
                model=model_path,
            )

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
        agent_llms=agent_llms,
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
    from science_infra.control.training import build_training_plan, launch_training

    plan = build_training_plan(exp_id)
    return launch_training(
        exp_id,
        request_id=uuid4().hex,
        preflight_revision=plan.revision,
        stop_local_llm=bool(stop_llm or confirm_gpu),
    )


def stop_train(exp_id: str) -> Dict[str, Any]:
    from science_infra.control.training import stop_training_run

    active = PROCS.active("train")
    if active is None or active.experiment_id != exp_id:
        return {"running": False}
    return stop_training_run(exp_id, active.run_id)
