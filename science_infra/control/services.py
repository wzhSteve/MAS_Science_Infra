"""Control services: LLM health/start, MAS collect, harness diagnose, RL train."""

from __future__ import annotations

import json
import mimetypes
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

import httpx
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
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ]
    # Align EPC-AW wrap completion budget with the local vLLM window.
    os.environ.setdefault("SCIENCE_EPC_AW_MAX_MODEL_LEN", "8192")
    os.environ.setdefault("SCIENCE_EPC_AW_MAX_COMPLETION", "2048")
    os.environ.setdefault("MAS_MAX_COMPLETION_TOKENS", "2048")
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


def _llm_log_tail(exp_id: str, run_id: str, *, limit: int = 12) -> str:
    proc = PROCS.get(run_id)
    path = proc.log_path if proc is not None else PROCS.run_dir(exp_id, run_id) / "stdout.log"
    if not path.is_file():
        return ""
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    return " | ".join(lines[-limit:])[:800]


def _served_model_ready(models: List[str], served_name: str) -> bool:
    if not served_name:
        return bool(models)
    return served_name in models or any(served_name in name for name in models)


def ensure_local_llm_ready(exp_id: str, *, timeout: float = 180.0) -> Dict[str, Any]:
    """Start local vLLM when needed and block until /v1/models lists the served model."""
    bundle = load_bundle(exp_id)
    resource = resolve_binding(exp_id, "inference")
    llm = resource.data["config"] if resource else bundle["llm"]
    if llm.get("kind") != "local":
        return {"started": False, "reused": False, "reason": "not_local"}
    model_path = str(llm.get("model_path") or llm.get("model") or "").strip()
    port = int(llm.get("port") or 8000)
    served_name = str(llm.get("model") or (Path(model_path).name if model_path else ""))
    base_url = f"http://127.0.0.1:{port}/v1"
    existing = _openai_models(base_url) or []
    if _served_model_ready(existing, served_name):
        return {
            "started": False,
            "reused": True,
            "base_url": base_url,
            "models": existing,
            "run_id": "reused",
        }
    if model_path and not Path(model_path).expanduser().exists():
        raise RuntimeError(f"本地模型目录不存在：{model_path}")
    started = start_local_llm(exp_id)
    if started.get("reused"):
        models = list(started.get("models") or existing)
        return {
            "started": False,
            "reused": True,
            "base_url": str(started.get("base_url") or base_url),
            "models": models,
            "run_id": started.get("run_id") or "reused",
        }
    run_id = str(started.get("run_id") or "")
    deadline = time.time() + max(5.0, float(timeout))
    while time.time() < deadline:
        models = _openai_models(base_url) or []
        if _served_model_ready(models, served_name):
            BUS.publish(
                exp_id,
                "llm_status",
                {"state": "ready", "run_id": run_id, "base_url": base_url, "models": models},
            )
            return {
                "started": True,
                "reused": False,
                "base_url": base_url,
                "models": models,
                "run_id": run_id,
            }
        proc = PROCS.get(run_id) if run_id else None
        if proc is not None and proc.popen is not None and proc.popen.poll() is not None:
            detail = _llm_log_tail(exp_id, run_id)
            raise RuntimeError(f"vLLM 进程已退出（code {proc.popen.returncode}）。{detail}")
        time.sleep(1.0)
    detail = _llm_log_tail(exp_id, run_id) if run_id else ""
    raise RuntimeError(f"等待 vLLM 就绪超时（{int(timeout)}s）。{detail}")


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


def rewrite_agl_payload(body: bytes, content_type: str) -> bytes:
    """Prefix bare ``/assets/`` so an older dashboard bundle still loads behind ``/agl``.

    Route paths and ``/v1/agl`` are left alone. The dashboard is built with Vite
    ``base: /agl/`` and a router basename, and it calls same-origin ``/v1/agl/*``.
    """
    ct = (content_type or "").lower()
    if not any(x in ct for x in ("text/html", "javascript", "text/css")):
        return body
    try:
        text = body.decode("utf-8")
    except Exception:
        return body
    text = text.replace("/agl/assets/", "\x00AGL_ASSETS\x00")
    text = text.replace("/assets/", "/agl/assets/")
    text = text.replace("\x00AGL_ASSETS\x00", "/agl/assets/")
    return text.encode("utf-8")


def is_agl_metrics_api(origin_path: str) -> bool:
    """True for the training-metrics API, not the dashboard ``/metrics`` page."""
    path = origin_path.strip("/")
    return path == "v1/agl/metrics" or path.endswith("/v1/agl/metrics")


def read_agl_static(origin_path: str) -> Optional[tuple]:
    """Built dashboard bytes for a proxied ``/agl/<path>`` when the store is down."""
    if not agl_dashboard_built():
        return None
    rel = origin_path.lstrip("/")
    root = agl_dashboard_dir().resolve()
    if rel.startswith("assets/"):
        target = (root / rel).resolve()
        if root != target and root not in target.parents:
            return None
        if not target.is_file():
            return None
        media = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        return target.read_bytes(), media
    if rel in ("",) or is_agl_spa_path(origin_path):
        index = root / "index.html"
        if not index.is_file():
            return None
        return index.read_bytes(), "text/html"
    return None


def mas_palette() -> Dict[str, Any]:
    _ensure_tir_on_path()
    try:
        from workflow.plugins import REGISTRY

        skills = REGISTRY.list_skills()
        roles = REGISTRY.list_roles()
    except Exception:
        skills = ["verifier"]
        roles = ["planner", "verifier"]
    from tools.tool_agents import PALETTE_TOOL_IDS

    tools = list(PALETTE_TOOL_IDS)
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
    # Bing and Wikipedia do not open result pages, so they have no pro tier.
    _TIERED_TOOL_IDS: set = set()

    def _default_fetch_n() -> int:
        try:
            return max(1, int(os.getenv("MAS_PRO_FETCH_N", "3") or 3))
        except (TypeError, ValueError):
            return 3

    try:
        from tools.tool_agents import PALETTE_TOOL_IDS, TOOL_AGENTS

        tools = list(PALETTE_TOOL_IDS)
        for tid in tools:
            _ta = TOOL_AGENTS.get(tid)
            if _ta is None:
                continue
            _has_tiers = tid in _TIERED_TOOL_IDS
            tool_agents.append(
                {
                    "id": _ta.id,
                    "backend": _ta.backend,
                    "llm_required": bool(_ta.llm_required),
                    "description": str(_ta.description or ""),
                    "has_tiers": _has_tiers,
                    "tiers": ["lite", "pro"] if _has_tiers else [],
                    "default_fetch_n": _default_fetch_n() if _has_tiers else 0,
                }
            )
    except Exception:
        tool_agents = [
            {
                "id": t,
                "backend": "llm",
                "llm_required": True,
                "description": "",
                "has_tiers": t in _TIERED_TOOL_IDS,
                "tiers": ["lite", "pro"] if t in _TIERED_TOOL_IDS else [],
                "default_fetch_n": _default_fetch_n() if t in _TIERED_TOOL_IDS else 0,
            }
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


def run_tool_agent_test(
    agent_id: str,
    *,
    tier: Optional[str] = None,
    args: Optional[Dict[str, Any]] = None,
    fetch_top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """Invoke a tool-agent directly for the UI test dialog.

    Resolves the tier, runs ``TOOL_AGENTS[agent_id].invoke`` and reports
    ``resolved_tier`` parsed from the ``tier=`` marker in the output. Errors
    are captured into ``ok=False`` instead of raising so the UI can render
    them. Unknown agent ids raise ``KeyError`` for the caller to map to 404.
    """
    import time


    _ensure_tir_on_path()
    from tools.tool_agents import TOOL_AGENTS, validate_tool_args

    agent = TOOL_AGENTS.get(agent_id)
    if agent is None:
        raise KeyError(agent_id)

    requested = str(tier or "lite").strip().lower()
    if requested not in ("lite", "pro"):
        raise ValueError(f"tier must be 'lite' or 'pro', got {tier!r}")
    payload: Dict[str, Any] = dict(args or {})
    # Request-level schema check (design §6.1): invalid args are a client
    # error (HTTP 400) raised before any invoke; runtime invoke failures stay
    # ok=False / HTTP 200 on purpose.
    args_ok, args_reason = validate_tool_args(agent_id, payload)
    if not args_ok:
        raise ValueError(f"invalid args for {agent_id}: {args_reason}")
    if fetch_top_n is not None and requested == "pro":
        # Clamp to the 1-10 contract window (design §5.2) BEFORE merging into
        # the copied args dict; None / non-int values stay unmerged so the
        # pro_tools default chain (args -> MAS_PRO_FETCH_N -> 3) applies.
        try:
            top_n = int(fetch_top_n)
        except (TypeError, ValueError):
            top_n = None
        if top_n is not None:
            payload["fetch_top_n"] = max(1, min(10, top_n))

    start = time.perf_counter()
    try:
        output = str(agent.invoke(payload, tier=requested))
        ok = True
    except Exception as exc:  # noqa: BLE001
        output = f"error: {exc}"
        ok = False
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Output contract: every agent prefixes its first line with "tier=<lite|pro>"
    # (pro_tools additionally marks degradations with "pro_fallback").
    resolved = requested
    first_line = output.split("\n", 1)[0].strip()
    if first_line.startswith("tier="):
        marker = first_line[5:].split()[0].strip().lower() if first_line[5:].split() else ""
        if marker in ("lite", "pro"):
            resolved = marker
    if not ok:
        resolved = requested

    return {
        "requested_tier": requested,
        "resolved_tier": resolved,
        "output": output,
        "elapsed_ms": elapsed_ms,
        "ok": ok,
    }


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


_LOSS_FIELDS = (
    ("actor/pg_loss", "pg_loss"),
    ("actor/entropy_loss", "entropy_loss"),
    ("actor/kl_loss", "kl_loss"),
)


def _metric_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_step(rec: Dict[str, Any], fallback: int) -> int:
    for key in ("training/global_step", "step"):
        raw = rec.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    return fallback


def _bump_index(index: int, previous: Optional[int]) -> int:
    # Restarted runs may repeat global_step (1, 2, 1). Keep the x-axis monotonic.
    if previous is not None and index <= previous:
        return previous + 1
    return index


def _parse_metrics_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split jsonl / store rows into reward, validation reward, and loss series."""
    step_rewards: List[Dict[str, Any]] = []
    val_rewards: List[Dict[str, Any]] = []
    step_losses: List[Dict[str, Any]] = []
    last_reward: Optional[int] = None
    last_val: Optional[int] = None
    last_loss: Optional[int] = None
    for rec in records:
        if not isinstance(rec, dict):
            continue
        reward = _metric_float(rec.get("training/reward"))
        if reward is None:
            reward = _metric_float(rec.get("critic/rewards/mean"))
        if reward is not None:
            index = _bump_index(_metric_step(rec, len(step_rewards) + 1), last_reward)
            last_reward = index
            step_rewards.append({"index": index, "reward": reward})
        val = _metric_float(rec.get("val/reward"))
        if val is not None:
            index = _bump_index(_metric_step(rec, len(val_rewards) + 1), last_val)
            last_val = index
            val_rewards.append({"index": index, "reward": val})
        loss_point: Dict[str, Any] = {}
        for src, dest in _LOSS_FIELDS:
            num = _metric_float(rec.get(src))
            if num is not None:
                loss_point[dest] = num
        if loss_point:
            index = _bump_index(_metric_step(rec, len(step_losses) + 1), last_loss)
            last_loss = index
            loss_point["index"] = index
            step_losses.append(loss_point)
    return {
        "step_rewards": step_rewards,
        "val_rewards": val_rewards,
        "step_losses": step_losses,
    }


def _empty_metrics() -> Dict[str, Any]:
    return {
        "step_rewards": [],
        "val_rewards": [],
        "step_losses": [],
        "rows": [],
        "path": None,
        "score": 0,
    }


def _read_metrics_file(path: str, limit: int = 2000) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if "step" not in rec and rec.get("training/global_step") is not None:
                    try:
                        rec = {**rec, "step": int(rec["training/global_step"])}
                    except (TypeError, ValueError):
                        pass
                rows.append(rec)
    except OSError:
        return _empty_metrics()
    if limit > 0 and len(rows) > limit:
        rows = rows[-limit:]
    parsed = _parse_metrics_records(rows)
    score = len(parsed["step_rewards"]) + len(parsed["val_rewards"]) + len(parsed["step_losses"])
    return {**parsed, "rows": rows, "path": os.path.abspath(path), "score": score}


def _metrics_candidate_paths(
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> tuple:
    """Return ``(ordered existing paths, pinned paths)``.

    Pinned paths belong to this experiment: stdout ``AGL_METRICS_JSONL=`` lines
    and checkpoint dirs whose name contains ``experiment_name``.
    """
    import glob as _glob
    import re as _re

    candidates: List[str] = []
    pinned: set = set()
    pat = _re.compile(r"^AGL_METRICS_JSONL=(\S+)", _re.MULTILINE)
    runs = [r for r in PROCS.list_runs(experiment_id) if r.get("kind") == "train"]
    runs = sorted(runs, key=lambda r: float(r.get("started_at") or 0.0), reverse=True)
    for row in runs:
        lp = row.get("log_path") or ""
        try:
            text = Path(lp).read_text(encoding="utf-8", errors="replace")[:200_000]
        except OSError:
            continue
        match = pat.search(text)
        if match:
            path = os.path.abspath(match.group(1))
            candidates.append(path)
            pinned.add(path)

    env_path = os.environ.get("AGL_METRICS_JSONL", "").strip()
    if env_path:
        candidates.append(os.path.abspath(env_path))

    repo = Path(__file__).resolve().parents[2]
    globbed = sorted(
        _glob.glob(str(repo / "mas" / "checkpoints" / "AgentLightning" / "*" / "metrics.jsonl"))
        + _glob.glob(str(repo / "checkpoints" / "AgentLightning" / "*" / "metrics.jsonl")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if experiment_name:
        preferred = [p for p in globbed if experiment_name in Path(p).parent.name]
        for path in preferred:
            pinned.add(os.path.abspath(path))
        candidates.extend(preferred)
    candidates.extend(globbed)

    ordered: List[str] = []
    seen = set()
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in seen or not os.path.isfile(cand):
            seen.add(cand)
            continue
        seen.add(cand)
        ordered.append(cand)
    return ordered, pinned


def _select_metrics_file(
    *,
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
    limit: int = 2000,
) -> Dict[str, Any]:
    """Pick the fullest metrics.jsonl for this experiment.

    A named experiment never falls through to another experiment's file.
    Among pinned files, the one with the most reward/loss points wins.
    """
    ordered, pinned = _metrics_candidate_paths(experiment_id, experiment_name)
    if experiment_name or pinned:
        pool = [path for path in ordered if path in pinned] if experiment_name else ordered
        if experiment_name and not pool:
            return _empty_metrics()
    else:
        pool = ordered
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[tuple] = None
    for path in pool:
        parsed = _read_metrics_file(path, limit=limit)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            mtime = 0.0
        key = (int(parsed["score"]), mtime)
        if best is None or best_key is None or key > best_key:
            best = parsed
            best_key = key
    return best or _empty_metrics()


def _select_recent_metrics(*, limit: int = 2000) -> Dict[str, Any]:
    """Fullest file among the few newest checkpoints.

    A one-line file touched last must not hide the completed run beside it,
    and an older unrelated file with more rows must not win on length alone.
    """
    ordered, _pinned = _metrics_candidate_paths()
    parsed: List[Dict[str, Any]] = []
    for path in ordered:
        item = _read_metrics_file(path, limit=limit)
        if int(item["score"]) <= 0:
            continue
        try:
            item["mtime"] = os.path.getmtime(path)
        except OSError:
            item["mtime"] = 0.0
        parsed.append(item)
    parsed.sort(key=lambda item: float(item["mtime"]), reverse=True)
    recent = parsed[:5]
    if not recent:
        return _empty_metrics()
    return max(recent, key=lambda item: (int(item["score"]), float(item["mtime"])))


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


def _agl_reward_series(
    agl: Dict[str, Any],
    *,
    experiment_id: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Episode rewards from the store; step reward/loss from the fuller series.

    A handful of live ``final_reward`` rows must not hide the jsonl curves.
    When both the store and disk have step rows, the longer series is kept.
    """
    train_rewards: List[Dict[str, Any]] = []
    for i, row in enumerate(agl.get("rollout_rewards") or []):
        if not isinstance(row, dict):
            continue
        rew = _metric_float(row.get("final_reward"))
        if rew is None:
            continue
        train_rewards.append(
            {
                "index": len(train_rewards) + 1,
                "id": str(row.get("rollout_id") or i),
                "reward": rew,
                "status": row.get("status"),
                "mode": row.get("mode"),
            }
        )
    live_steps = [step for step in (agl.get("training_steps") or []) if isinstance(step, dict)]
    live = _parse_metrics_records(live_steps)
    disk = _select_metrics_file(experiment_id=experiment_id, experiment_name=experiment_name)
    live_score = len(live["step_rewards"]) + len(live["val_rewards"]) + len(live["step_losses"])
    use_disk = int(disk["score"]) > 0 and int(disk["score"]) >= live_score
    chosen = disk if use_disk else live
    if use_disk and live_score > 0:
        source = "merged"
    elif use_disk:
        source = "offline_metrics_jsonl"
    elif agl:
        source = "agl_store"
    else:
        source = None
    return {
        "train_rewards": train_rewards,
        "step_rewards": chosen["step_rewards"],
        "val_rewards": chosen["val_rewards"],
        "step_losses": chosen["step_losses"],
        "metrics_path": disk.get("path") if use_disk else None,
        "agl_online": bool(agl),
        "agl_source": source,
    }


def local_training_metrics(
    *,
    step_limit: int = 2000,
    experiment_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Dashboard ``GET /v1/agl/metrics`` payload from metrics.jsonl.

    ``experiment_id`` pins the file the same way Monitor does. Without it, the
    fullest of the most recently touched checkpoints is used.
    """
    if step_limit <= 0:
        step_limit = 2000
    disk = _empty_metrics()
    if experiment_id:
        try:
            bundle = load_bundle(experiment_id)
            trainer = (bundle.get("rl") or {}).get("trainer") or {}
            experiment_name = str(trainer.get("experiment_name") or "") or None
            disk = _select_metrics_file(
                experiment_id=experiment_id,
                experiment_name=experiment_name,
                limit=step_limit,
            )
        except Exception:
            disk = _empty_metrics()
    if int(disk.get("score") or 0) <= 0:
        disk = _select_recent_metrics(limit=step_limit)
    ordered, _pinned = _metrics_candidate_paths(experiment_id if experiment_id else None)
    rows = list(disk["rows"][-step_limit:]) if disk.get("rows") else []
    # A restarted run can append step 1 again. The dashboard charts key on `step`,
    # so keep that field monotonic or the last update plots on top of step 1.
    last_step: Optional[int] = None
    normalized: List[Dict[str, Any]] = []
    for rec in rows:
        step = _bump_index(_metric_step(rec, (last_step or 0) + 1), last_step)
        last_step = step
        if rec.get("step") != step:
            rec = {**rec, "step": step}
        normalized.append(rec)
    return {
        "rollout_rewards": [],
        "training_steps": normalized,
        "source": {
            "metrics_jsonl": disk.get("path"),
            "exists": bool(disk.get("path")),
            "candidates": ordered[:10],
        },
    }


def monitor_model(exp_id: str) -> Dict[str, Any]:
    from science_infra.ui.dashboard import build_dashboard_model

    cpath = collect_path(exp_id)
    dpath = diagnose_path(exp_id)
    bundle = load_bundle(exp_id)
    rl_cfg = bundle.get("rl") if isinstance(bundle, dict) else {}
    experiment_name = None
    try:
        experiment_name = str((rl_cfg.get("trainer") or {}).get("experiment_name") or "") or None
    except AttributeError:
        experiment_name = None
    agl_series = _agl_reward_series(
        fetch_agl_training_metrics(),
        experiment_id=exp_id,
        experiment_name=experiment_name,
    )
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
    if any(agl_series.get(key) for key in ("train_rewards", "step_rewards", "val_rewards", "step_losses")):
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

    explicit = rl.get("algo") or (rl.get("algorithm") or {}).get("tir_algo")
    synced = apply_sample_policy(rl, sampling, algo=str(explicit) if explicit else None)
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
