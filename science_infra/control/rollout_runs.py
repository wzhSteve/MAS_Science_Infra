"""Single-task MAS rollout runs for the Control UI."""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional
from uuid import uuid4

import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from science_infra.control.experiments import (
    artifacts_dir,
    ensure_experiment,
    exp_dir,
    load_bundle,
    require_experiment,
    workflow_executable,
)
from science_infra.control.llm_config import resolve_llm_config, resource_llm_config
from science_infra.control.model_resources import get_resource
from science_infra.control.paths import tir_agent_root
from science_infra.control.process_manager import PROCS
from science_infra.control.training_logs import _redact

router = APIRouter(tags=["rollout-runs"])
RUN_ID = re.compile(r"^[a-f0-9]{16,64}$")
_eval_guard = threading.Lock()
_eval_running: Dict[str, str] = {}
_eval_cancel: Dict[str, threading.Event] = {}
_eval_release_llm: set[str] = set()


class RolloutTask(BaseModel):
    id: Optional[str] = None
    question: str


class RolloutRunRequest(BaseModel):
    workflow: Dict[str, Any]
    task: RolloutTask
    execution: Literal["mock", "live"] = "live"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runs_root(exp_id: str) -> Path:
    return artifacts_dir(exp_id) / "rollout-runs"


def _run_dir(exp_id: str, run_id: str) -> Path:
    if not RUN_ID.fullmatch(run_id):
        raise HTTPException(404, "运行记录不存在")
    root = _runs_root(exp_id).resolve()
    path = (root / run_id).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(404, "运行记录路径无效")
    return path


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(404, "运行记录不存在") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(500, "运行记录读取失败或内容损坏") from error
    if not isinstance(raw, dict):
        raise HTTPException(500, "运行记录格式无效")
    return raw


def _ensure_tir_on_path() -> None:
    path = str(tir_agent_root())
    if path not in sys.path:
        sys.path.insert(0, path)


def _agent_llms(workflow: Dict[str, Any]):
    from workflow.runtime import LLMConfig

    agent_llms: dict[str, LLMConfig] = {}
    for agent in workflow.get("agents") or []:
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
                api_key=config.api_key or "EMPTY",
            )
        else:
            model_path = str(resource.data["config"]["model_path"])
            agent_llms[agent_id] = LLMConfig(
                endpoint="http://127.0.0.1:8000/v1",
                model=model_path,
                api_key="EMPTY",
            )
    return agent_llms


def _counts(traj: Any) -> tuple[int, int]:
    kinds = [str(m.get("kind") or "") for m in (traj.messages or [])]
    if any(kinds):
        model_n = sum(1 for k in kinds if k in ("plan_step", "verify", "route_decision"))
        tool_n = sum(1 for k in kinds if k in ("tool_result", "tool_invoke"))
        if model_n or tool_n:
            return model_n, tool_n
    model_n = 0
    tool_n = 0
    for event in traj.events or []:
        kind = event.kind.value if hasattr(event.kind, "value") else str(event.kind)
        payload = event.payload or {}
        if kind == "tool_call":
            tool_n += 1
        elif kind in ("agent_message", "model_call") and payload.get("phase") != "enter":
            model_n += 1
    return model_n, tool_n


def _summary_from(
    *,
    run_id: str,
    execution: str,
    started_at: str,
    finished_at: Optional[str],
    status: str,
    traj: Any | None,
    model: Optional[Dict[str, Any]],
    error: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    model_n = tool_n = 0
    final_answer = None
    format_ok = None
    trajectory_id = None
    termination = None
    trace_status: str = "unavailable"
    if traj is not None:
        model_n, tool_n = _counts(traj)
        final_answer = traj.final_answer
        format_ok = bool(traj.format_ok)
        trajectory_id = traj.trajectory_id
        trace_status = "complete" if traj.events or traj.messages else "partial"
        if traj.meta.get("error"):
            termination = "error"
        elif final_answer:
            termination = "completed"
        else:
            termination = "no_answer"
    return {
        "run_id": run_id,
        "trajectory_id": trajectory_id,
        "status": status,
        "execution": execution,
        "started_at": started_at,
        "finished_at": finished_at,
        "model": model,
        "final_answer": final_answer,
        "termination_reason": termination,
        "format_ok": format_ok,
        "model_call_count": model_n,
        "tool_call_count": tool_n,
        "trace_status": trace_status,
        "error": error,
    }


def _load_record(exp_id: str, run_id: str) -> Dict[str, Any]:
    require_experiment(exp_id)
    return _read_json(_run_dir(exp_id, run_id) / "record.json")


def _context(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "run": record["run"],
        "workflow": record.get("workflow") or {},
        "task": record.get("task") or {"id": "", "question": ""},
        "model_binding": record.get("model_binding"),
        "redaction_applied": True,
    }


@router.post("/api/mas/rollout-runs")
def create_rollout(body: RolloutRunRequest, experiment_id: str = Query("demo")) -> Dict[str, Any]:
    ensure_experiment(experiment_id)
    question = (body.task.question or "").strip()
    if not question:
        raise HTTPException(422, "请填写任务问题。")
    exe = workflow_executable(body.workflow)
    if not exe.get("ok"):
        raise HTTPException(400, exe.get("reason") or "workflow 不可执行")

    run_id = uuid4().hex
    started_at = _now()
    run_dir = _run_dir(experiment_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    task_id = str(body.task.id or f"rollout-{run_id[:8]}")
    task = {"id": task_id, "question": question}
    spec_path = run_dir / "workflow.yaml"
    spec_path.write_text(
        yaml.safe_dump(body.workflow, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    bundle = load_bundle(experiment_id)
    effective = resolve_llm_config(experiment_id, llm=bundle["llm"])
    model_public = effective.public()
    model_summary = {
        "source": model_public.get("kind") or "api",
        "name": model_public.get("model") or "",
        "policy_version": None,
    }
    mock = body.execution == "mock"
    if not mock:
        issues = effective.issues(require_model=True)
        if issues:
            raise HTTPException(400, "；".join(item["message"] for item in issues))

    _ensure_tir_on_path()
    from workflow.runtime import ExecutionService, LLMConfig

    llm = None
    if not mock:
        llm = LLMConfig(
            endpoint=effective.base_url or "",
            model=effective.model,
            api_key=effective.api_key or "EMPTY",
        )
    error: Optional[Dict[str, str]] = None
    traj = None
    status = "succeeded"
    try:
        svc = ExecutionService(
            mock=mock,
            spec_path=str(spec_path),
            llm=llm,
            agent_llms=_agent_llms(body.workflow),
            archive_root=str(exp_dir(experiment_id) / "artifacts" / "archives"),
        )
        traj = svc.run(task)
        if traj.meta.get("error"):
            status = "failed"
            details = traj.meta.get("error_details") or {}
            error = {
                "stage": str(details.get("stage") or "graph_execution"),
                "code": str(details.get("error_type") or "execution_error"),
                "message": str(traj.meta.get("error") or "执行失败"),
            }
    except Exception as exc:  # noqa: BLE001
        status = "failed"
        error = {
            "stage": "graph_execution",
            "code": type(exc).__name__,
            "message": str(exc)[:1000],
        }

    finished_at = _now()
    summary = _summary_from(
        run_id=run_id,
        execution=body.execution,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        traj=traj,
        model=None if mock else model_summary,
        error=error,
    )
    record = {
        "run": summary,
        "workflow": body.workflow,
        "task": task,
        "model_binding": None if mock else model_public,
        "trajectory": traj.model_dump(mode="json") if traj is not None else None,
    }
    _atomic_json(run_dir / "record.json", _redact(record))
    return summary


@router.get("/api/mas/rollout-runs")
def list_rollouts(
    experiment_id: str = Query("demo"),
    limit: int = Query(15, ge=1, le=50),
    cursor: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        require_experiment(experiment_id)
    except FileNotFoundError as error:
        raise HTTPException(404, "实验不存在") from error
    root = _runs_root(experiment_id)
    items: List[Dict[str, Any]] = []
    if root.is_dir():
        for child in root.iterdir():
            record_path = child / "record.json"
            if not child.is_dir() or not record_path.is_file():
                continue
            try:
                record = _read_json(record_path)
            except HTTPException:
                items.append(
                    {
                        "run_id": child.name,
                        "run": None,
                        "question_preview": "",
                        "record_error": "记录损坏",
                    }
                )
                continue
            run = record.get("run") or {}
            question = str((record.get("task") or {}).get("question") or "")
            items.append(
                {
                    "run_id": str(run.get("run_id") or child.name),
                    "run": run,
                    "question_preview": question[:80],
                    "record_error": None,
                }
            )
    items.sort(
        key=lambda row: (
            str((row.get("run") or {}).get("started_at") or ""),
            str(row.get("run_id") or ""),
        ),
        reverse=True,
    )
    if cursor:
        idx = next((i for i, row in enumerate(items) if row["run_id"] == cursor), None)
        items = items[idx + 1 :] if idx is not None else items
    page = items[:limit]
    next_cursor = page[-1]["run_id"] if len(items) > limit else None
    return {"items": page, "next_cursor": next_cursor}


@router.get("/api/mas/rollout-runs/{run_id}")
def get_rollout(run_id: str, experiment_id: str = Query("demo")) -> Dict[str, Any]:
    return _load_record(experiment_id, run_id)["run"]


@router.get("/api/mas/rollout-runs/{run_id}/context")
def get_context(run_id: str, experiment_id: str = Query("demo")) -> Dict[str, Any]:
    return _context(_load_record(experiment_id, run_id))


@router.get("/api/mas/rollout-runs/{run_id}/trajectory")
def get_trajectory(
    run_id: str,
    experiment_id: str = Query("demo"),
    view: Optional[str] = None,
) -> Dict[str, Any]:
    record = _load_record(experiment_id, run_id)
    payload = _context(record)
    trajectory = record.get("trajectory") or {"events": [], "messages": [], "meta": {}}
    events = list(trajectory.get("events") or [])
    preview = view == "preview"
    if preview:
        events = events[:100]
    payload.update(
        {
            "trajectory": {**trajectory, "events": events},
            "preview": preview,
            "event_page": {
                "offset": 0,
                "total": len(record.get("trajectory", {}).get("events") or []),
                "next_offset": 100 if preview and len(record.get("trajectory", {}).get("events") or []) > 100 else None,
            },
            "snapshots": [],
        }
    )
    return payload


class EvalRunRequest(BaseModel):
    split: Literal["val", "test"] = "val"
    path: str = ""
    limit: int = 20


def registered_dataset(path: str) -> str:
    """Return a catalog path, or raise ValueError when it is not registered or missing."""
    from science_infra.control.dataset_resources import _read

    wanted = str(path or "").strip()
    match = next((item.path for item in _read().items if item.path == wanted), None)
    if not match:
        raise ValueError("请从模型与数据中选择数据集")
    if not Path(match).expanduser().is_file():
        raise ValueError(f"数据集不存在：{match}")
    return match


def eval_parquet_paths(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Validation file from rl.data.val_files.

    ``test_files`` wins when set. Otherwise the test file is ``test.parquet`` beside val.
    """
    data = dict((bundle.get("rl") or {}).get("data") or {})
    raw = str(data.get("val_files") or "").strip()
    explicit = str(data.get("test_files") or "").strip()
    val_path = Path(raw).expanduser() if raw else Path()
    if explicit:
        test_path = Path(explicit).expanduser()
    else:
        test_path = val_path.with_name("test.parquet") if raw else Path()
    return {
        "val_files": str(val_path) if raw else "",
        "val_exists": bool(raw) and val_path.is_file(),
        "test_files": str(test_path) if explicit or raw else "",
        "test_exists": bool(explicit or raw) and test_path.is_file(),
    }


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


@contextmanager
def _bind_episode_llm(llm: Any) -> Iterator[None]:
    """Overwrite process env so EPC-AW engines use this experiment endpoint, not Assistant/.env leftovers."""
    prev = {key: os.environ.get(key) for key in _LLM_ENV_KEYS}
    prev_extra = {
        key: os.environ.get(key)
        for key in (
            "SCIENCE_EPC_AW_MAX_MODEL_LEN",
            "SCIENCE_EPC_AW_MAX_COMPLETION",
            "MAS_MAX_COMPLETION_TOKENS",
        )
    }
    endpoint = str(getattr(llm, "endpoint", "") or "").rstrip("/")
    model = str(getattr(llm, "model", "") or "").strip()
    api_key = str(getattr(llm, "api_key", "") or "").strip() or "EMPTY"
    try:
        if endpoint:
            os.environ["OPENAI_API_BASE"] = endpoint
            os.environ["OPENAI_BASE_URL"] = endpoint
            os.environ["OPENAI_API_BASE_URL"] = endpoint
        if model:
            os.environ["OPENAI_MODEL"] = model
            os.environ["MODEL"] = model
            os.environ["MODEL_Name"] = model
            os.environ["MODEL_NAME"] = model
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ.setdefault("SCIENCE_EPC_AW_MAX_MODEL_LEN", "8192")
        os.environ.setdefault("SCIENCE_EPC_AW_MAX_COMPLETION", "2048")
        os.environ.setdefault("MAS_MAX_COMPLETION_TOKENS", "2048")
        yield
    finally:
        for key, value in prev.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        for key, value in prev_extra.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _execute_question(
    experiment_id: str,
    workflow: Dict[str, Any],
    task: Dict[str, Any],
    *,
    llm: Any,
    model_summary: Optional[Dict[str, Any]],
    model_public: Optional[Dict[str, Any]],
    on_window: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    run_id = uuid4().hex
    started_at = _now()
    run_dir = _run_dir(experiment_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "workflow.yaml"
    spec_path.write_text(yaml.safe_dump(workflow, allow_unicode=True, sort_keys=False), encoding="utf-8")
    from workflow.runtime import ExecutionService

    error: Optional[Dict[str, str]] = None
    traj = None
    status = "succeeded"
    window_token = None
    try:
        # Soft-reset EPC-AW / HIVE episode context so consecutive eval questions do not share outline/memory.
        ctx_mod = sys.modules.get("context")
        if ctx_mod is not None and hasattr(ctx_mod, "get_episode_context"):
            try:
                ctx_mod.get_episode_context(reset=True)
            except Exception:
                pass
        if on_window is not None:
            from workflow.user_gateway.loader import push_window_logger

            window_token = push_window_logger(on_window)
        svc = ExecutionService(
            mock=False,
            spec_path=str(spec_path),
            llm=llm,
            agent_llms=_agent_llms(workflow),
            archive_root=str(exp_dir(experiment_id) / "artifacts" / "archives"),
        )
        with _bind_episode_llm(llm):
            traj = svc.run(task)
        if traj.meta.get("error"):
            status = "failed"
            details = traj.meta.get("error_details") or {}
            error = {
                "stage": str(details.get("stage") or "graph_execution"),
                "code": str(details.get("error_type") or "execution_error"),
                "message": str(traj.meta.get("error") or "执行失败"),
            }
    except Exception as exc:  # noqa: BLE001
        from workflow.user_gateway.sandbox import EvalCancelled

        if isinstance(exc, EvalCancelled):
            raise
        status = "failed"
        error = {"stage": "graph_execution", "code": type(exc).__name__, "message": str(exc)[:1000]}
    finally:
        if window_token is not None:
            from workflow.user_gateway.loader import pop_window_logger

            pop_window_logger(window_token)
    finished_at = _now()
    summary = _summary_from(
        run_id=run_id,
        execution="live",
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        traj=traj,
        model=model_summary,
        error=error,
    )
    record = {
        "run": summary,
        "workflow": workflow,
        "task": {"id": task.get("id"), "question": task.get("question")},
        "model_binding": model_public,
        "trajectory": traj.model_dump(mode="json") if traj is not None else None,
    }
    _atomic_json(run_dir / "record.json", _redact(record))
    return summary, traj


@router.get("/api/mas/eval-sources")
def eval_sources(experiment_id: str = Query("demo")) -> Dict[str, Any]:
    try:
        bundle = load_bundle(experiment_id)
    except FileNotFoundError as error:
        raise HTTPException(404, "实验不存在") from error
    return eval_parquet_paths(bundle)


def _eval_log_limit() -> int:
    raw = os.environ.get("SCIENCE_EVAL_LOG_LIMIT") or "3000"
    try:
        return max(200, int(raw))
    except ValueError:
        return 3000


def _clip(text: Any, limit: int = 180) -> str:
    if isinstance(text, dict) and "error" in text:
        return json.dumps(text, ensure_ascii=False, default=str)[: max(limit, _eval_log_limit())]
    return " ".join(str(text or "").split())[:limit]


def _rule(title: str) -> str:
    body = f" {title} "
    return f"──{body}{'─' * max(8, 42 - len(body))}"


def _kv(label: str, value: Any, *, limit: int = 220) -> str:
    return f"{label:<6}{_clip(value, limit)}"


def _payload_brief(kind: str, payload: Any) -> str:
    if not isinstance(payload, dict):
        return _clip(payload, 160)
    if kind == "plan_step":
        return (
            f"next={payload.get('next')}  "
            f"goal={_clip(payload.get('sub_goal'), 80)}  done={payload.get('done')}"
        )
    if kind == "tool_invoke":
        return _clip(payload, 160)
    if kind == "tool_result":
        mark = "成功" if payload.get("ok") else "失败"
        return f"{mark}  {_clip(payload.get('output'), 160)}"
    if kind == "verify":
        state = "可停" if payload.get("ready_to_stop") else str(payload.get("step_conclusion") or "")
        return f"{state}  {_clip(payload.get('reason'), 120)}"
    if kind == "feedback":
        return _clip(payload.get("reason"), 160)
    if kind in ("final_answer", "error"):
        return _clip(payload.get("text") or payload.get("answer") or payload.get("error") or payload, 200)
    return _clip(payload, 160)


def _expand_payload_trace(prefix: str, payload: Any) -> List[str]:
    if not isinstance(payload, dict):
        return []
    trace = payload.get("trace")
    if not isinstance(trace, dict):
        return []
    limit = min(160, _eval_log_limit())
    lines: List[str] = []
    outline = trace.get("outline")
    if isinstance(outline, dict):
        for key, val in outline.items():
            lines.append(f"{prefix}   outline.{key} = {_clip(val, limit)}")
    elif outline:
        lines.append(f"{prefix}   outline = {_clip(outline, limit)}")
    tool = trace.get("tool") or trace.get("tool_name")
    if tool:
        lines.append(f"{prefix}   tool = {_clip(tool, limit)}")
    goal = trace.get("sub_goal") or trace.get("context")
    if goal:
        lines.append(f"{prefix}   goal = {_clip(goal, limit)}")
    bts = trace.get("bts")
    if isinstance(bts, dict):
        parts = [f"n={bts.get('n', '')}", f"selected={bts.get('selected', '')}"]
        if bts.get("planner_selected") is not None:
            parts.append(f"planner_selected={bts.get('planner_selected')}")
        lines.append(f"{prefix}   bts {' '.join(parts)}")
    for key, label in (
        ("command", "command"),
        ("output", "output"),
        ("analysis", "analysis"),
        ("explanation", "explanation"),
        ("new_info", "new_info"),
        ("conclusion", "conclusion"),
        ("outline_updated", "outline'"),
        ("llm_error", "llm_error"),
    ):
        val = trace.get(key)
        if val is None or val == "":
            continue
        if key == "llm_error":
            text = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False, default=str)
            lines.append(f"{prefix}   {label} = {_clip(text, limit)}")
            continue
        if key == "outline_updated" and isinstance(val, dict):
            items = [f"{k}={_clip(v, 80)}" for k, v in val.items()]
            lines.append(f"{prefix}   {label} = {'; '.join(items)}")
            continue
        lines.append(f"{prefix}   {label} = {_clip(val, limit)}")
    return lines


def format_eval_trace(prefix: str, messages: List[Dict[str, Any]], *, detail: bool = False) -> List[str]:
    """Compact hop list. Set detail=True to expand traces (download / debug)."""
    lines: List[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("src") and msg.get("dst"):
            kind = str(msg.get("kind") or "message")
            lines.append(
                f"{prefix} t{msg.get('turn', 0)}  "
                f"{msg.get('src')} → {msg.get('dst')}  {kind}  "
                f"{_payload_brief(kind, msg.get('payload'))}"
            )
            if detail:
                lines.extend(_expand_payload_trace(prefix, msg.get("payload")))
            continue
        role = str(msg.get("role") or msg.get("type") or "")
        calls = msg.get("tool_calls") if isinstance(msg.get("tool_calls"), list) else []
        if calls:
            for call in calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                name = call.get("name") or fn.get("name") or "tool"
                args = call.get("args") or call.get("arguments") or fn.get("arguments") or {}
                lines.append(f"{prefix} {role or 'assistant'} → {name}  tool_call  {_clip(args, 160)}")
            continue
        if role == "tool":
            lines.append(
                f"{prefix} {msg.get('name') or 'tool'} → assistant  tool_result  {_clip(msg.get('content'), 160)}"
            )
            continue
        content = msg.get("content")
        if role and content:
            lines.append(f"{prefix} {role}  {_clip(content, 160)}")
    if not lines:
        lines.append(f"{prefix} （本次没有 agent 消息）")
    return lines


def _run_eval_job(
    experiment_id: str,
    run_id: str,
    *,
    workflow: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    llm: Any,
    model_summary: Dict[str, Any],
    model_public: Optional[Dict[str, Any]],
    split: str,
    parquet: str,
    local_llm: bool = False,
) -> None:
    from rl.rewards.outcome import accuracy

    correct = 0
    scored = 0
    cancel = _eval_cancel.get(run_id)
    llm_started_by_us = False
    from workflow.user_gateway.sandbox import EvalCancelled, bind_eval_cancel, reset_eval_cancel

    cancel_token = bind_eval_cancel(cancel)
    try:
        total = len(tasks)
        model_name = str((model_public or {}).get("model") or model_summary.get("name") or "")
        model_base = str((model_public or {}).get("base_url") or "")
        PROCS.append_log(experiment_id, run_id, _rule("MAS 测试"))
        PROCS.append_log(experiment_id, run_id, _kv("数据集", f"{split} · {parquet}"))
        PROCS.append_log(experiment_id, run_id, _kv("题目数", total))
        if model_name or model_base:
            PROCS.append_log(
                experiment_id,
                run_id,
                _kv("模型", f"{model_name}" + (f" @ {model_base}" if model_base else "")),
            )
        if local_llm:
            PROCS.append_log(experiment_id, run_id, _kv("本地", "正在检查 vLLM"))
            try:
                from science_infra.control.services import ensure_local_llm_ready

                ready = ensure_local_llm_ready(experiment_id)
                llm_started_by_us = bool(ready.get("started")) and not bool(ready.get("reused"))
            except Exception as exc:  # noqa: BLE001
                PROCS.append_log(experiment_id, run_id, _kv("失败", f"vLLM 启动失败：{exc}"))
                PROCS.finish_inline(experiment_id, run_id, state="failed", message=str(exc)[:500], returncode=1)
                return
            models = "、".join(str(item) for item in (ready.get("models") or []))
            state = "已在服务" if ready.get("reused") else "已启动"
            PROCS.append_log(
                experiment_id,
                run_id,
                _kv(
                    "vLLM",
                    f"{state} {ready.get('base_url') or ''}" + (f" · {models}" if models else ""),
                ),
            )
        for index, task in enumerate(tasks, start=1):
            if cancel is not None and cancel.is_set():
                PROCS.append_log(experiment_id, run_id, "")
                PROCS.append_log(experiment_id, run_id, _rule("已停止"))
                PROCS.append_log(experiment_id, run_id, _kv("统计", f"已完成 {index - 1}/{total} 题后停止"))
                PROCS.finish_inline(
                    experiment_id,
                    run_id,
                    state="cancelled",
                    message=f"用户停止 · 已完成 {index - 1}/{total}",
                    returncode=0,
                )
                return
            question = _clip(task.get("question"), 280)
            PROCS.append_log(experiment_id, run_id, "")
            PROCS.append_log(experiment_id, run_id, _rule(f"题目 {index}/{total}"))
            PROCS.append_log(experiment_id, run_id, _kv("编号", task.get("id") or "-"))
            PROCS.append_log(experiment_id, run_id, _kv("问题", question))

            def _on_window(line: str, _index: int = index, _total: int = total) -> None:
                if cancel is not None and cancel.is_set():
                    return
                text = str(line or "")
                if text.startswith("· ") or text.startswith("  "):
                    PROCS.append_log(experiment_id, run_id, f"[{_index}/{_total}] {text}")
                else:
                    PROCS.append_log(experiment_id, run_id, f"[{_index}/{_total}]   {text}")

            summary, traj = _execute_question(
                experiment_id, workflow, task, llm=llm, model_summary=model_summary, model_public=model_public,
                on_window=_on_window,
            )
            gold = str(task.get("answer") or "")
            aliases = task.get("answers") if isinstance(task.get("answers"), list) else None
            matched = None
            if gold or aliases:
                score = accuracy(
                    summary.get("final_answer"),
                    gold,
                    source=str(task.get("source") or "gsm8k"),
                    aliases=aliases,
                )
                matched = score > 0
                scored += 1
                if matched:
                    correct += 1
            error = (summary.get("error") or {}).get("message") if isinstance(summary.get("error"), dict) else None
            verdict = "正确" if matched is True else "错误" if matched is False else summary.get("status") or "完成"
            PROCS.append_log(experiment_id, run_id, _rule(f"结果 {index}/{total}"))
            PROCS.append_log(experiment_id, run_id, _kv("判定", verdict))
            PROCS.append_log(
                experiment_id,
                run_id,
                _kv("答案", _clip(summary.get("final_answer"), 240) or "无"),
            )
            if gold:
                PROCS.append_log(experiment_id, run_id, _kv("标准", _clip(gold, 240)))
            if error:
                PROCS.append_log(experiment_id, run_id, _kv("错误", _clip(error, 240)))
            del traj
            if cancel is not None and cancel.is_set():
                PROCS.append_log(experiment_id, run_id, "")
                PROCS.append_log(experiment_id, run_id, _rule("已停止"))
                PROCS.append_log(experiment_id, run_id, _kv("统计", f"已完成 {index}/{total} 题后停止 · 答对 {correct}"))
                PROCS.finish_inline(
                    experiment_id,
                    run_id,
                    state="cancelled",
                    message=f"用户停止 · 已完成 {index}/{total}",
                    returncode=0,
                )
                return
        PROCS.append_log(experiment_id, run_id, "")
        PROCS.append_log(experiment_id, run_id, _rule("汇总"))
        message = f"完成 {total} 条 · 有标准答案 {scored} · 答对 {correct}"
        PROCS.append_log(experiment_id, run_id, _kv("统计", message))
        PROCS.finish_inline(experiment_id, run_id, state="succeeded", returncode=0)
    except EvalCancelled:
        PROCS.append_log(experiment_id, run_id, "")
        PROCS.append_log(experiment_id, run_id, _rule("已停止"))
        PROCS.append_log(experiment_id, run_id, _kv("统计", "用户停止，已中断当前题目"))
        PROCS.finish_inline(
            experiment_id,
            run_id,
            state="cancelled",
            message="用户停止",
            returncode=0,
        )
    except Exception as exc:  # noqa: BLE001
        message = f"测试失败：{exc}"
        PROCS.append_log(experiment_id, run_id, _kv("失败", message))
        PROCS.finish_inline(experiment_id, run_id, state="failed", message=str(exc)[:500], returncode=1)
    finally:
        reset_eval_cancel(cancel_token)
        with _eval_guard:
            release_llm = run_id in _eval_release_llm
            _eval_release_llm.discard(run_id)
        if llm_started_by_us or release_llm:
            try:
                from science_infra.control.services import stop_local_llm

                stop_local_llm(experiment_id)
                note = "用户停止，本地 vLLM 已关闭" if release_llm else "本轮启动的实例已停止"
                PROCS.append_log(experiment_id, run_id, _kv("vLLM", note))
            except Exception as exc:  # noqa: BLE001
                PROCS.append_log(experiment_id, run_id, _kv("vLLM", f"停止失败：{exc}"))
        with _eval_guard:
            if _eval_running.get(experiment_id) == run_id:
                _eval_running.pop(experiment_id, None)
            _eval_cancel.pop(run_id, None)


@router.post("/api/mas/eval-runs")
def create_eval_run(body: EvalRunRequest, experiment_id: str = Query("demo")) -> Dict[str, Any]:
    """Run the saved MAS on val.parquet or the sibling test.parquet. Does not train."""
    try:
        bundle = load_bundle(experiment_id)
    except FileNotFoundError as error:
        raise HTTPException(404, "实验不存在") from error
    split = body.split
    if body.path.strip():
        try:
            parquet = registered_dataset(body.path)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
    else:
        paths = eval_parquet_paths(bundle)
        parquet = paths["test_files"] if split == "test" else paths["val_files"]
        exists = paths["test_exists"] if split == "test" else paths["val_exists"]
        if not exists:
            label = "测试集 test.parquet" if split == "test" else "验证集"
            raise HTTPException(400, f"{label}不存在：{parquet or '未配置 data.val_files'}")
    workflow = bundle.get("workflow") or {}
    exe = workflow_executable(workflow)
    if not exe.get("ok"):
        raise HTTPException(400, exe.get("reason") or "workflow 不可执行")
    limit = max(1, min(int(body.limit or 20), 200))
    from science_infra.control.services import sample_parquet_tasks

    try:
        tasks = sample_parquet_tasks(parquet, n=limit, source="all")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, str(exc)) from exc

    effective = resolve_llm_config(experiment_id, llm=bundle["llm"])
    issues = effective.issues(require_model=True)
    if issues:
        raise HTTPException(400, "；".join(item["message"] for item in issues))
    model_public = effective.public()
    model_summary = {"source": model_public.get("kind") or "api", "name": model_public.get("model") or "", "policy_version": None}
    _ensure_tir_on_path()
    from workflow.runtime import LLMConfig

    llm = LLMConfig(endpoint=effective.base_url or "", model=effective.model, api_key=effective.api_key or "EMPTY")
    with _eval_guard:
        active = _eval_running.get(experiment_id)
        if active:
            raise HTTPException(409, f"已有测试在运行：{active}")
        row = PROCS.start_inline(
            kind="eval",
            experiment_id=experiment_id,
            meta={"split": split, "parquet": parquet, "limit": limit},
        )
        _eval_running[experiment_id] = row["run_id"]
        _eval_cancel[row["run_id"]] = threading.Event()
    threading.Thread(
        target=_run_eval_job,
        args=(experiment_id, row["run_id"]),
        kwargs={
            "workflow": workflow,
            "tasks": tasks,
            "llm": llm,
            "model_summary": model_summary,
            "model_public": model_public,
            "split": split,
            "parquet": parquet,
            "local_llm": effective.kind == "local",
        },
        daemon=True,
    ).start()
    return {
        "run_id": row["run_id"],
        "state": "running",
        "split": split,
        "parquet": parquet,
        "limit": limit,
    }


@router.post("/api/mas/eval-runs/{run_id}/stop")
def stop_eval_run(run_id: str, experiment_id: str = Query("demo")) -> Dict[str, Any]:
    """Collaborative cancel for MAS eval. Also stops the local vLLM for this experiment."""
    if not re.fullmatch(r"[a-f0-9]{8,64}", run_id):
        raise HTTPException(400, "run_id 无效")
    try:
        require_experiment(experiment_id)
    except FileNotFoundError as error:
        raise HTTPException(404, "实验不存在") from error
    with _eval_guard:
        active = _eval_running.get(experiment_id)
        cancel = _eval_cancel.get(run_id)
        if active != run_id or cancel is None:
            row = PROCS.disk_run(run_id, experiment_id) or {}
            state = str(row.get("state") or "")
            if state in {"cancelled", "succeeded", "failed", "interrupted"}:
                return {"run_id": run_id, "state": state, "message": "测试已结束"}
            raise HTTPException(404, "没有正在运行的测试，或 run_id 不匹配")
        cancel.set()
        _eval_release_llm.add(run_id)
    from workflow.user_gateway.sandbox import reap_search_browsers

    reap_search_browsers()
    PROCS.update_inline(experiment_id, run_id, state="stopping", message="正在停止测试…")
    PROCS.append_log(experiment_id, run_id, _kv("停止", "收到停止测试请求，正在关闭本地 vLLM"))
    try:
        from science_infra.control.services import stop_local_llm

        stop_local_llm(experiment_id)
        PROCS.append_log(experiment_id, run_id, _kv("vLLM", "本地 vLLM 已关闭"))
    except Exception as exc:  # noqa: BLE001
        PROCS.append_log(experiment_id, run_id, _kv("vLLM", f"关闭失败：{exc}"))
    return {"run_id": run_id, "state": "stopping"}


@router.get("/api/mas/rollout-runs/{run_id}/export")
def export_rollout(run_id: str, experiment_id: str = Query("demo")) -> Response:
    record = _redact(_load_record(experiment_id, run_id))
    body = json.dumps(record, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="rollout-{run_id}.json"'},
    )
