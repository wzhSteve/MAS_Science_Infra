"""Load user entry modules without putting ``user_space`` on ``sys.path``."""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from contextvars import ContextVar, Token
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Tuple

from workflow.protocol import AgentMessage, make_message, validate_payload

from .paths import assert_in_user_space, project_dir, validate_project_id
from .protocol import UserWindowRunner
from .registry import load_manifest
from .sandbox import EvalCancelled, assert_user_module_imports, run_isolated

_LOADED: Dict[str, ModuleType] = {}
_WINDOW_LOGGER: ContextVar[Optional[Callable[[str], None]]] = ContextVar("science_window_logger", default=None)


def push_window_logger(logger: Optional[Callable[[str], None]]) -> Token:
    return _WINDOW_LOGGER.set(logger)


def pop_window_logger(token: Token) -> None:
    _WINDOW_LOGGER.reset(token)


def note_window_progress(text: str) -> None:
    logger = _WINDOW_LOGGER.get()
    if logger is None:
        return
    line = " ".join(str(text or "").split())
    if line:
        logger(f"  {line}")


def _window_role(kind: str, label: str) -> str:
    blob = f"{kind} {label}".lower()
    if "plan" in blob:
        return "规划"
    if "verif" in blob or "diagnos" in blob:
        return "校验"
    if "tool" in blob or "execut" in blob:
        return "执行"
    return label or kind or "步骤"


def _short(text: Any, limit: int = 96) -> str:
    flat = " ".join(str(text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _trace_of(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("trace"), dict):
        return payload["trace"]
    return {}


def _error_text(value: Any) -> str:
    if isinstance(value, dict):
        return _short(value.get("message") or value.get("error") or value)
    return _short(value)


def window_detail_lines(message: Any) -> List[str]:
    """Short lines for one finished window. Empty when there is nothing useful to show."""
    payload = getattr(message, "payload", None)
    if not isinstance(payload, dict):
        payload = payload if isinstance(payload, dict) else {}
    kind = str(getattr(message, "kind", "") or "")
    trace = _trace_of(payload)
    lines: List[str] = []
    tool = trace.get("tool") or trace.get("tool_name") or payload.get("tool_name")
    if not tool and isinstance(payload.get("args"), dict):
        tool = payload["args"].get("tool_name")
    goal = trace.get("sub_goal") or payload.get("sub_goal")
    if tool:
        lines.append(f"  工具  {_short(tool)}")
    if goal:
        lines.append(f"  目标  {_short(goal)}")
    if trace.get("command"):
        lines.append(f"  命令  {_short(trace.get('command'))}")
    if kind == "tool_result":
        mark = "成功" if payload.get("ok") else "失败"
        output = trace.get("output") if trace.get("output") not in (None, "") else payload.get("output")
        lines.append(f"  结果  {mark}  {_short(output)}")
    elif kind == "verify":
        conclusion = trace.get("conclusion") or payload.get("step_conclusion") or ""
        ready = payload.get("ready_to_stop")
        state = "可以停止" if ready else ("继续" if str(conclusion).upper() == "CONTINUE" else _short(conclusion) or "未完成")
        lines.append(f"  结论  {state}")
        reason = payload.get("reason") or trace.get("analysis")
        if reason:
            lines.append(f"  说明  {_short(reason)}")
    elif kind == "error":
        lines.append(f"  错误  {_short(payload.get('error'))}")
    if trace.get("llm_error"):
        lines.append(f"  模型  {_error_text(trace.get('llm_error'))}")
    return lines


def _inject_science_user() -> None:
    if "science_user" in sys.modules:
        return
    from . import public as public_mod

    sys.modules["science_user"] = public_mod


def _load_file(module_name: str, path: Path) -> ModuleType:
    assert_in_user_space(path, must_exist=True)
    if path.suffix == ".py":
        assert_user_module_imports(path)
    _inject_science_user()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _parse_entry(entry: str) -> Tuple[str, str]:
    rel, _, qual = str(entry or "adapted/entry.py:UserAgent").partition(":")
    return rel or "adapted/entry.py", qual or "UserAgent"


def load_entry_module(project_id: str, *, reload: bool = False) -> ModuleType:
    pid = validate_project_id(project_id)
    key = f"_user_space_{pid}_entry"
    if not reload and key in _LOADED:
        return _LOADED[key]
    manifest = load_manifest(pid)
    rel, _ = _parse_entry(str(manifest.get("entry") or ""))
    path = assert_in_user_space(project_dir(pid) / rel, must_exist=True)
    module = _load_file(key, path)
    _LOADED[key] = module
    return module


def resolve_runner(project_id: str, *, reload: bool = False) -> UserWindowRunner:
    manifest = load_manifest(project_id)
    _, qual = _parse_entry(str(manifest.get("entry") or ""))
    module = load_entry_module(project_id, reload=reload)
    obj: Any = module
    for part in qual.split("."):
        if hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            obj = None
            break
    if obj is None:
        obj = getattr(module, "UserAgent", None) or getattr(module, "run_window", None)
    if obj is None:
        raise TypeError(f"{project_id} 缺少 UserAgent / run_window")
    instance = obj() if isinstance(obj, type) else obj
    if callable(instance) and not hasattr(instance, "run_window"):
        class _FnRunner:
            def run_window(self, message: Any) -> Any:
                return instance(message)

        return _FnRunner()
    if not hasattr(instance, "run_window"):
        raise TypeError(f"{project_id} 入口没有 run_window")
    return instance


def _coerce_message(
    result: Any,
    *,
    inbound: AgentMessage,
    default_kind: str,
    default_src: str,
) -> AgentMessage:
    if isinstance(result, AgentMessage):
        return result
    if isinstance(result, dict):
        kind = str(result.get("kind") or default_kind)
        payload = result.get("payload")
        if not isinstance(payload, dict):
            payload = {k: v for k, v in result.items() if k not in {"kind", "src", "dst", "msg_id", "task_id", "turn", "trace_ref"}}
            if "output" not in payload and "input" in result:
                payload = {"output": result.get("input"), "ok": True, "evidence_type": "DIRECT"}
        src = str(result.get("src") or default_src)
        dst = str(result.get("dst") or inbound.src or "orchestrator")
        return make_message(
            task_id=inbound.task_id,
            turn=inbound.turn,
            src=src,
            dst=dst,
            kind=kind,  # type: ignore[arg-type]
            payload=payload,
            trace_ref=inbound.msg_id,
        )
    text = str(result)
    return make_message(
        task_id=inbound.task_id,
        turn=inbound.turn,
        src=default_src,
        dst=inbound.src or "orchestrator",
        kind=default_kind,  # type: ignore[arg-type]
        payload={"output": text, "ok": True, "evidence_type": "DIRECT"} if default_kind == "tool_result" else {"answer": text},
        trace_ref=inbound.msg_id,
    )


def _window_timeout(timeout_s: Optional[float] = None) -> float:
    if timeout_s is not None:
        return float(timeout_s)
    raw = os.environ.get("SCIENCE_USER_WINDOW_TIMEOUT") or "300"
    try:
        return max(20.0, float(raw))
    except ValueError:
        return 300.0


def invoke_user_window(
    project_id: str,
    inbound: AgentMessage,
    *,
    agent_id: str = "",
    expected_kind: Optional[str] = None,
    timeout_s: Optional[float] = None,
    reload: bool = False,
) -> AgentMessage:
    # Whole window (incl. isolated worker) must sit under the MAS agent span so
    # TracerTraceToTriplet can match agent__{id} and keep Proxy token_ids.
    # Import lives outside workflow/ (AST-safe); see mas/rl_agent_span.py.
    from rl_agent_span import maybe_agent_window_span

    with maybe_agent_window_span(agent_id or str(getattr(inbound, "dst", "") or "")):
        return _invoke_user_window_body(
            project_id,
            inbound,
            agent_id=agent_id,
            expected_kind=expected_kind,
            timeout_s=timeout_s,
            reload=reload,
        )


def _invoke_user_window_body(
    project_id: str,
    inbound: AgentMessage,
    *,
    agent_id: str = "",
    expected_kind: Optional[str] = None,
    timeout_s: Optional[float] = None,
    reload: bool = False,
) -> AgentMessage:
    runner = resolve_runner(project_id, reload=reload)
    logger = _WINDOW_LOGGER.get()
    label = agent_id or str(getattr(inbound, "dst", "") or project_id)
    kind = str(getattr(inbound, "kind", "") or "")
    role = _window_role(kind, label)
    turn = int(getattr(inbound, "turn", 0) or 0)
    started = time.perf_counter()
    if logger is not None:
        logger(f"· 第{turn}轮  {role}")

    def _call() -> Any:
        return runner.run_window(inbound)

    try:
        raw = run_isolated(_call, timeout_s=_window_timeout(timeout_s))
    except EvalCancelled:
        raise
    except Exception as exc:  # noqa: BLE001
        if logger is not None:
            logger(f"  失败  {_short(exc)}")
            logger(f"  用时  {time.perf_counter() - started:.1f}s")
        return make_message(
            task_id=inbound.task_id,
            turn=inbound.turn,
            src=agent_id or inbound.dst or project_id,
            dst=inbound.src or "orchestrator",
            kind="error",
            payload={"agent_id": agent_id or project_id, "error": str(exc)},
            trace_ref=inbound.msg_id,
        )
    out_kind = expected_kind or inbound.kind
    if inbound.kind in ("tool_invoke",) and not expected_kind:
        out_kind = "tool_result"
    msg = _coerce_message(raw, inbound=inbound, default_kind=out_kind, default_src=agent_id or inbound.dst)
    if expected_kind:
        ok, reason = validate_payload(expected_kind, msg.payload)
        if not ok:
            msg = make_message(
                task_id=inbound.task_id,
                turn=inbound.turn,
                src=agent_id or inbound.dst,
                dst=inbound.src or "orchestrator",
                kind="error",
                payload={"agent_id": agent_id or project_id, "error": f"invalid {expected_kind}: {reason}"},
                trace_ref=inbound.msg_id,
            )
    if logger is not None:
        for line in window_detail_lines(msg):
            logger(line)
        logger(f"  用时  {time.perf_counter() - started:.1f}s")
    return msg


def maybe_invoke_user_node(node: Any, inbound: AgentMessage, **kwargs: Any) -> Optional[AgentMessage]:
    profile = dict(getattr(node, "profile", None) or {})
    if str(profile.get("backend") or "") != "user_space":
        return None
    project_id = str(profile.get("user_project") or "")
    if not project_id:
        parsed = None
        try:
            from .paths import parse_user_agent_id

            parsed = parse_user_agent_id(getattr(node, "id", ""))
        except Exception:
            parsed = None
        if parsed:
            project_id = parsed[0]
    if not project_id:
        return None
    return invoke_user_window(project_id, inbound, agent_id=str(getattr(node, "id", "")), **kwargs)
