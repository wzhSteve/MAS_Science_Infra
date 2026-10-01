"""Narrow re-exports that user-zone code may import.

User adapted modules should prefer ``import science_user`` (injected at load
time) or ``from workflow.user_gateway.public import ...``. Do not import
compiler, runtime, science_infra, or ``tools.tool_agents``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import uuid4

from workflow.protocol import (
    DEFAULT_INPUT_SCHEMA,
    DEFAULT_OUTPUT_SCHEMA,
    AgentMessage,
    make_message,
    validate_json_schema,
    validate_payload,
)

from .protocol import UserWindowRunner


def as_payload(message: Any) -> Dict[str, Any]:
    if isinstance(message, AgentMessage):
        return dict(message.payload or {})
    if isinstance(message, dict):
        if "payload" in message and isinstance(message["payload"], dict):
            return dict(message["payload"])
        return dict(message)
    return {"input": str(message)}


def input_text(message: Any) -> str:
    payload = as_payload(message)
    for key in ("input", "output", "query", "question", "text"):
        val = payload.get(key)
        if val:
            return str(val)
    return str(payload)


def tool_result(output: str, *, ok: bool = True, evidence_type: str = "DIRECT") -> Dict[str, Any]:
    return {
        "kind": "tool_result",
        "payload": {"output": str(output), "ok": bool(ok), "evidence_type": evidence_type},
    }


def plan_step(next_id: str, args: Optional[Dict[str, Any]] = None, *, sub_goal: str = "", done: bool = False) -> Dict[str, Any]:
    return {
        "kind": "plan_step",
        "payload": {"next": next_id, "args": dict(args or {}), "sub_goal": sub_goal, "done": done},
    }


def verify_result(ok: bool, reason: str = "", *, complete: bool = False) -> Dict[str, Any]:
    return {
        "kind": "verify",
        "payload": {
            "ok": bool(ok),
            "reason": reason,
            "step_conclusion": "COMPLETE" if complete or ok else "INCOMPLETE",
            "slot_updates": [],
        },
    }


def new_id() -> str:
    return uuid4().hex


def load_upload(project_id: str, rel_path: str) -> str:
    """Read a file from the project's locked ``upload/`` tree."""
    from .paths import assert_in_user_space, project_dir

    path = assert_in_user_space(project_dir(project_id) / "upload" / rel_path, must_exist=True)
    if path.is_dir():
        raise IsADirectoryError(str(path))
    return path.read_text(encoding="utf-8")


__all__ = [
    "AgentMessage",
    "DEFAULT_INPUT_SCHEMA",
    "DEFAULT_OUTPUT_SCHEMA",
    "UserWindowRunner",
    "as_payload",
    "input_text",
    "load_upload",
    "make_message",
    "new_id",
    "plan_step",
    "tool_result",
    "validate_json_schema",
    "validate_payload",
    "verify_result",
]
