"""Boundary records and credential redaction, independent of model providers."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from .contracts import EventKind, ExecutionEvent, ExecutionRecorder

_SECRET_KEY = re.compile(
    r"api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token"
    r"|cookie|credentials|private[_-]?key",
    re.I,
)


class Redactor:
    """Retain business content while removing explicit and environment credentials."""

    def __init__(self, secrets: Sequence[str] = ()) -> None:
        values = list(secrets) + [v for k, v in os.environ.items() if _SECRET_KEY.search(k)]
        self.secrets = sorted({v for v in values if v and v != "dummy"}, key=len, reverse=True)
        self.redacted = False

    def __call__(self, value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                if _SECRET_KEY.search(str(key)):
                    out[str(key)] = "[redacted]"
                    self.redacted = True
                else:
                    out[str(key)] = self(item)
            return out
        if isinstance(value, (list, tuple)):
            return [self(item) for item in value]
        if hasattr(value, "model_dump"):
            return self(value.model_dump(mode="json"))
        if isinstance(value, str):
            original = value
            for secret in self.secrets:
                value = value.replace(secret, "[redacted]")
            value = re.sub(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@", r"\1[redacted]@", value)
            value = re.sub(r"(?i)(Bearer\s+)[^\s\"',;]+", r"\1[redacted]", value)
            value = re.sub(
                r"(?i)((?:api[_-]?key|authorization|password|secret|access[_-]?token)"
                r"[\"']?\s*[:=]\s*[\"']?)[^\s&\"',;}]+",
                r"\1[redacted]", value,
            )
            self.redacted |= original != value
        return value


def message_records(messages: Sequence[Any]) -> list[Dict[str, Any]]:
    """Serialize visible messages without dropping provider evidence or content blocks."""
    result = []
    roles = {"human": "user", "ai": "assistant"}
    for message in messages:
        if isinstance(message, dict):
            result.append(dict(message))
            continue
        row = {
            "role": roles.get(message.type, message.type),
            "content": message.content,
        }
        for name in (
            "tool_calls", "tool_call_id", "name", "id", "additional_kwargs",
            "response_metadata", "usage_metadata", "invalid_tool_calls", "status",
        ):
            value = getattr(message, name, None)
            if value is not None and value != {} and value != []:
                row[name] = value
        result.append(row)
    return result


class BoundaryRecorder:
    """Attach run and Agent correlation at the moment an operation occurs."""

    def __init__(
        self, sink: ExecutionRecorder, *, run_id: str, trajectory_id: str,
        agent_id: str, agent_execution_id: str,
    ) -> None:
        self.sink = sink
        self.run_id = run_id
        self.trajectory_id = trajectory_id
        self.agent_id = agent_id
        self.agent_execution_id = agent_execution_id

    def emit(
        self, kind: EventKind, payload: Dict[str, Any], *,
        model_call_id: Optional[str] = None, tool_call_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> ExecutionEvent:
        return self.sink.append(ExecutionEvent(
            kind=kind, payload=payload, run_id=self.run_id, trajectory_id=self.trajectory_id,
            agent_id=self.agent_id, agent_execution_id=self.agent_execution_id,
            model_call_id=model_call_id, tool_call_id=tool_call_id, parent_id=parent_id,
            occurred_at=datetime.now(timezone.utc),
        ))
