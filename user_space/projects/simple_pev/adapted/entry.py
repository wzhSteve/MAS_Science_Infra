"""AgentFlow window dispatcher. Management runtime calls run_window only."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

from context import get_episode_context  # noqa: E402


def _field(message: Any, name: str) -> str:
    value = getattr(message, name, None)
    if value is None and isinstance(message, dict):
        value = message.get(name)
    return str(value or "")


class UserAgent:
    def run_window(self, message: Any) -> Any:
        ctx = get_episode_context()
        dst = _field(message, "dst")
        kind = _field(message, "kind")
        if "executor" in dst:
            return ctx.exec_window(message)
        if "verifier" in dst or kind == "verify":
            return ctx.verify_window(message)
        if "planner" in dst or kind == "plan_step":
            return ctx.plan_window(message)
        return ctx.exec_window(message)
