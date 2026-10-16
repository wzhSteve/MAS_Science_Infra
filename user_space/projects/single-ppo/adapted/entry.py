"""Single-PPO window dispatcher. Management runtime calls run_window only.

Mirrors agentflow/adapted/entry.py: route one inbound AgentMessage to the
matching episode-context window. The planner window is the single LLM ReAct
loop (APPO/ARPO paradigm); exec / verify windows are defensive mirrors so
unexpected dispatches never crash the episode.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

from context import get_episode_context  # noqa: E402


def _dst(message: Any) -> str:
    return str(getattr(message, "dst", None) or (message.get("dst") if isinstance(message, dict) else "") or "")


def _kind(message: Any) -> str:
    return str(getattr(message, "kind", None) or (message.get("kind") if isinstance(message, dict) else "") or "")


class UserAgent:
    def run_window(self, message: Any) -> Any:
        ctx = get_episode_context()
        dst = _dst(message)
        kind = _kind(message)
        if kind == "plan_step" or dst == "planner" or "planner" in dst:
            return ctx.plan_window(message)
        if kind == "verify" or "verifier" in dst or "diagnoser" in dst:
            return ctx.verify_window(message)
        if kind == "tool_invoke" or "executor" in dst or "solver" in dst:
            return ctx.exec_window(message)
        # Default: the single window is the solver itself.
        return ctx.plan_window(message)
