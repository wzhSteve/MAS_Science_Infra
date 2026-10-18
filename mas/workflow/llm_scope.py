"""Endpoint bound for the rollout that is currently executing.

User-space engines read this before process environment variables, so an
experiment's llm.yaml wins over Assistant credentials living in the UI process.
"""

from __future__ import annotations

import contextvars
from typing import Any, Dict, Optional

_BOUND: contextvars.ContextVar[Optional[Dict[str, str]]] = contextvars.ContextVar(
    "science_rollout_llm", default=None,
)


def current_rollout_llm() -> Dict[str, str]:
    return dict(_BOUND.get() or {})


def bind_rollout_llm(llm: Any):
    """Bind ``LLMConfig`` for user-space calls. Returns a reset token, or None."""
    endpoint = str(getattr(llm, "endpoint", "") or "").strip()
    model = str(getattr(llm, "model", "") or "").strip()
    if not endpoint or not model:
        return None
    return _BOUND.set({
        "endpoint": endpoint.rstrip("/"),
        "model": model,
        "api_key": str(getattr(llm, "api_key", None) or "EMPTY"),
    })


def reset_rollout_llm(token) -> None:
    if token is not None:
        _BOUND.reset(token)
