"""MAS RL agent parent spans for centralized / user_space windows.

Lives outside ``workflow/`` so AST isolation (no agentlightning in workflow/) holds.
TracerTraceToTriplet only keeps LLM calls under a matching agent subtree; the
``agent.name`` attribute must equal ``agent_node_name(agent_id)``.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Iterator


@contextmanager
def agent_window_span(agent_id: str) -> Iterator[None]:
    """Open the MAS RL agent parent span for one window. No-op without a tracer."""
    aid = str(agent_id or "").strip()
    if not aid:
        yield
        return
    try:
        import agentlightning as agl
        from agentlightning.tracer.base import get_active_tracer
        from workflow.agent_tracing import agent_node_name
    except Exception:
        yield
        return
    if get_active_tracer() is None:
        yield
        return
    name = agent_node_name(aid)
    with agl.operation(**{"agent.name": name}):
        yield


def maybe_agent_window_span(agent_id: str) -> Any:
    """Return a context manager; never raises on missing AGL."""
    try:
        return agent_window_span(agent_id)
    except Exception:
        return nullcontext()
