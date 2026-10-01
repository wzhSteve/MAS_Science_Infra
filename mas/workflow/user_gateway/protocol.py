"""User-window contract. User code may return a dict or an AgentMessage."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class UserWindowRunner(Protocol):
    """Single MAS window implemented in the user zone."""

    def run_window(self, message: Any) -> Any:
        """Accept AgentMessage or dict; return AgentMessage or dict."""
        ...
