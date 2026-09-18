"""Two-layer memory: per-agent buffer + optional MAS system store.

Phase 0 default: agent=messages, system=none (no system writes).

Agent-framework A4: per-agent scope routing.
- ``memory_scope: "agent"``  → private buffer per agent node id
- ``memory_scope: "shared"`` → shared hub buffer (default behaviour)
- ``profile.memory = {"policy": "append_latest", "max_items": N}`` truncation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .contracts import MemoryItem


class MemoryStore:
    """In-process memory. Not a vector DB; fork/resume still uses Archive snapshots."""

    def __init__(self) -> None:
        self._agent: Dict[str, List[MemoryItem]] = {}
        self._system: List[MemoryItem] = []

    def read(self, scope: str, owner: str = "hub") -> List[MemoryItem]:
        if scope == "system":
            return list(self._system)
        return list(self._agent.get(owner, []))

    def write(self, item: MemoryItem) -> MemoryItem:
        if item.scope == "system":
            self._system.append(item)
        else:
            self._agent.setdefault(item.owner or "hub", []).append(item)
        return item

    def latest(self, scope: str, owner: str = "hub") -> Optional[MemoryItem]:
        items = self.read(scope, owner)
        return items[-1] if items else None

    # ------------------------------------------------------------------
    # agent-framework A4: agent-scope routing
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_policy(items: List[MemoryItem], policy: Dict[str, Any]) -> List[MemoryItem]:
        if not policy:
            return items
        p = str(policy.get("policy") or "")
        max_items = policy.get("max_items")
        try:
            max_items = int(max_items) if max_items is not None else None
        except (TypeError, ValueError):
            max_items = None
        if p == "append_latest" and max_items is not None and max_items >= 0:
            return items[-max_items:]
        return items

    def read_agent(
        self,
        agent_id: str,
        *,
        memory_scope: str = "shared",
        policy: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryItem]:
        """Read an agent's memory honouring its node scope.

        "agent" → the agent's private buffer; "shared" → hub buffer
        (legacy default: MAS-layer history log stays in system/hub).
        """
        if memory_scope == "agent":
            items = list(self._agent.get(agent_id, []))
        else:
            items = list(self._agent.get("hub", []))
        return self._apply_policy(items, dict(policy or {}))

    def write_agent(
        self,
        agent_id: str,
        content: Dict[str, Any],
        *,
        memory_scope: str = "shared",
    ) -> MemoryItem:
        """Write to the buffer the agent's scope resolves to.

        "agent" → private buffer keyed by agent_id; "shared" → hub buffer so
        all agents see it (pre-existing behaviour, default empty).
        """
        owner = agent_id if memory_scope == "agent" else "hub"
        return self.write(MemoryItem(scope="agent", owner=owner, content=content))
