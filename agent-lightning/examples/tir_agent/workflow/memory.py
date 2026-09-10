"""Two-layer memory: per-agent buffer + optional MAS system store.

Phase 0 default: agent=messages, system=none (no system writes).
"""

from __future__ import annotations

from typing import Dict, List, Optional

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
