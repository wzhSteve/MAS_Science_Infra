"""Memory subpackage: MemoryStore / FactLedger / Archive (MAS refactor Phase 2).

Three explicit components (docs/MAS_design.md §4):
  - ``store``   : in-process three-scope memory (agent / shared hub / system)
  - ``facts``   : FactLedger — the single HIVE verify-gate classification
  - ``archive`` : persistent event stream + snapshots + resume index

``from workflow.memory import X`` keeps working (module → package transition;
see mas/workflow/memory.py history). Archive-specific names that lived at
``workflow.archive`` remain importable there via a 6-line shim (deletion
condition registered in docs/refactor/phase0_external_api.md §6).
"""

from __future__ import annotations

from .facts import FactLedger
from .store import MemoryItem, MemoryStore

__all__ = ["FactLedger", "MemoryItem", "MemoryStore", "store", "facts", "archive"]
