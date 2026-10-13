"""Transition shim: ``mas_agent`` now lives at ``legacy/structagent/mas_agent.py``.

Kept because the training hot path (``lit_tir_agent.py``, zero-move red line)
lazily imports the *top-level* name ``mas_agent`` under ``PYTHONPATH=mas``.
Deletion condition (docs/refactor/phase0_external_api.md §6): when
``lit_tir_agent.py`` switches to explicit legacy-path injection or the
parallel training entry goes away.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from legacy.structagent.mas_agent import (  # noqa: E402,F401
    LitMASAgent,
    bind_training_llm_env,
    bootstrap_mas_package,
    debug_mas_agent,
)

__all__ = ["LitMASAgent", "bind_training_llm_env", "bootstrap_mas_package", "debug_mas_agent"]
