"""Tool Capability Table + sub-goal required-capability inference.

This module implements the *Capability Consistency Check* used by the
Diagnoser: a tool selection is correct iff the tool's capability profile
covers the capability required by the sub-goal. This is a deterministic,
LLM-free gate that prevents Result-based misdiagnosis (e.g. a Python
``ImportError`` being mislabelled ``wrong_tool_class``).

Capabilities form a small closed ontology aligned with the MAS workflow
phases:

    retrieve   — fetch external facts (search tools)
    compute    — deterministic numeric / symbolic calculation (Python)
    perceive   — extract information from an image / figure (OCR / screenshot)
    synthesize — reason over accumulated evidence (Base_Generator)

The table is intentionally tiny (~10 lines of data) so it can be cited as
a deterministic component in the paper without ballooning the codebase.
"""

from __future__ import annotations

from typing import Set

# ------------------------------------------------------------------
# Tool Capability Profile  (N4 in the causal graph)
# ------------------------------------------------------------------
# Each tool maps to the set of capabilities it can satisfy. A tool may
# own more than one capability, but every required capability must be
# covered for the selection to be considered correct.

TOOL_CAPABILITY: dict[str, Set[str]] = {
    "Python_Coder_Tool": {"compute"},
    "Google_Search_Tool": {"retrieve"},
    "Web_Search_Tool": {"retrieve"},
    "Wikipedia_Search_Tool": {"retrieve"},
    "Screenshot_Tool": {"perceive"},
    "Vision_OCR_Tool": {"perceive"},
    "Base_Generator_Tool": {"synthesize"},
}

# Required-capability inference delegates to tool_router.infer_subgoal_kind
# (single ontology: acquisition↔retrieve, visual↔perceive, …).

_KIND_TO_CAPABILITY = {
    "compute": {"compute"},
    "visual": {"perceive"},
    "synthesis": {"synthesize"},
    "acquisition": {"retrieve"},
}


def infer_required_capability(sub_goal: str) -> Set[str]:
    """Infer capabilities required by ``sub_goal``.

    Delegates kind classification to ``tool_router.infer_subgoal_kind`` so
    ToolRouter and Diagnoser share one ontology (acquisition↔retrieve, etc.).
    """
    # Local import avoids import cycle at module load (tool_router → task_profile).
    from MAS.epc_aw.models.tool_router import infer_subgoal_kind

    kind = infer_subgoal_kind("", sub_goal)
    return set(_KIND_TO_CAPABILITY.get(kind, {"retrieve"}))


def get_tool_capability(tool_name: str) -> Set[str]:
    """Capability profile for ``tool_name`` (empty set if unknown)."""
    return set(TOOL_CAPABILITY.get(tool_name, set()))


def capability_matches(tool_name: str, sub_goal: str) -> bool:
    """True iff the tool's capability profile covers the sub-goal's need.

    This is the deterministic gate that replaces LLM-judged
    ``tool_appropriate`` for the *tool-selection* node of the causal
    graph. A result-level failure on a capability-matched tool cannot be
    a ``wrong_tool_class`` failure — it must be an execution /
    environment / command failure further down the chain.
    """
    tool_cap = get_tool_capability(tool_name)
    required = infer_required_capability(sub_goal)
    if not tool_cap or not required:
        return True  # unknown tool / unclear goal → defer to LLM judge
    return required.issubset(tool_cap)
