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

import re
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

# ------------------------------------------------------------------
# Required-capability inference from sub-goal text  (N1 -> N2)
# ------------------------------------------------------------------
# Deterministic keyword rules. Order matters: compute is checked first
# because "calculate the number of papers found by search" mentions both
# compute and search verbs but the *required* capability is compute.

_COMPUTE_MARKERS = (
    "calculate", "compute", "how many", "how much", "round up",
    "round to", "ceil", "floor", "multiply", "divide", "sum ", "subtract",
    "formula", "product of", "ratio", "percentage", "convert", "equation",
    "math.ceil", "math.floor", "derive the number", "determine the value of",
)
_RETRIEVE_MARKERS = (
    "retrieve", "find the", "find ", "search for", "search ", "look up",
    "locate", "what is the", "what was the", "who is", "who was",
    "when did", "where is", "how many articles published",  # retrieval, not compute
    "fetch", "get the", "obtain the", "identify the source",
)
_PERCEIVE_MARKERS = (
    "extract from image", "extract from figure", "read the text in",
    "identify from the image", "ocr", "screenshot", "axis label",
    "figure caption", "read the chart", "read the plot",
)
_SYNTHESIZE_MARKERS = (
    "synthesize", "combine the evidence", "reason about",
    "determine based on", "decide based on", "reconcile",
)


def infer_required_capability(sub_goal: str) -> Set[str]:
    """Infer the set of capabilities required to satisfy ``sub_goal``.

    Returns a set so callers can test ``tool_cap ⊇ required_cap``. When no
    marker matches we default to ``{"retrieve"}`` since the overwhelming
    majority of GAIA sub-goals are acquisition steps.
    """
    text = " " + str(sub_goal or "").lower() + " "
    if any(m in text for m in _COMPUTE_MARKERS):
        return {"compute"}
    if any(m in text for m in _PERCEIVE_MARKERS):
        return {"perceive"}
    if any(m in text for m in _SYNTHESIZE_MARKERS):
        return {"synthesize"}
    if any(m in text for m in _RETRIEVE_MARKERS):
        return {"retrieve"}
    return {"retrieve"}


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
