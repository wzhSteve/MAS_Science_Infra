"""
ToolRouter: subgoal kind classification and tool validation.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from MAS.epc_aw.models.task_profile import SlotGate, TaskProfile

ACQUISITION_TOOLS = frozenset({
    "Google_Search_Tool",
    "Web_Search_Tool",
    "Wikipedia_Search_Tool",
    "Screenshot_Tool",
    "Vision_OCR_Tool",
})

COMPUTE_TOOLS = frozenset({"Python_Coder_Tool"})
GENERATOR_TOOLS = frozenset({"Base_Generator_Tool"})
VISUAL_TOOLS = frozenset({"Screenshot_Tool", "Vision_OCR_Tool"})

ACQUISITION_MARKERS = (
    "extract", "retrieve", "find the", "locate", "confirm", "ocr",
    "screenshot", "search", "resolve", "identify the", "obtain",
    "download", "capture", "zip code", "arxiv",
)

SYNTHESIS_MARKERS = (
    "format and confirm", "final answer", "synthesize", "summariz",
    "produce the final", "base_generator",
)

COMPUTE_MARKERS = (
    "python_coder", "calculate", "multiply", "round up", "ceil",
    "compute",
)

VISUAL_MARKERS = (
    "screenshot", "image", "figure", "photo", "video", "on camera",
    "ocr", "diagram", "chart", "plot",
)

# Strong external-retrieval indicators. When present, the subgoal is acquisition
# even if compute/synthesis markers also appear (e.g. "Retrieve ... statistical
# significance" must NOT be classified as compute just because "statistical" is
# a compute-ish word). This prevents the v3 hard tool-class isolation from
# back-firing and forcing Python_Coder_Tool onto a retrieval subgoal.
#
# Matched with WORD BOUNDARIES so that past-participle adjectives describing
# already-retrieved INPUT data on a compute step ("the retrieved total",
# "the obtained value", "searched results") do NOT trigger a false acquisition
# classification. Only the imperative action form ("Retrieve the X", "Search
# Wikipedia for Y") counts as a strong retrieval signal.
STRONG_ACQUISITION_MARKERS = (
    "retrieve", "search", "obtain", "lookup", "find the", "fetch", "query",
)

import re as _re
_STRONG_ACQ_RE = _re.compile(
    r"\b(?:" + "|".join(_re.escape(m) for m in STRONG_ACQUISITION_MARKERS) + r")\b",
    _re.IGNORECASE,
)


def infer_subgoal_kind(target_information: str, sub_goal: str = "") -> str:
    text = f"{target_information} {sub_goal}".lower()
    # Visual is highest priority: visual tools are specialized acquisition
    # tools and a visual subgoal should never drop to generic compute.
    if any(m in text for m in VISUAL_MARKERS):
        return "visual"
    has_strong_acq = bool(_STRONG_ACQ_RE.search(text))
    # compute / synthesis only when the subgoal does NOT describe external
    # retrieval; otherwise "Retrieve ... statistical significance" etc. would
    # be misrouted to Python_Coder_Tool and hallucinate facts. Word boundaries
    # protect compute steps that merely reference "the retrieved total" as input.
    if any(m in text for m in COMPUTE_MARKERS) and not has_strong_acq:
        return "compute"
    if any(m in text for m in SYNTHESIS_MARKERS) and not has_strong_acq:
        return "synthesis"
    return "acquisition"


class ToolRouter:
    """Validate and redirect tool selection based on subgoal kind and task phase."""

    @staticmethod
    def allowed_tools(
        subgoal_kind: str,
        profile: Optional[TaskProfile],
        evidence_records: List[Dict[str, Any]],
    ) -> frozenset:
        # v3: strict tool-class isolation (pure field matching, no LLM).
        #   compute     -> Python only (no acquisition tools)
        #   acquisition -> retrieval only (no Python) — blocks hallucination
        #                  fallback where Python fabricates facts from parametric
        #                  knowledge (covers original F3 at the source)
        #   visual      -> Screenshot / Vision_OCR only
        #   synthesis   -> generator/compute (unchanged)
        if subgoal_kind == "compute":
            return COMPUTE_TOOLS
        if subgoal_kind == "visual":
            return VISUAL_TOOLS
        if subgoal_kind == "synthesis":
            if profile and SlotGate.can_enter_synthesize(profile, evidence_records):
                return GENERATOR_TOOLS | COMPUTE_TOOLS
            return ACQUISITION_TOOLS | COMPUTE_TOOLS
        return ACQUISITION_TOOLS

    @staticmethod
    def validate(
        tool_name: str,
        target_information: str,
        sub_goal: str,
        profile: Optional[TaskProfile],
        evidence_records: List[Dict[str, Any]],
        available_tools: List[str],
    ) -> Tuple[str, bool, str]:
        """
        Returns (resolved_tool, tool_appropriate, subgoal_kind).
        Redirects Base_Generator when acquisition/compute required.
        """
        kind = infer_subgoal_kind(target_information, sub_goal)
        allowed = ToolRouter.allowed_tools(kind, profile, evidence_records)

        if tool_name in GENERATOR_TOOLS and kind in ("acquisition", "compute", "visual"):
            alt = ToolRouter._pick_alternative(available_tools, kind)
            return alt or tool_name, False, kind

        if profile:
            forbidden = SlotGate.forbidden_tools(profile, evidence_records)
            if tool_name in forbidden:
                alt = ToolRouter._pick_alternative(available_tools, kind)
                return alt or tool_name, False, kind

        if tool_name not in allowed and tool_name in GENERATOR_TOOLS:
            alt = ToolRouter._pick_alternative(available_tools, kind)
            return alt or tool_name, False, kind

        # v3: hard tool-class enforcement for compute/visual. Any tool outside
        # the allowed set (e.g. Google picked for a compute subgoal, or Python
        # picked for an acquisition subgoal) is redirected to the kind's tool.
        if tool_name not in allowed and kind in ("compute", "visual"):
            alt = ToolRouter._pick_alternative(available_tools, kind)
            return alt or tool_name, False, kind

        appropriate = tool_name in allowed or tool_name not in GENERATOR_TOOLS
        return tool_name, appropriate, kind

    @staticmethod
    def _pick_alternative(available_tools: List[str], kind: str) -> Optional[str]:
        prefs: List[str] = []
        if kind == "compute":
            prefs = ["Python_Coder_Tool", "Google_Search_Tool", "Web_Search_Tool"]
        elif kind == "visual":
            prefs = ["Screenshot_Tool", "Vision_OCR_Tool", "Google_Search_Tool", "Web_Search_Tool"]
        else:
            prefs = [
                "Google_Search_Tool",
                "Web_Search_Tool",
                "Wikipedia_Search_Tool",
                "Screenshot_Tool",
                "Vision_OCR_Tool",
            ]
        for p in prefs:
            if p in available_tools:
                return p
        for t in available_tools:
            if t not in GENERATOR_TOOLS:
                return t
        return None
