"""FactLedger: the single HIVE verify-gate classification (MAS refactor Phase 2).

Auditable Fact Commit (MAS_design §4.3, Option A):
  - decides ONLY whether a verified tool_result is eligible to be marked as a
    committed fact in the event stream (a ``MEMORY_WRITE`` with
    ``payload.phase == "fact_commit"``);
  - is NOT a shared knowledge base and does NOT auto-inject anything into the
    next turn's readable context — inter-turn context flows via verifier
    feedback text in the planner window;
  - external contract frozen at Phase 0: ``protocol.commit_to_fact`` keeps its
    bool signature (tests depend on it) and the event payload keys stay
    ``{"phase": "fact_commit", "turn", "ok"}``.

Classification matrix (authoritative; mirrors the pre-refactor inline logic
in ``workflow/protocol.py::commit_to_fact`` byte-for-byte, including the
evidence_type fallback from tool_result when the verifier omits it):
"""

from __future__ import annotations

from typing import Any, Dict, Literal

FactDecision = Literal["COMMIT_FACT", "KEEP_WINDOW_ONLY", "REJECT"]


class FactLedger:
    """HIVE verification gate → fact-commit eligibility. No storage."""

    @staticmethod
    def classify(verify_payload: Dict[str, Any], tool_result_payload: Dict[str, Any]) -> FactDecision:
        # step_conclusion must COMPLETE; anything else stays window-only.
        if verify_payload.get("step_conclusion") != "COMPLETE":
            return "KEEP_WINDOW_ONLY"
        # Evidence must be DIRECT; verifier payload wins, fall back to the
        # tool_result's evidence_type when the verifier omits it.
        ev = verify_payload.get("evidence_type") or tool_result_payload.get("evidence_type")
        if ev != "DIRECT":
            return "REJECT"
        # All declared slots must be filled (empty slot_updates = no
        # constraint; pre-existing semantics).
        slots = verify_payload.get("slot_updates") or []
        if slots:
            if not all(bool(s.get("filled")) for s in slots if isinstance(s, dict)):
                return "KEEP_WINDOW_ONLY"
        return "COMMIT_FACT"

    @staticmethod
    def commits(decision: str) -> bool:
        """Bool bridge for the frozen ``commit_to_fact`` external signature."""
        return decision == "COMMIT_FACT"
