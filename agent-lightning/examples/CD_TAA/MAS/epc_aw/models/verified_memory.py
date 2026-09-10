"""StructAgent-inspired verified fact ledger for EPC-AW task memory.

Task-local truth is event-driven: only MemoryCommitEvent kinds can mutate
facts. Compact views and offline evolve should prefer this ledger over raw
tool output.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CommitKind(str, Enum):
    SATISFIED = "satisfied"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"
    VALUE_COMMITTED = "value_committed"


LIVE_VERIFICATION_STATUSES = frozenset(
    {CommitKind.SATISFIED.value, CommitKind.VALUE_COMMITTED.value}
)


@dataclass
class VerifiedFact:
    name: str
    value: Any
    source_evidence_id: Optional[str] = None
    source_step: int = 0
    outline_step: str = ""
    valid: bool = True


@dataclass
class MemoryFailureRecord:
    step: int
    outline_step: Optional[str]
    attributed_to: str
    reason: str
    recovery_hint: str = ""
    tool_name: Optional[str] = None


@dataclass
class MemoryCommitEvent:
    """The only object that may request a verified-memory transition."""

    kind: str
    step: int
    reason: str
    outline_step: Optional[str] = None
    evidence_id: Optional[str] = None
    facts: Dict[str, Any] = field(default_factory=dict)
    tool_name: Optional[str] = None
    attributed_to: str = "verifier"
    recovery_hint: str = ""
    confidence: str = "medium"

    def __post_init__(self) -> None:
        allowed = {item.value for item in CommitKind}
        if self.kind not in allowed:
            raise ValueError(f"Unsupported memory commit kind: {self.kind}")


@dataclass
class VerifiedTaskMemory:
    """Compact StructAgent-style ledger layered on EPC-AW SystemMemory."""

    facts: Dict[str, VerifiedFact] = field(default_factory=dict)
    events: List[MemoryCommitEvent] = field(default_factory=list)
    recent_failures: List[MemoryFailureRecord] = field(default_factory=list)
    max_failures: int = 5

    def apply_event(self, event: MemoryCommitEvent) -> None:
        if event.kind in {
            CommitKind.SATISFIED.value,
            CommitKind.VALUE_COMMITTED.value,
        }:
            for name, value in event.facts.items():
                self.facts[str(name)] = VerifiedFact(
                    name=str(name),
                    value=value,
                    source_evidence_id=event.evidence_id,
                    source_step=event.step,
                    outline_step=event.outline_step or "",
                    valid=True,
                )
        elif event.kind == CommitKind.INVALIDATED.value:
            if event.evidence_id:
                for fact in self.facts.values():
                    if fact.source_evidence_id == event.evidence_id:
                        fact.valid = False
            for name in event.facts:
                fact = self.facts.get(str(name))
                if fact is not None:
                    fact.valid = False
            self._append_failure(event)
        elif event.kind == CommitKind.REJECTED.value:
            self._append_failure(event)

        self.events.append(event)

    def _append_failure(self, event: MemoryCommitEvent) -> None:
        self.recent_failures.append(
            MemoryFailureRecord(
                step=event.step,
                outline_step=event.outline_step,
                attributed_to=event.attributed_to,
                reason=event.reason,
                recovery_hint=event.recovery_hint,
                tool_name=event.tool_name,
            )
        )
        if len(self.recent_failures) > self.max_failures * 4:
            self.recent_failures = self.recent_failures[-(self.max_failures * 4) :]

    def valid_facts(self) -> Dict[str, Any]:
        return {
            name: fact.value
            for name, fact in self.facts.items()
            if fact.valid
        }

    def invalidate_by_evidence(self, evidence_id: str, reason: str, step: int = 0) -> None:
        self.apply_event(
            MemoryCommitEvent(
                kind=CommitKind.INVALIDATED.value,
                step=step,
                reason=reason,
                evidence_id=evidence_id,
            )
        )

    def compact_view(
        self,
        *,
        question: str = "",
        outline_head: str = "",
        live_evidence: Optional[List[Dict[str, Any]]] = None,
        missing_slots: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "outline_head": outline_head,
            "facts": self.valid_facts(),
            "live_evidence": list(live_evidence or []),
            "recent_failures": [
                asdict(item) for item in self.recent_failures[-self.max_failures :]
            ],
            "missing_slots": list(missing_slots or []),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": {k: asdict(v) for k, v in self.facts.items()},
            "events": [asdict(e) for e in self.events],
            "recent_failures": [asdict(f) for f in self.recent_failures],
        }


def map_verification_to_commit_kind(
    *,
    step_conclusion: str,
    subgoal_complete: bool,
    evidence_type: str = "",
    claim_type: str = "fact",
    compute_real: Optional[bool] = None,
) -> str:
    """Map EPC VerificationResult fields onto MemoryCommitEvent.kind."""
    conclusion = (step_conclusion or "").upper()
    etype = (evidence_type or "").upper()
    if conclusion == "SUBGOAL_COMPLETE" and subgoal_complete:
        if compute_real is True:
            return CommitKind.VALUE_COMMITTED.value
        if etype in {"ABSENCE", "EMPTY"} or claim_type == "absence":
            # Absence may be recorded for diagnostics but is not a fact commit.
            return CommitKind.REJECTED.value
        return CommitKind.SATISFIED.value
    return CommitKind.REJECTED.value
