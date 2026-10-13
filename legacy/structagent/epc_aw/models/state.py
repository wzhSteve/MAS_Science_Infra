from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional


class MilestoneStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    INVALIDATED = "invalidated"


class EventKind(str, Enum):
    SATISFIED = "satisfied"
    INVALIDATED = "invalidated"
    REJECTED = "rejected"
    VALUE_COMMITTED = "value_committed"


@dataclass
class VerificationSpec:
    """Evidence contract authored at task initialization."""

    description: str
    required_fact_names: List[str] = field(default_factory=list)
    require_source: bool = True
    preferred_tools: List[str] = field(default_factory=list)


@dataclass
class Fact:
    name: str
    value: Any
    source_milestone_id: str
    source_step: int
    evidence_ids: List[str] = field(default_factory=list)
    valid: bool = True


@dataclass
class EvidenceRecord:
    id: str
    milestone_id: str
    step: int
    tool_name: str
    summary: str
    raw_result: Any
    source_urls: List[str] = field(default_factory=list)
    deterministic_checks: Dict[str, Any] = field(default_factory=dict)
    verdict: str = EventKind.REJECTED.value
    live: bool = True


@dataclass
class FailureRecord:
    step: int
    milestone_id: Optional[str]
    attributed_to: str
    reason: str
    recovery_hint: str = ""
    tool_name: Optional[str] = None


@dataclass
class VerifierEvent:
    """The only object allowed to request a State transition."""

    kind: str
    milestone_id: Optional[str]
    step: int
    reason: str
    evidence: Optional[EvidenceRecord] = None
    facts: Dict[str, Any] = field(default_factory=dict)
    confidence: str = "medium"

    def __post_init__(self) -> None:
        allowed = {item.value for item in EventKind}
        if self.kind not in allowed:
            raise ValueError(f"Unsupported verifier event kind: {self.kind}")


@dataclass
class ExecutionTrace:
    step: int
    milestone_id: str
    subgoal: str
    tool_name: str
    arguments: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    source_urls: List[str] = field(default_factory=list)


@dataclass
class Milestone:
    id: str
    description: str
    verify: VerificationSpec
    depends_on: List[str] = field(default_factory=list)
    status: MilestoneStatus = MilestoneStatus.PENDING
    evidence_ids: List[str] = field(default_factory=list)
    rejection_count: int = 0
    invalidation_count: int = 0
    last_updated_step: Optional[int] = None


@dataclass
class AgentState:
    question: str
    milestones: List[Milestone]
    facts: Dict[str, Fact] = field(default_factory=dict)
    evidence: List[EvidenceRecord] = field(default_factory=list)
    events: List[VerifierEvent] = field(default_factory=list)
    failures: List[FailureRecord] = field(default_factory=list)
    execution_traces: List[ExecutionTrace] = field(default_factory=list)
    final_answer: Optional[str] = None
    audit: Optional[Dict[str, Any]] = None
    max_invalidations: int = 3

    def __post_init__(self) -> None:
        ids = [m.id for m in self.milestones]
        if not ids:
            raise ValueError("AgentState requires at least one milestone")
        if len(ids) != len(set(ids)):
            raise ValueError("Milestone ids must be unique")
        valid_ids = set(ids)
        for milestone in self.milestones:
            unknown = set(milestone.depends_on) - valid_ids
            if unknown:
                raise ValueError(
                    f"Milestone {milestone.id} has unknown dependencies: {unknown}"
                )
            if milestone.id in milestone.depends_on:
                raise ValueError(f"Milestone {milestone.id} cannot depend on itself")

    def get_milestone(self, milestone_id: str) -> Optional[Milestone]:
        return next((m for m in self.milestones if m.id == milestone_id), None)

    def reachable_milestones(self) -> List[Milestone]:
        reachable: List[Milestone] = []
        for milestone in self.milestones:
            if milestone.status == MilestoneStatus.VERIFIED:
                continue
            if all(
                (dependency := self.get_milestone(dep)) is not None
                and dependency.status == MilestoneStatus.VERIFIED
                for dep in milestone.depends_on
            ):
                reachable.append(milestone)
        return reachable

    def leaf_milestones(self) -> List[Milestone]:
        depended_on = {
            dependency
            for milestone in self.milestones
            for dependency in milestone.depends_on
        }
        leaves = [m for m in self.milestones if m.id not in depended_on]
        return leaves or list(self.milestones)

    def can_attempt_done(self) -> bool:
        return all(
            milestone.status == MilestoneStatus.VERIFIED
            for milestone in self.leaf_milestones()
        )

    def live_evidence_for(self, milestone_id: str) -> List[EvidenceRecord]:
        return [
            item
            for item in self.evidence
            if item.milestone_id == milestone_id and item.live
        ]

    def apply_verifier_event(self, event: VerifierEvent) -> None:
        """Commit one verifier event.

        Planner and actor code receive no status mutator. Rejected evidence is
        logged without changing milestone status; invalidation only applies to
        a previously verified milestone.
        """

        milestone = (
            self.get_milestone(event.milestone_id)
            if event.milestone_id
            else None
        )
        if event.milestone_id and milestone is None:
            raise ValueError(f"Unknown milestone id: {event.milestone_id}")

        if event.evidence is not None:
            self.evidence.append(event.evidence)
            if milestone is not None:
                milestone.evidence_ids.append(event.evidence.id)

        if event.kind == EventKind.SATISFIED.value:
            if milestone is None:
                raise ValueError("satisfied event requires a milestone")
            milestone.status = MilestoneStatus.VERIFIED
            milestone.last_updated_step = event.step
            self._commit_facts(event, milestone.id)
        elif event.kind == EventKind.INVALIDATED.value:
            if milestone is None:
                raise ValueError("invalidated event requires a milestone")
            if (
                milestone.status == MilestoneStatus.VERIFIED
                and milestone.invalidation_count < self.max_invalidations
            ):
                milestone.status = MilestoneStatus.INVALIDATED
                milestone.invalidation_count += 1
                milestone.last_updated_step = event.step
                for evidence in self.evidence:
                    if evidence.milestone_id == milestone.id:
                        evidence.live = False
                for fact in self.facts.values():
                    if fact.source_milestone_id == milestone.id:
                        fact.valid = False
        elif event.kind == EventKind.REJECTED.value:
            if milestone is not None:
                milestone.rejection_count += 1
                milestone.last_updated_step = event.step
        elif event.kind == EventKind.VALUE_COMMITTED.value:
            self._commit_facts(event, event.milestone_id or "global")

        self.events.append(event)

    def _commit_facts(self, event: VerifierEvent, owner: str) -> None:
        evidence_ids = [event.evidence.id] if event.evidence else []
        for name, value in event.facts.items():
            self.facts[name] = Fact(
                name=name,
                value=value,
                source_milestone_id=owner,
                source_step=event.step,
                evidence_ids=evidence_ids,
            )

    def compact_view(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "milestones": [
                {
                    "id": m.id,
                    "description": m.description,
                    "status": m.status.value,
                    "depends_on": list(m.depends_on),
                    "verification": m.verify.description,
                    "preferred_tools": list(m.verify.preferred_tools),
                    "rejections": m.rejection_count,
                }
                for m in self.milestones
            ],
            "facts": {
                name: fact.value for name, fact in self.facts.items() if fact.valid
            },
            "recent_failures": [asdict(item) for item in self.failures[-5:]],
        }

    def add_traces(self, traces: Iterable[ExecutionTrace]) -> None:
        self.execution_traces.extend(traces)

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        for milestone in result["milestones"]:
            status = milestone.get("status")
            if isinstance(status, Enum):
                milestone["status"] = status.value
        return result
