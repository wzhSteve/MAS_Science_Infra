"""Turn an execution observation into a constraint candidate.

Deterministic rows come from the existing Ψ analyzer and may already be verified.
An LLM proposal is stored as an unverified candidate and cannot prune by itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from cghe_types import EvidenceOrigin, EvidenceStatus, ExecutionObservation
from failure_analyzer import FailureAnalysis, FailureAnalyzer


@dataclass
class ConstraintCandidate:
    constraint_id: str
    predicate: str
    arguments: dict[str, str]
    scope: dict[str, str]
    supporting_observations: list[str] = field(default_factory=list)
    contradicting_observations: list[str] = field(default_factory=list)
    inference_source: str = "llm"
    confidence: float = 0.0
    evidence_status: str = EvidenceStatus.UNVERIFIED_HYPOTHESIS.value
    observation_reliability: str = "unknown"
    constraint_validity: str = "untested"
    scope_complete: bool = False
    applicability: str = ""

    @property
    def admissible(self) -> bool:
        return (
            self.evidence_status == EvidenceStatus.VERIFIED_COUNTEREXAMPLE.value
            and self.observation_reliability == "traced"
            and self.constraint_validity == "supported"
            and self.scope_complete
            and not self.contradicting_observations
        )


def from_analyzer(analysis: FailureAnalysis, *, observation_id: str) -> Optional[ConstraintCandidate]:
    """Wrap a Ψ result. Only a fully bound direct counterexample is verified."""
    constraint = analysis.constraint
    if constraint is None or not analysis.verified_counterexample:
        return None
    scope = {
        "environment": constraint.scope.environment,
        "tool": constraint.scope.tool,
        "resource": constraint.scope.resource,
        "identity": constraint.scope.identity,
        "parameter": constraint.scope.parameter,
        "argument": constraint.scope.argument,
        "schema_signature": constraint.scope.schema_signature,
    }
    complete = bool(scope["environment"] and scope["tool"])
    if constraint.observation_code == "PERMISSION_DENIED":
        complete = complete and bool(scope["resource"] and scope["identity"])
    if constraint.observation_code in {"PARAM_ERROR", "SNIPPETS_INSUFFICIENT"}:
        complete = complete and bool(scope["parameter"] or scope["argument"])
    return ConstraintCandidate(
        constraint_id=constraint.constraint_id,
        predicate=constraint.condition.predicate,
        arguments={"tool": constraint.condition.tool, "argument": constraint.condition.argument},
        scope=scope,
        supporting_observations=[observation_id],
        inference_source="psi",
        confidence=constraint.confidence,
        evidence_status=constraint.evidence_status.value,
        observation_reliability="traced" if analysis.evidence.origin == EvidenceOrigin.DIRECT_OBSERVATION else "untraced",
        constraint_validity="supported" if analysis.evidence.direct else "untested",
        scope_complete=complete,
        applicability=constraint.observation_code,
    )


def propose_from_llm(payload: dict[str, Any], *, observation_id: str) -> ConstraintCandidate:
    """Accept a structured guess. The status is forced to unverified."""
    scope = dict(payload.get("scope") or {})
    return ConstraintCandidate(
        constraint_id=str(payload.get("constraint_id") or "llm-candidate"),
        predicate=str(payload.get("predicate") or ""),
        arguments={str(key): str(value) for key, value in dict(payload.get("arguments") or {}).items()},
        scope={str(key): str(value) for key, value in scope.items()},
        supporting_observations=[observation_id],
        inference_source="llm",
        confidence=float(payload.get("confidence") or 0.0),
        evidence_status=EvidenceStatus.UNVERIFIED_HYPOTHESIS.value,
        observation_reliability="untraced",
        constraint_validity="untested",
        scope_complete=bool(scope.get("environment") and scope.get("tool") and (scope.get("argument") or scope.get("resource"))),
        applicability="llm_proposal",
    )


def validate_candidate(
    candidate: ConstraintCandidate,
    *,
    traced_observation_ids: Sequence[str],
    contradictions: Sequence[str] = (),
) -> ConstraintCandidate:
    """Promote only a traced, scoped, uncontradicted deterministic candidate."""
    candidate.contradicting_observations = [str(item) for item in contradictions]
    traced = all(item in set(traced_observation_ids) for item in candidate.supporting_observations)
    candidate.observation_reliability = "traced" if traced else "untraced"
    if candidate.inference_source != "psi" or not traced or candidate.contradicting_observations or not candidate.scope_complete:
        candidate.evidence_status = EvidenceStatus.UNVERIFIED_HYPOTHESIS.value
        candidate.constraint_validity = "untested" if not candidate.contradicting_observations else "contradicted"
        return candidate
    candidate.constraint_validity = "supported"
    candidate.evidence_status = EvidenceStatus.VERIFIED_COUNTEREXAMPLE.value
    return candidate


def infer(observation: ExecutionObservation, *, observation_id: str, llm_payload: Optional[dict[str, Any]] = None) -> list[ConstraintCandidate]:
    analysis = FailureAnalyzer().analyze(observation)
    found: list[ConstraintCandidate] = []
    deterministic = from_analyzer(analysis, observation_id=observation_id)
    if deterministic is not None:
        found.append(validate_candidate(deterministic, traced_observation_ids=[observation_id]))
    elif llm_payload:
        found.append(propose_from_llm(llm_payload, observation_id=observation_id))
    return found
