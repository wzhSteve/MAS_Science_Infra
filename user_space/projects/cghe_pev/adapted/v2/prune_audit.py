"""Dry-run record around the existing pruner. Admission rules stay in HypothesisPruner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cghe_types import Constraint, FailureEvidence
from hypothesis_pruner import HypothesisPruner
from planning_graph import PlanningGraph


@dataclass(frozen=True)
class PruneAudit:
    constraint_id: str
    evidence_source: str
    evidence_id: str
    admissible: bool
    scope_tool: str
    scope_argument: str
    active_before: int
    active_after: int
    pruned_plan_ids: tuple[str, ...]
    applied: bool
    reason: str


def audit_prune(
    pruner: HypothesisPruner,
    graph: PlanningGraph,
    constraint: Optional[Constraint],
    *,
    evidence: Optional[FailureEvidence] = None,
    apply: bool = False,
) -> PruneAudit:
    result = pruner.prune(graph, constraint, evidence=evidence, apply=apply)
    scope = constraint.scope if constraint is not None else None
    return PruneAudit(
        constraint_id="" if constraint is None else constraint.constraint_id,
        evidence_source=result.evidence_source,
        evidence_id="" if evidence is None else evidence.evidence_id,
        admissible=bool(result.records and result.records[0].admissible),
        scope_tool="" if scope is None else scope.tool,
        scope_argument="" if scope is None else scope.argument,
        active_before=result.active_before,
        active_after=result.active_after,
        pruned_plan_ids=result.pruned_plan_ids,
        applied=apply,
        reason=result.reason,
    )
