"""Scope-checked elimination of candidate plans.

A plan leaves the active set through cross-path pruning only when every
admission check holds: verified counterexample, direct observation, confidence
at or above the threshold, predicate equal to the fixed mapping of the
observation, complete dependency labels, an explicit premise on that plan, and
a context contained in the counterexample scope.

A parameter error never drops other uses of the same tool. If labeling is
incomplete, cross-path pruning stays closed. A direct parameter or schema
failure may then retire only the executed plan. An empty active set asks for
another planning round and does not mark the task impossible.
"""

from __future__ import annotations

from typing import Optional

from cghe_types import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    INSTANCE_BOUND_OBSERVATIONS,
    PSI_OBSERVATION_TO_PREDICATE,
    AtomicPremise,
    Constraint,
    EvidenceOrigin,
    EvidenceStatus,
    FailureEvidence,
    GraphNode,
    NEEDS_REPLAN_MESSAGE,
    PlanStatus,
    PruneRecord,
    PruneResult,
    Scope,
    context_contained,
    new_id,
    norm,
)
from planning_graph import PlanningGraph


_INFERENCE_SOURCES = frozenset({"llm_hypothesis", "llm", "inference"})


class HypothesisPruner:
    """Apply definition 9, or the instance-only fallback, to one fixed universe."""

    def __init__(self, *, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
        self.confidence_threshold = float(confidence_threshold)

    def prune(
        self,
        graph: PlanningGraph,
        constraint: Optional[Constraint],
        *,
        evidence: Optional[FailureEvidence] = None,
        executed_plan_id: str = "",
        apply: bool = True,
    ) -> PruneResult:
        active_before_ids = graph.active_plan_ids()
        active_before = len(active_before_ids)
        admissible, dep_complete, rejection = self._global_admission(graph, constraint)
        pruned: list[str] = []
        retired: list[str] = []
        snippet_instance = bool(
            constraint is not None
            and constraint.observation_code == "SNIPPETS_INSUFFICIENT"
        )
        if admissible and constraint is not None and snippet_instance:
            # Query-instance scope: remove plans whose Step-1 / premise matches
            # this query argument. Other queries stay. Prefer the executed plan
            # when labels are incomplete.
            failed_node = str(getattr(evidence, "node_id", "") or "")
            matched = {
                plan_id
                for plan_id in active_before_ids
                if self._plan_admitted(graph, plan_id, constraint)
            }
            if failed_node and hasattr(graph, "plans_starting_with"):
                matched.update(graph.plans_starting_with(failed_node))
            if matched:
                pruned.extend(sorted(matched))
            elif executed_plan_id and executed_plan_id in active_before_ids:
                retired.append(executed_plan_id)
        elif admissible and constraint is not None:
            for plan_id in active_before_ids:
                if self._plan_admitted(graph, plan_id, constraint):
                    pruned.append(plan_id)
        pruned_ids = tuple(sorted(set(pruned)))

        if not pruned_ids and not retired and self._retire_executed_only(
            constraint, evidence, dep_complete, rejection
        ):
            if executed_plan_id and executed_plan_id in active_before_ids:
                retired.append(executed_plan_id)
        retired_ids = tuple(sorted(retired))

        if apply:
            graph.retire_plans(pruned_ids, status=PlanStatus.PRUNED)
            graph.retire_plans(retired_ids, status=PlanStatus.RETIRED)

        active_after = active_before - len(pruned_ids) - len(retired_ids)
        needs_replan = active_before > 0 and active_after == 0
        affected = tuple(sorted(set(pruned_ids) | set(retired_ids)))
        source = _evidence_source(constraint, evidence)
        reason_code, reason = _reason(
            pruned_ids=pruned_ids,
            retired_ids=retired_ids,
            needs_replan=needs_replan,
            admissible=admissible,
            rejection=rejection,
        )
        constraint_id = constraint.constraint_id if constraint is not None else ""
        status = constraint.evidence_status.value if constraint is not None else ""
        observation_code = constraint.observation_code if constraint is not None else ""
        if evidence is not None and not observation_code:
            observation_code = evidence.observation_code
        record = PruneRecord(
            record_id=new_id("prune"),
            active_before=active_before,
            active_after=active_after,
            reason=reason,
            reason_code=reason_code,
            evidence_source=source,
            affected_plan_ids=affected,
            constraint_id=constraint_id,
            admissible=admissible,
            needs_replan=needs_replan,
            evidence_status=status,
            observation_code=observation_code,
        )
        return PruneResult(
            active_before=active_before,
            active_after=active_after,
            pruned_plan_ids=pruned_ids,
            instance_retired_plan_ids=retired_ids,
            affected_plan_ids=affected,
            reason=reason,
            reason_code=reason_code,
            evidence_source=source,
            needs_replan=needs_replan,
            admissible=admissible,
            dep_complete=dep_complete,
            message=NEEDS_REPLAN_MESSAGE if needs_replan else "",
            records=(record,),
            round_id=graph.round_id,
            constraint_id=constraint_id,
        )

    def _global_admission(
        self,
        graph: PlanningGraph,
        constraint: Optional[Constraint],
    ) -> tuple[bool, bool, str]:
        if constraint is None:
            return False, False, "no constraint"
        if constraint.evidence_status != EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
            return False, False, "evidence is not a verified counterexample"
        if constraint.origin != EvidenceOrigin.DIRECT_OBSERVATION or not constraint.direct:
            return False, False, "evidence is not a direct observation"
        if norm(constraint.source) in _INFERENCE_SOURCES:
            return False, False, "source is an inference"
        if constraint.confidence < self.confidence_threshold:
            return False, False, "confidence is below the threshold"
        expected = PSI_OBSERVATION_TO_PREDICATE.get(constraint.observation_code, "")
        if not expected or constraint.condition.predicate != expected:
            return False, False, "predicate does not match the observation mapping"
        if not constraint.scope.tool or not constraint.scope.environment:
            return False, False, "scope must name the tool and environment"
        if constraint.observation_code == "PARAM_ERROR":
            if constraint.condition.predicate != "ValidID" or not constraint.scope.parameter_level():
                return False, False, "parameter error is not bound to an argument"
            if constraint.disables_tool:
                return False, False, "parameter error cannot disable the tool"
        if constraint.observation_code == "SNIPPETS_INSUFFICIENT":
            if constraint.condition.predicate != "SnippetsSufficient" or not constraint.scope.parameter_level():
                return False, False, "snippet gap is not bound to the search query"
            if constraint.disables_tool:
                return False, False, "a snippet gap cannot disable the search tool"
            # Still admissible for instance retirement; cross-path wipe is closed above.
        if constraint.observation_code == "PERMISSION_DENIED" and (
            not constraint.scope.resource or not constraint.scope.identity
        ):
            return False, False, "permission scope must name the current resource and identity"
        if constraint.observation_code == "SCHEMA_MISMATCH" and not constraint.scope.schema_signature:
            return False, False, "schema scope must name the call shape"
        if constraint.disables_tool and constraint.scope.parameter_level():
            return False, False, "a parameter-level scope cannot disable the tool"
        complete = dependency_complete(graph, constraint.condition)
        if not complete:
            return False, False, "dependency labeling is incomplete"
        return True, True, ""

    def _plan_admitted(self, graph: PlanningGraph, plan_id: str, constraint: Constraint) -> bool:
        plan = graph.plan(plan_id)
        dependent = [
            graph.node(node_id)
            for node_id in graph.plan_node_ids(plan_id)
            if graph.node(node_id).has_premise(constraint.condition)
        ]
        if not dependent:
            return False
        if constraint.observation_code == "PARAM_ERROR":
            # Same tool, different argument: the premise key already differs.
            # Refuse a tool-wide reading even if a caller widened the scope.
            if not constraint.scope.parameter_level():
                return False
        return all(
            context_contained(_effective_context(plan.context, constraint.condition, node), constraint.scope)
            for node in dependent
        )

    def _retire_executed_only(
        self,
        constraint: Optional[Constraint],
        evidence: Optional[FailureEvidence],
        dep_complete: bool,
        rejection: str,
    ) -> bool:
        """Definition 12's instance update, only for a direct bound call whose labels are incomplete."""
        if dep_complete or rejection != "dependency labeling is incomplete":
            return False
        code = ""
        direct = False
        origin = None
        confidence = 0.0
        status = None
        bound = False
        if evidence is not None:
            code = evidence.observation_code
            direct = evidence.direct
            origin = evidence.origin
            bound = evidence.bound_to_call
            confidence = 1.0 if evidence.direct else 0.0
            status = EvidenceStatus.VERIFIED_COUNTEREXAMPLE if evidence.direct else None
        if constraint is not None:
            code = constraint.observation_code or code
            direct = constraint.direct
            origin = constraint.origin
            confidence = constraint.confidence
            status = constraint.evidence_status
            bound = bound or constraint.observation_code in INSTANCE_BOUND_OBSERVATIONS
        if status != EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
            return False
        if not direct or origin != EvidenceOrigin.DIRECT_OBSERVATION:
            return False
        if confidence < self.confidence_threshold:
            return False
        if code not in INSTANCE_BOUND_OBSERVATIONS or not bound:
            return False
        expected = PSI_OBSERVATION_TO_PREDICATE.get(code, "")
        if constraint is not None and constraint.condition.predicate != expected:
            return False
        return True


def dependency_complete(graph: PlanningGraph, premise: AtomicPremise) -> bool:
    """Every node that ``¬premise`` would block already lists that premise.

    A missing label closes cross-path pruning for this premise. It does not
    by itself delete sibling plans.
    """
    for node in graph.nodes():
        if _blocked_by(node, premise) and not node.has_premise(premise):
            return False
    return True


def _blocked_by(node: GraphNode, premise: AtomicPremise) -> bool:
    item = premise.normalized()
    tool = item.tool
    if not tool or tool not in {norm(node.tool), norm(node.action)}:
        return False
    if item.predicate == "ToolExists":
        return True
    if item.predicate == "CanAccess":
        return bool(item.resource) and norm(node.resource) == item.resource
    if item.predicate == "SchemaOk":
        return bool(item.schema_signature) and norm(node.schema_signature) == item.schema_signature
    if item.predicate == "ValidID":
        if item.argument:
            return norm(node.argument) == item.argument
        if item.parameter:
            return norm(node.parameter) == item.parameter and not item.argument
        return False
    if item.predicate == "SnippetsSufficient":
        return bool(item.argument) and norm(node.tool) == tool and norm(node.argument) == item.argument
    return False


def _effective_context(plan_context: Scope, premise: AtomicPremise, node: GraphNode) -> Scope:
    base = plan_context.normalized()
    item = premise.normalized()
    return Scope(
        environment=base.environment,
        tool=item.tool or norm(node.tool) or base.tool,
        resource=item.resource or norm(node.resource) or base.resource,
        identity=item.identity or norm(node.identity) or base.identity,
        parameter=item.parameter or norm(node.parameter) or base.parameter,
        argument=item.argument or norm(node.argument) or base.argument,
        schema_signature=item.schema_signature or norm(node.schema_signature) or base.schema_signature,
        task_type=base.task_type,
        call_instance=base.call_instance,
    )


def _evidence_source(
    constraint: Optional[Constraint],
    evidence: Optional[FailureEvidence],
) -> str:
    if constraint is not None and constraint.source:
        return constraint.source
    if evidence is not None:
        return evidence.source
    return ""


def _reason(
    *,
    pruned_ids: tuple[str, ...],
    retired_ids: tuple[str, ...],
    needs_replan: bool,
    admissible: bool,
    rejection: str,
) -> tuple[str, str]:
    if needs_replan and pruned_ids:
        return "needs_replan", "admissible counterexample removed every active plan; " + NEEDS_REPLAN_MESSAGE
    if needs_replan and retired_ids:
        return "needs_replan", "the executed plan was the last active candidate; " + NEEDS_REPLAN_MESSAGE
    if pruned_ids:
        return "admissible_prune", "admissible counterexample removed plans that depend on the violated premise"
    if retired_ids:
        return (
            "instance_only",
            "failure is bound to this parameter or call, so only the executed plan is retired",
        )
    if admissible:
        return "no_change", "counterexample is admissible but no active plan depends on it inside the scope"
    if rejection:
        return "not_admissible", rejection
    return "no_change", "version space is unchanged"
