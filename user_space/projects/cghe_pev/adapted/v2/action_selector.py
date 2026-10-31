"""One selector for v2. EPV scores feedback partitions; hypotheses only supply them."""

from __future__ import annotations

from typing import Optional

from planning_graph import PlanningGraph

from v2.counterfactual import feedback_partition, generate_hypotheses
from v2.epv import FeedbackPrediction, ScoreBreakdown, score_action, select_scored


_SHARED_MODES = {"shared_only", "graph_only", "constraint_inference_only", "full", "v1"}
_EPV_MODES = {"epv_only", "epv_plus_counterfactual", "full_cghe_v2", "counterfactual_only"}


def _node_partition(graph: PlanningGraph, node_id: str) -> dict[str, list[str]]:
    active = list(graph.active_plan_ids())
    node = graph.node(node_id)
    inside = [plan_id for plan_id in node.candidate_plan_ids if plan_id in active]
    outside = [plan_id for plan_id in active if plan_id not in inside]
    outcomes = {"matches_action": inside}
    if outside:
        outcomes["other_plans"] = outside
    return outcomes


def choose_node(
    graph: PlanningGraph,
    *,
    mode: str = "full_cghe_v2",
    lambda_cost: float = 0.0,
    mu_risk: float = 0.0,
    failure_type: str = "SUBGOAL_NOT_SATISFIED",
) -> tuple[Optional[object], dict]:
    """Return ``(node, trace)``. A missing node means the caller uses shared-node order."""
    mode_name = str(mode or "full_cghe_v2")
    trace: dict = {"mode": mode_name, "scores": [], "selected": "", "fallback": False, "reason": ""}
    if mode_name in _SHARED_MODES or mode_name not in _EPV_MODES:
        trace["fallback"] = True
        trace["reason"] = "mode keeps the v1 shared-node rule"
        return None, trace
    ready = list(graph.executable_node_ids())
    if not ready:
        trace["fallback"] = True
        trace["reason"] = "no executable node"
        return None, trace
    rows: list[ScoreBreakdown] = []
    hypotheses = []
    if mode_name in {"counterfactual_only", "epv_plus_counterfactual", "full_cghe_v2"}:
        sample = graph.node(ready[0])
        hypotheses = generate_hypotheses(tool=sample.tool, argument=sample.argument, failure_type=failure_type, plans=graph.active_plan_ids())
    for node_id in ready:
        node = graph.node(node_id)
        outcomes = _node_partition(graph, node_id)
        if mode_name == "counterfactual_only":
            grouped = feedback_partition(hypotheses, graph.active_plan_ids())
            if grouped is None:
                trace["fallback"] = True
                trace["reason"] = "counterfactual hypotheses do not discriminate"
                return None, trace
            outcomes = grouped
        prediction = FeedbackPrediction(
            action_id=node_id,
            outcomes=outcomes,
            estimated_cost=float(node.estimated_cost),
            estimated_risk=0.0 if node.preconditions else 1.0,
            evidence_basis="explicit_premise" if mode_name != "counterfactual_only" else "counterfactual",
        )
        rows.append(score_action(prediction, lambda_cost=lambda_cost, mu_risk=mu_risk))
    chosen = select_scored(rows)
    trace["scores"] = [
        {"action_id": row.action_id, "epv": row.epv, "cost": row.cost, "risk": row.risk, "score": row.score, "fallback": row.fallback}
        for row in rows
    ]
    if chosen is None:
        trace["fallback"] = True
        trace["reason"] = "no legal EPV score"
        return None, trace
    trace["selected"] = chosen.action_id
    trace["reason"] = "argmax J(a)"
    return graph.node(chosen.action_id), trace
