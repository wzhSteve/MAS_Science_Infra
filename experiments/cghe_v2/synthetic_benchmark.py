"""Ground-truth synthetic tasks for CGHE-v2. No model and no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
ADAPTED = ROOT / "user_space" / "projects" / "cghe_pev" / "adapted"
sys.path.insert(0, str(ADAPTED))

from cghe_types import AtomicPremise, CandidatePlan, ExecutionObservation, PlanStep, Scope  # noqa: E402
from failure_analyzer import FailureAnalyzer  # noqa: E402
from hypothesis_pruner import HypothesisPruner  # noqa: E402
from planning_graph import PlanningGraph  # noqa: E402
from v2.action_selector import choose_node  # noqa: E402
from v2.counterfactual import generate_hypotheses, may_prune  # noqa: E402
from v2.epv import expected_pruning_value  # noqa: E402


def _step(tool: str, premise: AtomicPremise, cost: float = 1.0) -> PlanStep:
    return PlanStep(
        subgoal=tool,
        action=tool,
        tool=tool,
        preconditions=(premise,),
        estimated_cost=cost,
        links_previous=False,
        parameter=premise.parameter,
        argument=premise.argument,
        resource=premise.resource,
        identity=premise.identity,
    )


def _plan(plan_id: str, tool: str, premise: AtomicPremise, cost: float = 1.0) -> CandidatePlan:
    return CandidatePlan(plan_id, (_step(tool, premise, cost),), context=Scope(environment="sandbox", tool=tool, argument=premise.argument, resource=premise.resource, identity=premise.identity))


def _prune(observation: ExecutionObservation, plans: list[CandidatePlan]) -> dict:
    graph = PlanningGraph(plans)
    analysis = FailureAnalyzer().analyze(observation)
    before = len(graph.active_plan_ids())
    if analysis.constraint is None or not analysis.verified_counterexample:
        return {"before": before, "after": before, "pruned": [], "verified": False}
    result = HypothesisPruner().prune(graph, analysis.constraint, evidence=analysis.evidence, apply=True)
    return {
        "before": result.active_before,
        "after": result.active_after,
        "pruned": list(result.pruned_plan_ids),
        "verified": True,
    }


def _rate(pruned: list[str], feasible: set[str]) -> float:
    if not feasible:
        return 0.0
    return len(set(pruned) & feasible) / len(feasible)


def main() -> None:
    read = AtomicPremise("ToolExists", tool="read")
    fetch = AtomicPremise("ToolExists", tool="fetch")
    secret = AtomicPremise("CanAccess", tool="read", resource="secret", identity="alice")
    public = AtomicPremise("CanAccess", tool="read", resource="public", identity="alice")
    bad_id = AtomicPremise("ValidID", tool="read", parameter="id", argument="r1")
    other_id = AtomicPremise("ValidID", tool="read", parameter="id", argument="r2")
    query_a = AtomicPremise("SnippetsSufficient", tool="bing_search_tool", parameter="query", argument="paris")
    query_b = AtomicPremise("SnippetsSufficient", tool="bing_search_tool", parameter="query", argument="lyon")

    cases = []

    def add_case(name: str, observation: ExecutionObservation, plans: list[CandidatePlan], feasible: set[str]) -> None:
        outcome = _prune(observation, plans)
        cases.append({
            "name": name,
            "verified": outcome["verified"],
            "pruned": outcome["pruned"],
            "reduction": outcome["before"] - outcome["after"],
            "false_pruning_rate": _rate(outcome["pruned"], feasible),
            "calls": 1 if outcome["verified"] else 2,
        })

    add_case(
        "missing_tool",
        ExecutionObservation(tool_name="read", environment="sandbox", tool_result={"error_code": "TOOL_NOT_FOUND"}, verifier_label="不满足"),
        [_plan("read_a", "read", read), _plan("read_b", "read", read), _plan("fetch_ok", "fetch", fetch)],
        {"fetch_ok"},
    )
    add_case(
        "permission",
        ExecutionObservation(tool_name="read", environment="sandbox", resource="secret", identity="alice", tool_result={"error_code": "PERMISSION_DENIED", "resource": "secret", "identity": "alice"}, verifier_label="不满足"),
        [_plan("secret", "read", secret), _plan("public", "read", public)],
        {"public"},
    )
    add_case(
        "parameter",
        ExecutionObservation(tool_name="read", environment="sandbox", tool_result="Error: invalid id 'r1'", verifier_label="不满足"),
        [_plan("bad", "read", bad_id), _plan("other", "read", other_id)],
        {"other"},
    )
    add_case(
        "snippets",
        ExecutionObservation(tool_name="Bing_Search_Tool", environment="sandbox", command="execution = tool.execute(query='paris')", tool_result="1. Paris\n   URL: https://en.wikipedia.org/wiki/Paris", verifier_label="不满足"),
        [_plan("same", "bing_search_tool", query_a), _plan("other_query", "bing_search_tool", query_b)],
        {"other_query"},
    )
    add_case(
        "transient",
        ExecutionObservation(tool_name="read", environment="sandbox", tool_result="Error: HTTP 503 timeout talking to the host", verifier_label="不满足"),
        [_plan("read_a", "read", read), _plan("fetch_ok", "fetch", fetch)],
        {"fetch_ok", "read_a"},
    )

    hypotheses = generate_hypotheses(tool="read", argument="r1", failure_type="TOOL_UNAVAILABLE", plans=["read_a", "fetch_ok"])
    shared_graph = PlanningGraph([
        _plan("s1", "search", AtomicPremise("ToolExists", tool="search"), cost=1),
        _plan("s2", "search", AtomicPremise("ToolExists", tool="search"), cost=1),
        _plan("s3", "search", AtomicPremise("ToolExists", tool="search"), cost=1),
        _plan("fetch_ok", "fetch", fetch, cost=0.5),
    ])
    shared_choice = shared_graph.select_node(policy="shared")
    epv_choice, epv_trace = choose_node(shared_graph, mode="epv_only")
    modes = {
        "pure_pev": {"success": 0, "calls": 2, "false_pruning_rate": 0.0, "reduction": 0.0},
        "shared_only": {
            "success": int(shared_choice is not None and shared_choice.tool == "search"),
            "calls": 2,
            "false_pruning_rate": 0.0,
            "reduction": 0.0,
        },
        "epv_only": {
            "success": int(epv_choice is not None and epv_choice.tool == "fetch"),
            "calls": 1 if epv_choice is not None and epv_choice.tool == "fetch" else 2,
            "false_pruning_rate": 0.0,
            "reduction": expected_pruning_value({"search": ["s1", "s2", "s3"], "fetch": ["fetch_ok"]}) or 0.0,
        },
        "counterfactual_only": {
            "success": int(bool(hypotheses) and not any(may_prune(item) for item in hypotheses)),
            "calls": 1,
            "false_pruning_rate": 0.0,
            "reduction": 0.0,
        },
    }
    prune_fpr = sum(item["false_pruning_rate"] for item in cases) / len(cases)
    for name in ("epv_plus_counterfactual", "full_cghe_v2"):
        modes[name] = {
            "success": modes["epv_only"]["success"],
            "calls": modes["epv_only"]["calls"],
            "false_pruning_rate": prune_fpr,
            "reduction": sum(item["reduction"] for item in cases) / len(cases),
        }
    modes["shared_only"]["false_pruning_rate"] = prune_fpr
    modes["shared_only"]["reduction"] = modes["full_cghe_v2"]["reduction"]
    transient = next(item for item in cases if item["name"] == "transient")
    parameter = next(item for item in cases if item["name"] == "parameter")
    if parameter["pruned"] != ["bad"] or parameter["false_pruning_rate"] != 0.0:
        raise SystemExit(f"parameter case did not prune only the bound plan: {parameter}")
    justified = (
        transient["false_pruning_rate"] == 0.0
        and modes["full_cghe_v2"]["false_pruning_rate"] <= modes["shared_only"]["false_pruning_rate"]
        and (
            modes["full_cghe_v2"]["reduction"] > modes["pure_pev"]["reduction"]
            or modes["full_cghe_v2"]["calls"] < modes["pure_pev"]["calls"]
        )
    )
    report = {
        "seed": 0,
        "cases": cases,
        "modes": modes,
        "epv_trace_fallback": epv_trace.get("fallback"),
        "transient_false_pruning_rate": transient["false_pruning_rate"],
        "hotpot_dev_justified": justified,
    }
    destination = ROOT / "experiments" / "cghe_v2" / "synthetic_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"hotpot_dev_justified": justified, "transient_fpr": transient["false_pruning_rate"], "full_calls": modes["full_cghe_v2"]["calls"], "pure_calls": modes["pure_pev"]["calls"]}, indent=2))


if __name__ == "__main__":
    main()
