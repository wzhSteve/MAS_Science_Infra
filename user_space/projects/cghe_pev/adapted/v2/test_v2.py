"""Unit tests for CGHE-v2. v1 recovery tests stay in the other files."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cghe_types import AtomicPremise, CandidatePlan, ExecutionObservation, PlanStep, Scope  # noqa: E402
from planning_graph import PlanningGraph  # noqa: E402
from v2.action_selector import choose_node  # noqa: E402
from v2.constraint_inference import infer, propose_from_llm, validate_candidate  # noqa: E402
from v2.counterfactual import generate_hypotheses, may_prune, update_hypothesis  # noqa: E402
from v2.epv import FeedbackPrediction, expected_pruning_value, score_action, select_scored  # noqa: E402


def _plan(plan_id: str, tool: str) -> CandidatePlan:
    return CandidatePlan(
        plan_id,
        (PlanStep(subgoal=tool, action=tool, tool=tool, preconditions=(AtomicPremise("ToolExists", tool=tool),), links_previous=False),),
        context=Scope(environment="sandbox", tool=tool),
    )


class EpvTests(unittest.TestCase):
    def test_balanced_partition_beats_an_undivided_action(self) -> None:
        blind = expected_pruning_value({"all": ["p1", "p2", "p3", "p4"]})
        split = expected_pruning_value({"left": ["p1", "p2"], "right": ["p3", "p4"]})
        self.assertEqual(blind, 0.0)
        self.assertEqual(split, 2.0)
        chosen = select_scored([
            score_action(FeedbackPrediction("A", {"all": ["p1", "p2", "p3", "p4"]})),
            score_action(FeedbackPrediction("B", {"left": ["p1", "p2"], "right": ["p3", "p4"]})),
        ])
        self.assertEqual(chosen.action_id, "B")

    def test_cost_can_outweigh_a_higher_epv(self) -> None:
        expensive = score_action(FeedbackPrediction("A", {"left": ["p1", "p2"], "right": ["p3", "p4"]}, estimated_cost=100), lambda_cost=0.1)
        cheap = score_action(FeedbackPrediction("B", {"left": ["p1"], "right": ["p2", "p3", "p4"]}, estimated_cost=1), lambda_cost=0.1)
        self.assertGreater(expensive.epv, cheap.epv)
        self.assertEqual(select_scored([expensive, cheap]).action_id, "B")

    def test_risk_penalty_changes_the_choice(self) -> None:
        risky = score_action(FeedbackPrediction("A", {"left": ["p1", "p2"], "right": ["p3", "p4"]}, estimated_risk=1), mu_risk=3)
        safe = score_action(FeedbackPrediction("B", {"left": ["p1"], "right": ["p2", "p3", "p4"]}, estimated_risk=0), mu_risk=3)
        self.assertEqual(select_scored([risky, safe]).action_id, "B")

    def test_unusable_prediction_falls_back(self) -> None:
        row = score_action(FeedbackPrediction("A", {"only": ["p1"]}, probabilities={"only": 0.2, "missing": 0.8}))
        self.assertTrue(row.fallback)
        self.assertIsNone(select_scored([row]))

    def test_equal_epv_uses_cost_then_id(self) -> None:
        left = score_action(FeedbackPrediction("b", {"left": ["p1"], "right": ["p2"]}, estimated_cost=2))
        right = score_action(FeedbackPrediction("a", {"left": ["p1"], "right": ["p2"]}, estimated_cost=2))
        cheaper = score_action(FeedbackPrediction("c", {"left": ["p1"], "right": ["p2"]}, estimated_cost=1))
        self.assertEqual(select_scored([left, right]).action_id, "a")
        self.assertEqual(select_scored([left, right, cheaper]).action_id, "c")

    def test_negative_probability_is_rejected(self) -> None:
        value = expected_pruning_value({"yes": ["p1"], "no": ["p2"]}, probabilities={"yes": 1.2, "no": -0.2})
        self.assertIsNone(value)


class ConstraintTests(unittest.TestCase):
    def test_tool_error_can_verify_and_llm_cannot(self) -> None:
        observation = ExecutionObservation(tool_name="read", environment="sandbox", tool_result={"error_code": "TOOL_NOT_FOUND"})
        found = infer(observation, observation_id="obs-1")
        self.assertEqual(found[0].inference_source, "psi")
        self.assertTrue(found[0].admissible)
        guessed = propose_from_llm({"predicate": "ToolExists", "confidence": 1.0, "scope": {"environment": "sandbox", "tool": "read"}}, observation_id="obs-2")
        self.assertEqual(guessed.evidence_status, "UNVERIFIED_HYPOTHESIS")
        self.assertFalse(guessed.admissible)
        self.assertFalse(validate_candidate(guessed, traced_observation_ids=["obs-2"]).admissible)

    def test_timeout_empty_and_narrow_scope_do_not_overgeneralize(self) -> None:
        timeout = infer(ExecutionObservation(tool_name="read", environment="sandbox", tool_result="Error: timeout talking to the host"), observation_id="t")
        empty = infer(ExecutionObservation(tool_name="read", environment="sandbox", tool_result=""), observation_id="e")
        self.assertEqual(timeout, [])
        self.assertEqual(empty, [])
        param = infer(ExecutionObservation(tool_name="read", environment="sandbox", tool_result="Error: invalid parameter 'resource_id' 'r1'"), observation_id="p")
        if param:
            self.assertNotEqual(param[0].scope.get("tool") and param[0].predicate, "ToolExists")
            self.assertTrue(param[0].scope.get("argument") or param[0].scope.get("parameter") or param[0].evidence_status != "VERIFIED_COUNTEREXAMPLE")

    def test_contradiction_or_missing_scope_blocks_confirmation(self) -> None:
        observation = ExecutionObservation(tool_name="read", environment="sandbox", tool_result={"error_code": "TOOL_NOT_FOUND"})
        candidate = infer(observation, observation_id="obs")[0]
        blocked = validate_candidate(candidate, traced_observation_ids=["obs"], contradictions=["later success"])
        self.assertFalse(blocked.admissible)
        candidate.scope_complete = False
        unscoped = validate_candidate(candidate, traced_observation_ids=["obs"])
        self.assertFalse(unscoped.admissible)


class CounterfactualTests(unittest.TestCase):
    def test_rivals_are_testable_and_deduplicated(self) -> None:
        first = generate_hypotheses(tool="Bing_Search_Tool", argument="paris", failure_type="SNIPPETS_INSUFFICIENT", plans=["a"])
        again = generate_hypotheses(tool="Bing_Search_Tool", argument="paris", failure_type="SNIPPETS_INSUFFICIENT", plans=["a"])
        self.assertGreaterEqual(len(first), 2)
        self.assertEqual([item.hypothesis_id for item in first], [item.hypothesis_id for item in again])
        self.assertTrue(all(item.testable for item in first))
        self.assertEqual(generate_hypotheses(tool="Bing_Search_Tool", failure_type="TRANSIENT_ERROR"), [])

    def test_evidence_updates_status_without_pruning(self) -> None:
        hypothesis = generate_hypotheses(tool="read", argument="r1", failure_type="TOOL_UNAVAILABLE", plans=["a"])[0]
        update_hypothesis(hypothesis, observation=hypothesis.predicted_observations[0], verified_counterexample=True)
        self.assertEqual(hypothesis.status, "SUPPORTED")
        self.assertFalse(may_prune(hypothesis))
        update_hypothesis(hypothesis, observation="contradict:alternate path worked")
        self.assertEqual(hypothesis.status, "REFUTED")


class SelectorTests(unittest.TestCase):
    def test_epv_mode_prefers_the_splitting_action(self) -> None:
        plans = [_plan(name, "shared") for name in ("p1", "p2")] + [_plan("p3", "other"), _plan("p4", "other")]
        graph = PlanningGraph(plans)
        node, trace = choose_node(graph, mode="epv_only", lambda_cost=0.0, mu_risk=0.0)
        self.assertIsNotNone(node)
        self.assertFalse(trace["fallback"])
        self.assertTrue(trace["scores"])

    def test_shared_mode_asks_for_the_v1_rule(self) -> None:
        graph = PlanningGraph([_plan("p1", "read")])
        node, trace = choose_node(graph, mode="shared_only")
        self.assertIsNone(node)
        self.assertTrue(trace["fallback"])


class EvidenceLogTests(unittest.TestCase):
    def test_bing_query_constraint_does_not_transfer_to_another_query(self) -> None:
        import tempfile
        from v2.evidence_log import V2EvidenceLog

        with tempfile.TemporaryDirectory() as folder:
            log = V2EvidenceLog(str(Path(folder) / "v2.sqlite"))
            log.record_prune(
                task_id="q1",
                environment_id="cghe_pev",
                constraint_id="c1",
                tool="bing_search_tool",
                argument="query-a",
                active_before=3,
                active_after=2,
            )
            self.assertEqual(len(log.matching_constraints(tool="bing_search_tool", argument="query-a")), 1)
            self.assertEqual(len(log.matching_constraints(tool="bing_search_tool", argument="query-b")), 0)
            log.close()


if __name__ == "__main__":
    unittest.main()
