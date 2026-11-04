"""Spec checks for Step-1 path coverage, pruning, and bounded regeneration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cghe_types import (  # noqa: E402
    AtomicPremise,
    CandidatePlan,
    CausalRelation,
    Constraint,
    EvidenceOrigin,
    EvidenceStatus,
    PlanStatus,
    PlanStep,
    Scope,
)
from cghe_loop import (  # noqa: E402
    CGHESession,
    make_regeneration_context,
    prune_by_counterexample,
    validate_dag,
)
from failure_analyzer import FailureAnalyzer  # noqa: E402
from hypothesis_pruner import HypothesisPruner  # noqa: E402
from planning_graph import PlanningGraph  # noqa: E402


def _step(name: str, *premises: AtomicPremise) -> PlanStep:
    return PlanStep(subgoal=name, action=name, tool=name, preconditions=premises, links_previous=True)


def _plan(plan_id: str, names: list[str], premises: tuple[AtomicPremise, ...] = ()) -> CandidatePlan:
    steps = []
    for index, name in enumerate(names):
        steps.append(_step(name, *(premises if index == 0 else ())))
    return CandidatePlan(plan_id, tuple(steps), context=Scope(environment="sandbox", tool=names[0]))


def _node(graph: PlanningGraph, action: str) -> str:
    want = action.casefold()
    for node in graph.nodes():
        if str(node.action).casefold() == want or str(node.tool).casefold() == want:
            return node.node_id
    raise AssertionError(action)


class Step1CoverageTests(unittest.TestCase):
    def test_shared_first_node_beats_a_single_path(self) -> None:
        graph = PlanningGraph([
            _plan("p1", ["A", "X"]),
            _plan("p2", ["A", "Y"]),
            _plan("p3", ["B", "Z"]),
        ])
        coverage = graph.compute_step1_coverage()
        self.assertEqual(coverage[_node(graph, "A")], 2)
        self.assertEqual(coverage[_node(graph, "B")], 1)
        selected = graph.select_step1()
        assert selected is not None
        self.assertEqual(str(graph.node(selected[0]).action).casefold(), "a")
        self.assertEqual(selected[1], 2)

    def test_pruned_plans_leave_step1_coverage(self) -> None:
        graph = PlanningGraph([
            _plan("p1", ["A", "X"]),
            _plan("p2", ["A", "Y"]),
            _plan("p3", ["B", "Z"]),
        ])
        graph.retire_plans(["p1", "p2"], status=PlanStatus.PRUNED, prune_reason="e1")
        coverage = graph.compute_step1_coverage()
        self.assertNotIn(_node(graph, "A"), coverage)
        self.assertEqual(coverage[_node(graph, "B")], 1)

    def test_inactive_statuses_do_not_count(self) -> None:
        graph = PlanningGraph([
            _plan("done", ["A"]),
            _plan("cut", ["A"]),
            _plan("left", ["A"]),
            _plan("drop", ["B"]),
            _plan("keep", ["C"]),
        ])
        graph.complete_plans(["done"])
        graph.retire_plans(["cut"], status=PlanStatus.PRUNED)
        graph.retire_plans(["left"], status=PlanStatus.ABANDONED)
        graph.retire_plans(["drop"], status=PlanStatus.RETIRED)
        coverage = graph.compute_step1_coverage()
        self.assertEqual(coverage, {_node(graph, "C"): 1})

    def test_more_paths_beat_more_descendants_and_more_successors(self) -> None:
        graph = PlanningGraph([
            _plan("only", ["A", "A2", "A3", "A4"]),
            _plan("b1", ["B", "Z"]),
            _plan("b2", ["B", "Z"]),
            _plan("b3", ["B", "Z"]),
        ])
        selected = graph.select_step1()
        assert selected is not None
        self.assertEqual(str(graph.node(selected[0]).action).casefold(), "b")
        self.assertEqual(selected[1], 3)
        self.assertGreater(graph.out_degree(_node(graph, "A")) + 3, graph.out_degree(_node(graph, "B")))

    def test_initial_tie_does_not_regenerate(self) -> None:
        graph = PlanningGraph([_plan("p1", ["A"]), _plan("p2", ["B"]), _plan("p3", ["C"])])
        self.assertEqual(graph.attempt_count, 0)
        self.assertEqual(set(graph.compute_step1_coverage().values()), {1})
        self.assertFalse(graph.coverage_saturated())
        self.assertEqual(graph.regeneration_reasons(), [])

    def test_equal_coverage_after_an_attempt_is_saturated(self) -> None:
        graph = PlanningGraph([_plan("p1", ["A"]), _plan("p2", ["B"])])
        node = _node(graph, "A")
        graph.record_step1_attempt(node, attempt_id="a1", selected_plan_ids=("p1",), coverage=1)
        self.assertTrue(graph.coverage_saturated())
        self.assertIn("COVERAGE_SATURATED", graph.regeneration_reasons())

    def test_one_remaining_step1_is_not_saturated(self) -> None:
        graph = PlanningGraph([_plan("p1", ["A"]), _plan("p2", ["B"])])
        graph.retire_plans(["p2"])
        graph.record_step1_attempt(_node(graph, "A"), attempt_id="a1", selected_plan_ids=("p1",), coverage=1)
        self.assertFalse(graph.coverage_saturated())
        selected = graph.select_step1()
        assert selected is not None
        self.assertEqual(str(graph.node(selected[0]).action).casefold(), "a")

    def test_no_eligible_step1_asks_for_regeneration_without_counting(self) -> None:
        class Controller:
            graph = None

        graph = PlanningGraph([_plan("p1", ["A"])])
        graph.mark_failed(_node(graph, "A"))
        controller = Controller()
        controller.graph = graph
        session = CGHESession(controller)
        decision, reasons = session.next_decision()
        self.assertEqual(decision, "REGENERATE")
        self.assertEqual(reasons, ["NO_EXECUTABLE_CANDIDATE"])
        again, _ = session.next_decision()
        self.assertEqual(again, "REGENERATE")
        self.assertEqual(graph.attempt_count, 0)
        self.assertEqual(session.global_attempts, 0)

    def test_three_real_calls_reach_the_per_dag_budget(self) -> None:
        graph = PlanningGraph([_plan("p1", ["A", "X"]), _plan("p2", ["A", "Y"])])
        node = _node(graph, "A")
        for index in range(3):
            graph.record_step1_attempt(node, attempt_id=f"a{index}", selected_plan_ids=("p1", "p2"), coverage=2)
        self.assertIn("MAX_STEP1_ATTEMPTS", graph.regeneration_reasons())
        graph.record_step1_attempt(node, attempt_id="a0", selected_plan_ids=("p1",), coverage=2)
        self.assertEqual(graph.attempt_count, 3)

    def test_invalid_plans_do_not_spend_an_attempt(self) -> None:
        class Controller:
            graph = None

        controller = Controller()
        session = CGHESession(controller)
        ctx = make_regeneration_context(query="q", trigger_reasons=["MAX_STEP1_ATTEMPTS"])
        self.assertIsNone(session.regenerate([], ctx))
        self.assertEqual(session.global_attempts, 0)
        self.assertIsNone(controller.graph)

    def test_new_dag_resets_its_counter_and_keeps_the_global_one(self) -> None:
        class Controller:
            graph = None

        old = PlanningGraph([_plan("p1", ["A"]), _plan("p2", ["B"])])
        old.record_step1_attempt(_node(old, "A"), attempt_id="a1", selected_plan_ids=("p1",), coverage=1)
        controller = Controller()
        controller.graph = old
        session = CGHESession(controller)
        session.global_attempts = 4
        ctx = make_regeneration_context(
            query="q",
            trigger_reasons=["MAX_STEP1_ATTEMPTS", "COVERAGE_SATURATED"],
            verified_counterexamples=[{"record_id": "m-1"}],
            verified_successes=[{"record_id": "m+1"}],
            unverified_failures=[{"note": "hint"}],
        )
        fresh = session.regenerate([_plan("n1", ["C"]), _plan("n2", ["D"])], ctx)
        assert fresh is not None
        self.assertEqual(fresh.attempt_count, 0)
        self.assertNotEqual(fresh.dag_id, old.dag_id)
        self.assertEqual(session.global_attempts, 4)
        self.assertEqual(session.regeneration_count, 1)
        events = [item for item in session.events if item["event"] == "dag_regenerated"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["trigger_reasons"], ["MAX_STEP1_ATTEMPTS", "COVERAGE_SATURATED"])
        self.assertIn("m-1", events[0]["inherited_ids"])
        self.assertFalse(ctx["unverified_failures_are_hard_constraints"])
        self.assertIn("NOT hard constraints", __import__("cghe_loop").context_guidance(ctx))


class PruneTests(unittest.TestCase):
    def _constraint(self, *, predicate: str = "ToolExists", tool: str = "A", status=EvidenceStatus.VERIFIED_COUNTEREXAMPLE, direct: bool = True, environment: str = "sandbox", resource: str = "", identity: str = "") -> Constraint:
        return Constraint(
            constraint_id="c1",
            condition=AtomicPremise(predicate=predicate, tool=tool, resource=resource, identity=identity),
            scope=Scope(environment=environment, tool=tool, resource=resource, identity=identity),
            evidence_status=status,
            source="tool",
            confidence=1.0,
            direct=direct,
            origin=EvidenceOrigin.DIRECT_OBSERVATION if direct else EvidenceOrigin.INFERENCE,
            observation_code="TOOL_NOT_FOUND" if predicate == "ToolExists" else "PERMISSION_DENIED",
            evidence_id="e1",
        )

    def test_timeout_does_not_prune(self) -> None:
        from cghe_types import ExecutionObservation

        graph = PlanningGraph([
            _plan("p1", ["read"], (AtomicPremise(predicate="ToolExists", tool="read"),)),
            _plan("p2", ["other"]),
        ])
        analysis = FailureAnalyzer().analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result="Error: HTTP 503 timeout talking to the host",
            verifier_label="不满足",
        ))
        self.assertFalse(analysis.verified_counterexample)
        self.assertEqual(prune_by_counterexample(graph, analysis.constraint, evidence=analysis.evidence), [])
        self.assertEqual(set(graph.active_plan_ids()), {"p1", "p2"})

    def test_incomplete_or_mismatched_scope_does_not_prune(self) -> None:
        premise = AtomicPremise(predicate="CanAccess", tool="fetch", resource="r", identity="alice")
        graph = PlanningGraph([
            CandidatePlan("p1", (_step("fetch", premise),), context=Scope(environment="sandbox", tool="fetch", resource="r", identity="alice")),
        ])
        incomplete = self._constraint(predicate="CanAccess", tool="fetch")
        self.assertEqual(HypothesisPruner().prune(graph, incomplete, apply=True).pruned_plan_ids, ())
        mismatched = self._constraint(predicate="CanAccess", tool="fetch", resource="other", identity="bob")
        self.assertEqual(HypothesisPruner().prune(graph, mismatched, apply=True).pruned_plan_ids, ())
        self.assertEqual(graph.active_plan_ids(), ("p1",))

    def test_only_explicit_dependents_are_pruned_and_shared_nodes_stay(self) -> None:
        premise = AtomicPremise(predicate="ToolExists", tool="A")
        other = AtomicPremise(predicate="ToolExists", tool="Q")
        graph = PlanningGraph([
            CandidatePlan("h1", (_step("A", premise), _step("S"), _step("X")), context=Scope(environment="sandbox", tool="A")),
            CandidatePlan("h2", (_step("A", premise), _step("S"), _step("Y")), context=Scope(environment="sandbox", tool="A")),
            CandidatePlan("free", (_step("Q", other), _step("S"), _step("Z")), context=Scope(environment="sandbox", tool="Q")),
        ])
        shared = _node(graph, "S")
        pruned = prune_by_counterexample(graph, self._constraint())
        self.assertEqual(pruned, ["h1", "h2"])
        self.assertEqual(graph.active_plan_ids(), ("free",))
        self.assertEqual(graph.node(shared).status.value, "pending")
        self.assertNotEqual(graph.node(_node(graph, "A")).status.value, "pending")


class HandoffTests(unittest.TestCase):
    def test_step1_success_is_not_task_completion(self) -> None:
        class Controller:
            graph = None
            memory = None

        graph = PlanningGraph([_plan("p1", ["A", "X"]), _plan("p2", ["B"])])
        controller = Controller()
        controller.graph = graph
        session = CGHESession(controller)
        outcome = session.handoff_after_step1_success(graph, _node(graph, "A"), None)
        self.assertFalse(outcome["task_completed"])
        self.assertEqual(outcome["decision"], "continue_pev")
        self.assertFalse(outcome["graph_cleared"])
        self.assertEqual(str(outcome["remaining"][0]["action"]).casefold(), "x")
        self.assertIs(controller.graph, graph)
        self.assertFalse(validate_dag(graph))


class CausalMemoryTests(unittest.TestCase):
    def test_observed_cause_requires_direct_evidence(self) -> None:
        from evidence_memory import EvidenceMemory
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        memory = EvidenceMemory(str(Path(tmp.name) / "m.sqlite"))
        try:
            refused = memory.add_causal_edge("a", "b", CausalRelation.OBSERVED_TO_CAUSE, direct=False)
            kept = memory.add_causal_edge("a", "b", CausalRelation.OBSERVED_TO_CAUSE, direct=True)
            self.assertIsNone(refused)
            self.assertIsNotNone(kept)
            self.assertEqual(len(memory.causal_edges()), 1)
        finally:
            memory.close()
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
