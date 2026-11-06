"""Unit tests for the CGHE planning graph, failure analyzer and pruner."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cghe_types import (  # noqa: E402
    NEEDS_REPLAN_MESSAGE,
    AtomicPremise,
    CandidatePlan,
    Constraint,
    EvidenceOrigin,
    EvidenceStatus,
    ExecutionObservation,
    FailureType,
    PlanStep,
    Scope,
    schema_signature,
)
from failure_analyzer import FailureAnalyzer  # noqa: E402
from hypothesis_pruner import HypothesisPruner  # noqa: E402
from planning_graph import PlanningGraph, UniverseFrozenError  # noqa: E402


def _step(
    tool: str,
    *,
    subgoal: str = "",
    premises: tuple[AtomicPremise, ...] = (),
    state: tuple[str, ...] = (),
    cost: float = 1.0,
    resource: str = "",
    argument: str = "",
    parameter: str = "",
    schema: str = "",
    identity: str = "",
    links_previous: bool = True,
) -> PlanStep:
    return PlanStep(
        subgoal=subgoal or tool,
        action=tool,
        tool=tool,
        preconditions=premises,
        required_state=state,
        estimated_cost=cost,
        resource=resource,
        argument=argument,
        parameter=parameter,
        schema_signature=schema,
        identity=identity,
        links_previous=links_previous,
    )


def _plan(plan_id: str, steps: tuple[PlanStep, ...], *, environment: str = "sandbox") -> CandidatePlan:
    return CandidatePlan(plan_id, steps, context=Scope(environment=environment))


def _exists(tool: str) -> AtomicPremise:
    return AtomicPremise(predicate="ToolExists", tool=tool)


def _access(tool: str, resource: str, identity: str = "alice") -> AtomicPremise:
    return AtomicPremise(
        predicate="CanAccess",
        tool=tool,
        resource=resource,
        identity=identity,
    )


def _valid(tool: str, argument: str, parameter: str = "id") -> AtomicPremise:
    return AtomicPremise(
        predicate="ValidID",
        tool=tool,
        parameter=parameter,
        argument=argument,
    )


def _schema(tool: str, signature: str) -> AtomicPremise:
    return AtomicPremise(predicate="SchemaOk", tool=tool, schema_signature=signature)


class PlanningGraphTests(unittest.TestCase):
    def test_compatible_steps_merge_and_incompatible_steps_do_not(self) -> None:
        shared = _exists("read")
        merged = PlanningGraph([
            _plan("p1", (_step("read", premises=(shared,), state=("ready",), subgoal="one"),)),
            _plan("p2", (_step("Read", premises=(shared,), state=("ready",), subgoal="two"),)),
        ])
        self.assertEqual(len(merged.nodes()), 1)
        node = merged.nodes()[0]
        self.assertEqual(node.candidate_plan_ids, ("p1", "p2"))
        self.assertEqual(merged.sharing_score(node.node_id), 2)
        self.assertEqual(node.subgoal, "one")

        different_premise = PlanningGraph([
            _plan("p1", (_step("read", premises=(shared,)),)),
            _plan("p2", (_step("read"),)),
        ])
        self.assertEqual(len(different_premise.nodes()), 2)
        self.assertTrue(all(different_premise.sharing_score(item.node_id) == 1 for item in different_premise.nodes()))

        different_state = PlanningGraph([
            _plan("p1", (_step("read", premises=(shared,), state=("have_id",)),)),
            _plan("p2", (_step("read", premises=(shared,), state=("have_token",)),)),
        ])
        self.assertEqual(len(different_state.nodes()), 2)

    def test_universe_is_fixed_and_node_ids_do_not_depend_on_plan_order(self) -> None:
        left = _plan("a", (_step("read", premises=(_exists("read"),)),))
        right = _plan("b", (_step("export", premises=(_exists("export"),)),))
        forward = PlanningGraph([left, right], round_id="e1")
        backward = PlanningGraph([right, left], round_id="e1")
        self.assertEqual(
            tuple(node.node_id for node in forward.nodes()),
            tuple(node.node_id for node in backward.nodes()),
        )
        self.assertEqual(forward.candidate_plan_ids, ("a", "b"))
        with self.assertRaises(UniverseFrozenError):
            forward.add_candidate(_plan("c", (_step("read"),)))
        self.assertEqual(forward.candidate_plan_ids, ("a", "b"))

    def test_selection_uses_active_sharing_then_cost_then_node_id(self) -> None:
        long_plan = _plan("long", (
            _step("read", premises=(_exists("read"),), cost=1),
            _step("write", premises=(_exists("write"),), cost=1),
            _step("export", premises=(_exists("export"),), cost=1),
        ))
        short_a = _plan("s1", (_step("ping", premises=(_exists("ping"),), cost=9),))
        short_b = _plan("s2", (_step("ping", premises=(_exists("ping"),), cost=9),))
        graph = PlanningGraph([long_plan, short_a, short_b])
        selected = graph.select_step1()
        self.assertIsNotNone(selected)
        assert selected is not None
        chosen = graph.node(selected[0])
        self.assertEqual(chosen.tool, "ping")
        self.assertEqual(selected[1], 2)
        self.assertEqual(graph.compute_step1_coverage()[chosen.node_id], 2)
        read_id = next(node.node_id for node in graph.nodes() if node.tool == "read")
        self.assertEqual(graph.sharing_score(read_id), 1)
        self.assertGreater(len(graph.node(read_id).dependencies) + 2, graph.sharing_score(read_id))

        cheap = _plan("cheap", (_step("cheap", cost=1),))
        costly = _plan("costly", (_step("costly", cost=5),))
        tied = PlanningGraph([cheap, costly])
        # Legacy shared policy still prefers lower cost. Step-1 coverage does not.
        self.assertEqual(tied.select_node(policy="shared").tool, "cheap")
        step1 = tied.select_step1()
        assert step1 is not None
        self.assertEqual(step1[1], 1)
        self.assertEqual(tied.node(step1[0]).node_id, min(tied.compute_step1_coverage()))

        low = _plan("low", (_step("tool", state=("a",), cost=3),))
        high = _plan("high", (_step("tool", state=("b",), cost=3),))
        stable = PlanningGraph([high, low])
        ready = stable.executable_node_ids()
        self.assertEqual(stable.select_node().node_id, min(ready))
        self.assertEqual(graph.select_node(policy="first_plan").tool, "read")

    def test_pruned_plans_drop_out_of_the_sharing_score(self) -> None:
        graph = PlanningGraph([
            _plan("p1", (_step("shared", cost=5),)),
            _plan("p2", (_step("shared", cost=5),)),
            _plan("p3", (_step("private", cost=1),)),
        ])
        self.assertEqual(graph.select_node().tool, "shared")
        graph.retire_plans(["p1", "p2"])
        shared = next(node for node in graph.nodes() if node.tool == "shared")
        self.assertEqual(shared.candidate_plan_ids, ("p1", "p2"))
        self.assertEqual(graph.sharing_score(shared.node_id), 0)
        self.assertEqual(graph.select_node().tool, "private")

    def test_a_node_waits_until_its_own_plan_prefix_is_satisfied(self) -> None:
        graph = PlanningGraph([
            _plan("p1", (
                _step("setup", cost=10),
                _step("finish", cost=0.1),
            )),
            _plan("p2", (
                _step("setup", cost=10),
                _step("finish", cost=0.1),
            )),
        ])
        first = graph.select_node()
        self.assertEqual(first.tool, "setup")
        self.assertNotIn("finish", graph.executable_node_ids() and [
            graph.node(node_id).tool for node_id in graph.executable_node_ids()
        ])
        graph.mark_satisfied(first.node_id)
        second = graph.select_node()
        self.assertEqual(second.tool, "finish")
        self.assertEqual(graph.sharing_score(second.node_id), 2)


class FailureAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = FailureAnalyzer()

    def _observe(self, **kwargs: object) -> ExecutionObservation:
        base = dict(tool_name="read", environment="sandbox", identity="alice")
        base.update(kwargs)
        return ExecutionObservation(**base)

    def test_direct_tool_permission_and_schema_mappings(self) -> None:
        missing = self.analyzer.analyze(self._observe(
            tool_result={"error_code": "TOOL_NOT_FOUND", "tool": "read"},
            llm_hypothesis="the resource does not exist",
        ))
        self.assertTrue(missing.verified_counterexample)
        self.assertEqual(missing.mapped_predicate, "ToolExists")
        self.assertEqual(missing.evidence.failure_type, FailureType.TOOL_UNAVAILABLE)
        self.assertEqual(missing.constraint.evidence_status, EvidenceStatus.VERIFIED_COUNTEREXAMPLE)
        self.assertEqual(missing.constraint.origin, EvidenceOrigin.DIRECT_OBSERVATION)
        self.assertTrue(missing.constraint.direct)
        self.assertEqual(missing.constraint.scope.tool, "read")
        self.assertEqual(missing.constraint.scope.environment, "sandbox")
        self.assertFalse(missing.constraint.scope.resource)
        self.assertEqual(missing.constraint.source, "tool_result")

        denied = self.analyzer.analyze(self._observe(
            resource="secret",
            tool_result="Error: permission denied",
            execution_log="TOOL_NOT_FOUND",
        ))
        self.assertEqual(denied.mapped_predicate, "CanAccess")
        self.assertNotEqual(denied.constraint.condition.predicate, "ToolExists")
        self.assertEqual(denied.constraint.scope.resource, "secret")
        self.assertEqual(denied.constraint.scope.identity, "alice")
        self.assertEqual(denied.constraint.source, "tool_result")

        unscoped = self.analyzer.analyze(self._observe(
            identity="",
            tool_result={"error_code": "PERMISSION_DENIED"},
        ))
        self.assertFalse(unscoped.verified_counterexample)
        self.assertEqual(unscoped.constraint.evidence_status, EvidenceStatus.UNVERIFIED_HYPOTHESIS)

        signature = schema_signature({"mode": "full"})
        mismatched = self.analyzer.analyze(self._observe(
            parameters={"mode": "full"},
            tool_result={"error_code": "SCHEMA_MISMATCH"},
        ))
        self.assertEqual(mismatched.mapped_predicate, "SchemaOk")
        self.assertEqual(mismatched.constraint.condition.schema_signature, signature)
        self.assertNotEqual(mismatched.constraint.condition.predicate, "ToolExists")
        self.assertFalse(mismatched.constraint.disables_tool)

    def test_parameter_errors_stay_on_the_argument(self) -> None:
        bound = self.analyzer.analyze(self._observe(
            parameters={"id": "bad"},
            tool_result="parameter error: invalid id 'bad'",
        ))
        self.assertTrue(bound.verified_counterexample)
        self.assertEqual(bound.mapped_predicate, "ValidID")
        self.assertEqual(bound.constraint.condition.argument, "bad")
        self.assertTrue(bound.constraint.parameter_level)
        self.assertFalse(bound.constraint.disables_tool)
        self.assertTrue(bound.evidence.bound_to_call)
        self.assertNotEqual(bound.constraint.condition.predicate, "ToolExists")

        ambiguous = self.analyzer.analyze(self._observe(
            parameters={"id": "bad", "mode": "full"},
            tool_result={"error_code": "PARAM_ERROR"},
        ))
        self.assertFalse(ambiguous.verified_counterexample)
        self.assertEqual(ambiguous.constraint.evidence_status, EvidenceStatus.UNVERIFIED_HYPOTHESIS)

    def test_non_counterexamples_stay_unverified(self) -> None:
        cases = [
            ({"tool_result": "Error: timeout talking to the host"}, FailureType.TRANSIENT_ERROR),
            ({"tool_result": "Error: something broke"}, FailureType.GENERIC_FAILURE),
            ({"tool_result": ""}, FailureType.MISSING_INFORMATION),
            ({"tool_result": None, "execution_log": ""}, FailureType.MISSING_INFORMATION),
            ({"verifier_label": "uncertain", "tool_result": "a normal page"}, FailureType.VERIFIER_UNCERTAIN),
            ({"verifier_label": "不满足", "tool_result": "a normal page"}, FailureType.SUBGOAL_NOT_SATISFIED),
        ]
        for kwargs, failure_type in cases:
            analysis = self.analyzer.analyze(self._observe(**kwargs))
            self.assertFalse(analysis.verified_counterexample, kwargs)
            self.assertEqual(analysis.evidence.failure_type, failure_type)
            self.assertIsNotNone(analysis.constraint)
            self.assertEqual(analysis.constraint.evidence_status, EvidenceStatus.UNVERIFIED_HYPOTHESIS)
            self.assertFalse(analysis.constraint.direct)

        guessed = self.analyzer.analyze(self._observe(
            tool_result="Error: something broke",
            llm_hypothesis="TOOL_NOT_FOUND the read tool was never installed",
        ))
        self.assertEqual(guessed.evidence.failure_type, FailureType.GENERIC_FAILURE)
        self.assertFalse(guessed.verified_counterexample)
        self.assertNotEqual(guessed.constraint.evidence_status, EvidenceStatus.VERIFIED_COUNTEREXAMPLE)

        inferred = self.analyzer.analyze(self._observe(
            tool_result="page text without an error code",
            llm_hypothesis="TOOL_NOT_FOUND",
        ))
        self.assertEqual(inferred.evidence.origin, EvidenceOrigin.INFERENCE)
        self.assertFalse(inferred.verified_counterexample)
        self.assertEqual(inferred.constraint.source, "llm_hypothesis")

        from_log = self.analyzer.analyze(self._observe(
            tool_result=None,
            execution_log="TOOL_NOT_FOUND",
        ))
        self.assertTrue(from_log.verified_counterexample)
        self.assertEqual(from_log.constraint.source, "execution_log")


class HypothesisPrunerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pruner = HypothesisPruner()
        self.analyzer = FailureAnalyzer()

    def _graph(self) -> PlanningGraph:
        return PlanningGraph([
            _plan("read_a", (_step("read", premises=(_exists("read"),), resource="secret"),)),
            _plan("read_b", (_step("read", premises=(_exists("read"),), resource="other"),)),
            _plan("export", (_step("export", premises=(_exists("export"),)),)),
        ], round_id="e1")

    def test_only_an_admissible_counterexample_prunes_dependent_plans(self) -> None:
        graph = self._graph()
        analysis = self.analyzer.analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result={"error_code": "TOOL_NOT_FOUND"},
        ))
        result = self.pruner.prune(
            graph,
            analysis.constraint,
            evidence=analysis.evidence,
            executed_plan_id="read_a",
        )
        self.assertTrue(result.admissible)
        self.assertEqual(result.pruned_plan_ids, ("read_a", "read_b"))
        self.assertEqual(result.active_before, 3)
        self.assertEqual(result.active_after, 1)
        self.assertEqual(result.evidence_source, "tool_result")
        self.assertFalse(result.needs_replan)
        self.assertFalse(result.task_impossible)
        self.assertEqual(graph.active_plan_ids(), ("export",))
        self.assertEqual(result.records[0].affected_plan_ids, ("read_a", "read_b"))
        self.assertIn("read_a", result.records[0].affected_plan_ids)

        unverified = self.analyzer.analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            llm_hypothesis="TOOL_NOT_FOUND",
            tool_result="still looking",
        ))
        untouched = PlanningGraph([
            _plan("read_a", (_step("read", premises=(_exists("read"),)),)),
            _plan("export", (_step("export", premises=(_exists("export"),)),)),
        ])
        skipped = self.pruner.prune(untouched, unverified.constraint, evidence=unverified.evidence)
        self.assertFalse(skipped.admissible)
        self.assertEqual(skipped.pruned_plan_ids, ())
        self.assertEqual(untouched.active_plan_ids(), ("read_a", "export"))

    def test_confidence_scope_and_missing_labels_block_cross_path_pruning(self) -> None:
        analysis = self.analyzer.analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result="TOOL_NOT_FOUND",
        ))
        low = analysis.constraint
        low.confidence = 0.2
        graph = self._graph()
        result = self.pruner.prune(graph, low, evidence=analysis.evidence, executed_plan_id="read_a")
        self.assertFalse(result.admissible)
        self.assertEqual(result.pruned_plan_ids, ())
        self.assertEqual(result.instance_retired_plan_ids, ())
        self.assertEqual(graph.active_plan_ids(), ("read_a", "read_b", "export"))

        mismatched = self._graph()
        scoped = self.analyzer.analyze(ExecutionObservation(
            tool_name="read",
            environment="other-env",
            tool_result="TOOL_NOT_FOUND",
        ))
        scoped_result = self.pruner.prune(mismatched, scoped.constraint, evidence=scoped.evidence)
        self.assertTrue(scoped_result.dep_complete)
        self.assertEqual(scoped_result.pruned_plan_ids, ())
        self.assertEqual(mismatched.active_plan_ids(), ("read_a", "read_b", "export"))

        incomplete = PlanningGraph([
            _plan("labeled", (_step("read", premises=(_exists("read"),)),)),
            _plan("unlabeled", (_step("read"),)),
        ])
        observed = self.analyzer.analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result="TOOL_NOT_FOUND",
        ))
        blocked = self.pruner.prune(
            incomplete,
            observed.constraint,
            evidence=observed.evidence,
            executed_plan_id="labeled",
        )
        self.assertFalse(blocked.dep_complete)
        self.assertEqual(blocked.pruned_plan_ids, ())
        self.assertEqual(blocked.instance_retired_plan_ids, ())
        self.assertEqual(incomplete.active_plan_ids(), ("labeled", "unlabeled"))

    def test_parameter_error_does_not_disable_the_tool(self) -> None:
        signature = ""
        bad = _valid("read", "bad")
        good = _valid("read", "good")
        graph = PlanningGraph([
            _plan("bad", (_step("read", premises=(bad,), argument="bad", parameter="id"),)),
            _plan("also_bad", (_step("read", premises=(bad,), argument="bad", parameter="id"),)),
            _plan("good", (_step("read", premises=(good,), argument="good", parameter="id"),)),
            _plan("export", (_step("export", premises=(_exists("export"),)),)),
        ])
        analysis = self.analyzer.analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            parameters={"id": "bad"},
            tool_result="parameter error: invalid id 'bad'",
        ))
        self.assertNotEqual(signature, analysis.constraint.condition.predicate)
        result = self.pruner.prune(
            graph,
            analysis.constraint,
            evidence=analysis.evidence,
            executed_plan_id="bad",
        )
        self.assertEqual(result.pruned_plan_ids, ("also_bad", "bad"))
        self.assertEqual(graph.active_plan_ids(), ("good", "export"))
        self.assertFalse(analysis.constraint.disables_tool)

        leaked = PlanningGraph([
            _plan("bad", (_step("read", premises=(bad,), argument="bad", parameter="id"),)),
            _plan("unlabeled", (_step("read", argument="bad", parameter="id"),)),
            _plan("good", (_step("read", premises=(good,), argument="good", parameter="id"),)),
        ])
        retired = self.pruner.prune(
            leaked,
            analysis.constraint,
            evidence=analysis.evidence,
            executed_plan_id="bad",
        )
        self.assertEqual(retired.pruned_plan_ids, ())
        self.assertEqual(retired.instance_retired_plan_ids, ("bad",))
        self.assertEqual(leaked.active_plan_ids(), ("unlabeled", "good"))
        self.assertEqual(retired.reason_code, "instance_only")

    def test_exhausting_the_universe_asks_for_replanning(self) -> None:
        graph = PlanningGraph([
            _plan("read_a", (_step("read", premises=(_exists("read"),)),)),
            _plan("read_b", (_step("read", premises=(_exists("read"),)),)),
        ])
        analysis = self.analyzer.analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result={"error_code": "TOOL_NOT_FOUND"},
        ))
        preview = self.pruner.prune(
            graph,
            analysis.constraint,
            evidence=analysis.evidence,
            apply=False,
        )
        self.assertEqual(graph.active_plan_ids(), ("read_a", "read_b"))
        self.assertTrue(preview.needs_replan)
        result = self.pruner.prune(graph, analysis.constraint, evidence=analysis.evidence)
        self.assertEqual(result.active_before, 2)
        self.assertEqual(result.active_after, 0)
        self.assertTrue(result.needs_replan)
        self.assertEqual(result.message, NEEDS_REPLAN_MESSAGE)
        self.assertFalse(result.task_impossible)
        self.assertNotIn("不可完成", result.reason)
        self.assertEqual(result.reason_code, "needs_replan")
        self.assertEqual(graph.active_plan_ids(), ())
        forced = result.__class__(
            active_before=1,
            active_after=0,
            pruned_plan_ids=(),
            affected_plan_ids=(),
            reason="x",
            reason_code="needs_replan",
            evidence_source="",
            needs_replan=True,
            admissible=False,
            dep_complete=False,
            task_impossible=True,
        )
        self.assertFalse(forced.task_impossible)


class SnippetGapTests(unittest.TestCase):
    def test_search_with_urls_is_soft_unverified(self) -> None:
        observation = ExecutionObservation(
            tool_name="Bing_Search_Tool",
            environment="cghe_pev",
            command="execution = tool.execute(query='royal flash')",
            tool_result="1. Reed\n   URL: https://en.wikipedia.org/wiki/Oliver_Reed",
            verifier_label="不满足",
        )
        analysis = FailureAnalyzer().analyze(observation)
        self.assertFalse(analysis.verified_counterexample)
        self.assertEqual(analysis.evidence.failure_type, FailureType.SUBGOAL_NOT_SATISFIED)
        self.assertEqual(analysis.evidence.observation_code, "SUBGOAL_NOT_SATISFIED")

    def test_empty_search_is_hard_snippet_ce(self) -> None:
        from planning_graph import plans_for_snippet_gap

        observation = ExecutionObservation(
            tool_name="Bing_Search_Tool",
            environment="cghe_pev",
            command="execution = tool.execute(query='royal flash')",
            tool_result="No results",
            verifier_label="不满足",
        )
        analysis = FailureAnalyzer().analyze(observation)
        self.assertTrue(analysis.verified_counterexample)
        self.assertEqual(analysis.evidence.failure_type, FailureType.SNIPPETS_INSUFFICIENT)
        self.assertFalse(analysis.constraint.disables_tool)
        plans = plans_for_snippet_gap(
            {
                "tool_name": "Bing_Search_Tool",
                "command": observation.command,
                "result": "1. Reed\n   URL: https://en.wikipedia.org/wiki/Oliver_Reed",
            },
            environment="cghe_pev",
        )
        graph = PlanningGraph(plans)
        result = HypothesisPruner().prune(graph, analysis.constraint, evidence=analysis.evidence)
        self.assertIn("same_query", result.pruned_plan_ids)
        self.assertTrue(any(plan_id.startswith("fetch_") for plan_id in graph.active_plan_ids()))
        self.assertEqual(graph.select_node().tool, "web_fetch_tool")
        ordinary = FailureAnalyzer().analyze(ExecutionObservation(
            tool_name="Bing_Search_Tool",
            environment="cghe_pev",
            verifier_label="不满足",
            tool_result="a normal page without any link marker here",
        ))
        self.assertFalse(ordinary.verified_counterexample)
        self.assertEqual(ordinary.evidence.failure_type, FailureType.SUBGOAL_NOT_SATISFIED)

    def test_plan_successors_and_short_answer(self) -> None:
        from planning_graph import plans_for_snippet_gap
        from context import _plan_successors, wrap_answer

        plans = plans_for_snippet_gap(
            {
                "tool_name": "Bing_Search_Tool",
                "command": "execution = tool.execute(query='royal flash')",
                "result": "1. Reed\n   URL: https://en.wikipedia.org/wiki/Oliver_Reed",
            },
            environment="cghe_pev",
        )
        graph = PlanningGraph(plans)
        bing = next(n.node_id for n in graph.nodes() if "bing" in str(n.tool).casefold())
        successors = _plan_successors(graph, bing)
        self.assertTrue(successors)
        # Downstream may be Fetch or shared Extract/Generalist depending on outline merge.
        self.assertTrue(
            any(
                any(tok in str(graph.node(n).tool).casefold() for tok in ("fetch", "generalist", "extract"))
                for n in successors
            )
        )
        short = wrap_answer(
            "<answer>Sanming is located in Fujian Province, China, and Jiutai District is located in Jilin Province, China.</answer>"
        )
        self.assertIn("<answer>", short)
        self.assertLess(len(short), 120)
        self.assertIn("Sanming", short)


class FrontierVoteTests(unittest.TestCase):
    def test_later_shared_node_does_not_jump_the_frontier(self) -> None:
        first = _step("Bing_Search_Tool", subgoal="find the fact", cost=1, links_previous=False)
        graph = PlanningGraph([
            _plan("a", (first, _step("fetch", subgoal="open the page", cost=1))),
            _plan("b", (
                _step("Bing_Search_Tool", subgoal="find the fact", cost=1, links_previous=False),
                _step("fetch", subgoal="open the page", cost=1),
            )),
            _plan("c", (_step("fetch", subgoal="open the page", cost=1, links_previous=False),)),
        ])
        shared = graph.select_node(policy="shared")
        frontier = graph.select_node(policy="frontier")
        self.assertEqual(shared.tool, "fetch")
        self.assertEqual(graph.sharing_score(shared.node_id), 3)
        self.assertEqual(frontier.tool, "bing_search_tool")
        self.assertEqual(graph.frontier_vote(frontier.node_id), 2)

    def test_failure_removes_only_the_voted_outlines_suffix(self) -> None:
        from planning_graph import plans_for_snippet_gap

        # Hard CE requires empty / URL-less search; prune fixture still needs URLs for fetch plans.
        observation = ExecutionObservation(
            tool_name="Bing_Search_Tool",
            environment="sandbox",
            command="execution = tool.execute(query='paris')",
            tool_result="No results",
            verifier_label="不满足",
        )
        plans = list(plans_for_snippet_gap(
            {
                "tool_name": "Bing_Search_Tool",
                "command": observation.command,
                "result": "1. Paris\n   URL: https://en.wikipedia.org/wiki/Paris",
            },
            environment="sandbox",
        ))
        self.assertGreaterEqual(len(plans[0].steps), 2)
        graph = PlanningGraph(plans)
        analysis = FailureAnalyzer().analyze(observation)
        self.assertTrue(analysis.verified_counterexample)
        result = HypothesisPruner().prune(graph, analysis.constraint, evidence=analysis.evidence)
        self.assertIn("same_query", result.pruned_plan_ids)
        self.assertNotIn("same_query", graph.active_plan_ids())
        chosen = graph.select_node(policy="frontier")
        self.assertEqual(chosen.tool, "web_fetch_tool")
        self.assertNotIn("same_query", chosen.candidate_plan_ids)

    def test_different_arguments_do_not_merge_and_timeout_does_not_prune(self) -> None:
        bad = CandidatePlan(
            "bad",
            (_step("read", subgoal="read the record", premises=(_valid("read", "r1"),), argument="r1", parameter="id", links_previous=False),),
            context=Scope(environment="sandbox", tool="read", parameter="id", argument="r1"),
        )
        other = CandidatePlan(
            "other",
            (_step("read", subgoal="read the record", premises=(_valid("read", "r2"),), argument="r2", parameter="id", links_previous=False),),
            context=Scope(environment="sandbox", tool="read", parameter="id", argument="r2"),
        )
        graph = PlanningGraph([bad, other])
        self.assertEqual(len([node for node in graph.nodes() if node.tool == "read"]), 2)
        self.assertEqual(graph.frontier_vote(graph.frontier_node_id("bad")), 2)
        observation = ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result="Error: invalid id 'r1'",
            verifier_label="不满足",
        )
        analysis = FailureAnalyzer().analyze(observation)
        result = HypothesisPruner().prune(graph, analysis.constraint, evidence=analysis.evidence)
        self.assertEqual(result.pruned_plan_ids, ("bad",))
        self.assertEqual(graph.active_plan_ids(), ("other",))
        self.assertFalse(analysis.constraint.disables_tool)
        before = graph.frontier_vote(graph.frontier_node_id("other"))
        timeout = FailureAnalyzer().analyze(ExecutionObservation(
            tool_name="read",
            environment="sandbox",
            tool_result="Error: HTTP 503 timeout talking to the host",
            verifier_label="不满足",
        ))
        self.assertFalse(timeout.verified_counterexample)
        self.assertEqual(graph.active_plan_ids(), ("other",))
        self.assertEqual(graph.frontier_vote(graph.frontier_node_id("other")), before)


if __name__ == "__main__":
    unittest.main()
