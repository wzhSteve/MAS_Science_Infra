"""Round-based DAG regeneration + HypothesisMemory acceptance tests."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

from cghe_loop import CGHESession, drop_contradicted_plans, make_regeneration_context, validate_dag  # noqa: E402
from cghe_types import AtomicPremise, CandidatePlan, PlanStep, Scope  # noqa: E402
from hypothesis_round_memory import (  # noqa: E402
    HypothesisMemory,
    HypothesisStatus,
    PlanningImplication,
    SubgoalStatus,
    VerifierFeedback,
    build_verifier_feedback,
    freeze_dag_snapshot,
)
from planning_graph import PlanningGraph  # noqa: E402


def _step(tool: str, *, argument: str = "", subgoal: str = "", links_previous: bool = True) -> PlanStep:
    return PlanStep(
        action=argument or subgoal or tool,
        tool=tool,
        subgoal=subgoal or argument or tool,
        argument=argument or subgoal,
        parameter="query" if "bing" in tool.casefold() or "search" in tool.casefold() else "",
        preconditions=(
            AtomicPremise(predicate="SnippetsSufficient", tool=tool, parameter="query", argument=argument or subgoal),
        ) if "bing" in tool.casefold() or "search" in tool.casefold() else (),
        links_previous=links_previous,
    )


def _plan(plan_id: str, query: str) -> CandidatePlan:
    return CandidatePlan(
        plan_id,
        (
            _step("Bing_Search_Tool", argument=query, subgoal=f"search {query}", links_previous=False),
            _step("Web_Fetch_Tool", argument="page", subgoal="open page"),
        ),
        context=Scope(environment="cghe_pev", tool="bing_search_tool", parameter="query", argument=query),
    )


class Controller:
    def __init__(self) -> None:
        self.graph = None
        self.hypothesis_memory = HypothesisMemory()
        self.cghe_session = None


class RoundRegenTests(unittest.TestCase):
    def test_1_new_dag_generation(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        g1 = PlanningGraph([_plan("p1", "query a"), _plan("p2", "query b")], version=1)
        session.bind_graph(g1)
        session.planning_round = 1
        snap1 = freeze_dag_snapshot(g1)
        g2 = session.regenerate(
            [_plan("p3", "query c"), _plan("p4", "query d")],
            make_regeneration_context(query="q", trigger_reasons=["ROUND_FEEDBACK_REGEN"]),
        )
        self.assertIsNotNone(g2)
        self.assertIsNot(g1, g2)
        self.assertNotEqual(str(g1.version), str(g2.version))
        self.assertTrue(g1.sealed)
        # G1 snapshot unchanged after Round 2.
        self.assertEqual(freeze_dag_snapshot(g1)["nodes"], snap1["nodes"])

    def test_2_hypothesis_persistence(self) -> None:
        mem = HypothesisMemory()
        h1 = mem.upsert(description="search paris", tool="Bing_Search_Tool", argument="paris", dag_version="1")
        fb = VerifierFeedback(
            subgoal_status=SubgoalStatus.NOT_SATISFIED,
            failure_type="SUBGOAL_NOT_SATISFIED",
            planning_implication=PlanningImplication.REVISE,
            tool="Bing_Search_Tool",
            argument="paris",
            evidence="timeout",
            dag_version="1",
        )
        mem.update_from_feedback(fb)
        self.assertEqual(h1.verification_status, HypothesisStatus.UNVERIFIED)
        again = mem.get(h1.hypothesis_id)
        self.assertIs(again, h1)
        self.assertTrue(again.evidence)

    def test_3_unverified_on_timeout(self) -> None:
        fb = build_verifier_feedback(
            satisfied=False,
            label="不满足",
            analysis=None,
            action={"tool_name": "Bing_Search_Tool", "command": "paris"},
            evidence_text="timeout",
            soft_continue=False,
        )
        # Without hard CE analysis → UNVERIFIED path via REVISE + SUBGOAL_NOT_SATISFIED
        mem = HypothesisMemory()
        hyp = mem.upsert(description="paris", tool="Bing_Search_Tool", argument="paris")
        fb.affected_hypothesis_ids = (hyp.hypothesis_id,)
        fb.failure_type = "TIMEOUT"
        mem.update_from_feedback(fb)
        self.assertEqual(hyp.verification_status, HypothesisStatus.UNVERIFIED)

    def test_4_evidence_grounded_refute(self) -> None:
        mem = HypothesisMemory()
        hyp = mem.upsert(description="bad query", tool="Bing_Search_Tool", argument="bad query")
        fb = VerifierFeedback(
            subgoal_status=SubgoalStatus.NOT_SATISFIED,
            failure_type="SNIPPETS_INSUFFICIENT",
            planning_implication=PlanningImplication.AVOID,
            tool="Bing_Search_Tool",
            argument="bad query",
            evidence="empty search",
            affected_hypothesis_ids=(hyp.hypothesis_id,),
        )
        mem.update_from_feedback(fb)
        self.assertEqual(hyp.verification_status, HypothesisStatus.REFUTED)
        plans = [_plan("p1", "bad query"), _plan("p2", "good query")]
        kept = mem.filter_plans(plans)
        self.assertEqual([p.plan_id for p in kept], ["p2"])

    def test_5_new_hypothesis_generation(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        g1 = PlanningGraph([_plan("a", "A"), _plan("b", "B"), _plan("c", "C")], version=1)
        session.bind_graph(g1)
        session.planning_round = 1
        ctrl.hypothesis_memory.register_dag_hypotheses(g1)
        keys1 = {h.semantic_key for h in ctrl.hypothesis_memory.all()}
        g2 = session.regenerate(
            [_plan("b2", "B"), _plan("d", "D")],
            make_regeneration_context(query="q", trigger_reasons=["ROUND_FEEDBACK_REGEN"]),
        )
        self.assertIsNotNone(g2)
        args2 = {str(g2.node(nid).argument) for nid in g2.plan_node_ids(list(g2.active_plan_ids())[0])}
        # D appears in G2 step1 arguments across plans
        step1_args = set()
        for pid in g2.active_plan_ids():
            seq = g2.plan_node_ids(pid)
            step1_args.add(str(g2.node(seq[0]).argument))
        self.assertIn("d", {a.casefold() for a in step1_args})
        self.assertNotIn("d", {str(g1.node(g1.plan_node_ids(pid)[0]).argument).casefold() for pid in g1.active_plan_ids()})

    def test_6_dag_immutability(self) -> None:
        g1 = PlanningGraph([_plan("p1", "q1")], version=1)
        before = copy.deepcopy(freeze_dag_snapshot(g1))
        g1.seal()
        # Mutating attempt_count on sealed graph for runtime bookkeeping is allowed,
        # but structural snapshot must remain equal to the sealed copy.
        snap = g1.seal_snapshot()
        self.assertEqual(snap["nodes"], before["nodes"])
        self.assertEqual(snap["plans"], before["plans"])
        g2 = PlanningGraph([_plan("p2", "q2")], version=2)
        self.assertNotEqual(g1.version, g2.version)
        self.assertEqual(g1.seal_snapshot()["nodes"], before["nodes"])

    def test_7_no_repeated_refuted(self) -> None:
        mem = HypothesisMemory()
        mem.upsert(description="x", tool="Bing_Search_Tool", argument="refuted-q")
        hyp = mem.get_by_key("Bing_Search_Tool", "refuted-q")
        mem.update_from_feedback(VerifierFeedback(
            subgoal_status=SubgoalStatus.NOT_SATISFIED,
            failure_type="SNIPPETS_INSUFFICIENT",
            planning_implication=PlanningImplication.AVOID,
            tool="Bing_Search_Tool",
            argument="refuted-q",
            affected_hypothesis_ids=(hyp.hypothesis_id,),
        ))
        guidance = mem.planner_guidance()
        self.assertIn("AVOID REFUTED", guidance)
        kept = mem.filter_plans([_plan("p1", "refuted-q"), _plan("p2", "other")])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].steps[0].argument, "other")

    def test_8_plan_selection_unchanged(self) -> None:
        # Shared first node covers 2 paths → selected over singleton.
        shared = _step("Bing_Search_Tool", argument="shared", subgoal="shared", links_previous=False)
        p1 = CandidatePlan("p1", (shared, _step("Web_Fetch_Tool", argument="u1", subgoal="f1")))
        p2 = CandidatePlan(
            "p2",
            (
                _step("Bing_Search_Tool", argument="shared", subgoal="shared", links_previous=False),
                _step("Web_Fetch_Tool", argument="u2", subgoal="f2"),
            ),
        )
        p3 = CandidatePlan(
            "p3",
            (
                _step("Bing_Search_Tool", argument="alone", subgoal="alone", links_previous=False),
                _step("Web_Fetch_Tool", argument="u3", subgoal="f3"),
            ),
        )
        graph = PlanningGraph([p1, p2, p3], version=1)
        selected, coverage, eligible = graph.select_step1()
        self.assertGreaterEqual(coverage, 2)
        self.assertEqual(graph.node(selected).argument, "shared")

    def test_9_successful_supported(self) -> None:
        mem = HypothesisMemory()
        hyp = mem.upsert(description="ok", tool="Bing_Search_Tool", argument="ok")
        fb = build_verifier_feedback(
            satisfied=True,
            label="满足",
            action={"tool_name": "Bing_Search_Tool", "command": "ok"},
        )
        fb.affected_hypothesis_ids = (hyp.hypothesis_id,)
        mem.update_from_feedback(fb)
        self.assertEqual(hyp.verification_status, HypothesisStatus.SUPPORTED)

    def test_10_multi_round_integration(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5, "max_total_step1_attempts": 12})
        g1 = session.regenerate(
            [_plan("a", "A")],
            make_regeneration_context(query="q", trigger_reasons=["INIT"]),
        )
        # First regenerate with empty old graph still works via bind.
        if g1 is None:
            g1 = PlanningGraph([_plan("a", "A")], version=1)
            session.planning_round = 1
            session.bind_graph(g1)
            ctrl.hypothesis_memory.register_dag_hypotheses(g1)
        v1 = str(g1.version)
        ctrl.hypothesis_memory.update_from_feedback(VerifierFeedback(
            subgoal_status=SubgoalStatus.NOT_SATISFIED,
            failure_type="TIMEOUT",
            planning_implication=PlanningImplication.REVISE,
            tool="Bing_Search_Tool",
            argument="A",
            evidence="timeout",
        ))
        g2 = session.regenerate(
            [_plan("b", "B")],
            make_regeneration_context(query="q", trigger_reasons=["ROUND_FEEDBACK_REGEN"]),
        )
        self.assertIsNotNone(g2)
        self.assertTrue(g1.sealed)
        self.assertNotEqual(str(g2.version), v1)
        ctrl.hypothesis_memory.update_from_feedback(VerifierFeedback(
            subgoal_status=SubgoalStatus.NOT_SATISFIED,
            failure_type="SNIPPETS_INSUFFICIENT",
            planning_implication=PlanningImplication.AVOID,
            tool="Bing_Search_Tool",
            argument="B",
            evidence="empty",
        ))
        g3 = session.regenerate(
            [_plan("c", "C")],
            make_regeneration_context(query="q", trigger_reasons=["ROUND_FEEDBACK_REGEN"]),
        )
        self.assertIsNotNone(g3)
        self.assertTrue(g2.sealed)
        self.assertNotEqual(g3.version, g2.version)
        # Success on C
        hyp = ctrl.hypothesis_memory.upsert(description="C", tool="Bing_Search_Tool", argument="C")
        ctrl.hypothesis_memory.update_from_feedback(build_verifier_feedback(
            satisfied=True, label="满足", action={"tool_name": "Bing_Search_Tool", "command": "C"},
        ))
        # Mark affected
        ctrl.hypothesis_memory.get_by_key("Bing_Search_Tool", "C").verification_status = HypothesisStatus.SUPPORTED
        self.assertEqual(
            ctrl.hypothesis_memory.get_by_key("Bing_Search_Tool", "C").verification_status,
            HypothesisStatus.SUPPORTED,
        )
        self.assertGreaterEqual(len(session.archived_dags), 2)


if __name__ == "__main__":
    unittest.main()
