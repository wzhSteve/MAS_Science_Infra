"""Soft-Bing soft-exhaust / deepen / budget-force acceptance tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

from cghe_loop import CGHESession  # noqa: E402
from cghe_types import AtomicPremise, CandidatePlan, PlanStatus, PlanStep  # noqa: E402
from context import (  # noqa: E402
    _looks_like_abstention,
    _should_brake_regen,
    _soft_budget_ok,
    _soft_exhaust_sibling_or_regen,
    _switch_after_graph_failure,
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
            AtomicPremise(
                predicate="SnippetsSufficient",
                tool=tool,
                parameter="query",
                argument=argument or subgoal,
            ),
        )
        if "bing" in tool.casefold() or "search" in tool.casefold()
        else (),
        links_previous=links_previous,
    )


def _chain_plan(plan_id: str, query: str) -> CandidatePlan:
    return CandidatePlan(
        plan_id,
        (
            _step("Bing_Search_Tool", argument=query, subgoal=f"search {query}", links_previous=False),
            _step("Web_Fetch_Tool", argument=f"url:{query}", subgoal="open page"),
            _step(
                "Generalist_Solution_Generator_Tool",
                argument=f"extract:{query}",
                subgoal="extract answer",
            ),
        ),
    )


class Controller:
    def __init__(self) -> None:
        self.graph = None
        self.hypothesis_memory = None
        self.cghe_session = None
        self.memory = MagicMock()

    def _remember_failure(self, *args: Any, **kwargs: Any) -> None:
        return None


def _episode(*, step_count: int = 2) -> Any:
    return SimpleNamespace(
        question="Who founded Sinclair Oil?",
        step_count=step_count,
        _cghe_open_attempt_id="",
        _cghe_graph_mode="",
        _cghe_pending_reasons=[],
        _cghe_directive={},
        _cghe_pages=[],
        memory=SimpleNamespace(actions={}),
    )


def _leaf_action(graph: PlanningGraph, plan_id: str) -> dict:
    leaf = graph.plan_node_ids(plan_id)[-1]
    return {
        "cghe_node_id": leaf,
        "cghe_plan_id": plan_id,
        "tool_name": "Generalist_Solution_Generator_Tool",
        "sub_goal": "extract answer",
        "result": "long enough page body without the answer key facts " * 5,
    }


class SoftExhaustTests(unittest.TestCase):
    def test_soft_exhaust_selects_sibling(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        graph = PlanningGraph(
            [_chain_plan("p1", "query a"), _chain_plan("p2", "query b")],
            version=1,
        )
        session.bind_graph(graph)
        ep = _episode()
        leaf = graph.plan_node_ids("p1")[-1]
        ok = _soft_exhaust_sibling_or_regen(
            ep,
            ctrl,
            node_id=leaf,
            action={"cghe_plan_id": "p1", "cghe_node_id": leaf},
            reasons=["SOFT_CHAIN_EXHAUSTED"],
            label="不满足",
        )
        self.assertTrue(ok)
        self.assertEqual(graph.plan("p1").status, PlanStatus.SOFT_EXHAUSTED)
        self.assertEqual(graph.plan("p2").status, PlanStatus.ACTIVE)
        self.assertEqual(ep._cghe_graph_mode, "switch")
        self.assertIn("sibling step1", str((ep._cghe_directive or {}).get("reason") or ""))
        selected = str((ep._cghe_directive or {}).get("node_id") or "")
        self.assertEqual(selected, graph.plan_node_ids("p2")[0])
        events = [e.get("event") for e in session.events]
        self.assertIn("plans_soft_exhausted", events)

    def test_soft_exhaust_both_triggers_regen(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        graph = PlanningGraph(
            [_chain_plan("p1", "query a"), _chain_plan("p2", "query b")],
            version=1,
        )
        session.bind_graph(graph)
        ep = _episode(step_count=2)
        leaf1 = graph.plan_node_ids("p1")[-1]
        _soft_exhaust_sibling_or_regen(
            ep,
            ctrl,
            node_id=leaf1,
            action={"cghe_plan_id": "p1", "cghe_node_id": leaf1},
            reasons=["SOFT_CHAIN_EXHAUSTED"],
        )
        leaf2 = graph.plan_node_ids("p2")[-1]
        ep2 = _episode(step_count=2)
        ok = _soft_exhaust_sibling_or_regen(
            ep2,
            ctrl,
            node_id=leaf2,
            action={"cghe_plan_id": "p2", "cghe_node_id": leaf2},
            reasons=["SOFT_CHAIN_EXHAUSTED"],
        )
        self.assertTrue(ok)
        self.assertEqual(graph.plan("p1").status, PlanStatus.SOFT_EXHAUSTED)
        self.assertEqual(graph.plan("p2").status, PlanStatus.SOFT_EXHAUSTED)
        self.assertEqual(ep2._cghe_graph_mode, "rebuild")
        self.assertIn("SOFT_CHAIN_EXHAUSTED", list(ep2._cghe_pending_reasons or []))

    def test_soft_budget_allows_three(self) -> None:
        graph = PlanningGraph(
            [_chain_plan("p1", "query a"), _chain_plan("p2", "query b")],
            version=1,
        )
        graph.note_soft_advance(["p1"])
        graph.note_soft_advance(["p1"])
        self.assertEqual(graph.soft_advance_count("p1"), 2)
        self.assertTrue(_soft_budget_ok(graph, ["p1"]))
        graph.note_soft_advance(["p1"])
        self.assertEqual(graph.soft_advance_count("p1"), 3)
        self.assertFalse(_soft_budget_ok(graph, ["p1"]))
        self.assertTrue(_soft_budget_ok(graph, ["p2"]))

    def test_leaf_fail_deepens_before_exhaust(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        graph = PlanningGraph(
            [_chain_plan("p1", "query a"), _chain_plan("p2", "query b")],
            version=1,
        )
        session.bind_graph(graph)
        ep = _episode()
        ok = _switch_after_graph_failure(ep, ctrl, _leaf_action(graph, "p1"), "不满足", False)
        self.assertTrue(ok)
        self.assertEqual(graph.plan("p1").status, PlanStatus.ACTIVE)
        self.assertEqual(graph.leaf_fail_count("p1"), 1)
        self.assertIn("deepen_after_leaf_fail", str((ep._cghe_directive or {}).get("reason") or ""))
        events = [e.get("event") for e in session.events]
        self.assertIn("leaf_deepen", events)
        self.assertNotIn("plans_soft_exhausted", events)

    def test_second_leaf_fail_soft_exhausts(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        graph = PlanningGraph(
            [_chain_plan("p1", "query a"), _chain_plan("p2", "query b")],
            version=1,
        )
        session.bind_graph(graph)
        ep = _episode()
        _switch_after_graph_failure(ep, ctrl, _leaf_action(graph, "p1"), "不满足", False)
        self.assertEqual(graph.plan("p1").status, PlanStatus.ACTIVE)
        ep2 = _episode()
        ok = _switch_after_graph_failure(ep2, ctrl, _leaf_action(graph, "p1"), "不满足", False)
        self.assertTrue(ok)
        self.assertEqual(graph.leaf_fail_count("p1"), 2)
        self.assertEqual(graph.plan("p1").status, PlanStatus.SOFT_EXHAUSTED)
        self.assertEqual(graph.plan("p2").status, PlanStatus.ACTIVE)
        self.assertIn("sibling step1", str((ep2._cghe_directive or {}).get("reason") or ""))

    def test_switch_leaf_fail_soft_exhausts(self) -> None:
        """Backward-compatible name: two leaf fails exhaust (Deepen-then-Explore)."""
        self.test_second_leaf_fail_soft_exhausts()

    def test_abstention_detector(self) -> None:
        self.assertTrue(_looks_like_abstention(
            "<answer>The information provided in the retrieved steps does not contain the name</answer>"
        ))
        self.assertTrue(_looks_like_abstention("<answer>No valid answer</answer>"))
        self.assertTrue(_looks_like_abstention("<answer>None</answer>"))
        self.assertTrue(_looks_like_abstention("<answer>Cannot be determined from the provided information</answer>"))
        self.assertTrue(_looks_like_abstention(""))
        self.assertFalse(_looks_like_abstention("<answer>Harry F. Sinclair</answer>"))
        self.assertFalse(_looks_like_abstention("<answer>Oldham County</answer>"))

    def test_hard_prune_keeps_sibling(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        graph = PlanningGraph(
            [_chain_plan("p1", "query a"), _chain_plan("p2", "query b")],
            version=1,
        )
        session.bind_graph(graph)
        ep = _episode()
        bing = graph.plan_node_ids("p1")[0]
        action = {
            "cghe_node_id": bing,
            "cghe_plan_id": "p1",
            "tool_name": "Bing_Search_Tool",
            "sub_goal": "search query a",
            "result": "",
            "raw_tool_result": "",
        }
        ok = _switch_after_graph_failure(ep, ctrl, action, "不满足", False)
        self.assertTrue(ok)
        self.assertEqual(graph.plan("p1").status, PlanStatus.PRUNED)
        self.assertEqual(graph.plan("p2").status, PlanStatus.ACTIVE)
        self.assertEqual(ep._cghe_graph_mode, "switch")
        self.assertIn("after verified prune", str((ep._cghe_directive or {}).get("reason") or ""))
        self.assertNotEqual(ep._cghe_graph_mode, "rebuild")

    def test_regen_brake_near_budget(self) -> None:
        ctrl = Controller()
        session = CGHESession(ctrl, settings={"round_regen": True, "max_regenerations": 5})
        graph = PlanningGraph([_chain_plan("p1", "query a")], version=1)
        session.bind_graph(graph)
        ep = _episode(step_count=9)
        with patch("context.load_runtime", return_value={"max_steps": 10, "max_time": 600}):
            self.assertTrue(_should_brake_regen(ep, ctrl))
        ep2 = _episode(step_count=2)
        with patch("context.load_runtime", return_value={"max_steps": 10, "max_time": 600}):
            self.assertFalse(_should_brake_regen(ep2, ctrl))

    def test_budget_force_answer_nonempty(self) -> None:
        """out_of_budget + rebuild hang should still produce a nonempty answer path."""
        from context import EpisodeContext

        class FakePlanner:
            def generate_final_output(self, *a, **k):
                return "Harry F. Sinclair"

            def generate_direct_output(self, *a, **k):
                return "Harry F. Sinclair"

        class FakeVerifier:
            def verificate_context(self, *a, **k):
                return "Conclusion: STOP\nReason: budget"

            def extract_conclusion(self, text):
                return "budget", "STOP"

        ctx = EpisodeContext()
        ep = SimpleNamespace(
            question="Who founded Sinclair Oil?",
            step_count=10,
            start=0.0,
            source="test",
            query_analysis="",
            planner=FakePlanner(),
            verifier=FakeVerifier(),
            memory=SimpleNamespace(
                actions={},
                obtained_information=["Sinclair Oil founded by Harry F. Sinclair"],
                get_actions=lambda: {},
            ),
            final_output="",
            direct_output="",
            json_data={},
            done=False,
            _cghe_pages=["page about Sinclair"],
            _cghe=None,
            _cghe_graph_mode="rebuild",
            _cghe_directive={},
            cghe_controller=lambda: None,
            export_state=lambda **k: {},
        )

        def fake_episode(self, payload):
            return ep

        with patch.object(EpisodeContext, "episode", fake_episode), \
             patch.object(EpisodeContext, "pin_engines", lambda self: None), \
             patch.object(EpisodeContext, "_trace", lambda self, extra=None: dict(extra or {})), \
             patch("context.load_runtime", return_value={"max_steps": 10, "max_time": 99999}), \
             patch("context.cghe_enabled", return_value=True), \
             patch("context._cghe_on_verdict", lambda *a, **k: None), \
             patch("context._complete_cghe_plans", lambda *a, **k: None), \
             patch("context._graph_for", return_value=None), \
             patch("context._cghe_trace", return_value={}), \
             patch("context._metrics", return_value={}), \
             patch("context.time") as mock_time:
            mock_time.time.return_value = 0.0
            out = ctx.verify_window({"payload": {}})
        payload = out.get("payload") or {}
        self.assertTrue(payload.get("ready_to_stop"))
        self.assertTrue(str(payload.get("answer") or "").strip())
        self.assertIn("Sinclair", str(payload.get("answer") or ""))
        self.assertTrue(ep.done)
        self.assertIn("budget_force_answer", str(payload.get("reason") or ""))


if __name__ == "__main__":
    unittest.main()
