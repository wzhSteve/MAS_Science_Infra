"""Planner outline limits and the graph built from those outlines."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planning_graph import PlanningGraph, render_planner_graph_markdown  # noqa: E402
from sft_prompts import generate_next_step, parse_candidate_plans, remaining_outline_steps  # noqa: E402


def _step(index: int, tool: str, *, argument: str = "") -> dict:
    return {
        "subgoal": f"step {index}",
        "action": f"step {index}",
        "tool": tool,
        "argument": argument,
        "parameter": "query" if argument else "",
    }


def _plans(count: int, steps: int, *, last_tool: str = "Web_Fetch_Tool") -> list:
    rows = []
    for plan_index in range(count):
        body = [_step(i, "Bing_Search_Tool") for i in range(steps - 1)]
        body.append(_step(steps - 1, last_tool))
        rows.append({"plan_id": f"p{plan_index}", "steps": body})
    return rows


class PlannerOutlineLimitTests(unittest.TestCase):
    def test_remaining_steps_follow_the_step_budget(self) -> None:
        self.assertEqual(remaining_outline_steps(2, 6), 4)
        self.assertEqual(remaining_outline_steps(6, 6), 1)

    def test_extra_outlines_and_steps_are_truncated(self) -> None:
        raw = json.dumps({"plans": _plans(12, 6)})
        plans = parse_candidate_plans(raw, limit=10, max_steps=4)
        self.assertEqual(len(plans), 10)
        self.assertTrue(all(len(plan.steps) == 4 for plan in plans))

    def test_last_step_may_be_search_or_fetch(self) -> None:
        raw = json.dumps({"plans": [
            {"plan_id": "fetch", "steps": [_step(0, "Bing_Search_Tool"), _step(1, "Web_Fetch_Tool")]},
            {"plan_id": "search", "steps": [_step(0, "Web_Fetch_Tool"), _step(1, "Bing_Search_Tool")]},
        ]})
        plans = parse_candidate_plans(raw, limit=10, max_steps=4)
        self.assertEqual([plan.steps[-1].tool for plan in plans], ["Web_Fetch_Tool", "Bing_Search_Tool"])

    def test_frontier_vote_beats_a_later_shared_node(self) -> None:
        first = {"subgoal": "find the fact", "action": "find the fact", "tool": "Bing_Search_Tool"}
        later = {"subgoal": "open the page", "action": "open the page", "tool": "Web_Fetch_Tool"}
        raw = json.dumps({"plans": [
            {"plan_id": "a", "steps": [first, later]},
            {"plan_id": "b", "steps": [dict(first), dict(later)]},
            {"plan_id": "c", "steps": [dict(later)]},
        ]})
        plans = parse_candidate_plans(raw, limit=10, max_steps=4)
        graph = PlanningGraph(plans)
        shared = graph.select_node(policy="shared")
        frontier = graph.select_node(policy="frontier")
        self.assertEqual(shared.tool, "web_fetch_tool")
        self.assertEqual(graph.sharing_score(shared.node_id), 3)
        self.assertEqual(frontier.tool, "bing_search_tool")
        self.assertEqual(graph.frontier_vote(frontier.node_id), 2)

    def test_different_arguments_do_not_merge(self) -> None:
        raw = json.dumps({"plans": [
            {"plan_id": "bad", "steps": [_step(0, "Bing_Search_Tool", argument="r1")]},
            {"plan_id": "other", "steps": [_step(0, "Bing_Search_Tool", argument="r2")]},
        ]})
        plans = parse_candidate_plans(raw, limit=10, max_steps=4)
        graph = PlanningGraph(plans)
        reads = [node for node in graph.nodes() if node.tool == "bing_search_tool"]
        self.assertEqual(len(reads), 2)

    def test_unquoted_plan_keys_still_parse(self) -> None:
        raw = """
        {"plans":[
          {"plan_id":"p1", steps:[{"subgoal":"search the film","action":"search","tool":"Bing_Search_Tool","estimated_cost":1}]},
          {plan_id: p2, steps:[{"subgoal":"open the page","action":"open","tool":"Web_Fetch_Tool","estimated_cost":1}]}
        ]}
        """
        plans = parse_candidate_plans(raw, limit=10, max_steps=4)
        self.assertEqual([plan.plan_id for plan in plans], ["p1", "p2"])
        self.assertEqual(plans[1].steps[-1].tool, "Web_Fetch_Tool")

    def test_generate_next_step_keeps_the_capped_plans(self) -> None:
        payload = {
            "outline": ["find the fact"],
            "subgoal": "find the fact",
            "selected_tool": "Bing_Search_Tool",
            "tool_input": "oliver reed royal flash",
            "prediction": "YES",
            "prediction_reason": "search first",
            "plans": _plans(12, 6, last_tool="Web_Fetch_Tool"),
        }

        class Planner:
            def __init__(self) -> None:
                self.llm_engine = lambda _prompt: json.dumps(payload)
                self._last_plan = {}

        class Memory:
            outline: list = []
            obtained_information: list = []

            def get_actions(self) -> dict:
                return {}

        planner = Planner()
        generate_next_step(planner, "Who is it?", None, "", Memory(), 2, 6)
        kept = planner._last_plan["candidate_plans"]
        self.assertEqual(len(kept), 10)
        self.assertTrue(all(len(plan.steps) <= 4 for plan in kept))
        text = render_planner_graph_markdown(
            question="Who is it?",
            n=10,
            step_count=2,
            max_steps=6,
            plans=kept,
        )
        self.assertIn("票数", text)
        self.assertIn("出度", text)
        self.assertIn("剩余步数：4", text)

    def test_line_outlines_fill_the_ten_routes(self) -> None:
        lines = []
        for index in range(1, 11):
            lines.append(
                f"P{index}: Bing_Search_Tool | query {index} || Web_Fetch_Tool | page {index}"
            )
        plans = parse_candidate_plans("\n".join(lines), limit=10, max_steps=10)
        self.assertEqual(len(plans), 10)
        self.assertTrue(all(len(plan.steps) == 2 for plan in plans))
        self.assertEqual(len({plan.steps[0].action for plan in plans}), 10)

    def test_layers_share_the_final_extract_node(self) -> None:
        class Engine:
            def __call__(self, prompt: str) -> str:
                if "Write at most" in prompt:
                    return (
                        "L0-1: Bing_Search_Tool | oliver reed role\n"
                        "L0-2: Bing_Search_Tool | royal flash plot\n"
                    )
                return (
                    "FROM L0-1:\n"
                    "N1: Web_Fetch_Tool | wikipedia royal flash\n"
                    "N2: Web_Fetch_Tool | imdb royal flash\n"
                    "FROM L0-2:\n"
                    "N1: Web_Fetch_Tool | britannica royal flash\n"
                    "N2: Generalist_Solution_Generator_Tool | read the plot snippet\n"
                )

        from sft_prompts import build_layered_outlines

        plans = build_layered_outlines(Engine(), "Who is it?", roots=4, branches=2, depth=3)
        graph = PlanningGraph(plans)
        self.assertEqual(len(plans), 4)
        sink = [node for node in graph.nodes() if "extract the asked fact" in node.subgoal]
        self.assertEqual(len(sink), 1)
        self.assertEqual(graph.sharing_score(sink[0].node_id), 4)
        self.assertGreaterEqual(len(sink[0].dependencies), 2)
        chosen = graph.select_node(policy="frontier")
        self.assertEqual(chosen.tool, "bing_search_tool")
        self.assertEqual(graph.frontier_vote(chosen.node_id), 2)


if __name__ == "__main__":
    unittest.main()
