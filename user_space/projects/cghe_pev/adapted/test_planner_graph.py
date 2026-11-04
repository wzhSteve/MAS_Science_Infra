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

    def test_layers_keep_extract_instances_per_path(self) -> None:
        class Engine:
            def __call__(self, prompt: str) -> str:
                if "Write at most" in prompt:
                    return (
                        "L0-1: Bing_Search_Tool | oliver reed role\n"
                        "L0-2: Bing_Search_Tool | royal flash plot\n"
                    )
                return (
                    "FROM L0-1:\n"
                    "N1: Web_Fetch_Tool | open wikipedia result\n"
                    "N2: Web_Fetch_Tool | open imdb result\n"
                    "FROM L0-2:\n"
                    "N1: Web_Fetch_Tool | open britannica result\n"
                    "N2: Web_Fetch_Tool | open cast result\n"
                )

        from sft_prompts import build_layered_outlines

        plans = build_layered_outlines(Engine(), "Who is it?", roots=4, branches=2, depth=3)
        graph = PlanningGraph(plans)
        self.assertEqual(len(plans), 4)
        extracts = [node for node in graph.nodes() if "extract the asked fact" in node.subgoal]
        self.assertEqual(len(extracts), 4)
        self.assertTrue(all(node.task_id == "extract_fact" for node in extracts))
        self.assertTrue(all("page_content" in node.consumes for node in extracts))
        opens = [node for node in graph.nodes() if "search_url" in node.consumes]
        self.assertEqual(len(opens), 4)
        chosen = graph.select_node(policy="frontier")
        self.assertEqual(chosen.tool, "bing_search_tool")
        self.assertEqual(graph.frontier_vote(chosen.node_id), 2)
        # Extract is not READY from search snippets alone.
        search = chosen.node_id
        graph.mark_satisfied(search)
        graph.store_search_results(search, ["https://example.com/a", "https://example.com/b"])
        ready_ids = list(graph.executable_node_ids())
        open_ready = [nid for nid in ready_ids if "search_url" in graph.node(nid).consumes]
        self.assertGreaterEqual(len(open_ready), 1)
        self.assertTrue(graph.bound_url(open_ready[0]).startswith("https://"))
        # Still no extract without page body.
        ready_tools = {graph.node(nid).tool for nid in ready_ids}
        self.assertNotIn("generalist_solution_generator_tool", ready_tools)
        # Bound open pages beat unfinished sibling searches when votes tie.
        nxt = graph.select_node(policy="frontier")
        self.assertIsNotNone(nxt)
        self.assertIn("search_url", nxt.consumes)


class OneshotOutDegreePruneTests(unittest.TestCase):
    def test_parse_outline_forest_and_oneshot_builder(self) -> None:
        from sft_prompts import build_outlines_oneshot, parse_outline_forest

        text = """
O1: Bing_Search_Tool | oliver reed royal flash
O1.1: Web_Fetch_Tool | open wikipedia result
O1.2: Generalist_Solution_Generator_Tool | extract the asked fact from the retrieved evidence
O2: Bing_Search_Tool | royal flash nationality
O2.1: Web_Fetch_Tool | open cast page
O2.2: Generalist_Solution_Generator_Tool | extract the asked fact from the retrieved evidence
O3: Bing_Search_Tool | flashman character
O3.1: Web_Fetch_Tool | open biography page
"""
        forest = parse_outline_forest(text, limit=8, depth=3)
        self.assertEqual(len(forest), 3)
        self.assertEqual(forest[0][0]["tool"], "Bing_Search_Tool")

        class Engine:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, prompt: str) -> str:
                self.calls += 1
                return (
                    "S1 | Bing_Search_Tool | Oliver Reed Royal Flash | SnippetsSufficient | primary\n"
                    "S2 | Bing_Search_Tool | Royal Flash 1975 film | SnippetsSufficient |\n"
                    "S1 | Web_Fetch_Tool | open encyclopedia page for Oliver Reed | CanAccess | fetch:oliver reed\n"
                    "S1 | Web_Fetch_Tool | open encyclopedia page for Royal Flash | CanAccess | fetch:royal flash\n"
                    "S1 | Web_Fetch_Tool | open cast page | CanAccess | fetch:cast\n"
                    "S2 | Web_Fetch_Tool | open encyclopedia page for Royal Flash film | CanAccess | fetch:royal flash film\n"
                    "S1 | Generalist_Solution_Generator_Tool | extract asked fact | PageContainsAnswer | extract:answer\n"
                )

        engine = Engine()
        plans = build_outlines_oneshot(engine, "Who is it?", n=6, depth=3)
        self.assertEqual(engine.calls, 1)
        self.assertGreaterEqual(len(plans), 2)
        self.assertTrue(all(plan.steps for plan in plans))
        self.assertTrue(any("search_url" in step.consumes for plan in plans for step in plan.steps))

    def test_oneshot_falls_back_to_layered(self) -> None:
        from sft_prompts import build_outlines_oneshot

        class Engine:
            def __call__(self, prompt: str) -> str:
                return "not a plan"

        plans = build_outlines_oneshot(Engine(), "Who?", n=4, depth=3)
        self.assertEqual(plans, [])

    def test_step1_coverage_is_path_count_not_out_degree(self) -> None:
        from cghe_types import CandidatePlan, PlanStep

        def step(action: str, tool: str) -> PlanStep:
            return PlanStep(subgoal=action, action=action, tool=tool, links_previous=True)

        # Three ACTIVE paths share A and one successor. B is one path but is also
        # a later node of another plan, so it has two direct successors.
        plans = [
            CandidatePlan("p1", (step("A", "search"), step("S", "fetch"), step("X", "extract"))),
            CandidatePlan("p2", (step("A", "search"), step("S", "fetch"), step("Y", "extract"))),
            CandidatePlan("p3", (step("A", "search"), step("S", "fetch"), step("Z", "extract"))),
            CandidatePlan("p4", (step("B", "search"), step("U", "fetch"), step("V", "extract"))),
            CandidatePlan("p5", (step("Q", "search"), step("B", "search"), step("W", "fetch"))),
        ]
        graph = PlanningGraph(plans)
        step1 = {str(node.action).casefold(): node.node_id for node in graph.nodes()}
        self.assertEqual(graph.compute_step1_coverage()[step1["a"]], 3)
        self.assertEqual(graph.compute_step1_coverage()[step1["b"]], 1)
        self.assertEqual(graph.out_degree(step1["a"]), 1)
        self.assertGreater(graph.out_degree(step1["b"]), graph.out_degree(step1["a"]))
        selected = graph.select_step1()
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(str(graph.node(selected[0]).action).casefold(), "a")
        self.assertEqual(selected[1], 3)

    def test_prune_unique_downstream_keeps_shared_nodes(self) -> None:
        from cghe_types import candidate_plan_from_dict

        shared_open = {
            "subgoal": "open shared", "action": "open shared", "tool": "Web_Fetch_Tool",
            "task_id": "open_page", "consumes": ["search_url"], "produces": ["page_content"],
            "instance_key": "", "links_previous": True, "argument": "https://example.com/shared",
        }
        plans = [
            candidate_plan_from_dict({
                "plan_id": "a",
                "steps": [
                    {"subgoal": "q-a", "action": "q-a", "tool": "Bing_Search_Tool",
                     "task_id": "search", "produces": ["search_results"], "links_previous": False},
                    dict(shared_open),
                    {"subgoal": "extract a", "action": "extract a", "tool": "Generalist_Solution_Generator_Tool",
                     "task_id": "extract_fact", "consumes": ["page_content"],
                     "instance_key": "a:extract", "links_previous": True},
                ],
            }),
            candidate_plan_from_dict({
                "plan_id": "b",
                "steps": [
                    {"subgoal": "q-b", "action": "q-b", "tool": "Bing_Search_Tool",
                     "task_id": "search", "produces": ["search_results"], "links_previous": False},
                    dict(shared_open),
                    {"subgoal": "extract b", "action": "extract b", "tool": "Generalist_Solution_Generator_Tool",
                     "task_id": "extract_fact", "consumes": ["page_content"],
                     "instance_key": "b:extract", "links_previous": True},
                ],
            }),
        ]
        graph = PlanningGraph(plans)
        search_a = next(n for n in graph.nodes() if n.action == "q-a")
        open_node = next(n for n in graph.nodes() if "search_url" in n.consumes)
        extract_a = next(n for n in graph.nodes() if n.instance_key == "a:extract")
        extract_b = next(n for n in graph.nodes() if n.instance_key == "b:extract")
        graph.prune_unique_downstream(search_a.node_id)
        self.assertEqual(graph.node(search_a.node_id).status.value, "failed")
        self.assertEqual(graph.node(extract_a.node_id).status.value, "failed")
        self.assertEqual(graph.node(open_node.node_id).status.value, "pending")
        self.assertEqual(graph.node(extract_b.node_id).status.value, "pending")
        self.assertEqual(graph.active_plan_ids(), ("b",))

    def test_extract_not_ready_without_page(self) -> None:
        from sft_prompts import build_outlines_oneshot

        class Engine:
            def __call__(self, prompt: str) -> str:
                return (
                    "S1 | Bing_Search_Tool | query one | SnippetsSufficient | primary\n"
                    "S2 | Bing_Search_Tool | query two | SnippetsSufficient |\n"
                    "S1 | Web_Fetch_Tool | open wiki | CanAccess | fetch:wiki\n"
                    "S1 | Web_Fetch_Tool | open bio | CanAccess | fetch:bio\n"
                    "S2 | Web_Fetch_Tool | open other | CanAccess | fetch:other\n"
                    "S1 | Generalist_Solution_Generator_Tool | extract asked fact | PageContainsAnswer | extract:answer\n"
                )

        plans = build_outlines_oneshot(Engine(), "Q?", n=4, depth=3)
        graph = PlanningGraph(plans)
        search = graph.select_node(policy="max_out_degree")
        self.assertIsNotNone(search)
        graph.mark_satisfied(search.node_id)
        graph.store_search_results(search.node_id, ["https://example.com/page"])
        ready_tools = {graph.node(nid).tool for nid in graph.executable_node_ids()}
        self.assertNotIn("generalist_solution_generator_tool", ready_tools)

    def test_role_bind_prefers_wikipedia_over_rank(self) -> None:
        from cghe_types import candidate_plan_from_dict

        plans = [
            candidate_plan_from_dict({
                "plan_id": "a",
                "steps": [
                    {
                        "subgoal": "q", "action": "q", "tool": "Bing_Search_Tool",
                        "task_id": "search", "produces": ["search_results"],
                    },
                    {
                        "subgoal": "role=wiki_entity open encyclopedia",
                        "action": "role=wiki_entity open encyclopedia",
                        "tool": "Web_Fetch_Tool",
                        "task_id": "open_page", "consumes": ["search_url"],
                        "produces": ["page_content"], "instance_key": "a:open",
                        "links_previous": True,
                    },
                ],
            })
        ]
        graph = PlanningGraph(plans)
        search = next(n for n in graph.nodes() if n.task_id == "search")
        graph.mark_satisfied(search.node_id)
        graph.store_search_results(
            search.node_id,
            [
                "https://pinterest.com/spam",
                "https://en.wikipedia.org/wiki/Good",
                "https://example.com/other",
            ],
            results=[
                {"title": "spam", "url": "https://pinterest.com/spam", "snippet": "pin"},
                {"title": "Good", "url": "https://en.wikipedia.org/wiki/Good", "snippet": "encyclopedia"},
                {"title": "other", "url": "https://example.com/other", "snippet": "misc"},
            ],
        )
        open_id = next(n.node_id for n in graph.nodes() if "search_url" in n.consumes)
        self.assertTrue(graph._try_bind_open(open_id))
        self.assertIn("wikipedia.org", graph.bound_url(open_id))

    def test_no_role_bind_ablation_uses_first_url(self) -> None:
        import os
        from cghe_types import candidate_plan_from_dict

        plans = [
            candidate_plan_from_dict({
                "plan_id": "a",
                "steps": [
                    {
                        "subgoal": "q", "action": "q", "tool": "Bing_Search_Tool",
                        "task_id": "search", "produces": ["search_results"],
                    },
                    {
                        "subgoal": "role=wiki_entity open encyclopedia",
                        "action": "role=wiki_entity open encyclopedia",
                        "tool": "Web_Fetch_Tool",
                        "task_id": "open_page", "consumes": ["search_url"],
                        "produces": ["page_content"], "instance_key": "a:open",
                        "links_previous": True,
                    },
                ],
            })
        ]
        graph = PlanningGraph(plans)
        search = next(n for n in graph.nodes() if n.task_id == "search")
        graph.mark_satisfied(search.node_id)
        open_id = next(n.node_id for n in graph.nodes() if "search_url" in n.consumes)
        old = os.environ.get("CGHE_NO_ROLE_BIND")
        os.environ["CGHE_NO_ROLE_BIND"] = "1"
        try:
            graph.store_search_results(
                search.node_id,
                ["https://example.com/first", "https://en.wikipedia.org/wiki/Good"],
                results=[
                    {"title": "first", "url": "https://example.com/first", "snippet": ""},
                    {"title": "Good", "url": "https://en.wikipedia.org/wiki/Good", "snippet": ""},
                ],
            )
            self.assertTrue(graph._try_bind_open(open_id))
            self.assertEqual(graph.bound_url(open_id), "https://example.com/first")
        finally:
            if old is None:
                os.environ.pop("CGHE_NO_ROLE_BIND", None)
            else:
                os.environ["CGHE_NO_ROLE_BIND"] = old


if __name__ == "__main__":
    unittest.main()
