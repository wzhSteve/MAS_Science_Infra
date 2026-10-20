"""CPU test: <search> / <python> become tool.execute(query=...) or url=."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_prompts import (  # noqa: E402
    _planner_for_parser,
    actual_label,
    add_obtained_information,
    apply_outline,
    budget_unresolved_reason,
    parse_planner_payload,
    parse_verifier_text,
    tags_to_execute,
)


class _Tool:
    def execute(self, query=None, url=None):
        return {"query": query, "url": url}


def _run(command: str):
    tool = _Tool()
    ns = {"tool": tool}
    exec(command, ns)  # noqa: S102
    return ns["execution"]


class TagToExecuteTest(unittest.TestCase):
    def test_bing_wikipedia_and_generalist_use_query(self) -> None:
        for name in (
            "Bing_Search_Tool",
            "Wikipedia_RAG_Search_Tool",
            "Generalist_Solution_Generator_Tool",
        ):
            command = tags_to_execute(name, "note <search>Eleanor of Lancaster</search>", "No command found.")
            self.assertEqual(_run(command), {"query": "Eleanor of Lancaster", "url": None})

    def test_web_fetch_uses_url(self) -> None:
        command = tags_to_execute(
            "Web_Fetch_Tool",
            "<search>https://en.wikipedia.org/wiki/Example</search>",
            "No command found.",
        )
        self.assertEqual(
            _run(command),
            {"query": None, "url": "https://en.wikipedia.org/wiki/Example"},
        )

    def test_python_uses_python_tag(self) -> None:
        command = tags_to_execute(
            "Python_Code_Generator_Tool",
            "<python>1 + 2 * 3</python>",
            "No command found.",
        )
        self.assertEqual(_run(command), {"query": "1 + 2 * 3", "url": None})

    def test_python_ignores_search_tag(self) -> None:
        extracted = "No command found."
        command = tags_to_execute("Python_Code_Generator_Tool", "<search>1+1</search>", extracted)
        self.assertEqual(command, extracted)

    def test_no_tag_keeps_extracted_command(self) -> None:
        extracted = 'execution = tool.execute(query="already")'
        command = tags_to_execute("Bing_Search_Tool", "no tags here", extracted)
        self.assertEqual(command, extracted)
        self.assertEqual(_run(command)["query"], "already")

    def test_tag_overrides_existing_execute(self) -> None:
        raw = 'execution = tool.execute(query="old")\n<search>new query</search>'
        command = tags_to_execute("Bing_Search_Tool", raw, 'execution = tool.execute(query="old")')
        self.assertEqual(_run(command)["query"], "new query")

    def test_arpo_search_query_tag_is_a_fallback(self) -> None:
        raw = "<|begin_search_query|>Carl Johan Alonsens fødelsdato<|end_search_query|>"
        command = tags_to_execute("Web_Fetch_Tool", raw, "No command found.")
        self.assertEqual(_run(command)["url"], "Carl Johan Alonsens fødelsdato")
        command = tags_to_execute("Bing_Search_Tool", raw, "No command found.")
        self.assertEqual(_run(command)["query"], "Carl Johan Alonsens fødelsdato")

    def test_search_tag_wins_over_arpo_tag(self) -> None:
        raw = "<|begin_search_query|>old<|end_search_query|>\n<search>kept</search>"
        command = tags_to_execute("Bing_Search_Tool", raw, "No command found.")
        self.assertEqual(_run(command)["query"], "kept")

    def test_planner_strips_brackets_and_think_from_tool_name(self) -> None:
        raw = (
            "Context: birth years\n"
            "Sub-Goal: search both names\n"
            "Tool Name: <Bing_Search_Tool></think>\n"
            "extra ramble\n"
        )
        cleaned = _planner_for_parser(raw)
        self.assertIn("Tool Name: Bing_Search_Tool", cleaned)
        self.assertNotIn("extra ramble", cleaned)
        self.assertNotIn("</think>", cleaned)

    def test_planner_keeps_a_clean_tool_name(self) -> None:
        raw = "Context: facts\nSub-Goal: open the page\nTool Name: Web_Fetch_Tool\n"
        cleaned = _planner_for_parser(raw)
        self.assertTrue(cleaned.endswith("Tool Name: Web_Fetch_Tool"))

    def test_last_tag_wins_and_quotes_are_exec_safe(self) -> None:
        raw = "<search>first</search>\n<search>say \"hi\"\nand more</search>"
        command = tags_to_execute("Bing_Search_Tool", raw, "No command found.")
        self.assertEqual(_run(command)["query"], 'say "hi"\nand more')


class _Memory:
    def __init__(self):
        self.actions = {
            "Action Step 1": {
                "tool_name": "Bing_Search_Tool",
                "sub_goal": "find dates",
                "command": "execution = tool.execute(query='old')",
                "result": "snippet https://en.wikipedia.org/wiki/A",
            }
        }
        self.outline = []
        self.obtained_information = []

    def get_actions(self):
        return self.actions


class MemoryPromptTest(unittest.TestCase):
    def test_json_becomes_the_three_lines(self) -> None:
        raw = """<think>look</think>
{"outline":[{"subgoal":"find birth dates","status":"pending"}],
 "subgoal":"search both names","selected_tool":"Bing_Search_Tool",
 "tool_input":"William Green birth","prediction":"partial",
 "prediction_reason":"snippets only"}
"""
        cleaned = _planner_for_parser(raw)
        self.assertIn("Context: William Green birth", cleaned)
        self.assertIn("Sub-Goal: search both names", cleaned)
        self.assertIn("Tool Name: Bing_Search_Tool", cleaned)
        payload = parse_planner_payload(raw)
        assert payload is not None
        self.assertEqual(payload["prediction"], "partial")
        self.assertEqual(payload["outline"][0]["status"], "pending")

    def test_broken_json_keeps_the_three_lines(self) -> None:
        raw = "not json {broken\nContext: birth years\nSub-Goal: search both names\nTool Name: Bing_Search_Tool\n"
        self.assertIsNone(parse_planner_payload(raw))
        cleaned = _planner_for_parser(raw)
        self.assertIn("Tool Name: Bing_Search_Tool", cleaned)
        self.assertIn("Context: birth years", cleaned)

    def test_outline_update_keeps_history(self) -> None:
        memory = _Memory()
        apply_outline(memory, [{"subgoal": "read the page", "status": "pending"}])
        self.assertEqual(memory.outline[0]["subgoal"], "read the page")
        self.assertIn("Action Step 1", memory.get_actions())
        self.assertIn("query='old'", memory.get_actions()["Action Step 1"]["command"])

    def test_obtained_fact_and_url_are_deduped(self) -> None:
        memory = _Memory()
        self.assertTrue(add_obtained_information(memory, "Eleanor was born in 1318"))
        self.assertFalse(add_obtained_information(memory, "Eleanor   was born in 1318"))
        self.assertFalse(add_obtained_information(memory, "NONE"))
        self.assertTrue(add_obtained_information(memory, "https://en.wikipedia.org/wiki/A"))
        self.assertFalse(add_obtained_information(memory, "https://en.wikipedia.org/wiki/A"))
        self.assertEqual(len(memory.obtained_information), 2)

    def test_nonempty_search_is_partial(self) -> None:
        self.assertEqual(actual_label("titles https://example.com"), "partial")
        self.assertEqual(actual_label(""), "failure")
        self.assertEqual(actual_label("search_unavailable"), "failure")
        self.assertEqual(actual_label("No result was generated because the tool was not found."), "failure")

    def test_budget_reason_lists_unfinished_subgoals(self) -> None:
        memory = _Memory()
        memory.outline = [
            {"subgoal": "birth date of Eleanor", "status": "pending"},
            {"subgoal": "already found", "status": "done"},
            {"subgoal": "birth date of Green", "status": "failed"},
        ]
        text = budget_unresolved_reason("hit the step cap", memory)
        self.assertIn("birth date of Eleanor", text)
        self.assertIn("birth date of Green", text)
        self.assertNotIn("already found", text)

    def test_verifier_json_maps_onto_conclusion(self) -> None:
        verdict = parse_verifier_text(
            '{"decision":"continue","reason":"missing the second date","obtained":"Eleanor was born in 1318"}'
        )
        self.assertEqual(verdict["conclusion"], "CONTINUE")
        self.assertEqual(verdict["obtained"], "Eleanor was born in 1318")
        lines = parse_verifier_text("Conclusion: STOP\nReason: facts are enough\nObtained: NONE")
        self.assertEqual(lines["conclusion"], "STOP")
        self.assertEqual(lines["obtained"], "NONE")

    def test_copied_example_json_is_skipped_for_a_later_tool(self) -> None:
        raw = """
{"outline":[{"subgoal":"task","status":"pending"}],"subgoal":"this step","selected_tool":"one of Bing_Search_Tool, Wikipedia_RAG_Search_Tool, Web_Fetch_Tool, Python_Code_Generator_Tool, Generalist_Solution_Generator_Tool","tool_input":"query or one URL","prediction":"partial","prediction_reason":"one short line"}
{"outline":[{"subgoal":"birth date of Eleanor","status":"pending"}],"subgoal":"open the page","selected_tool":"Wikipedia_RAG_Search_Tool","tool_input":"Eleanor of Lancaster birth","prediction":"partial","prediction_reason":"snippet has no year"}
"""
        payload = parse_planner_payload(raw)
        assert payload is not None
        self.assertEqual(payload["tool_name"], "Wikipedia_RAG_Search_Tool")
        self.assertEqual(payload["subgoal"], "open the page")
        self.assertEqual(payload["outline"], [{"subgoal": "birth date of Eleanor", "status": "pending"}])
        cleaned = _planner_for_parser(raw)
        self.assertIn("Tool Name: Wikipedia_RAG_Search_Tool", cleaned)
        self.assertNotIn("this step", cleaned)
        self.assertNotIn("[pending] task", cleaned)

    def test_example_json_alone_does_not_become_a_plan(self) -> None:
        raw = (
            '{"outline":[{"subgoal":"task","status":"pending"}],"subgoal":"this step",'
            '"selected_tool":"one of Bing_Search_Tool, Wikipedia_RAG_Search_Tool",'
            '"tool_input":"query or one URL","prediction_reason":"one short line"}'
        )
        self.assertIsNone(parse_planner_payload(raw))
        memory = _Memory()
        memory.outline = [{"subgoal": "keep the old plan", "status": "pending"}]
        apply_outline(memory, [{"subgoal": "task", "status": "pending"}])
        self.assertEqual(memory.outline[0]["subgoal"], "keep the old plan")

    def test_placeholder_subgoal_uses_tool_input(self) -> None:
        raw = (
            '{"outline":[{"subgoal":"task","status":"pending"}],"subgoal":"this step",'
            '"selected_tool":"Wikipedia_RAG_Search_Tool","tool_input":"Eleanor of Lancaster birth",'
            '"prediction_reason":"one short line"}'
        )
        payload = parse_planner_payload(raw)
        assert payload is not None
        self.assertEqual(payload["tool_name"], "Wikipedia_RAG_Search_Tool")
        self.assertEqual(payload["subgoal"], "Eleanor of Lancaster birth")
        self.assertEqual(payload["outline"], [])
        self.assertEqual(payload["prediction_reason"], "")

    def test_placeholder_three_lines_are_not_a_tool(self) -> None:
        raw = "Context: <query or one URL>\nSub-Goal: <query or one URL>\nTool Name: <query or one URL>\n"
        self.assertIsNone(parse_planner_payload(raw))
        cleaned = _planner_for_parser(raw)
        self.assertNotIn("Tool Name: query", cleaned)
        self.assertNotIn("Bing_Search_Tool", cleaned)

    def test_none_obtained_does_not_enter_memory(self) -> None:
        memory = _Memory()
        self.assertFalse(add_obtained_information(memory, "None (birth date not listed on the page)"))
        self.assertFalse(add_obtained_information(memory, "null"))
        self.assertFalse(add_obtained_information(memory, "n/a"))
        self.assertEqual(memory.obtained_information, [])
        verdict = parse_verifier_text(
            "Conclusion: CONTINUE\nReason:\nObtained: None (birth date not listed on the page)"
        )
        self.assertEqual(verdict["obtained"], "NONE")
        self.assertEqual(verdict["reason"], "关键事实还没有")
        stop = parse_verifier_text("Conclusion: STOP\nReason:\nObtained: NONE")
        self.assertEqual(stop["reason"], "现有信息够回答")


if __name__ == "__main__":
    unittest.main()
