"""CPU test: <search> / <python> become tool.execute(query=...) or url=."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_prompts import _planner_for_parser, tags_to_execute  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
