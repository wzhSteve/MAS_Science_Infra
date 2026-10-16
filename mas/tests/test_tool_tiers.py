"""lite and pro tool-agent tiers. Network is mocked.

Search tools return title, URL, and a truncated description. Neither
wikipedia_search nor bing_search opens a result page. web_fetch opens
exactly one URL and never searches.

No test in this module may touch a real network endpoint.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.pro_tools import run_pro, select_titles  # noqa: E402
from tools.search import _parse_ddg_lite  # noqa: E402
from tools.tool_agents import TOOL_AGENTS, effective_tier, validate_tool_args  # noqa: E402
from tools.wikipedia import wikipedia_search  # noqa: E402
from workflow.spec import MASSpec  # noqa: E402
from workflow.templates import load_template_workflow  # noqa: E402


class _Resp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


class TestToolTiers(unittest.TestCase):
    def test_missing_template_tier_stays_lite(self):
        spec = MASSpec.model_validate(load_template_workflow("centralized"))
        wiki = next(agent for agent in spec.agents if agent.id == "wikipedia_search")
        self.assertNotEqual(wiki.profile.get("tier"), "pro")
        self.assertEqual(effective_tier(wiki.profile.get("tier")), "lite")

    def test_kernel_switch_forces_lite(self):
        self.assertEqual(effective_tier("pro"), "lite")
        saved = os.environ.pop("SCIENCE_INFRA_TOOL_KERNEL", None)
        try:
            self.assertEqual(effective_tier("pro"), "pro")
            self.assertEqual(effective_tier(None), "lite")
        finally:
            if saved is not None:
                os.environ["SCIENCE_INFRA_TOOL_KERNEL"] = saved

    def test_lite_wikipedia_title_url_and_description_never_opens_pages(self):
        calls = []

        def fake_get(url, **_kwargs):
            calls.append(url)
            if "api.php" in url:
                return _Resp({"query": {"search": [
                    {"title": "Ada Lovelace", "snippet": "math"},
                    {"title": "Analytical Engine", "snippet": "engine"},
                ]}})
            raise AssertionError(f"wikipedia_search opened {url}")

        with patch("tools.wikipedia.requests.get", fake_get):
            text = wikipedia_search("Ada Lovelace")
        self.assertIn("https://en.wikipedia.org/wiki/Ada_Lovelace", text)
        self.assertIn("https://en.wikipedia.org/wiki/Analytical_Engine", text)
        self.assertIn("math", text)
        self.assertIn("engine", text)
        self.assertNotIn("retrieved_information:", text)
        self.assertTrue(all("api.php" in url for url in calls))

    def test_pro_wikipedia_does_not_open_pages(self):
        lite = "wikipedia results for: Ada\n1. Ada Lovelace\n   URL: https://en.wikipedia.org/wiki/Ada_Lovelace\n   math"
        fetched = []

        def _fetch(url, max_chars=2000):
            fetched.append(url)
            return "should not run"

        with patch("tools.search.fetch_page", side_effect=_fetch), patch(
            "tools.pro_tools._rag_summary", side_effect=AssertionError("pro must not RAG")
        ):
            text = run_pro("wikipedia_search", {"query": "Ada Lovelace"}, lambda _args: lite)
        self.assertTrue(text.startswith("tier=pro"))
        self.assertNotIn("pro_fallback", text)
        self.assertIn("Ada Lovelace", text)
        self.assertNotIn("retrieved_information:", text)
        self.assertEqual(fetched, [])
        self.assertEqual(select_titles("Ada Lovelace", ["Ada Lovelace", "Potato"])[0], "Ada Lovelace")

    def test_lite_bing_title_url_and_description_never_opens_pages(self):
        fetched = []

        def _request(method, url, **_kwargs):
            return _BING_HTML, ""

        def _fetch(url, max_chars=2000):
            fetched.append(url)
            return "should not run"

        with patch("tools.search._request", side_effect=_request), patch(
            "tools.search._brightdata_enabled", return_value=False
        ), patch(
            "tools.search.fetch_page", side_effect=_fetch
        ):
            text = TOOL_AGENTS["bing_search"].invoke({"query": "Ada Lovelace"})
        self.assertTrue(text.startswith("tier=lite"))
        self.assertIn("https://example.com/ada", text)
        self.assertIn("Ada Lovelace", text)
        self.assertIn("English mathematician", text)
        self.assertNotIn("abstract:", text)
        self.assertEqual(fetched, [])

    def test_pro_bing_does_not_open_pages(self):
        fetched = []

        def _fetch(url, max_chars=2000):
            fetched.append(url)
            return "should not run"

        with patch("tools.search.fetch_page", side_effect=_fetch), patch(
            "tools.pro_tools._rag_summary", side_effect=AssertionError("pro must not RAG")
        ):
            text = run_pro("bing_search", {"query": "Ada"}, lambda _args: "lite serp")
        self.assertTrue(text.startswith("tier=pro\n"))
        self.assertNotIn("pro_fallback", text)
        self.assertEqual(text, "tier=pro\nlite serp")
        self.assertEqual(fetched, [])

    def test_pro_bing_empty_query_falls_back(self):
        text = run_pro("bing_search", {"query": "  "}, lambda _args: "unused")
        self.assertIn("pro_fallback", text)
        self.assertIn("empty search query", text)

    def test_pro_python_does_not_call_hive_backend(self):
        agent = TOOL_AGENTS["python_coder"]
        saved_env = os.environ.pop("SCIENCE_INFRA_TOOL_KERNEL", None)
        saved_llm = agent._llm_invoke

        def _boom(_args):
            raise AssertionError("hive backend")

        agent._llm_invoke = _boom
        try:
            text = agent.invoke({"code": "result = 6 * 7"}, tier="pro")
        finally:
            agent._llm_invoke = saved_llm
            if saved_env is not None:
                os.environ["SCIENCE_INFRA_TOOL_KERNEL"] = saved_env
        self.assertTrue(text.startswith("tier=pro"))
        self.assertIn("42", text)


_BING_HTML = """
<html><body><ul>
<li class="b_algo"><h2><a href="https://example.com/ada">Ada Lovelace</a></h2>
<p class="b_caption">English mathematician and writer</p></li>
<li class="b_algo"><h2><a href="https://example.com/two">Second</a></h2></li>
</ul></body></html>
"""

_DDG_LITE = """
<html><body><ol>
  <li><a rel="nofollow" class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fada&amp;rut=abc">Ada Lovelace</a></li>
  <li><a rel="nofollow" class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo&amp;rut=def">Second</a></li>
  <li><a rel="nofollow" class="result-link" href="https://example.com/three">Third</a></li>
</ol></body></html>
"""


class TestEachToolAgent(unittest.TestCase):
    def test_lite_invoke_prefixes_every_agent(self):
        samples = {
            "wikipedia_search": {"query": "Ada"},
            "bing_search": {"query": "Ada"},
            "web_search": {"query": "Ada", "url": "https://example.com/ada"},
            "python_coder": {"code": "result = 1"},
            "think": {"text": "reason here"},
        }
        for name, args in samples.items():
            agent = TOOL_AGENTS[name]
            if name == "bing_search":
                with patch("tools.search._request", return_value=(_BING_HTML, "")), patch(
                    "tools.search.fetch_page", side_effect=AssertionError("lite must not open a page"),
                ), patch("tools.search._brightdata_enabled", return_value=False), patch(
                    "tools.pro_tools._rag_summary", side_effect=AssertionError("pro RAG must not run under kernel=1"),
                ):
                    text = agent.invoke(args, tier="pro")
            elif name == "web_search":
                with patch("tools.search.fetch_by_type", return_value="url: https://example.com/ada\nAda Lovelace wrote notes."):
                    text = agent.invoke(args)
            elif name == "wikipedia_search":
                def fake_get(url, **_kwargs):
                    if "api.php" in url:
                        return _Resp({"query": {"search": [{"title": "Ada", "snippet": "s"}]}})
                    raise AssertionError(url)

                with patch("tools.wikipedia.requests.get", fake_get):
                    text = agent.invoke(args)
            else:
                text = agent.invoke(args, tier="pro")
            self.assertTrue(str(text).startswith("tier=lite"), name)
        self.assertIn("42", TOOL_AGENTS["python_coder"].invoke({"code": "result = 6 * 7"}))
        self.assertIn("reason here", TOOL_AGENTS["think"].invoke({"text": "reason here"}))

    def test_ddg_parser_unwraps_redirect_and_keeps_direct(self):
        hits = _parse_ddg_lite(_DDG_LITE, 10)
        self.assertEqual(
            [hit["url"] for hit in hits],
            ["https://example.com/ada", "https://example.com/two", "https://example.com/three"],
        )
        self.assertEqual(hits[0]["title"], "Ada Lovelace")

    def test_bing_empty_query_and_offline_pro(self):
        lite = run_pro("bing_search", {"query": "  "}, lambda _args: "unused")
        self.assertIn("pro_fallback", lite)
        self.assertIn("empty search query", lite)
        with patch.dict(os.environ, {"TIR_OFFLINE_SEARCH": "1"}):
            text = run_pro(
                "bing_search",
                {"query": "Ada"},
                lambda _args: "search_unavailable: TIR_OFFLINE_SEARCH=1",
            )
        self.assertTrue(text.startswith("tier=pro"))
        self.assertIn("TIR_OFFLINE_SEARCH", text)

    def test_web_fetch_does_not_search_and_requires_url(self):
        # url required
        ok, reason = validate_tool_args("web_search", {"query": "Ada"})
        self.assertFalse(ok)
        self.assertIn("url", reason)

        def _fetch(url, max_chars=2000):
            # a search-engine query page must never be requested by web tools
            self.assertNotIn("duckduckgo.com", url)
            return "Ada Lovelace published notes. Unrelated sentence."

        with patch("tools.search.fetch_by_type", side_effect=_fetch):
            web = TOOL_AGENTS["web_fetch"].invoke({"url": "https://example.com/ada"})
        self.assertTrue(web.startswith("tier=lite"))
        self.assertIn("Ada Lovelace published notes.", web)

    def test_web_pro_highlights_query_matches(self):
        def _fetch(url, max_chars=4000):
            self.assertEqual(max_chars, 4000)
            return "url: x\nAda Lovelace published notes. Unrelated sentence."

        with patch("tools.search.fetch_page", side_effect=_fetch):
            web = run_pro("web_search", {"query": "Ada", "url": "https://example.com/ada"}, lambda _args: "lite page")
        self.assertTrue(web.startswith("tier=pro"))
        self.assertIn("query_matches:", web)
        self.assertIn("Ada Lovelace published notes.", web)

        python = run_pro("python_coder", {"code": "print('hi')\nresult = 2"}, TOOL_AGENTS["python_coder"]._kernel)
        self.assertTrue(python.startswith("tier=pro"))
        self.assertIn("stdout:", python)
        self.assertIn("hi", python)
        self.assertIn("result = 2", python)

        think = run_pro("think", {"text": "only the input"}, None)
        self.assertEqual(think, "tier=pro\nonly the input")

        with patch("tools.search.fetch_page", return_value="fetch_unavailable: down"):
            failed = run_pro("web_search", {"query": "Ada", "url": "https://example.com"}, lambda _args: "lite page")
        self.assertIn("pro_fallback", failed)
        self.assertIn("fetch_unavailable", failed)


if __name__ == "__main__":
    unittest.main()
