"""Bing SERP: Bright Data is the live path; datacenter HTML is not a fallback."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tools.search as search_mod  # noqa: E402
from tools.search import bing_search  # noqa: E402
from tools.search_cache import reload_search_cache  # noqa: E402


class _FakeResp:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class TestBingBrightDataOnly(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cache_path = Path(self._tmp.name) / "search_cache.json"
        saved = os.environ.pop("MAS_SEARCH_CACHE", None)
        if saved is not None:
            self.addCleanup(os.environ.__setitem__, "MAS_SEARCH_CACHE", saved)
        search_mod._BD_ZONE_DENIED = False

    def test_bd_http_error_does_not_scrape_bing_html(self) -> None:
        fake = _FakeResp(429, text="rate limited")
        with mock.patch.dict(
            os.environ,
            {
                "TIR_OFFLINE_SEARCH": "0",
                "BRIGHT_DATA_API_KEY": "test-key",
                "BRIGHT_DATA_MODE": "force",
                "BRIGHT_DATA_MAX_RETRIES": "1",
                "BRIGHT_DATA_TIMEOUT": "2",
                "MAS_SEARCH_CACHE": str(self.cache_path),
                "TIR_BING_MIN_INTERVAL": "0",
            },
            clear=False,
        ), mock.patch.object(search_mod, "requests") as req, mock.patch.object(
            search_mod, "_request", side_effect=AssertionError("must not GET bing.com")
        ):
            req.post.return_value = fake
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertTrue(out.startswith("search_unavailable:"), out)
        self.assertIn("brightdata_failed", out)
        self.assertNotIn("html=", out)
        self.assertEqual(req.post.call_count, 1)
        if self.cache_path.is_file():
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self.assertFalse(any(v == "search_unavailable_in_cache" for v in data.values()))

    def test_empty_organic_does_not_scrape_bing_html(self) -> None:
        envelope = {"body": json.dumps({"organic": []})}
        fake = _FakeResp(200, payload=envelope)
        with mock.patch.dict(
            os.environ,
            {
                "TIR_OFFLINE_SEARCH": "0",
                "BRIGHT_DATA_API_KEY": "test-key",
                "BRIGHT_DATA_MODE": "force",
                "BRIGHT_DATA_MAX_RETRIES": "1",
                "MAS_SEARCH_CACHE": str(self.cache_path),
                "TIR_BING_MIN_INTERVAL": "0",
            },
            clear=False,
        ), mock.patch.object(search_mod, "requests") as req, mock.patch.object(
            search_mod, "_request", side_effect=AssertionError("must not GET bing.com")
        ):
            req.post.return_value = fake
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertTrue(out.startswith("search_unavailable:"), out)
        self.assertIn("empty", out)

    def test_success_stores_cache_second_call_skips_network(self) -> None:
        organic = {
            "body": json.dumps({
                "organic": [{
                    "title": "Ada Lovelace",
                    "link": "https://example.com/ada",
                    "description": "mathematician",
                }]
            })
        }
        fake = _FakeResp(200, payload=organic)
        with mock.patch.dict(
            os.environ,
            {
                "TIR_OFFLINE_SEARCH": "0",
                "BRIGHT_DATA_API_KEY": "test-key",
                "BRIGHT_DATA_MODE": "auto",
                "BRIGHT_DATA_MAX_RETRIES": "1",
                "MAS_SEARCH_CACHE": str(self.cache_path),
                "TIR_BING_MIN_INTERVAL": "0",
            },
            clear=False,
        ), mock.patch.object(search_mod, "requests") as req:
            req.post.return_value = fake
            reload_search_cache()
            first = bing_search("Ada Lovelace")
            self.assertIn("bing results for: Ada Lovelace", first)
            req.post.reset_mock()
            second = bing_search("Ada Lovelace")
        self.assertEqual(first, second)
        req.post.assert_not_called()

    def test_denied_falls_back_to_html_in_auto_mode(self) -> None:
        envelope = {
            "status_code": 407,
            "headers": {"proxy-status": 'brd.superproxy.io; error="http_request_denied"'},
            "body": "",
        }
        fake = _FakeResp(200, payload=envelope, text=json.dumps(envelope))
        html = (
            '<html><body><ul><li class="b_algo"><h2>'
            '<a href="https://example.com/ada">Ada Lovelace</a></h2></li></ul></body></html>'
        )
        with mock.patch.dict(
            os.environ,
            {
                "TIR_OFFLINE_SEARCH": "0",
                "BRIGHT_DATA_API_KEY": "test-key",
                "BRIGHT_DATA_MODE": "auto",
                "BRIGHT_DATA_MAX_RETRIES": "1",
                "MAS_SEARCH_CACHE": str(self.cache_path),
                "TIR_BING_MIN_INTERVAL": "0",
            },
            clear=False,
        ), mock.patch.object(search_mod, "requests") as req, mock.patch.object(
            search_mod, "_request", return_value=(html, "")
        ):
            req.post.return_value = fake
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertIn("bing results for: Ada Lovelace", out)
        self.assertIn("https://example.com/ada", out)

    def test_irrelevant_organic_falls_through_to_wikipedia(self) -> None:
        html = (
            "<html><body><ol id=\"b_results\">"
            "<li class=\"b_algo\"><h2><a href=\"https://www.ada.gov/\">ADA.gov</a></h2></li>"
            "<li class=\"b_algo\"><h2><a href=\"https://www.howard-hotels.com.tw/\">Howard Hotels</a></h2></li>"
            "</ol></body></html>"
        )
        wiki = [{"title": "Ada Lovelace", "url": "https://en.wikipedia.org/wiki/Ada_Lovelace"}]
        with mock.patch.dict(
            os.environ,
            {
                "TIR_OFFLINE_SEARCH": "0",
                "BRIGHT_DATA_MODE": "0",
                "MAS_SEARCH_CACHE": str(self.cache_path),
                "TIR_BING_MIN_INTERVAL": "0",
            },
            clear=False,
        ), mock.patch.object(search_mod, "_request", return_value=(html, "")), mock.patch.object(
            search_mod, "_wikipedia_search_hits", return_value=wiki
        ):
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertIn("wikipedia results for: Ada Lovelace", out)
        self.assertIn("https://en.wikipedia.org/wiki/Ada_Lovelace", out)
        self.assertNotIn("howard-hotels", out)

    def test_empty_serp_is_named_in_the_error(self) -> None:
        html = (
            "<html><body><ol id=\"b_results\"><li class=\"b_no\">"
            "<h1>There are no results for Ada Lovelace</h1></li></ol></body></html>"
        )
        with mock.patch.dict(
            os.environ,
            {
                "TIR_OFFLINE_SEARCH": "0",
                "BRIGHT_DATA_MODE": "0",
                "MAS_SEARCH_CACHE": str(self.cache_path),
                "TIR_BING_MIN_INTERVAL": "0",
            },
            clear=False,
        ), mock.patch.object(search_mod, "_request", return_value=(html, "")), mock.patch.object(
            search_mod, "_wikipedia_search_hits", return_value=[]
        ):
            reload_search_cache()
            out = bing_search("Ada Lovelace")
        self.assertIn("empty_serp", out)
        self.assertTrue(out.startswith("search_unavailable:"), out)

    def test_timeout_disables_brightdata_for_later_auto_calls(self) -> None:
        html = "<html><body><ol id=\"b_results\"><li class=\"b_no\">There are no results</li></ol></body></html>"
        with mock.patch.dict(
            os.environ,
            {
                "TIR_OFFLINE_SEARCH": "0",
                "BRIGHT_DATA_API_KEY": "test-key",
                "BRIGHT_DATA_MODE": "auto",
                "BRIGHT_DATA_MAX_RETRIES": "1",
                "MAS_SEARCH_CACHE": str(self.cache_path),
                "TIR_BING_MIN_INTERVAL": "0",
            },
            clear=False,
        ), mock.patch.object(search_mod, "requests") as req, mock.patch.object(
            search_mod, "_request", return_value=(html, "")
        ), mock.patch.object(search_mod, "_wikipedia_search_hits", return_value=[]):
            req.post.side_effect = TimeoutError("timed out")
            reload_search_cache()
            first = bing_search("Ada Lovelace")
            self.assertIn("search_unavailable:", first)
            self.assertTrue(search_mod._BD_ZONE_DENIED)
            req.post.reset_mock()
            bing_search("Grace Hopper")
        req.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
