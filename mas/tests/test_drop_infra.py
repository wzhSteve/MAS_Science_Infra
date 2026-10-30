"""Drop infra-fault rollouts from a GRPO group without changing the algorithm."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestKeepIndicesDropInfra(unittest.TestCase):
    def test_drops_fault_rows_and_undersized_groups(self):
        from workflow.mas_health import keep_indices_drop_infra

        ids = ["a0", "a1", "a2", "a3", "b0", "b1", "c0", "c1"]
        groups = ["A", "A", "A", "A", "B", "B", "C", "C"]
        faults = ["proxy_down", "", "", "", "http 429", "", "x", "y"]
        keep = keep_indices_drop_infra(ids, groups, faults, min_group=2)
        self.assertEqual(keep, [1, 2, 3])

    def test_all_clean_group_is_kept(self):
        from workflow.mas_health import keep_indices_drop_infra

        keep = keep_indices_drop_infra(
            ["r0", "r1"],
            ["u", "u"],
            ["", ""],
            min_group=2,
        )
        self.assertEqual(keep, [0, 1])


class TestWikiProxyFailFast(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("TIR_PROXY_FORCE", None)

    def test_wikipedia_search_returns_proxy_down_without_retry(self):
        os.environ["TIR_PROXY_FORCE"] = "down"
        os.environ.pop("MAS_SEARCH_CACHE", None)
        from mas.tools.search import _proxy_alive_memo
        from mas.tools.wikipedia import wikipedia_search

        _proxy_alive_memo["t"] = 0.0
        _proxy_alive_memo["ok"] = False
        out = wikipedia_search("citibank")
        self.assertIn("proxy_down", out)

    def test_wikipedia_retries_429_then_succeeds(self):
        os.environ["TIR_PROXY_FORCE"] = "up"
        os.environ["TIR_WIKI_MIN_INTERVAL"] = "0"
        os.environ.pop("MAS_SEARCH_CACHE", None)
        from unittest.mock import patch

        from mas.tools.wikipedia import wikipedia_search

        class _Resp:
            def __init__(self, status, text="", headers=None):
                self.status_code = status
                self.text = text
                self.headers = headers or {}

        payload = '{"query":{"search":[{"title":"Citibank","snippet":"bank"}]}}'
        seq = [_Resp(429, headers={"Retry-After": "0"}), _Resp(200, payload)]

        def _get(*_a, **_k):
            return seq.pop(0)

        with patch("mas.tools.search.requests.get", _get):
            out = wikipedia_search("Citibank")
        self.assertIn("wikipedia results for: Citibank", out)
        self.assertIn("Citibank", out)

    def test_wiki_fetch_returns_proxy_down(self):
        os.environ["TIR_PROXY_FORCE"] = "down"
        from mas.tools.search import _proxy_alive_memo, fetch_page

        _proxy_alive_memo["t"] = 0.0
        out = fetch_page("https://en.wikipedia.org/wiki/Citibank")
        self.assertEqual(out, "fetch_unavailable: proxy_down")


if __name__ == "__main__":
    unittest.main()
