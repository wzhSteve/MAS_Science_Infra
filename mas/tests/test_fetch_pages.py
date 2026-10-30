"""Multi-URL fetch keeps each page and does not open a fourth URL."""

from __future__ import annotations

import os
import sys
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
from tools.search import fetch_pages  # noqa: E402


class TestFetchPages(unittest.TestCase):
    def test_three_pages_keep_a_failure_and_drop_the_fourth(self) -> None:
        seen: list[str] = []

        def fake(url: str, max_chars: int = 2000, _repair: bool = True) -> str:
            del max_chars, _repair
            seen.append(url)
            if "bad" in url:
                return "fetch_unavailable: GET 500"
            return f"url: {url}\nbody"

        urls = [
            "https://a.example/one",
            "https://b.example/two",
            "https://bad.example/three",
            "https://c.example/four",
        ]
        with mock.patch.object(search_mod, "fetch_page", side_effect=fake):
            text = fetch_pages(urls)
        self.assertEqual(seen, urls[:3])
        self.assertIn("url: https://a.example/one", text)
        self.assertIn("url: https://b.example/two", text)
        self.assertIn("fetch_unavailable: GET 500", text)
        self.assertNotIn("four", text)

    def test_blocked_host_is_skipped(self) -> None:
        with mock.patch.object(search_mod, "fetch_page", side_effect=AssertionError("blocked")):
            text = fetch_pages(["https://www.imdb.com/title/tt1/"])
        self.assertIn("skipped_blocked: https://www.imdb.com/title/tt1/", text)


if __name__ == "__main__":
    unittest.main()
