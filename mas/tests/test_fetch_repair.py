"""Fetch repair: Wikipedia title correction and 403-host search fallback."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.search import (  # noqa: E402
    fetch_page,
    is_blocked_fetch_host,
    query_from_url,
    wikipedia_url_parts,
)


class TestFetchRepairHelpers(unittest.TestCase):
    def test_wikipedia_url_parts(self) -> None:
        lang, title = wikipedia_url_parts(
            "https://en.wikipedia.org/wiki/Gisherbrum_II"
        )
        self.assertEqual(lang, "en")
        self.assertEqual(title, "Gisherbrum II")
        self.assertIsNone(wikipedia_url_parts("https://www.imdb.com/title/tt1/"))

    def test_blocked_hosts(self) -> None:
        self.assertTrue(is_blocked_fetch_host("https://www.imdb.com/title/tt0073639/"))
        self.assertTrue(is_blocked_fetch_host("https://www.allmusic.com/album/x"))
        self.assertTrue(is_blocked_fetch_host("https://www.discogs.com/artist/1"))
        self.assertFalse(is_blocked_fetch_host("https://en.wikipedia.org/wiki/Ada"))

    def test_query_from_url(self) -> None:
        self.assertIn("Gisherbrum II", query_from_url("https://en.wikipedia.org/wiki/Gisherbrum_II"))
        self.assertIn("imdb", query_from_url("https://www.imdb.com/title/tt0073639/").lower())


class TestFetchRepair(unittest.TestCase):
    def test_wikipedia_404_uses_mediawiki_search(self) -> None:
        calls: list[str] = []

        def fake_request(_method: str, url: str, **_kwargs):
            calls.append(url)
            if "api.php" in url:
                payload = {
                    "query": {
                        "search": [
                            {"title": "Gasherbrum II"},
                        ]
                    }
                }
                return json.dumps(payload), ""
            if "Gisherbrum" in url:
                return "", "GET 404"
            if "Gasherbrum_II" in url:
                return "<html><body><p>Gasherbrum II is a mountain.</p></body></html>", ""
            return "", "GET 404"

        with patch("tools.search._request", side_effect=fake_request), patch(
            "tools.search.bing_search", side_effect=AssertionError("wiki repair must not need bing")
        ):
            out = fetch_page("https://en.wikipedia.org/wiki/Gisherbrum_II")
        self.assertIn("url: https://en.wikipedia.org/wiki/Gasherbrum_II", out)
        self.assertIn("corrected_from: https://en.wikipedia.org/wiki/Gisherbrum_II", out)
        self.assertIn("Gasherbrum II is a mountain", out)
        self.assertTrue(any("api.php" in url for url in calls))

    def test_imdb_skips_get_and_falls_back_to_bing(self) -> None:
        with patch(
            "tools.search._request",
            side_effect=AssertionError("blocked host must not GET"),
        ), patch("tools.search.bing_search", return_value="bing results for: tt0073639") as bing:
            out = fetch_page("https://www.imdb.com/title/tt0073639/")
        self.assertTrue(out.startswith("fetch_blocked: https://www.imdb.com/title/tt0073639/"))
        self.assertIn("used_search_fallback: bing", out)
        self.assertIn("bing results for: tt0073639", out)
        bing.assert_called_once()

    def test_generic_404_falls_back_to_bing(self) -> None:
        with patch("tools.search._request", return_value=("", "GET 404")), patch(
            "tools.search.bing_search", return_value="bing results for: missing page"
        ):
            out = fetch_page("https://www.historynet.com/george-s-patton-ii-postwar-career.html")
        self.assertIn("fetch_blocked:", out)
        self.assertIn("GET 404", out)
        self.assertIn("bing results for: missing page", out)

    def test_bot_wall_falls_back(self) -> None:
        html = "<html><body>Bot Verification Verifying that you are not a robot...</body></html>"
        with patch("tools.search._request", return_value=(html, "")), patch(
            "tools.search.bing_search", return_value="bing results for: Afroman"
        ):
            out = fetch_page("https://gropedia.com/wiki/Afroman_discography")
        self.assertIn("fetch_blocked:", out)
        self.assertIn("bot_wall", out)

    def test_wikipedia_follows_searchinfo_suggestion(self) -> None:
        def fake_request(_method: str, url: str, **_kwargs):
            if "api.php" in url:
                if "Gisherbrum" in url:
                    return json.dumps({
                        "query": {
                            "searchinfo": {"suggestion": "gasherbrum ii"},
                            "search": [],
                        }
                    }), ""
                if "gasherbrum" in url.lower():
                    return json.dumps({"query": {"search": [{"title": "Gasherbrum II"}]}}), ""
                return json.dumps({"query": {"search": []}}), ""
            if "Gisherbrum" in url:
                return "", "GET 404"
            if "Gasherbrum_II" in url:
                return "<html><body><p>Gasherbrum II is in the Karakoram.</p></body></html>", ""
            return "", "GET 404"

        with patch("tools.search._request", side_effect=fake_request):
            out = fetch_page("https://en.wikipedia.org/wiki/Gisherbrum_II")
        self.assertIn("corrected_from:", out)
        self.assertIn("Gasherbrum II is in the Karakoram", out)

    def test_successful_wikipedia_fetch_unchanged(self) -> None:
        html = "<html><body><p>Afroman is a rapper.</p></body></html>"
        with patch("tools.search._request", return_value=(html, "")), patch(
            "tools.search.bing_search", side_effect=AssertionError("success path must not search")
        ):
            out = fetch_page("https://en.wikipedia.org/wiki/Afroman")
        self.assertEqual(out, "url: https://en.wikipedia.org/wiki/Afroman\nAfroman is a rapper.")


if __name__ == "__main__":
    unittest.main()
