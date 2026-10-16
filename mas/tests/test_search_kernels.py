"""Search kernel unit tests: parsers, anomaly detection, bing /ck/ unpacking.

Network calls are mocked: these tests lock in the parsing contract, not the
availability of any upstream SERP.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.search import (  # noqa: E402
    _is_anomaly_page,
    _organic_hits,
    _parse_bing,
    _parse_ddg_html,
    _parse_ddg_lite,
    _unpack_bing_ck,
    format_title_url,
    truncate_description,
)


DDG_LITE_HTML = """
<html><body>
<a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FRoyal_Flash_(film)&rut=abc">Royal Flash (film) - Wikipedia</a>
<a class="result-link" href="https://www.imdb.com/title/tt0073639/">Royal Flash (1975) - IMDb</a>
</body></html>
"""

DDG_HTML_HTML = """
<html><body>
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FOliver_Reed&rut=x">Oliver Reed - Wikipedia</a></div>
<div class="result"><a class="result__a" href="https://example.com/a">Example A</a></div>
</body></html>
"""

ANOMALY_HTML = "<html><head></head><body>If this error persists, please let us know (anomaly detection)</body></html>"

BING_HTML = """
<html><body><ul>
<li class="b_algo"><h2><a href="https://en.wikipedia.org/wiki/Royal_Flash_(film)">Royal Flash (film) - Wikipedia</a></h2></li>
<li class="b_algo"><h2><a href="https://www.bing.com/ck/a?!&&p=1&u=a1aHR0cHM6Ly93d3cuaW1kYi5jb20vdGl0bGUvdHQwMDczNjM5Lw&ntb=1">Royal Flash (1975) - IMDb</a></h2></li>
</ul></body></html>
"""


def test_parse_ddg_lite_lists_http_titles() -> None:
    hits = _parse_ddg_lite(DDG_LITE_HTML, 5)
    assert hits[0]["title"] == "Royal Flash (film) - Wikipedia"
    assert hits[0]["url"] == "https://en.wikipedia.org/wiki/Royal_Flash_(film)"
    assert hits[1]["url"] == "https://www.imdb.com/title/tt0073639/"


def test_parse_ddg_html_extracts_listed_hits() -> None:
    hits = _parse_ddg_html(DDG_HTML_HTML, 5)
    assert len(hits) == 2
    assert hits[0]["title"] == "Oliver Reed - Wikipedia"
    assert hits[0]["url"] == "https://en.wikipedia.org/wiki/Oliver_Reed"


def test_anomaly_page_detected() -> None:
    assert _is_anomaly_page(ANOMALY_HTML)
    assert not _is_anomaly_page(DDG_LITE_HTML)
    assert not _is_anomaly_page("")


def test_unpack_bing_ck_decodes_real_url() -> None:
    # a1 + base64url of "https://www.imdb.com/title/tt0073639/"
    raw = base64.urlsafe_b64encode(b"https://www.imdb.com/title/tt0073639/").decode().rstrip("=")
    ck = f"https://www.bing.com/ck/a?!&&p=1&u=a1{raw}&ntb=1"
    assert _unpack_bing_ck(ck) == "https://www.imdb.com/title/tt0073639/"
    # plain url passthrough
    assert _unpack_bing_ck("https://example.com/x") == "https://example.com/x"


def test_parse_bing_decodes_ck_links() -> None:
    hits = _parse_bing(BING_HTML, 5)
    assert len(hits) == 2
    assert hits[0]["url"] == "https://en.wikipedia.org/wiki/Royal_Flash_(film)"
    assert hits[1]["url"] == "https://www.imdb.com/title/tt0073639/"


def test_format_title_url_numbered_output() -> None:
    text = format_title_url([{"title": "T1", "url": "https://u1"}, {"title": "T2", "url": "https://u2"}], header="H")
    lines = text.splitlines()
    assert lines[0] == "H"
    assert lines[1] == "1. T1"
    assert lines[2] == "   URL: https://u1"
    assert lines[3] == "2. T2"


def test_truncate_description_strips_and_cuts() -> None:
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"TIR_SEARCH_DESC_CHARS": "10"}):
        assert truncate_description("American <span>bank</span> founded") == "American b ...[truncated]"
        assert truncate_description("short") == "short"
        assert truncate_description("") == ""
        assert truncate_description("&quot;NA&quot;") == '"NA"'


def test_organic_hits_keep_title_link_description() -> None:
    body = {
        "organic": [
            {"title": "Ada Lovelace", "link": "https://example.com/ada", "description": "A" * 250},
            {"title": "skip", "link": ""},
        ]
    }
    hits = _organic_hits(body, 10)
    assert hits == [{
        "title": "Ada Lovelace",
        "url": "https://example.com/ada",
        "description": "A" * 250,
    }]


def test_format_title_url_truncates_description() -> None:
    import os
    from unittest.mock import patch

    with patch.dict(os.environ, {"TIR_SEARCH_DESC_CHARS": "20"}):
        text = format_title_url(
            [{"title": "Ada", "url": "https://u", "description": "X" * 80}],
            header="bing results for: Ada",
        )
    assert "1. Ada" in text
    assert "URL: https://u" in text
    assert "XXXXXXXXXXXXXXXXXXXX ...[truncated]" in text
    assert ("X" * 21) not in text.replace(" ...[truncated]", "")
