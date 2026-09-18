#!/usr/bin/env python3
"""No-LLM sanity check: python sandbox + wikipedia + web_search must work."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

os.environ.pop("TIR_OFFLINE_SEARCH", None)
if not os.environ.get("WEB_SEARCH_PROXY") and not os.environ.get("GOOGLE_CHROME_PROXY"):
    os.environ["WEB_SEARCH_PROXY"] = "http://127.0.0.1:7890"

from python_tool import execute_python  # noqa: E402
from tools.search import web_search  # noqa: E402
from tools.wikipedia import wikipedia_search  # noqa: E402


def _ok(name: str, text: str, needle: str) -> None:
    print(f"===== {name} =====")
    print(text[:800])
    print()
    if needle.lower() not in text.lower():
        raise SystemExit(f"FAIL: {name} did not contain {needle!r}")
    if text.lower().startswith("search_unavailable"):
        raise SystemExit(f"FAIL: {name} unavailable: {text[:300]}")
    print(f"PASS: {name} contains {needle!r}")


def main() -> None:
    py = execute_python("result = 12 * 15")
    _ok("execute_python", py, "180")

    wiki = wikipedia_search("capital of France")
    _ok("wikipedia_search", wiki, "Paris")

    web = web_search("capital of France")
    if web.lower().startswith("search_unavailable") or "No web results" in web:
        raise SystemExit(f"FAIL: web_search: {web[:400]}")
    print("===== web_search =====")
    print(web[:800])
    print()
    print("PASS: web_search returned snippets")
    print("All tools OK.")


if __name__ == "__main__":
    main()
