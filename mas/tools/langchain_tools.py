"""LangChain @tool wrappers with short Hermes-friendly names.

Isolation contract:
  - bing_search / wikipedia_search: query -> top-10 title, URL, truncated description.
  - web_fetch: url -> page content extraction. Never searches.
"""

from __future__ import annotations

from langchain_core.tools import tool

from python_tool import execute_python as _execute_python

from .search import bing_search as _bing_search
from .search import fetch_page as _fetch_page
from .wikipedia import wikipedia_search as _wikipedia_search


@tool
def bing_search(query: str) -> str:
    """Search the web with Bing. Returns up to 10 entries, each with a title, URL, and truncated description. Does not open pages."""
    return _bing_search(query)


@tool
def wikipedia_search(query: str) -> str:
    """Search Wikipedia. Returns up to 10 entries, each with a title, URL, and truncated description. Does not open pages."""
    return _wikipedia_search(query)


@tool
def web_fetch(url: str) -> str:
    """Open one URL and extract its readable text content. Does not search."""
    return _fetch_page(url)


@tool
def execute_python(code: str) -> str:
    """Run a short Python snippet. No imports. Assign the final value to `result`."""
    return _execute_python(code)


TOOLS = [bing_search, wikipedia_search, web_fetch, execute_python]
TOOL_MAP = {t.name: t for t in TOOLS}
