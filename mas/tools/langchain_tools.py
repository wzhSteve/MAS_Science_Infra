"""LangChain @tool wrappers with short Hermes-friendly names."""

from __future__ import annotations

from langchain_core.tools import tool

from python_tool import execute_python as _execute_python

from .search import web_search as _web_search
from .wikipedia import wikipedia_search as _wikipedia_search


@tool
def web_search(query: str) -> str:
    """Search the public web for facts, news, or pages. Use for open-domain questions."""
    return _web_search(query)


@tool
def wikipedia_search(query: str) -> str:
    """Look up Wikipedia pages and short extracts. Prefer this for well-known entities."""
    return _wikipedia_search(query)


@tool
def execute_python(code: str) -> str:
    """Run a short Python snippet. No imports. Assign the final value to `result`."""
    return _execute_python(code)


TOOLS = [web_search, wikipedia_search, execute_python]
TOOL_MAP = {t.name: t for t in TOOLS}
