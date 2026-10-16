"""AgentFlow toolbox. Search tools return title, URL, and a truncated description.

``Bing_Search_Tool`` is local: it calls ``mas.tools.search.bing_search`` and does
not open result pages. ``Web_Fetch_Tool`` is local: it calls
``mas.tools.search.fetch_page`` with one URL. ``Wikipedia_Search_Tool`` keeps
the official class for metadata shape, but ``execute`` returns the same
plain-text kernel output and does not fill ``retrieved_information``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Tuple

# Short class names accepted in runtime.yaml -> official module path.
# Bing has no upstream class; an empty path marks the local tool.
OFFICIAL_TOOLS: Dict[str, str] = {
    "Base_Generator_Tool": "agentflow.tools.base_generator.tool",
    "Python_Coder_Tool": "agentflow.tools.python_coder.tool",
    "Bing_Search_Tool": "",
    "Wikipedia_Search_Tool": "agentflow.tools.wikipedia_search.tool",
    "Web_Fetch_Tool": "",
}
DEFAULT_TOOLS = list(OFFICIAL_TOOLS)

_BING_DESCRIPTION = (
    "Bing web search. Returns up to 10 entries, each with a title, its URL, "
    "and a short truncated description. Does not open pages. "
    "Read a page only with a separate fetch step that is given one of these URLs."
)
_WIKI_DESCRIPTION = (
    "Wikipedia search. Returns up to 10 entries, each with a title, its URL, "
    "and a short truncated description. Does not open pages and does not call web search."
)
_FETCH_DESCRIPTION = (
    "Open exactly one URL and extract its readable text. Does not search. "
    "Get a URL from Bing_Search_Tool or Wikipedia_Search_Tool first."
)


@contextmanager
def _placeholder_keys() -> Iterator[None]:
    """Official constructors refuse to build without provider keys."""
    added = [k for k in ("GOOGLE_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    for key in added:
        os.environ[key] = "local-placeholder"
    try:
        yield
    finally:
        for key in added:
            os.environ.pop(key, None)


def _unavailable(text: str) -> bool:
    low = str(text or "").lstrip().lower()
    return low.startswith(("search_unavailable", "fetch_unavailable", "error"))


def local_bing_search(query: str) -> str:
    from mas.tools.search import bing_search

    return str(bing_search(str(query or "")) or "")


def local_wikipedia_search(query: str) -> str:
    from mas.tools.wikipedia import wikipedia_search

    return str(wikipedia_search(str(query or "")) or "")


def local_web_fetch(url: str) -> str:
    from mas.tools.search import fetch_page

    return str(fetch_page(str(url or "")) or "")


class Bing_Search_Tool:
    """Local Bing tool. Matches official ``BaseTool`` methods the Executor calls.

    Official ``Executor.execute_tool_command`` always does
    ``tool.set_custom_output_dir(query_cache_dir)`` before ``tool.execute``.
    This class is not in the upstream package, so those methods live here.
    """

    tool_name = "Bing_Search_Tool"
    tool_description = _BING_DESCRIPTION
    tool_version = "1.0.0"
    input_types = {"query": "str - the search query."}
    output_type = "str - numbered entries with title, URL, and truncated description."
    demo_commands = [{
        "command": 'execution = tool.execute(query="Citibank founded")',
        "description": "Search the web with Bing.",
    }]
    user_metadata: Dict[str, Any] = {}
    require_llm_engine = False

    def __init__(self) -> None:
        self.output_dir = None
        self.model_string = None
        self.user_metadata = dict(type(self).user_metadata)

    def set_custom_output_dir(self, output_dir: Any) -> None:
        self.output_dir = output_dir

    def set_llm_engine(self, model_string: Any) -> None:
        self.model_string = model_string

    def execute(self, query: str, **_kwargs: Any) -> str:
        return local_bing_search(query)


class Web_Fetch_Tool:
    """Local page fetch. Matches official ``BaseTool`` methods the Executor calls.

    Input is a single URL. Does not search and does not reuse official
    ``Web_Search_Tool`` / Web_RAG.
    """

    tool_name = "Web_Fetch_Tool"
    tool_description = _FETCH_DESCRIPTION
    tool_version = "1.0.0"
    input_types = {"url": "str - the page URL to open."}
    output_type = "str - truncated visible text from the page."
    demo_commands = [{
        "command": 'execution = tool.execute(url="https://example.com")',
        "description": "Read one page.",
    }]
    user_metadata: Dict[str, Any] = {}
    require_llm_engine = False

    def __init__(self) -> None:
        self.output_dir = None
        self.model_string = None
        self.user_metadata = dict(type(self).user_metadata)

    def set_custom_output_dir(self, output_dir: Any) -> None:
        self.output_dir = output_dir

    def set_llm_engine(self, model_string: Any) -> None:
        self.model_string = model_string

    def execute(self, url: str, **_kwargs: Any) -> str:
        return local_web_fetch(url)


_LOCAL_TOOL_CLASSES = {
    "Bing_Search_Tool": Bing_Search_Tool,
    "Web_Fetch_Tool": Web_Fetch_Tool,
}


def _metadata(instance: Any, cls: Any) -> Dict[str, Any]:
    """Same fields as ``Initializer._load_single_tool``."""
    return {
        "tool_name": getattr(instance, "tool_name", "Unknown"),
        "tool_description": getattr(instance, "tool_description", "No description"),
        "tool_version": getattr(instance, "tool_version", "Unknown"),
        "input_types": getattr(instance, "input_types", {}),
        "output_type": getattr(instance, "output_type", "Unknown"),
        "demo_commands": getattr(instance, "demo_commands", []),
        "user_metadata": getattr(instance, "user_metadata", {}),
        "require_llm_engine": getattr(cls, "require_llm_engine", False),
    }


def build_toolbox(
    enabled: List[str], tool_engine: Callable[[], Any]
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """Return ``(available_tools, toolbox_metadata, tool_instances_cache)``.

    Keys are the external tool names, sorted like ``run_demo_commands``.
    ``tool_engine()`` builds the engine for tools that call an LLM.
    """
    import importlib

    instances: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    for short in enabled:
        if short not in OFFICIAL_TOOLS:
            raise ValueError(f"not an AgentFlow tool: {short} (allowed: {', '.join(OFFICIAL_TOOLS)})")
        with _placeholder_keys():
            if short in _LOCAL_TOOL_CLASSES:
                cls = _LOCAL_TOOL_CLASSES[short]
                inst = cls()
            else:
                cls = getattr(importlib.import_module(OFFICIAL_TOOLS[short]), short)
                inst = cls(model_string="gpt-4o-mini")
                if short == "Wikipedia_Search_Tool":
                    inst.execute = local_wikipedia_search
                    inst.tool_description = _WIKI_DESCRIPTION
                    inst.output_type = "str - numbered entries with title, URL, and truncated description."
        if hasattr(inst, "llm_engine"):
            inst.llm_engine = tool_engine()
        instances[inst.tool_name] = inst
        metadata[inst.tool_name] = _metadata(inst, cls)
    available = sorted(metadata)
    return available, {name: metadata[name] for name in available}, instances
