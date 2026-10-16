"""Five HIVE-style tool-agents for the centralized MAS.

Each id is an encapsulated agent: a tool kernel plus an LLM that reads the
kernel output (or writes code, or reasons) and returns a ``tool_result``.
Pure functions in ``mas/tools`` stay as kernels used *inside* the agent;
they are not first-class MAS nodes.

Palette ids: ``wikipedia_search``, ``bing_search``, ``web_fetch``, ``python_coder``,
``think``. ``web_search`` resolves to ``web_fetch``. ``execute_python`` maps to
``python_coder``. ``google_search`` is not a tool.

Tool isolation contract:
  - ``wikipedia_search`` / ``bing_search``: query -> top-10 entries of title,
    URL, and a truncated description. Neither tier opens a result page.
  - ``web_fetch`` / ``web_search``: url -> page content extraction. They never search.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _import_epc_aw_tool(module_path: str, attr: str) -> Any:
    """Import a vendored EPC-AW tool class from ``legacy/structagent``.

    Resolution order (both sys.path layouts must work):
      1. ``legacy.structagent.epc_aw...`` — repo root on sys.path (normal runs).
      2. Fallback: insert repo root (parent of ``mas/``) and retry once —
         covers ``PYTHONPATH=mas`` single-root processes (check scripts).
    Raises ImportError when the vendored stack is unavailable; callers
    treat that as an explicit missing-dependency degradation (not a crash).
    """
    try:
        mod = importlib.import_module(f"legacy.structagent.{module_path}")
    except ImportError:
        repo_root = str(Path(__file__).resolve().parents[2])
        if repo_root in sys.path:
            raise
        sys.path.insert(0, repo_root)
        mod = importlib.import_module(f"legacy.structagent.{module_path}")
    return getattr(mod, attr)


def effective_tier(tier: Optional[str]) -> str:
    """``pro`` only when requested and the kernel-only test switch is off."""
    forced = os.getenv("SCIENCE_INFRA_TOOL_KERNEL", "").strip().lower() in ("1", "true", "yes")
    if forced or str(tier or "").strip().lower() != "pro":
        return "lite"
    return "pro"


_logged_tier_mismatches: set = set()


def log_effective_tier(tool_id: str, configured_tier: Optional[str]) -> str:
    """Startup structured log for tier resolution (MAS_design §5.2, Phase 4).

    Emits ONCE per (tool, mismatch) at warning level:
      {"tool", "configured_tier", "effective_tier", "override_reason"}
    ``override_reason`` is ``SCIENCE_INFRA_TOOL_KERNEL`` when the kernel-only
    switch forced a downgrade, else empty (clean resolution stays quiet).
    """
    requested = str(configured_tier or "").strip().lower()
    forced = os.getenv("SCIENCE_INFRA_TOOL_KERNEL", "").strip().lower() in ("1", "true", "yes")
    resolved = effective_tier(configured_tier)
    mismatch = requested == "pro" and resolved != "pro"
    key = (tool_id, requested, resolved)
    if mismatch and key not in _logged_tier_mismatches:
        _logged_tier_mismatches.add(key)
        import logging

        logging.getLogger(__name__).warning(
            "tool tier downgraded: %s",
            {
                "tool": tool_id,
                "configured_tier": requested or "lite",
                "effective_tier": resolved,
                "override_reason": "SCIENCE_INFRA_TOOL_KERNEL" if forced else "",
            },
        )
    return resolved


class ToolAgent:
    """Encapsulated tool-agent: LLM-in-tool by default, kernel as fallback."""

    def __init__(
        self,
        agent_id: str,
        kernel: Callable[[Dict[str, Any]], Any],
        *,
        description: str = "",
        trainable: bool = True,
        profile: Optional[Dict[str, Any]] = None,
        llm_invoke: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self.id = agent_id
        self.kind = "tool"
        self.description = description
        self.trainable = trainable
        self.profile = profile or {}
        self._kernel = kernel
        self._llm_invoke = llm_invoke

    @property
    def backend(self) -> str:
        return "llm" if self._llm_invoke is not None else "kernel"

    @property
    def llm_required(self) -> bool:
        return True

    @property
    def args_schema(self) -> Dict[str, Any]:
        return dict(self.profile.get("args_schema") or {})

    def invoke(self, args: Dict[str, Any], *, prefer_llm: bool = True, tier: Optional[str] = None) -> str:
        """Run this tool-agent.

        ``lite`` uses the HTTP or local kernel. ``pro`` adds page selection and
        landing-page excerpts. ``SCIENCE_INFRA_TOOL_KERNEL=1`` forces lite.
        ``prefer_llm`` is ignored so a missing HIVE import cannot exit the process.
        """
        del prefer_llm
        resolved = log_effective_tier(self.id, tier)
        payload = args if isinstance(args, dict) else {}
        if resolved == "pro":
            from .pro_tools import run_pro

            return run_pro(self.id, payload, self._kernel)
        if self._kernel is None:
            text = str(payload.get("text") or payload.get("query") or payload.get("input") or "")
            return f"tier=lite\n{text}".rstrip()
        body = str(self._kernel(payload))
        if body.startswith("tier="):
            return body
        return f"tier=lite\n{body}"

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolAgent(id={self.id!r}, backend={self.backend!r}, trainable={self.trainable})"


def _model_name() -> str:
    import os

    return (
        os.getenv("MODEL_Name")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("SCIENCE_INFRA_MODEL")
        or "Qwen3-4B"
    )


def _epc_aw_llm_backend(name: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Lazy epc_aw / HIVE LLM-in-tool backend factory."""
    try:
        if name == "python_coder":
            def _py(args: Dict[str, Any]) -> Any:
                Python_Coder_Tool = _import_epc_aw_tool("epc_aw.tools.python_coder.tool", "Python_Coder_Tool")

                t = Python_Coder_Tool(model_string=_model_name())
                return t.execute(args.get("code") or args.get("query") or args.get("question") or "")

            return _py
        if name == "wikipedia_search":
            def _wiki(args: Dict[str, Any]) -> Any:
                Wikipedia_Search_Tool = _import_epc_aw_tool("epc_aw.tools.wikipedia_search.tool", "Wikipedia_Search_Tool")

                t = Wikipedia_Search_Tool(model_string=_model_name())
                return t.execute(args.get("query") or "")

            return _wiki
        if name == "web_search":
            def _web(args: Dict[str, Any]) -> Any:
                Web_Search_Tool = _import_epc_aw_tool("epc_aw.tools.web_search.tool", "Web_Search_Tool")

                t = Web_Search_Tool(model_string=_model_name())
                return t.execute(args.get("query") or "", args.get("url") or "")

            return _web
        if name == "think":
            def _think(args: Dict[str, Any]) -> Any:
                Base_Generator_Tool = _import_epc_aw_tool("epc_aw.tools.base_generator.tool", "Base_Generator_Tool")

                t = Base_Generator_Tool(model_string=_model_name())
                query = str(args.get("text") or args.get("query") or args.get("input") or "")
                return t.execute(query)

            return _think
    except Exception:
        return None
    return None


PALETTE_TOOL_IDS = (
    "base_generator",
    "python_coder",
    "bing_search",
    "wikipedia_search",
    "web_fetch",
)
# ``web_search`` is the user-facing name of the same url->content tool as ``web_fetch``.
_LEGACY_TOOL_ALIASES = {
    "web_search": "web_fetch",
}

TOOL_AGENT_ARGS_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "wikipedia_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    },
    "bing_search": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    },
    "web_fetch": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "url": {"type": "string"}},
        "required": ["url"],
    },
    "web_search": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "url": {"type": "string"}},
        "required": ["url"],
    },
    "python_coder": {"type": "object", "properties": {"query": {"type": "string"}, "code": {"type": "string"}}, "required": []},
    "base_generator": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "text": {"type": "string"}},
        "required": [],
    },
    "think": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
}


def _kernel(name: str) -> Optional[Callable[[Dict[str, Any]], str]]:
    """Internal kernels (fetch / exec / wiki REST). Not MAS nodes."""
    try:
        if name == "python_coder":
            from python_tool import execute_python as _ep

            def _py(args: Dict[str, Any]) -> str:
                return str(_ep(args.get("code") or args.get("query") or args.get("question") or ""))

            return _py
        if name == "wikipedia_search":
            def _wiki(args: Dict[str, Any]) -> str:
                # Late import so tests can patch tools.wikipedia.
                from .wikipedia import wikipedia_search as _wiki_fn

                return str(_wiki_fn(args.get("query") or ""))

            return _wiki
        if name == "bing_search":
            def _bing(args: Dict[str, Any]) -> str:
                from .search import bing_search as _bs

                return str(_bs(args.get("query") or ""))

            return _bing
        if name in ("web_fetch", "web_search"):
            def _typed_fetch(args: Dict[str, Any]) -> str:
                # Late import so tests can patch tools.search.fetch_by_type.
                from .search import fetch_by_type as _typed

                return str(_typed(str(args.get("url") or ""), str(args.get("query") or "")))

            return _typed_fetch
        if name == "base_generator":
            def _gen(args: Dict[str, Any]) -> str:
                return str(args.get("query") or args.get("text") or args.get("input") or "")

            return _gen
        if name == "think":
            return None
    except Exception:
        return None
    return None


_DESCRIPTIONS = {
    "wikipedia_search": (
        "Search Wikipedia. Returns up to 10 entries, each with a title, its URL, and a short "
        "truncated description. Does not open pages. Use web_fetch with one URL to read a full page."
    ),
    "bing_search": (
        "Search the web with Bing. Returns up to 10 entries, each with a title, its URL, and a short "
        "truncated description. Does not open pages. Use web_fetch with one URL to read a full page."
    ),
    "web_fetch": (
        "Input is a URL. Opens exactly that page and extracts its readable text content. "
        "This tool does not search and does not accept a search query as input."
    ),
    "web_search": (
        "Input is a URL. Opens exactly that page and extracts its readable text content. "
        "Alias of web_fetch. This tool does not search."
    ),
    "python_coder": "Write and execute Python to solve the task",
    "base_generator": (
        "Generalist answer with no web search and no page fetch. "
        "Same role as AgentFlow Base_Generator_Tool."
    ),
    "think": "Reason about the given text with no external side effects",
}


def _default_registry() -> Dict[str, ToolAgent]:
    out: Dict[str, ToolAgent] = {}
    for aid in (
        "wikipedia_search",
        "bing_search",
        "web_fetch",
        "web_search",  # alias of web_fetch: url -> content, kept callable
        "python_coder",
        "base_generator",
        "think",
    ):
        kernel = _kernel(aid)
        llm_fn = _epc_aw_llm_backend(aid)
        if kernel is None and llm_fn is None:
            continue
        out[aid] = ToolAgent(
            aid,
            kernel or (lambda _args: ""),
            description=_DESCRIPTIONS[aid],
            trainable=False,
            profile={"llm_required": True, "args_schema": TOOL_AGENT_ARGS_SCHEMAS[aid]},
            llm_invoke=llm_fn,
        )
        if kernel is None:
            out[aid]._kernel = None  # think has no echo kernel
    return out


TOOL_AGENTS: Dict[str, ToolAgent] = _default_registry()


def get_tool_agent(agent_id: str) -> Optional[ToolAgent]:
    """Look up a tool-agent by id. ``execute_python`` maps to ``python_coder``.

    User-space tools are resolved through an overlay and never written into
    ``TOOL_AGENTS``.
    """
    if agent_id == "execute_python":
        agent_id = "python_coder"
    try:
        from workflow.user_gateway.tools import get_user_tool_agent

        user = get_user_tool_agent(agent_id)
        if user is not None:
            return user  # duck-typed: invoke(args) -> str
    except Exception:
        pass
    agent_id = _LEGACY_TOOL_ALIASES.get(agent_id, agent_id)
    return TOOL_AGENTS.get(agent_id)


def validate_tool_args(agent_id: str, args: Dict[str, Any]) -> tuple:
    """Validate ``tool_invoke`` args against the declared schema. Returns (ok, reason)."""
    from workflow.protocol import validate_json_schema

    if agent_id == "execute_python":
        agent_id = "python_coder"
    schema = TOOL_AGENT_ARGS_SCHEMAS.get(agent_id)
    if schema is None:
        return True, ""
    payload = args if isinstance(args, dict) else {}
    ok, reason = validate_json_schema(payload, schema)
    if not ok:
        return False, reason
    if agent_id == "python_coder":
        if not any(str(payload.get(k) or "").strip() for k in ("query", "code", "question")):
            return False, "python_coder requires query or code"
    return True, ""


class BlankAgentAdapter:
    """Single-window LLM hop for a kind=blank agent (centralized runtime uses
    WindowLLM directly; this adapter remains for tests and graph fallback)."""

    def __init__(self, spec: Any, llm_factory: Any = None) -> None:
        self.spec = spec
        self.id = str(getattr(spec, "id", "") or "")
        self.kind = "blank"
        self._llm = None
        self._llm_factory = llm_factory
        self.profile = dict(getattr(spec, "profile", None) or {})
        self.trainable = bool(getattr(spec, "trainable", False))
        self.description = str(self.profile.get("description") or f"{self.id}: custom agent")

    @property
    def tool_name(self) -> str:
        return self.id

    @classmethod
    def from_agent(cls, spec: Any, llm_factory: Any = None) -> "BlankAgentAdapter":
        return cls(spec, llm_factory)

    def bind_llm(self, llm: Any) -> "BlankAgentAdapter":
        self._llm = llm
        return self

    def invoke(self, args: Dict[str, Any], *, prefer_llm: bool = True) -> str:
        del prefer_llm
        llm = self._llm
        if llm is None and callable(self._llm_factory):
            try:
                built = self._llm_factory(self.spec)
                llm = built if hasattr(built, "invoke") else None
            except Exception:
                llm = None
        if llm is None:
            return f"Error: blank agent '{self.id}' has no bound LLM."
        parts: List[str] = [self.system_prompt or f"You are {self.id}."]
        user_input = args.get("input") or args.get("query") or args.get("question") or args.get("text") or ""
        parts.append(f"[Input]\n{user_input}")
        prompt = "\n\n".join(p for p in parts if p)
        try:
            out = llm.invoke(prompt)
            return str(getattr(out, "content", out) or "")
        except Exception as e:  # noqa: BLE001
            return f"Error invoking blank agent: {e}"

    @property
    def system_prompt(self) -> str:
        p = dict(getattr(self.spec, "profile", None) or {})
        return str(p.get("system_prompt") or getattr(self.spec, "system_prompt", "") or "")

    def __repr__(self) -> str:  # pragma: no cover
        return f"BlankAgentAdapter(id={self.id!r}, kind=blank)"


def register_tool_agent(agent: ToolAgent) -> None:
    TOOL_AGENTS[agent.id] = agent
