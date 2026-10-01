"""Five HIVE-style tool-agents for the centralized MAS.

Each id is an encapsulated agent: a tool kernel plus an LLM that reads the
kernel output (or writes code, or reasons) and returns a ``tool_result``.
Pure functions in ``mas/tools`` stay as kernels used *inside* the agent;
they are not first-class MAS nodes.

Ids: ``wikipedia_search``, ``google_search``, ``web_search``, ``python_coder``,
``think``. ``execute_python`` maps to ``python_coder``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional


def effective_tier(tier: Optional[str]) -> str:
    """``pro`` only when requested and the kernel-only test switch is off."""
    forced = os.getenv("SCIENCE_INFRA_TOOL_KERNEL", "").strip().lower() in ("1", "true", "yes")
    if forced or str(tier or "").strip().lower() != "pro":
        return "lite"
    return "pro"


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
        resolved = effective_tier(tier)
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
                from MAS_structagent.epc_aw.tools.python_coder.tool import Python_Coder_Tool

                t = Python_Coder_Tool(model_string=_model_name())
                return t.execute(args.get("code") or args.get("query") or args.get("question") or "")

            return _py
        if name == "wikipedia_search":
            def _wiki(args: Dict[str, Any]) -> Any:
                from MAS_structagent.epc_aw.tools.wikipedia_search.tool import Wikipedia_Search_Tool

                t = Wikipedia_Search_Tool(model_string=_model_name())
                return t.execute(args.get("query") or "")

            return _wiki
        if name == "google_search":
            def _gog(args: Dict[str, Any]) -> Any:
                from MAS_structagent.epc_aw.tools.google_search.tool import Google_Search_Tool

                t = Google_Search_Tool(model_string=_model_name())
                return t.execute(args.get("query") or "")

            return _gog
        if name == "web_search":
            def _web(args: Dict[str, Any]) -> Any:
                from MAS_structagent.epc_aw.tools.web_search.tool import Web_Search_Tool

                t = Web_Search_Tool(model_string=_model_name())
                return t.execute(args.get("query") or "", args.get("url") or "")

            return _web
        if name == "think":
            def _think(args: Dict[str, Any]) -> Any:
                from MAS_structagent.epc_aw.tools.base_generator.tool import Base_Generator_Tool

                t = Base_Generator_Tool(model_string=_model_name())
                query = str(args.get("text") or args.get("query") or args.get("input") or "")
                return t.execute(query)

            return _think
    except Exception:
        return None
    return None


TOOL_AGENT_ARGS_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "wikipedia_search": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "google_search": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    "web_search": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "url": {"type": "string"}},
        "required": ["query", "url"],
    },
    "python_coder": {"type": "object", "properties": {"query": {"type": "string"}, "code": {"type": "string"}}, "required": []},
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
            from .wikipedia import wikipedia_search as _wiki_fn

            def _wiki(args: Dict[str, Any]) -> str:
                return str(_wiki_fn(args.get("query") or ""))

            return _wiki
        if name == "google_search":
            from .search import web_search as _ws

            def _gog(args: Dict[str, Any]) -> str:
                return str(_ws(args.get("query") or ""))

            return _gog
        if name == "web_search":
            from .search import fetch_page as _fp

            def _web(args: Dict[str, Any]) -> str:
                url = str(args.get("url") or "")
                query = str(args.get("query") or "")
                page = str(_fp(url))
                if query:
                    return f"query: {query}\n{page}"
                return page

            return _web
        if name == "think":
            return None
    except Exception:
        return None
    return None


_DESCRIPTIONS = {
    "wikipedia_search": "Read Wikipedia and extract content relevant to the query",
    "google_search": "Search the web and summarize results",
    "web_search": "Open a URL and answer the query from the page",
    "python_coder": "Write and execute Python to solve the task",
    "think": "Reason about the given text with no external side effects",
}


def _default_registry() -> Dict[str, ToolAgent]:
    out: Dict[str, ToolAgent] = {}
    for aid in ("wikipedia_search", "google_search", "web_search", "python_coder", "think"):
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
