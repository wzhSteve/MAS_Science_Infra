"""Tool-agent registry (schema 0.3, new_framework P1).

Wraps existing tools with an agent contract ({id, kind, invoke}) while keeping
the original @tool functions untouched. LLM-in-tool wrappers (epc_aw) can be
registered here without changing their internals.

Dual-mode backends (agent-framework A1):
- ``pure``: mas/tools pure functions (RL-safe, no external LLM calls)
- ``llm``: epc_aw LLM-in-tool implementations (test-mode enhancement; lazy)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class ToolAgent:
    """Agent-contract shell around a tool function with dual backends."""

    def __init__(
        self,
        agent_id: str,
        invoke: Callable[[Dict[str, Any]], Any],
        *,
        description: str = "",
        trainable: bool = False,
        profile: Optional[Dict[str, Any]] = None,
        llm_invoke: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> None:
        self.id = agent_id
        self.kind = "tool"
        self.description = description
        self.trainable = trainable
        self.profile = profile or {}
        self._invoke = invoke
        self._llm_invoke = llm_invoke

    @property
    def backend(self) -> str:
        return "llm" if self.llm_required and self._llm_invoke is not None else "pure"

    @property
    def llm_required(self) -> bool:
        return bool(self.profile.get("llm_required"))

    def invoke(self, args: Dict[str, Any], *, prefer_llm: bool = False) -> str:
        """Run this tool-agent.

        ``prefer_llm`` (test mode / llm.mode=api): use the epc_aw LLM-in-tool
        backend when the agent declares ``profile.llm_required`` and one is
        registered. Otherwise the pure mas/tools function runs (RL-safe).
        """
        if prefer_llm and self.llm_required and self._llm_invoke is not None:
            return str(self._llm_invoke(args))
        return str(self._invoke(args))

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolAgent(id={self.id!r}, backend={self.backend!r}, trainable={self.trainable})"


def _wrap_langchain_tool(name: str) -> Optional[ToolAgent]:
    from .langchain_tools import TOOL_MAP

    fn = TOOL_MAP.get(name)
    if fn is None:
        return None

    def _invoke(args: Dict[str, Any]) -> str:
        return str(fn.invoke(args))

    return ToolAgent(
        name,
        _invoke,
        description=str(getattr(fn, "description", "") or ""),
    )


def _epc_aw_llm_backend(name: str) -> Optional[Callable[[Dict[str, Any]], Any]]:
    """Lazy epc_aw LLM-in-tool backend factory (agent-framework A1).

    Returns a callable or None when epc_aw is unavailable (training path never
    imports this eagerly). Import is deferred to call time so merely loading
    the registry cannot pull external deps into RL workers.
    """
    try:
        if name == "execute_python":
            def _py(args: Dict[str, Any]) -> Any:
                import os

                from MAS_structagent.epc_aw.tools.python_coder.tool import Python_Coder_Tool

                t = Python_Coder_Tool(model_string=os.getenv("MODEL_Name"))
                return t.execute(args.get("code") or args.get("query") or "")

            return _py
        if name == "wikipedia_search":
            def _wiki(args: Dict[str, Any]) -> Any:
                import os

                from MAS_structagent.epc_aw.tools.wikipedia_search.tool import Wikipedia_Search_Tool

                t = Wikipedia_Search_Tool(model_string=os.getenv("MODEL_Name"))
                return t.execute(args.get("query") or "")

            return _wiki
        if name == "web_search":
            def _web(args: Dict[str, Any]) -> Any:
                import os

                from MAS_structagent.epc_aw.tools.google_search.tool import Google_Search_Tool

                t = Google_Search_Tool(model_string=os.getenv("MODEL_Name"))
                return t.execute(args.get("query") or "")

            return _web
        if name == "web_search_llm":
            def _rag(args: Dict[str, Any]) -> Any:
                import os

                from MAS_structagent.epc_aw.tools.web_search.tool import Web_Search_Tool

                t = Web_Search_Tool(model_string=os.getenv("MODEL_Name"))
                return t.execute(args.get("query") or "", args.get("url") or "")

            return _rag
    except Exception:
        return None
    return None


def _default_registry() -> Dict[str, ToolAgent]:
    out: Dict[str, ToolAgent] = {}
    try:
        from .langchain_tools import TOOLS

        for t in TOOLS:
            wrapped = _wrap_langchain_tool(t.name)
            if wrapped is not None:
                out[wrapped.id] = wrapped
    except Exception:
        pass
    return out


TOOL_AGENTS: Dict[str, ToolAgent] = _default_registry()

# epc_aw LLM-enhanced variants (separate ids so the pure trio stays untouched).
for _name in ("execute_python", "wikipedia_search", "web_search"):
    _base = TOOL_AGENTS.get(_name)
    if _base is not None:
        _llm_fn = _epc_aw_llm_backend(_name)
        if _llm_fn is not None:
            _llm_agent = ToolAgent(
                _name,
                _base._invoke,
                description=_base.description,
                profile={"llm_required": True},
                llm_invoke=_llm_fn,
            )
            TOOL_AGENTS[_name] = _llm_agent


def register_tool_agent(agent: ToolAgent) -> None:
    TOOL_AGENTS[agent.id] = agent
