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


class BlankAgentAdapter(ToolAgent):
    """Wrap a kind=blank agent as a tool_call shell (agent-framework W1).

    Adapter-mode routing: router candidates that are blank agents get invoked
    through the same tool-call surface as tool-agents. One ``llm.invoke`` hop
    with the agent's system_prompt; output is a plain str (the caller wraps it
    in a ToolMessage) — structurally identical to A2 router tool-agents.
    """

    def __init__(self, spec: Any, llm_factory: Any = None) -> None:
        self.spec = spec
        self._llm = None
        self._llm_factory = llm_factory
        profile = dict(getattr(spec, "profile", None) or {})
        skills = profile.get("skills") or [str(s) for s in (getattr(spec, "skills", None) or [])]
        desc = str(profile.get("description") or "")
        if not desc:
            desc = f"{spec.id}: {', '.join(str(s) for s in skills)}" if skills else f"{spec.id}: blank agent"
        super().__init__(
            f"blank:{spec.id}",
            self._invoke,
            description=desc,
            trainable=False,
            profile=profile,
        )
        self.kind = "blank"  # agent contract: this shell wraps a blank agent

    @classmethod
    def from_agent(cls, spec: Any, llm_factory: Any = None) -> "BlankAgentAdapter":
        """Build from an AgentNodeSpec (kind=blank) + optional LLM factory."""
        return cls(spec, llm_factory)

    def bind_llm(self, llm: Any) -> "BlankAgentAdapter":
        """Bind a concrete LLM (any object with .invoke(prompt) -> output)."""
        self._llm = llm
        return self

    def _invoke(self, args: Dict[str, Any]) -> str:
        llm = self._llm
        if llm is None and callable(self._llm_factory):
            # factory takes precedence over None; may itself be a builder llm
            try:
                built = self._llm_factory(self.spec)
                llm = built if hasattr(built, "invoke") else None
            except Exception:
                llm = None
        if llm is None:
            return (
                f"Error: blank agent '{self.spec.id}' has no bound LLM. "
                "Ensure the router candidates include a valid agent or the runtime "
                "binds an LLM before invocation."
            )
        # build the one-hop prompt: system_prompt + optional history + input
        parts: List[str] = [self.system_prompt or f"You are {self.spec.id}."]
        hist = args.get("history")
        if hist:
            parts.append(f"[History]\n{hist}")
        user_input = args.get("input") or args.get("query") or args.get("question") or ""
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

    @property
    def agent_kind(self) -> str:
        return "blank"

    @property
    def agent_id(self) -> str:
        return str(self.spec.id)

    def __repr__(self) -> str:  # pragma: no cover
        return f"BlankAgentAdapter(id={self.id!r}, kind=blank)"


def register_tool_agent(agent: ToolAgent) -> None:
    TOOL_AGENTS[agent.id] = agent
