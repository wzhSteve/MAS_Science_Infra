"""Agent registry: unified agent model for the MAS layer (agent-framework A1).

Two agent classes + routers (new_framework §2.1/2.2):
- packaged agents: planner / tool-agent / verifier (kind != blank)
- blank agents: custom profile (system_prompt + skills + memory policy)
- AgentRouter: runtime selection among candidates (adapter mode → tool_calls)

Layer rule: no AGL/verl/ray imports here. epc_aw LLM-in-tool backends are
lazy-loaded inside tools.tool_agents (never at import time).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .spec import AgentNodeSpec, MASSpec, RouterSpec


@dataclass
class BlankAgent:
    """kind=blank agent: user-defined profile, invoked as an LLM hop."""

    spec: AgentNodeSpec

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def system_prompt(self) -> str:
        return str(self.spec.system_prompt or self.spec.profile.get("system_prompt") or "")

    @property
    def skills(self) -> List[str]:
        p_skills = self.spec.profile.get("skills")
        if isinstance(p_skills, list) and p_skills:
            return [str(s) for s in p_skills]
        return list(self.spec.skills or [])

    @property
    def memory_policy(self) -> Dict[str, Any]:
        m = self.spec.profile.get("memory")
        return dict(m) if isinstance(m, dict) else {}


@dataclass
class ResolvedToolAgent:
    """A kind=tool node bound to a concrete backend invocation."""

    spec: AgentNodeSpec
    agent: Any  # ToolAgent from tools.tool_agents

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def backend(self) -> str:
        return getattr(self.agent, "backend", "pure")

    def invoke(self, args: Dict[str, Any], *, prefer_llm: bool = False) -> str:
        return self.agent.invoke(args, prefer_llm=prefer_llm)


class AgentRegistry:
    """Runtime registry over a MASSpec: agents + routers resolved once."""

    def __init__(self) -> None:
        self.agents: Dict[str, AgentNodeSpec] = {}
        self.routers: Dict[str, RouterSpec] = {}
        self.tool_agents: Dict[str, ResolvedToolAgent] = {}
        self.blank_agents: Dict[str, BlankAgent] = {}

    @classmethod
    def from_spec(cls, spec: MASSpec, *, allow_llm_backends: bool = False) -> "AgentRegistry":
        """Build from spec. ``allow_llm_backends`` gates epc_aw LLM-in-tool
        usage (test mode / llm.kind=api); training/collect always pure."""
        reg = cls()
        from .compiler import _agent_map

        reg.agents = _agent_map(spec)
        # sugar fallback: specs built directly (not via load_spec) may carry
        # only top-level tools — expand them as implicit kind=tool nodes.
        if not any(a.kind == "tool" for a in reg.agents.values()):
            for t in getattr(spec, "tools", None) or []:
                if t not in reg.agents:
                    node = AgentNodeSpec(id=t, kind="tool", trainable=False)
                    reg.agents[t] = node
        reg.routers = {r.id: r for r in (spec.routers or [])}
        from tools.tool_agents import TOOL_AGENTS

        for aid, node in reg.agents.items():
            if node.kind == "tool":
                ta = TOOL_AGENTS.get(aid)
                if ta is None:
                    continue
                # llm backend only bound when allowed AND the profile asks for it
                if not allow_llm_backends and ta.llm_required:
                    # degrade to pure by re-wrapping the pure invoke
                    reg.tool_agents[aid] = ResolvedToolAgent(
                        spec=node,
                        agent=_PureOnlyView(ta),
                    )
                else:
                    reg.tool_agents[aid] = ResolvedToolAgent(spec=node, agent=ta)
            elif node.kind == "blank":
                reg.blank_agents[aid] = BlankAgent(spec=node)
        return reg

    def tool_ids(self) -> List[str]:
        return sorted(self.tool_agents)

    def invocable_ids(self) -> List[str]:
        """Ids the TirAgent tool-call surface may bind (adapter mode)."""
        return sorted(self.tool_agents)

    def router_for(self, agent_id: str) -> Optional[RouterSpec]:
        for r in self.routers.values():
            if agent_id in (r.candidates or []):
                return r
        return None

    def describe(self) -> Dict[str, Any]:
        return {
            "n_agents": len(self.agents),
            "n_routers": len(self.routers),
            "tool_agents": {
                aid: {"backend": ta.backend, "llm_required": bool(ta.spec.profile.get("llm_required"))}
                for aid, ta in self.tool_agents.items()
            },
            "blank_agents": sorted(self.blank_agents),
        }


class _PureOnlyView:
    """Force the pure backend even when the underlying ToolAgent has an
    epc_aw LLM variant (training / collect path safety)."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    @property
    def id(self) -> str:
        return getattr(self._agent, "id", "")

    @property
    def backend(self) -> str:
        return "pure"

    @property
    def profile(self) -> Dict[str, Any]:
        return {k: v for k, v in dict(getattr(self._agent, "profile", {})).items() if k != "llm_required"}

    def invoke(self, args: Dict[str, Any], *, prefer_llm: bool = False) -> str:
        del prefer_llm
        return self._agent.invoke(args, prefer_llm=False)
