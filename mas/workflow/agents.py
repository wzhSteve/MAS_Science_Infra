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
class UserBoundAgent:
    """kind=blank/tool node whose window lives in user_space."""

    spec: AgentNodeSpec
    project_id: str

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def backend(self) -> str:
        return "user_space"


def _user_project_id(node: AgentNodeSpec) -> str:
    profile = dict(node.profile or {})
    return str(profile.get("user_project") or "")


def _is_user_backend(node: AgentNodeSpec) -> bool:
    return str((node.profile or {}).get("backend") or "") == "user_space"


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
        return getattr(self.agent, "backend", "llm")

    def invoke(self, args: Dict[str, Any], **_kwargs) -> str:
        return self.agent.invoke(args)


class AgentRegistry:
    """Runtime registry over a MASSpec: agents + routers resolved once."""

    def __init__(self) -> None:
        self.agents: Dict[str, AgentNodeSpec] = {}
        self.routers: Dict[str, RouterSpec] = {}
        self.tool_agents: Dict[str, ResolvedToolAgent] = {}
        self.blank_agents: Dict[str, BlankAgent] = {}
        self.user_agents: Dict[str, UserBoundAgent] = {}

    @classmethod
    def from_spec(cls, spec: MASSpec, *, allow_llm_backends: bool = True) -> "AgentRegistry":
        """Build from spec. Tool-agents are HIVE-style LLM-in-tool wrappers."""
        del allow_llm_backends
        reg = cls()
        from .compiler import _agent_map

        reg.agents = _agent_map(spec)
        if not any(a.kind == "tool" for a in reg.agents.values()):
            for t in getattr(spec, "tools", None) or []:
                if t not in reg.agents:
                    node = AgentNodeSpec(id=t, kind="tool", trainable=False)
                    reg.agents[t] = node
        reg.routers = {r.id: r for r in (spec.routers or [])}
        from tools.tool_agents import TOOL_AGENTS, get_tool_agent

        for aid, node in reg.agents.items():
            if _is_user_backend(node):
                pid = _user_project_id(node)
                if pid:
                    try:
                        from workflow.user_gateway.tools import bind_project_tools

                        bind_project_tools(pid)
                    except Exception:
                        pass
                    reg.user_agents[aid] = UserBoundAgent(spec=node, project_id=pid)
            if node.kind == "tool":
                ta = get_tool_agent(aid) or TOOL_AGENTS.get(aid)
                if ta is None and _is_user_backend(node):
                    try:
                        from workflow.user_gateway.tools import get_user_tool_agent

                        ta = get_user_tool_agent(aid)
                    except Exception:
                        ta = None
                if ta is None:
                    continue
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
            "user_agents": sorted(self.user_agents),
        }
