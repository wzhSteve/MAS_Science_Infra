"""Topology Compiler: MASSpec graph → executable handoff plan.

Must not import agentlightning. Compiles route / message / feedback / tool_call
into a walk the GraphRunner can execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .spec import AgentNodeSpec, MASSpec

KNOWN_TOOLS = {"web_search", "wikipedia_search", "execute_python"}
KNOWN_ROLES = {
    "hub",
    "orchestrator",
    "planner",
    "executor",
    "verifier",
    "critic",
    "agent",
}
FEEDBACK_SOURCES = {"verifier", "critic"}
FEEDBACK_TARGETS = {"hub", "planner", "orchestrator"}


@dataclass
class CompiledWorkflow:
    ok: bool
    reason: str
    entry_agent: str
    agents: Dict[str, AgentNodeSpec]
    tools_for: Dict[str, List[str]]
    route_out: Dict[str, str]
    message_out: Dict[str, str]
    feedback_to: Dict[str, str]
    issues: List[str] = field(default_factory=list)
    hub_subset: bool = False

    @property
    def multi_agent(self) -> bool:
        ids = [i for i in self.agents if i not in KNOWN_TOOLS]
        extra = [i for i in ids if i not in {"hub", "verifier"}]
        return bool(extra)


def _agent_map(spec: MASSpec) -> Dict[str, AgentNodeSpec]:
    out: Dict[str, AgentNodeSpec] = {}
    for a in spec.agents:
        out[a.id] = a
    if "hub" not in out:
        out["hub"] = AgentNodeSpec(
            id="hub",
            role=spec.hub.role or "orchestrator",
            skills=list(spec.hub.skills or ["react_loop"]),
            tools=list(spec.tools),
            system_prompt=spec.hub.system_prompt or "",
            trainable=True,
        )
    elif not out["hub"].system_prompt and spec.hub.system_prompt:
        out["hub"].system_prompt = spec.hub.system_prompt
    if spec.hub.verify and "verifier" not in out:
        out["verifier"] = AgentNodeSpec(
            id="verifier",
            role="verifier",
            skills=[spec.hub.verify],
            trainable=False,
        )
    return out


def compile_spec(spec: MASSpec) -> CompiledWorkflow:
    issues: List[str] = []
    agents = _agent_map(spec)
    tool_ids = set(KNOWN_TOOLS)
    agent_ids = set(agents)

    entry = str(spec.entry_agent or "hub").strip() or "hub"
    if entry not in agents:
        if spec.topology in ("hub_react", "single", "") and entry == "hub":
            pass
        else:
            issues.append(f"entry_agent {entry!r} is not an agent node")

    inbound_route: Dict[str, int] = {}
    route_out: Dict[str, str] = {}
    message_out: Dict[str, str] = {}
    feedback_to: Dict[str, str] = {}
    tools_for: Dict[str, List[str]] = {aid: list(a.tools or []) for aid, a in agents.items()}

    for e in spec.edges:
        src, dst, kind = e.source, e.target, e.kind
        src_is_tool = src in tool_ids
        dst_is_tool = dst in tool_ids
        src_is_agent = src in agent_ids
        dst_is_agent = dst in agent_ids

        if kind == "tool_call":
            if not src_is_agent:
                issues.append(f"tool_call source {src!r} must be an agent")
            if not dst_is_tool:
                issues.append(f"tool_call target {dst!r} must be a tool")
            if src_is_agent and dst_is_tool:
                if dst not in tools_for.setdefault(src, []):
                    tools_for[src].append(dst)
            continue

        if src_is_tool or dst_is_tool:
            issues.append(f"{kind} cannot involve tool node ({src}->{dst})")
            continue
        if not src_is_agent:
            issues.append(f"edge source {src!r} is not an agent")
        if not dst_is_agent:
            issues.append(f"edge target {dst!r} is not an agent")
            continue

        src_role = (agents.get(src).role if src in agents else "") or src
        dst_role = (agents.get(dst).role if dst in agents else "") or dst

        if kind == "route":
            inbound_route[dst] = inbound_route.get(dst, 0) + 1
            if inbound_route[dst] > 1:
                issues.append(f"agent {dst!r} has more than one inbound route")
            route_out[src] = dst
        elif kind == "message":
            message_out[src] = dst
        elif kind == "feedback":
            if src_role not in FEEDBACK_SOURCES and src not in FEEDBACK_SOURCES:
                issues.append(f"feedback source {src!r} must be verifier/critic")
            if dst_role not in FEEDBACK_TARGETS and dst not in FEEDBACK_TARGETS:
                issues.append(f"feedback target {dst!r} must be hub/planner")
            feedback_to[src] = dst
        else:
            issues.append(f"unknown edge kind {kind!r}")

    # Hub tools fallback
    if not tools_for.get("hub"):
        tools_for["hub"] = list(spec.tools)

    extra = [i for i in agents if i not in {"hub", "verifier"}]
    hub_subset = not extra
    if spec.topology in ("hub_react", "single", ""):
        ok = not issues
        reason = "hub_react" if ok else "; ".join(issues)
        return CompiledWorkflow(
            ok=ok,
            reason=reason,
            entry_agent="hub",
            agents=agents,
            tools_for=tools_for,
            route_out=route_out,
            message_out=message_out,
            feedback_to=feedback_to,
            issues=issues,
            hub_subset=True,
        )

    if spec.topology != "graph":
        return CompiledWorkflow(
            ok=False,
            reason=f"unknown topology {spec.topology}",
            entry_agent=entry,
            agents=agents,
            tools_for=tools_for,
            route_out=route_out,
            message_out=message_out,
            feedback_to=feedback_to,
            issues=issues,
        )

    if issues:
        return CompiledWorkflow(
            ok=False,
            reason="; ".join(issues),
            entry_agent=entry,
            agents=agents,
            tools_for=tools_for,
            route_out=route_out,
            message_out=message_out,
            feedback_to=feedback_to,
            issues=issues,
            hub_subset=hub_subset,
        )

    if entry not in agents:
        return CompiledWorkflow(
            ok=False,
            reason=f"entry_agent {entry!r} missing",
            entry_agent=entry,
            agents=agents,
            tools_for=tools_for,
            route_out=route_out,
            message_out=message_out,
            feedback_to=feedback_to,
            issues=issues,
            hub_subset=hub_subset,
        )

    if hub_subset:
        return CompiledWorkflow(
            ok=True,
            reason="graph_hub_subset",
            entry_agent=entry if entry in {"hub", "verifier"} else "hub",
            agents=agents,
            tools_for=tools_for,
            route_out=route_out,
            message_out=message_out,
            feedback_to=feedback_to,
            issues=issues,
            hub_subset=True,
        )

    return CompiledWorkflow(
        ok=True,
        reason="graph_compiled",
        entry_agent=entry,
        agents=agents,
        tools_for=tools_for,
        route_out=route_out,
        message_out=message_out,
        feedback_to=feedback_to,
        issues=issues,
        hub_subset=False,
    )


def next_agent(compiled: CompiledWorkflow, current: str) -> Optional[str]:
    if current in compiled.route_out:
        return compiled.route_out[current]
    if current in compiled.message_out:
        return compiled.message_out[current]
    return None


def trainable_agents(spec: MASSpec) -> List[str]:
    compiled = compile_spec(spec)
    names = [aid for aid, a in compiled.agents.items() if a.trainable and aid not in KNOWN_TOOLS]
    if not names:
        return [compiled.entry_agent]
    return names


def prompt_for(spec: MASSpec, agent_id: str) -> str:
    compiled = compile_spec(spec)
    node = compiled.agents.get(agent_id)
    if node and str(node.system_prompt or "").strip():
        return str(node.system_prompt).strip()
    if agent_id == "hub" and spec.hub.system_prompt:
        return spec.hub.system_prompt.strip()
    return ""
