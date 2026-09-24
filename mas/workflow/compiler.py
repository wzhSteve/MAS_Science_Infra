"""Topology Compiler: MASSpec graph → executable handoff plan.

Must not import agentlightning. Compiles route / message / feedback / tool_call
into a walk the GraphRunner can execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .spec import AgentNodeSpec, MASSpec, RouterSpec

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
    # agent-framework A2: routers available at runtime (id -> RouterSpec)
    routers: Dict[str, "RouterSpec"] = field(default_factory=dict)  # type: ignore[name-defined]

    @property
    def multi_agent(self) -> bool:
        ids = [i for i in self.agents if i not in KNOWN_TOOLS and getattr(self.agents[i], "kind", None) != "tool"]
        extra = [i for i in ids if i not in {"hub", "verifier"}]
        return bool(extra)


def _agent_map(spec: MASSpec) -> Dict[str, AgentNodeSpec]:
    out: Dict[str, AgentNodeSpec] = {}
    for a in spec.agents:
        out[a.id] = a
    if "hub" not in out:
        out["hub"] = AgentNodeSpec(
            id="hub",
            kind="hub",
            role=spec.hub.role or "orchestrator",
            skills=list(spec.hub.skills or ["react_loop"]),
            tools=list(spec.tools),
            system_prompt=spec.hub.system_prompt or "",
            trainable=True,
        )
    else:
        if not out["hub"].kind or out["hub"].kind == "blank":
            out["hub"].kind = "hub"
        if not out["hub"].system_prompt and spec.hub.system_prompt:
            out["hub"].system_prompt = spec.hub.system_prompt
    if spec.hub.verify and "verifier" not in out:
        out["verifier"] = AgentNodeSpec(
            id="verifier",
            kind="verifier",
            role="verifier",
            skills=[spec.hub.verify],
            trainable=False,
        )
    return out


def tool_agent_ids(spec: MASSpec) -> set:
    """Tool-agent registry (schema 0.3): kind=tool agents ∪ legacy top-level tools."""
    return {a.id for a in spec.agents if a.kind == "tool"} | set(spec.tools or [])


def _router_upstream(spec: MASSpec, router_id: str) -> Optional[str]:
    """Agent feeding this router (edge source targeting router id), if any."""
    for e in spec.edges or []:
        if str(getattr(e, "target", "") or "") == router_id:
            return str(getattr(e, "source", "") or "")
    return None


def compile_spec(spec: MASSpec) -> CompiledWorkflow:
    issues: List[str] = []
    agents = _agent_map(spec)
    tool_ids = tool_agent_ids(spec)
    agent_ids = set(agents)
    # W1: blank agents routable via router candidates (blank:<id> tool-call shell)
    blank_ids = {a.id for a in agents.values() if getattr(a, "kind", None) == "blank"}

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

    # Router ids known up-front so edge validation can skip routing sugar.
    router_ids: set = set(str(r.id) for r in (getattr(spec, "routers", None) or []))

    for e in spec.edges:
        src, dst, kind = e.source, e.target, e.kind
        # agent-framework A2: edges touching a router node are routing sugar;
        # the router's upstream/first candidate mapping is handled via
        # _router_upstream + tools_for — no agent-agent validation here.
        if src in router_ids or dst in router_ids:
            continue
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

        if kind == "sample_barrier":
            # Declaration only: allows RL daemon to fork at this agent/tool boundary.
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

    # Router validation (schema 0.3): candidates must exist among agents.
    # W1: blank candidates may be written either as the bare agent id or with
    # the ``blank:`` tool-call prefix (UI writes blank:<id>) — both accepted.
    routers_out: Dict[str, RouterSpec] = {}
    for r in getattr(spec, "routers", None) or []:
        router_ids.add(r.id)
        routers_out[r.id] = r
        for c in r.candidates or []:
            bare = c[len("blank:"):] if c.startswith("blank:") else c
            if bare not in agent_ids and bare not in tool_ids:
                issues.append(f"router {r.id!r} candidate {c!r} is not an agent node")
        if r.strategy == "score" and r.scorer and r.scorer not in agent_ids:
            issues.append(f"router {r.id!r} scorer {r.scorer!r} is not an agent node")
        # agent-framework A2 (adapter mode): a router's tool-agent candidates
        # become routable via the upstream agent's tool_call surface — same
        # semantics as tool_call edges, pure sugar over the existing protocol.
        # W1: kind=blank candidates join with a ``blank:`` id prefix — same
        # tool-call surface, dispatched to BlankAgentAdapter (one llm hop).
        upstream = _router_upstream(spec, r.id)
        if upstream and upstream in agents:
            cur = tools_for.setdefault(upstream, [])
            for c in r.candidates or []:
                if c in tool_ids:
                    if c not in cur:
                        cur.append(c)
                else:
                    bare = c[len("blank:"):] if c.startswith("blank:") else c
                    if bare in blank_ids:
                        pref = f"blank:{bare}"
                        if pref not in cur:
                            cur.append(pref)

    extra = [i for i in agents if i not in {"hub", "verifier"} and i not in tool_ids and i not in router_ids]
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
            routers=routers_out,
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
            routers=routers_out,
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
            routers=routers_out,
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
            routers=routers_out,
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
            routers=routers_out,
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
        routers=routers_out,
    )


def next_agent(compiled: CompiledWorkflow, current: str) -> Optional[str]:
    if current in compiled.route_out:
        return compiled.route_out[current]
    if current in compiled.message_out:
        return compiled.message_out[current]
    return None


def trainable_agents(spec: MASSpec) -> List[str]:
    compiled = compile_spec(spec)
    names = [
        aid
        for aid, a in compiled.agents.items()
        if a.trainable and aid not in KNOWN_TOOLS
    ]
    return names


def prompt_for(spec: MASSpec, agent_id: str) -> str:
    compiled = compile_spec(spec)
    node = compiled.agents.get(agent_id)
    if node and str(node.system_prompt or "").strip():
        return str(node.system_prompt).strip()
    if agent_id == "hub" and spec.hub.system_prompt:
        return spec.hub.system_prompt.strip()
    return ""
