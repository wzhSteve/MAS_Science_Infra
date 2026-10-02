"""Topology Compiler: MASSpec graph → executable handoff plan.

Must not import agentlightning. Compiles route / message / feedback / tool_call
into a walk the GraphRunner can execute. Routers are first-class graph nodes
(centralized topology): edges pointing at a router populate ``route_out``, and
edges leaving a router populate ``message_out`` / ``route_out``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .spec import AgentNodeSpec, MASSpec, RouterSpec

# Legacy pure-tool ids that are no longer first-class agents (mapped to
# python_coder in spec._normalize_schema03). Kept only to recognise tool nodes
# for the multi_agent / hub_subset heuristics; not a trainable blacklist.
KNOWN_TOOL_IDS = {"web_search", "wikipedia_search", "execute_python", "google_search", "python_coder", "think"}
KNOWN_ROLES = {
    "planner",
    "verifier",
    "critic",
    "agent",
}
FEEDBACK_SOURCES = {"verifier", "critic"}
FEEDBACK_TARGETS = {"planner"}


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
    routers: Dict[str, "RouterSpec"] = field(default_factory=dict)
    # multi-router fan-out: planner id -> list of router ids it feeds
    fan_out: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def multi_agent(self) -> bool:
        ids = [i for i in self.agents if getattr(self.agents[i], "kind", None) != "tool"]
        extra = [i for i in ids if i not in {"planner", "verifier"} and i not in self.routers]
        return bool(extra)


def _agent_map(spec: MASSpec) -> Dict[str, AgentNodeSpec]:
    out: Dict[str, AgentNodeSpec] = {}
    for a in spec.agents:
        out[a.id] = a
    # Synthesize a planner node from HubSpec when none exists. The orchestrator
    # id is "planner" (hub was removed in the centralized redesign).
    if "planner" not in out:
        out["planner"] = AgentNodeSpec(
            id="planner",
            kind="planner",
            role=spec.hub.role or "planner",
            skills=list(spec.hub.skills or []),
            tools=list(spec.tools),
            system_prompt=spec.hub.system_prompt or "",
            trainable=True,
        )
    else:
        if not out["planner"].system_prompt and spec.hub.system_prompt:
            out["planner"].system_prompt = spec.hub.system_prompt
        if not out["planner"].tools:
            out["planner"].tools = list(spec.tools)
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
    """Tool-agent registry (schema 0.3): kind=tool agents ∪ top-level tools."""
    return {a.id for a in spec.agents if a.kind == "tool"} | set(spec.tools or [])


def _is_pool_id(node_id: str) -> bool:
    """Canvas pool ids are transparent: router → pool → next becomes router → next."""
    return str(node_id).startswith("pool_")


def _is_tool_target(node_id: str, tool_ids: set) -> bool:
    """True if ``node_id`` is a tool-agent, including the execute_python alias."""
    if node_id in tool_ids:
        return True
    if node_id == "execute_python" and "python_coder" in tool_ids:
        return True
    if node_id == "python_coder" and "execute_python" in tool_ids:
        return True
    return False


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
    blank_ids = {a.id for a in agents.values() if getattr(a, "kind", None) == "blank"}

    entry = str(spec.entry_agent or "planner").strip() or "planner"
    # hub_react sugar: entry "hub" maps to "planner"
    if entry == "hub" and "planner" in agents and "hub" not in agents:
        entry = "planner"
    if entry not in agents:
        if spec.topology in ("centralized", "hub_react", "single", "") and entry in ("planner", "hub"):
            entry = "planner" if "planner" in agents else entry
            if entry not in agents:
                pass
        else:
            issues.append(f"entry_agent {entry!r} is not an agent node")

    inbound_route: Dict[str, int] = {}
    route_out: Dict[str, str] = {}
    message_out: Dict[str, str] = {}
    feedback_to: Dict[str, str] = {}
    tools_for: Dict[str, List[str]] = {aid: list(a.tools or []) for aid, a in agents.items()}
    fan_out: Dict[str, List[str]] = {}

    # Router ids known up-front so edge validation can treat them as nodes.
    router_ids: set = set(str(r.id) for r in (getattr(spec, "routers", None) or []))

    for e in spec.edges:
        src, dst, kind = e.source, e.target, e.kind
        if _is_pool_id(src) or _is_pool_id(dst):
            continue
        # Router is a first-class graph node (centralized topology):
        # - edge X -> router (route/message): route_out[X] = router
        # - edge router -> Y (message/route): message_out/router_out of router
        if dst in router_ids:
            if src in agent_ids and kind in ("route", "message"):
                fan_out.setdefault(src, [])
                if dst not in fan_out[src]:
                    fan_out[src].append(dst)
                if src not in route_out:
                    route_out[src] = dst
            continue
        if src in router_ids:
            if kind == "message":
                message_out[src] = dst
            elif kind == "route":
                route_out[src] = dst
            else:
                issues.append(f"router {src!r} edge kind {kind!r} not allowed")
            continue
        src_is_tool = src in tool_ids or (src == "execute_python" and "python_coder" in tool_ids)
        dst_is_tool = _is_tool_target(dst, tool_ids)
        src_is_agent = src in agent_ids
        dst_is_agent = dst in agent_ids
        src_is_blank = src in blank_ids
        dst_is_blank = dst in blank_ids

        if kind == "tool_call":
            # Candidate declaration only (not walked as a graph hop).
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

        # Post-tool pipeline: tool/blank --message--> blank/verifier (not a tool node).
        if kind == "message" and (src_is_tool or src_is_blank) and (
            dst_is_blank or (dst_is_agent and not dst_is_tool)
        ):
            message_out[src] = dst
            continue

        if src_is_tool or dst_is_tool:
            issues.append(f"{kind} cannot involve tool node ({src}->{dst})")
            continue
        if not src_is_agent:
            issues.append(f"edge source {src!r} is not an agent")
            if not dst_is_agent:
                continue
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
                issues.append(f"feedback target {dst!r} must be planner")
            feedback_to[src] = dst
        else:
            issues.append(f"unknown edge kind {kind!r}")

    pool_of_router: Dict[str, str] = {}
    pool_next: Dict[str, str] = {}
    pool_members: Dict[str, List[str]] = {}
    for e in spec.edges:
        if e.source in router_ids and _is_pool_id(e.target):
            pool_of_router[e.source] = e.target
            raw_members = e.meta.get("members") if isinstance(e.meta, dict) else None
            if isinstance(raw_members, list):
                pool_members[e.source] = [str(item) for item in raw_members]
        if _is_pool_id(e.source) and e.kind in ("message", "route") and not _is_pool_id(e.target):
            pool_next[e.source] = e.target
    for router_id, pool_id in pool_of_router.items():
        nxt = pool_next.get(pool_id)
        if nxt:
            message_out[router_id] = nxt
        elif pool_id not in pool_next:
            issues.append(f"router {router_id!r} pool {pool_id!r} has no downstream agent")
        declared = pool_members.get(router_id)
        if declared is not None:
            current = list(next((r.candidates for r in (spec.routers or []) if r.id == router_id), []) or [])
            if current != declared:
                issues.append(
                    f"router {router_id!r} candidates {current!r} must equal pool members {declared!r}"
                )

    # Planner tools fallback
    if not tools_for.get("planner"):
        tools_for["planner"] = list(spec.tools)

    # Router validation: candidates must exist among agents (tool-agents or blank).
    # Bare ids only. A leftover ``blank:`` prefix from old UI YAML is stripped.
    routers_out: Dict[str, RouterSpec] = {}
    for r in getattr(spec, "routers", None) or []:
        router_ids.add(r.id)
        cleaned = []
        for c in r.candidates or []:
            bare = c[len("blank:"):] if str(c).startswith("blank:") else str(c)
            if bare not in agent_ids and bare not in tool_ids:
                issues.append(f"router {r.id!r} candidate {c!r} is not an agent node")
            else:
                cleaned.append(bare)
        if cleaned != list(r.candidates or []):
            r.candidates = cleaned
        routers_out[r.id] = r
        if r.strategy == "score" and r.scorer and r.scorer not in agent_ids:
            issues.append(f"router {r.id!r} scorer {r.scorer!r} is not an agent node")
        upstream = _router_upstream(spec, r.id)
        if upstream and upstream in agents:
            cur = tools_for.setdefault(upstream, [])
            for c in r.candidates or []:
                if c not in cur and (c in tool_ids or c in blank_ids):
                    cur.append(c)
            fan_out.setdefault(upstream, [])
            if r.id not in fan_out[upstream]:
                fan_out[upstream].append(r.id)

    # Implicit router: planner's tool-agents become the agent set when the
    # planner has no explicit route and no direct message into a user window.
    # A user wrap (EPC-AW executor) already connected by a message edge is
    # dispatched directly and must not grow a synthetic router.
    if not routers_out and "planner" not in route_out:
        downstream_id = message_out.get("planner")
        downstream = agents.get(downstream_id) if downstream_id else None
        downstream_profile = dict(getattr(downstream, "profile", None) or {}) if downstream else {}
        direct_user_window = downstream is not None and (
            getattr(downstream, "kind", None) == "blank"
            or str(downstream_profile.get("backend") or "") == "user_space"
        )
        planner_set = [t for t in tools_for.get("planner", []) if t in tool_ids or t in blank_ids]
        if not planner_set:
            planner_set = [t for t in tool_ids if t != "planner"]
        if planner_set and not direct_user_window:
            impl_id = "_implicit_route"
            routers_out[impl_id] = RouterSpec(
                id=impl_id,
                candidates=planner_set,
                strategy="from_plan",
                output_contract="json",
            )
            route_out["planner"] = impl_id
            fan_out.setdefault("planner", []).append(impl_id)

    extra = [i for i in agents if i not in {"planner", "verifier"} and i not in tool_ids and i not in router_ids]
    hub_subset = not extra
    # hub_react / single / centralized always run as centralized
    if spec.topology in ("centralized", "hub_react", "single", ""):
        ok = not issues
        reason = "centralized" if ok else "; ".join(issues)
        return CompiledWorkflow(
            ok=ok,
            reason=reason,
            entry_agent="planner" if entry in ("hub", "planner") else entry,
            agents=agents,
            tools_for=tools_for,
            route_out=route_out,
            message_out=message_out,
            feedback_to=feedback_to,
            issues=issues,
            hub_subset=True,
            routers=routers_out,
            fan_out=fan_out,
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
            fan_out=fan_out,
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
            fan_out=fan_out,
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
            fan_out=fan_out,
        )

    if hub_subset:
        return CompiledWorkflow(
            ok=True,
            reason="graph_hub_subset",
            entry_agent=entry if entry in {"planner", "verifier"} else "planner",
            agents=agents,
            tools_for=tools_for,
            route_out=route_out,
            message_out=message_out,
            feedback_to=feedback_to,
            issues=issues,
            hub_subset=True,
            routers=routers_out,
            fan_out=fan_out,
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
        fan_out=fan_out,
    )


def next_agent(compiled: CompiledWorkflow, current: str) -> Optional[str]:
    if current in compiled.route_out:
        return compiled.route_out[current]
    if current in compiled.message_out:
        return compiled.message_out[current]
    return None


def trainable_agents(spec: MASSpec) -> List[str]:
    """Trainable agent ids. Tool-agents are never trainable."""
    compiled = compile_spec(spec)
    return [
        aid for aid, a in compiled.agents.items()
        if a.trainable and getattr(a, "kind", None) != "tool"
    ]


def prompt_for(spec: MASSpec, agent_id: str) -> str:
    compiled = compile_spec(spec)
    node = compiled.agents.get(agent_id)
    if node and str(node.system_prompt or "").strip():
        return str(node.system_prompt).strip()
    if agent_id == "planner" and spec.hub.system_prompt:
        return spec.hub.system_prompt.strip()
    return ""
