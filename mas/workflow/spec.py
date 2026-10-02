"""Static Framework spec (YAML / Pydantic). Not a Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .contracts import SamplePolicy

DEFAULT_SPEC_PATH = Path(__file__).resolve().parent.parent / "specs" / "hub_react.yaml"


class HubSpec(BaseModel):
    """Planner defaults (YAML key remains ``hub:`` for experiment compatibility).

    This is not an agent node. The orchestrator on the graph is ``kind: planner``.
    """

    role: str = "planner"
    skills: List[str] = Field(default_factory=list)
    verify: Optional[str] = None
    max_feedback_hops: int = 1
    system_prompt: str = ""

    model_config = {"extra": "forbid"}


class LLMBinding(BaseModel):
    kind: Literal["api", "local", "rl_endpoint"] = "api"
    model: str = ""
    base_url: Optional[str] = None

    model_config = {"extra": "forbid"}


class MemorySpec(BaseModel):
    agent: str = "messages"
    system: str = "none"

    model_config = {"extra": "forbid"}


class ArchiveSpec(BaseModel):
    window: str = "post_first_tool"

    model_config = {"extra": "forbid"}


class AgentNodeSpec(BaseModel):
    """Graph node (schema 0.3). kind unifies agents and tools (new_framework P1).

    ``hub`` was removed in the centralized redesign — the orchestrator is now a
    ``planner``. ``memory_scope`` defaults to ``none`` (agent-layer memory is
    opt-in via ``profile.memory``); the MAS-layer blackboard log is always on.
    """

    id: str
    kind: Literal["planner", "tool", "verifier", "blank"] = "blank"
    role: str = "agent"
    label: Optional[str] = None  # canvas display name; runtime routing still uses id
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)  # routable tool-agent ids (planner)
    memory_scope: str = "none"
    system_prompt: str = ""
    model: str = "inherit"
    trainable: bool = True
    profile: Dict[str, Any] = Field(default_factory=dict)  # blank agent custom profile
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class RouterSpec(BaseModel):
    """Graph node whose downstream is an agent set (``candidates``).

    Unpacks an upstream ``plan_step`` (or other AgentMessage) into one or more
    ``tool_invoke`` hops. ``from_plan`` / ``llm_choice`` only parse the already
    produced payload — the router does not re-plan. ``score`` and
    ``round_robin`` pick among the same set.
    """

    id: str
    candidates: List[str] = Field(default_factory=list)
    strategy: Literal["from_plan", "llm_choice", "score", "round_robin"] = "from_plan"
    scorer: Optional[str] = None  # scoring agent id when strategy=score
    output_contract: str = "json"  # expected output serialization (json | text)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class EdgeSpec(BaseModel):
    """Communication edge between agents (UI / future compiler)."""

    source: str = Field(alias="from")
    target: str = Field(alias="to")
    kind: Literal["message", "tool_call", "feedback", "route", "sample_barrier"] = "message"
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid", "populate_by_name": True}


class MASSpec(BaseModel):
    schema_version: str = "0.3"
    topology: str = "centralized"
    hub: HubSpec = Field(default_factory=HubSpec)
    tools: List[str] = Field(
        default_factory=lambda: ["wikipedia_search", "google_search", "web_search", "python_coder", "think"]
    )
    llm: LLMBinding = Field(default_factory=LLMBinding)
    memory: MemorySpec = Field(default_factory=MemorySpec)
    archive: ArchiveSpec = Field(default_factory=ArchiveSpec)
    # schema 0.2 graph extensions
    agents: List[AgentNodeSpec] = Field(default_factory=list)
    edges: List[EdgeSpec] = Field(default_factory=list)
    # schema 0.3: agent routers (new_framework P1)
    routers: List[RouterSpec] = Field(default_factory=list)
    entry_agent: str = "planner"
    sampling: SamplePolicy = Field(default_factory=SamplePolicy)

    model_config = {"extra": "forbid"}

    def is_executable(self) -> tuple[bool, str]:
        """Centralized topology always runs; graph topologies need a valid compile()."""
        if self.topology in ("centralized", "hub_react", "single", ""):
            return True, "centralized"
        if self.topology == "graph":
            from .compiler import compile_spec

            compiled = compile_spec(self)
            return compiled.ok, compiled.reason
        return False, f"unknown topology {self.topology}"


def load_spec(path: Optional[str] = None) -> MASSpec:
    p = Path(path) if path else DEFAULT_SPEC_PATH
    if not p.is_file():
        return MASSpec()
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text) or {}
    except Exception:
        raw = _parse_simple_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError(f"spec must be a mapping: {p}")
    # Normalize edge keys from / from_ / source
    edges = raw.get("edges")
    if isinstance(edges, list):
        norm = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            item = dict(e)
            if "from" in item and "source" not in item:
                pass
            elif "source" in item and "from" not in item:
                item["from"] = item.pop("source")
            if "to" in item and "target" not in item:
                pass
            elif "target" in item and "to" not in item:
                item["to"] = item.pop("target")
            norm.append(item)
        raw["edges"] = norm
    _normalize_schema03(raw)
    return MASSpec.model_validate(raw)


def _normalize_schema03(raw: Dict[str, Any]) -> None:
    """Sugar expansion to schema 0.3 (in place, old YAML untouched on disk).

    1. Top-level ``tools: [t1, t2]`` -> implicit ``kind=tool`` AgentNodeSpec
       (skipped when an agent with the same id already exists).
    2. Legacy agents without ``kind`` get one inferred from ``role``
       (planner/verifier -> same name; tool-ish roles -> tool; else blank).
    3. ``hub`` / ``orchestrator`` agents are dropped and merged into planner.
       ``executor`` agents are dropped; their tools join the router pool.
       User wraps (``profile.backend=user_space`` or ``meta.wraps`` starting with
       ``EPC-AW.``) stay on the canvas as their own windows.
    4. ``execute_python`` (legacy pure-tool id) maps to ``python_coder``.
    5. ``kind: tool`` agents are forced ``trainable: false``.
    """
    # 4. execute_python -> python_coder (tools list + agent ids + router candidates)
    _LEGACY_TOOL_MAP = {"execute_python": "python_coder"}

    def _remap_id(node_id: str) -> str:
        return _LEGACY_TOOL_MAP.get(str(node_id), str(node_id))

    tools = raw.get("tools")
    if isinstance(tools, list):
        raw["tools"] = [_remap_id(t) for t in tools]
    for a in raw.get("agents") or []:
        if isinstance(a, dict):
            if "id" in a:
                a["id"] = _remap_id(a["id"])
            if "tools" in a and isinstance(a["tools"], list):
                a["tools"] = [_remap_id(t) for t in a["tools"]]
    for r in raw.get("routers") or []:
        if isinstance(r, dict) and "candidates" in r and isinstance(r["candidates"], list):
            r["candidates"] = [_remap_id(c) for c in r["candidates"]]
    for e in raw.get("edges") or []:
        if isinstance(e, dict):
            if "from" in e:
                e["from"] = _remap_id(e["from"])
            if "to" in e:
                e["to"] = _remap_id(e["to"])

    agents = raw.get("agents")
    if not isinstance(agents, list):
        agents = []
    agent_ids = {a.get("id") for a in agents if isinstance(a, dict)}
    for t in raw.get("tools") or []:
        t = str(t)
        if t and t not in agent_ids:
            agents.append({"id": t, "kind": "tool", "trainable": False})
            agent_ids.add(t)
    for a in agents:
        if isinstance(a, dict) and not a.get("kind"):
            role = str(a.get("role") or "").lower()
            if role in ("planner", "verifier"):
                a["kind"] = role
            elif role == "tool":
                a["kind"] = "tool"
            elif role not in ("executor", "hub", "orchestrator"):
                a["kind"] = "blank"

    def _is_user_wrap(agent: Dict[str, Any]) -> bool:
        profile = agent.get("profile") if isinstance(agent.get("profile"), dict) else {}
        meta = agent.get("meta") if isinstance(agent.get("meta"), dict) else {}
        wraps = str(meta.get("wraps") or "")
        return str(profile.get("backend") or "") == "user_space" or wraps.startswith("EPC-AW.")

    def _is_hub(agent: Dict[str, Any]) -> bool:
        if _is_user_wrap(agent):
            return False
        kind = str(agent.get("kind") or "").lower()
        role = str(agent.get("role") or "").lower()
        return str(agent.get("id") or "") == "hub" or kind == "hub" or role in ("hub", "orchestrator")

    def _is_executor(agent: Dict[str, Any]) -> bool:
        if _is_user_wrap(agent):
            return False
        kind = str(agent.get("kind") or "").lower()
        role = str(agent.get("role") or "").lower()
        return str(agent.get("id") or "") == "executor" or kind == "executor" or role == "executor"

    kept: List[Dict[str, Any]] = []
    drop_ids: set = set()
    hub_prompt = ""
    hub_skills: List[str] = []
    executor_tools: List[str] = []
    for a in agents:
        if not isinstance(a, dict):
            continue
        if _is_hub(a):
            drop_ids.add(str(a.get("id") or ""))
            if not hub_prompt and a.get("system_prompt"):
                hub_prompt = str(a.get("system_prompt") or "")
            for skill in a.get("skills") or []:
                if skill not in hub_skills:
                    hub_skills.append(str(skill))
            continue
        if _is_executor(a):
            drop_ids.add(str(a.get("id") or ""))
            for tool in a.get("tools") or []:
                if tool not in executor_tools:
                    executor_tools.append(str(tool))
            continue
        if str(a.get("kind") or "") == "tool":
            a["trainable"] = False
        kept.append(a)

    planner = next((a for a in kept if a.get("id") == "planner" or a.get("kind") == "planner"), None)
    if (hub_prompt or hub_skills) and planner is None:
        planner = {"id": "planner", "kind": "planner", "role": "planner", "trainable": True}
        kept.insert(0, planner)
    if planner is not None:
        if hub_prompt and not planner.get("system_prompt"):
            planner["system_prompt"] = hub_prompt
        if hub_skills and not planner.get("skills"):
            planner["skills"] = hub_skills
    raw["agents"] = kept

    executor_next = None
    edges = []
    for e in raw.get("edges") or []:
        if not isinstance(e, dict):
            continue
        src, dst = str(e.get("from") or ""), str(e.get("to") or "")
        if src in drop_ids and e.get("kind") == "message" and dst not in drop_ids:
            executor_next = dst
        if src in drop_ids or dst in drop_ids:
            if e.get("kind") == "feedback" and src not in drop_ids:
                item = dict(e)
                item["to"] = "planner"
                edges.append(item)
            continue
        if dst == "hub" and e.get("kind") == "feedback":
            item = dict(e)
            item["to"] = "planner"
            edges.append(item)
            continue
        edges.append(e)

    routers = raw.get("routers") if isinstance(raw.get("routers"), list) else []
    if executor_tools and not routers:
        routers = [{"id": "route_exec", "candidates": [], "strategy": "from_plan"}]
        if not any(e.get("to") == "route_exec" for e in edges):
            edges.append({"from": "planner", "to": "route_exec", "kind": "route"})
        if executor_next and not any(e.get("from") == "route_exec" for e in edges):
            edges.append({"from": "route_exec", "to": executor_next, "kind": "message"})
    for router in routers:
        if not isinstance(router, dict):
            continue
        candidates = [str(c) for c in (router.get("candidates") or []) if str(c) not in drop_ids]
        if executor_tools and router is routers[0]:
            for tool in executor_tools:
                if tool not in candidates:
                    candidates.append(tool)
        router["candidates"] = candidates
    if routers:
        raw["routers"] = routers
    if raw.get("edges") is not None or edges:
        raw["edges"] = edges

    known = {a.get("id") for a in kept if isinstance(a, dict)}
    for tool in executor_tools:
        if tool and tool not in known:
            kept.append({"id": tool, "kind": "tool", "trainable": False})
            known.add(tool)
    raw["agents"] = kept


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Minimal fallback when PyYAML is absent (hub_react.yaml subset)."""
    data: Dict[str, Any] = {}
    current: Optional[str] = None
    nested: Dict[str, Any] = {}
    tools: List[str] = []
    skills: List[str] = []
    in_tools = False
    in_skills = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        s = line.strip()
        if indent == 0 and ":" in s and not s.startswith("-"):
            key, _, rest = s.partition(":")
            key = key.strip()
            rest = rest.strip().strip('"').strip("'")
            if rest:
                data[key] = rest
                continue
            if current == "hub":
                data["hub"] = nested
            elif current == "llm":
                data["llm"] = nested
            elif current == "memory":
                data["memory"] = nested
            elif current == "archive":
                data["archive"] = nested
            current = key
            nested = {}
            in_tools = key == "tools"
            in_skills = False
            if in_tools:
                tools = []
            continue
        if s.startswith("- "):
            item = s[2:].strip()
            if in_tools:
                tools.append(item)
            elif in_skills:
                skills.append(item)
            continue
        if indent > 0 and ":" in s:
            k, v = s.split(":", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k == "skills":
                in_skills = True
                skills = []
                if v.startswith("[") and v.endswith("]"):
                    inner = v[1:-1].strip()
                    skills = [x.strip() for x in inner.split(",") if x.strip()]
                    in_skills = False
                    nested[k] = skills
                continue
            nested[k] = v
    if current == "hub":
        if skills:
            nested["skills"] = skills
        data["hub"] = nested
    elif current == "llm":
        data["llm"] = nested
    elif current == "memory":
        data["memory"] = nested
    elif current == "archive":
        data["archive"] = nested
    if tools:
        data["tools"] = tools
    return data
