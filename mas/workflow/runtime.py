"""ExecutionService + single-path episode runner (LangGraph TirAgent adapter).

Must not import agentlightning. TirAgent is loaded via importlib (AST-safe).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable
from uuid import uuid4

from .archive import Archive, branch_point_to_resume_task_fields, register_archive
from .compiler import compile_spec, next_agent
from .contracts import BranchPoint, EventKind, ExecutionEvent, MemoryItem, Trajectory
from .memory import MemoryStore
from .plugins import invoke_hub_skills, invoke_skill, REGISTRY
from .rewards import has_answer_format
from .spec import MASSpec, load_spec


@dataclass
class LLMConfig:
    endpoint: str
    model: str
    temperature: float = 0.3
    max_turns: int = 8
    max_tokens: int = 1024
    max_model_len: Optional[int] = None
    request_logprobs: bool = False
    enabled_tools: Optional[Sequence[str]] = None
    langchain_callbacks: Optional[List[Any]] = None


@dataclass
class EpisodeRaw:
    messages: List[Dict[str, Any]]
    final_answer: Optional[str] = None
    format_ok: bool = False
    n_search: int = 0
    n_python: int = 0
    h_root: float = 0.0
    h_tool: float = 0.0
    consecutive_high: int = 0
    branch_messages: List[Dict[str, Any]] = field(default_factory=list)
    window_snapshots: List[Dict[str, Any]] = field(default_factory=list)
    window_events: List[Dict[str, Any]] = field(default_factory=list)
    turn_records: List[Dict[str, Any]] = field(default_factory=list)
    obs_hashes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    lc_messages: Any = None
    skill_results: List[Dict[str, Any]] = field(default_factory=list)


def append_events_from_messages(archive: Archive, messages: List[Dict[str, Any]]) -> None:
    """Translate serialized chat into MasEvents (tool / result / answer)."""
    for msg in messages:
        role = str(msg.get("role") or "")
        if role == "assistant" and msg.get("tool_calls"):
            for tc in msg.get("tool_calls") or []:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = tc.get("name") or (fn.get("name") if fn else None)
                args = tc.get("args") or tc.get("arguments") or (fn.get("arguments") if fn else {})
                archive.append(
                    ExecutionEvent(
                        kind=EventKind.TOOL_CALL,
                        payload={"name": name, "args": args, "id": tc.get("id")},
                    )
                )
            continue
        if role == "tool":
            archive.append(
                ExecutionEvent(
                    kind=EventKind.TOOL_RESULT,
                    payload={
                        "name": msg.get("name"),
                        "content": str(msg.get("content") or "")[:2000],
                    },
                )
            )
            continue
        if role == "assistant":
            content = str(msg.get("content") or "")
            if "<answer>" in content.lower() or "###" in content:
                archive.append(
                    ExecutionEvent(kind=EventKind.FINAL_ANSWER, payload={"text": content[:4000]})
                )
            elif content:
                archive.append(
                    ExecutionEvent(kind=EventKind.AGENT_MESSAGE, payload={"text": content[:500]})
                )


def _select_mock_tool(spec: Optional[MASSpec]) -> str:
    tools = list(spec.tools) if spec else ["execute_python"]
    if "execute_python" in tools:
        return "execute_python"
    return tools[0] if tools else "execute_python"


def _node_memory_settings(spec: MASSpec, agent_id: str) -> tuple:
    """agent-framework A4: resolve a node's memory_scope + profile.memory policy."""
    for a in getattr(spec, "agents", None) or []:
        if str(getattr(a, "id", "")) == agent_id:
            scope = str(getattr(a, "memory_scope", None) or "shared")
            policy = getattr(a, "profile", None) or {}
            mem = policy.get("memory") if isinstance(policy, dict) else None
            return scope, dict(mem) if isinstance(mem, dict) else {}
    return "shared", {}


def _open_construct(
    task: Dict[str, Any],
    archive: Archive,
    spec: MASSpec,
    memory: MemoryStore,
    agent_id: str = "hub",
) -> List[Dict[str, Any]]:
    archive.append(ExecutionEvent(kind=EventKind.TASK_START, payload={"task_id": task.get("id")}))
    # agent-framework A4: honour node memory_scope (agent | shared) and the
    # profile.memory truncation policy when reading the agent layer.
    mem_scope, mem_policy = _node_memory_settings(spec, agent_id)
    agent_items = memory.read_agent(agent_id, memory_scope=mem_scope, policy=mem_policy)
    archive.append(
        ExecutionEvent(
            kind=EventKind.MEMORY_READ,
            payload={
                "scope": "agent",
                "owner": agent_id if mem_scope == "agent" else "hub",
                "kind": spec.memory.agent,
                "n": len(agent_items),
            },
        )
    )
    ctx: Dict[str, Any] = {
        "task": task,
        "spec": spec,
        "memory_agent": [i.model_dump(mode="json") for i in agent_items],
    }
    return invoke_hub_skills(spec, ctx)


def _close_construct(
    archive: Archive,
    spec: MASSpec,
    memory: MemoryStore,
    raw: EpisodeRaw,
    task: Dict[str, Any],
    agent_id: str = "hub",
) -> None:
    # agent-framework A4: write to the buffer the node's scope resolves to
    # (private per-agent buffer, or the shared hub buffer by default).
    mem_scope, _policy = _node_memory_settings(spec, agent_id)
    agent_item = memory.write_agent(
        agent_id,
        {"final_answer": raw.final_answer, "n_messages": len(raw.messages)},
        memory_scope=mem_scope,
    )
    archive.append(
        ExecutionEvent(
            kind=EventKind.MEMORY_WRITE,
            payload={
                "scope": "agent",
                "owner": agent_item.owner,
                "id": agent_item.id,
                "memory_scope": mem_scope,
            },
        )
    )
    if spec.memory.system != "none" and not raw.error:
        sys_item = memory.write(
            MemoryItem(
                scope="system",
                owner="mas",
                content={"task_id": task.get("id"), "final_answer": raw.final_answer},
            )
        )
        archive.append(
            ExecutionEvent(
                kind=EventKind.MEMORY_WRITE,
                payload={"scope": "system", "owner": "mas", "id": sys_item.id},
            )
        )


def episode_to_trajectory(
    task: Dict[str, Any],
    raw: EpisodeRaw,
    archive: Archive,
    *,
    collector: str,
    spec: Optional[MASSpec] = None,
    memory: Optional[MemoryStore] = None,
) -> Trajectory:
    rid = str(task.get("_rollout_id") or uuid4().hex)
    bps = []
    if raw.branch_messages or (raw.messages and not raw.error):
        snap_msgs = raw.branch_messages or raw.messages
        snap = archive.snapshot(
            messages=snap_msgs,
            meta={
                "reason": "post_tool",
                "h_root": raw.h_root,
                "h_tool": raw.h_tool,
                "consecutive_high": raw.consecutive_high,
            },
            rollout_id=rid,
        )
        bps = [archive.to_branch_point(snap.snapshot_id, reason="post_tool")]
    mem_dump = [MemoryItem(scope="agent", owner="hub").model_dump(mode="json")]
    if memory is not None:
        latest = memory.latest("agent", "hub")
        mem_dump = [latest.model_dump(mode="json")] if latest else mem_dump
        sys_latest = memory.latest("system", "mas")
        if sys_latest:
            mem_dump.append(sys_latest.model_dump(mode="json"))
    spec_obj = spec or load_spec()
    routes = [r.get("route") for r in raw.skill_results if r.get("route")]
    traj = Trajectory(
        task=dict(task),
        messages=list(raw.messages),
        final_answer=raw.final_answer,
        format_ok=bool(raw.format_ok),
        n_search=int(raw.n_search),
        n_python=int(raw.n_python),
        archive=archive.ref(),
        branch_points=bps,
        events=list(archive._events),
        meta={
            "collector": collector,
            "h_root": raw.h_root,
            "h_tool": raw.h_tool,
            "consecutive_high": raw.consecutive_high,
            "rollout_id": rid,
            "memory": mem_dump,
            "tools": list(spec_obj.tools),
            "skills": [r.get("skill") for r in raw.skill_results],
        },
    )
    if routes:
        traj.meta["skill_route"] = routes[-1]
    if raw.error:
        traj.meta["error"] = raw.error
    traj.sync_tir_meta()
    return traj


def run_mock_episode(
    task: Dict[str, Any],
    archive: Archive,
    *,
    spec: Optional[MASSpec] = None,
    memory: Optional[MemoryStore] = None,
    agent_id: str = "hub",
) -> EpisodeRaw:
    spec = spec or load_spec()
    memory = memory or MemoryStore()
    skill_results = _open_construct(task, archive, spec, memory, agent_id=agent_id)
    question = str(task.get("question") or "")
    gold = str(task.get("answer") or "0")
    forced = task.get("_mock_answer")
    answer = str(forced) if forced is not None else gold
    hop = int(task.get("_feedback_hop") or 0)
    force_error = task.get("_mock_error")
    if hop >= 1 and not task.get("_mock_error_persist"):
        force_error = None
    tool_name = _select_mock_tool(spec)
    if tool_name == "execute_python":
        args: Dict[str, Any] = {"code": "result = 1"}
        tool_out = "1"
        n_python, n_search = 1, 0
    else:
        args = {"query": question[:80]}
        tool_out = "mock-obs"
        n_python, n_search = 0, 1
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "mock"},
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": tool_name, "arguments": str(args)},
                    "name": tool_name,
                    "args": args,
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": tool_name, "content": tool_out},
        {"role": "assistant", "content": f"<answer>{answer}</answer>"},
    ]
    if force_error:
        archive.append(ExecutionEvent(kind=EventKind.ERROR, payload={"error": str(force_error)}))
        raw_err = EpisodeRaw(
            messages=messages,
            final_answer=None,
            format_ok=False,
            error=str(force_error),
            skill_results=skill_results,
        )
        _close_construct(archive, spec, memory, raw_err, task, agent_id=agent_id)
        return raw_err
    append_events_from_messages(archive, messages)
    raw = EpisodeRaw(
        messages=messages,
        final_answer=answer,
        format_ok=True,
        n_search=n_search,
        n_python=n_python,
        h_root=0.3,
        h_tool=1.0,
        consecutive_high=1,
        branch_messages=messages[:4],
        window_snapshots=[{"agent_id": agent_id, "kind": "post_first_tool", "messages": messages[:4]}],
        window_events=[
            {"agent_id": agent_id, "kind": "after_tool", "tool_id": tool_name, "turn": 1, "snapshot_ref": f"{agent_id}:0", "metrics": {}}
        ],
        skill_results=skill_results,
    )
    _close_construct(archive, spec, memory, raw, task, agent_id=agent_id)
    return raw


def run_episode(
    task: Dict[str, Any],
    llm: LLMConfig,
    archive: Archive,
    *,
    spec: Optional[MASSpec] = None,
    memory: Optional[MemoryStore] = None,
    agent_id: str = "hub",
    tools_override: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
) -> EpisodeRaw:
    """Invoke TirAgent graph. Caller must supply a live endpoint."""
    spec = spec or load_spec()
    memory = memory or MemoryStore()
    skill_results = _open_construct(task, archive, spec, memory, agent_id=agent_id)

    tir = importlib.import_module("tir_agent")
    arpo = importlib.import_module("rl.hooks.arpo_rollout")
    lc_messages = importlib.import_module("langchain_core.messages")

    question = str(task["question"])
    resume_msgs = tir._parse_resume_messages(task.get("resume_messages"))
    enabled = (
        list(tools_override)
        if tools_override is not None
        else (list(llm.enabled_tools) if llm.enabled_tools is not None else list(spec.tools))
    )
    from_spec = getattr(tir.TirAgent, "from_spec", None)
    if callable(from_spec):
        agent = from_spec(
            spec,
            endpoint=llm.endpoint,
            model_name=llm.model,
            temperature=llm.temperature,
            max_turns=llm.max_turns,
            max_tokens=llm.max_tokens,
            max_model_len=llm.max_model_len,
            request_logprobs=llm.request_logprobs,
            enabled_tools=enabled,
            system_prompt=system_prompt,
            agent_id=agent_id,
        )
    else:
        agent = tir.TirAgent(
            endpoint=llm.endpoint,
            model_name=llm.model,
            temperature=llm.temperature,
            max_turns=llm.max_turns,
            max_tokens=llm.max_tokens,
            max_model_len=llm.max_model_len,
            request_logprobs=llm.request_logprobs,
            enabled_tools=enabled,
            system_prompt=system_prompt,
            agent_id=agent_id,
        )
    # agent-framework A1: test mode (llm.kind=api) may use epc_aw LLM-in-tool
    # backends for tool-agents declaring profile.llm_required; training/collect
    # keeps the pure mas/tools functions.
    invoker = getattr(agent, "tool_agent_invoker", None)
    if invoker is not None:
        try:
            invoker.prefer_llm = bool(
                getattr(getattr(spec, "llm", None), "kind", "") == "api"
                and any(
                    bool(getattr(a, "profile", {}).get("llm_required"))
                    for a in getattr(spec, "agents", []) or []
                )
            )
        except Exception:
            invoker.prefer_llm = False
    sys_text = getattr(agent, "system_prompt", None) or tir.SYSTEM_PROMPT
    if resume_msgs:
        initial = {
            "question": question,
            "num_turns": sum(1 for m in resume_msgs if isinstance(m, lc_messages.AIMessage)),
            "asked_finalize": False,
            "messages": resume_msgs,
            "turn_records": [],
            "n_search": 0,
            "n_python": 0,
            "h_root": 0.0,
            "h_tool": 0.0,
            "consecutive_high": 0,
            "last_entropy": 0.0,
            "branch_messages": [],
        }
    else:
        initial = {
            "question": question,
            "num_turns": 0,
            "asked_finalize": False,
            "messages": [
                lc_messages.SystemMessage(content=sys_text),
                lc_messages.HumanMessage(content=question),
            ],
            "turn_records": [],
            "n_search": 0,
            "n_python": 0,
            "h_root": 0.0,
            "h_tool": 0.0,
            "consecutive_high": 0,
            "last_entropy": 0.0,
            "branch_messages": [],
        }
    cfg: Dict[str, Any] = {"recursion_limit": 40}
    if llm.langchain_callbacks:
        cfg["callbacks"] = llm.langchain_callbacks
    try:
        result_state = agent.graph().invoke(initial, cfg)
        messages = list(result_state["messages"])
        prediction = tir.extract_answer_from_messages(messages)
        raw_text = tir.last_assistant_text(messages)
        ser = arpo.serialize_messages(messages)
        append_events_from_messages(archive, ser)
        records = list(result_state.get("turn_records") or [])
        hashes = [str(r.get("obs_hash") or "") for r in records]
        branch_messages = list(result_state.get("branch_messages") or [])
        window_snapshots = list(result_state.get("window_snapshots") or [])
        window_events = list(result_state.get("window_events") or [])
        if not window_snapshots and branch_messages:
            # legacy graph path: derive one post_first_tool window snapshot
            window_snapshots = [{"agent_id": "hub", "kind": "post_first_tool", "messages": branch_messages}]
        if not window_events and window_snapshots:
            # P2 fallback: derive WindowEndEvent dicts from snapshots (kind after_tool)
            window_events = [
                {
                    "agent_id": str(s.get("agent_id") or "hub"),
                    "kind": "after_tool",
                    "turn": int(s.get("turn") or 0),
                    "snapshot_ref": f"{s.get('agent_id') or 'hub'}:{i}",
                    "metrics": {},
                }
                for i, s in enumerate(window_snapshots)
            ]
        raw = EpisodeRaw(
            messages=ser,
            final_answer=prediction,
            format_ok=bool(has_answer_format(raw_text)),
            n_search=int(result_state.get("n_search") or 0),
            n_python=int(result_state.get("n_python") or 0),
            h_root=float(result_state.get("h_root") or 0.0),
            h_tool=float(result_state.get("h_tool") or 0.0),
            consecutive_high=int(result_state.get("consecutive_high") or 0),
            branch_messages=branch_messages or ser,
            window_snapshots=window_snapshots,
            window_events=window_events,
            turn_records=records,
            obs_hashes=hashes,
            lc_messages=messages,
            skill_results=skill_results,
        )
        _close_construct(archive, spec, memory, raw, task, agent_id=agent_id)
        return raw
    except Exception as e:
        archive.append(ExecutionEvent(kind=EventKind.ERROR, payload={"error": str(e)}))
        raw_err = EpisodeRaw(messages=[], error=str(e), skill_results=skill_results)
        _close_construct(archive, spec, memory, raw_err, task, agent_id=agent_id)
        return raw_err


@runtime_checkable
class EpisodeRunner(Protocol):
    def run(
        self,
        task: Dict[str, Any],
        llm: Optional[LLMConfig],
        archive: Archive,
        spec: MASSpec,
        memory: MemoryStore,
        agent_id: str = "hub",
    ) -> EpisodeRaw:
        ...


class MockRunner:
    def run(
        self,
        task: Dict[str, Any],
        llm: Optional[LLMConfig],
        archive: Archive,
        spec: MASSpec,
        memory: MemoryStore,
        agent_id: str = "hub",
    ) -> EpisodeRaw:
        del llm
        return run_mock_episode(task, archive, spec=spec, memory=memory, agent_id=agent_id)


class TirRunner:
    def run(
        self,
        task: Dict[str, Any],
        llm: Optional[LLMConfig],
        archive: Archive,
        spec: MASSpec,
        memory: MemoryStore,
        agent_id: str = "hub",
    ) -> EpisodeRaw:
        if llm is None or not llm.endpoint:
            raise RuntimeError("Set LLMConfig.endpoint or pass mock=True")
        prompt = (spec.hub.system_prompt or "").strip() or None
        return run_episode(
            task,
            llm,
            archive,
            spec=spec,
            memory=memory,
            agent_id=agent_id,
            system_prompt=prompt,
        )


def apply_verifier_feedback(
    task: Dict[str, Any],
    raw: EpisodeRaw,
    llm: Optional[LLMConfig],
    archive: Archive,
    spec: MASSpec,
    memory: MemoryStore,
    runner: EpisodeRunner,
) -> EpisodeRaw:
    """If hub.verify is set, run that skill after the episode. route=hub reruns at most max_feedback_hops."""
    name = str(getattr(spec.hub, "verify", None) or "").strip()
    if not name:
        return raw
    REGISTRY.get_role("verifier")
    max_hops = max(0, int(spec.hub.max_feedback_hops or 0))
    hops = 0
    while True:
        ctx: Dict[str, Any] = {
            "task": task,
            "final_answer": raw.final_answer,
            "format_ok": raw.format_ok,
            "error": raw.error,
            "hop": hops,
        }
        row = invoke_skill(name, ctx)
        raw.skill_results = list(raw.skill_results) + [row]
        route = row.get("route")
        will_rerun = route == "hub" and hops < max_hops
        archive.append(
            ExecutionEvent(
                kind=EventKind.FEEDBACK,
                agent_id="verifier",
                payload={
                    "skill": name,
                    "ok": row.get("ok"),
                    "route": route,
                    "hop": hops,
                    "rerun": will_rerun,
                },
            )
        )
        if not will_rerun:
            return raw
        hops += 1
        retry_task = dict(task)
        retry_task["_feedback_hop"] = hops
        prev_skills = list(raw.skill_results)
        raw = runner.run(retry_task, llm, archive, spec, memory)
        raw.skill_results = prev_skills + list(raw.skill_results)


def run_compiled_episode(
    task: Dict[str, Any],
    llm: Optional[LLMConfig],
    archive: Archive,
    spec: MASSpec,
    memory: MemoryStore,
    runner: EpisodeRunner,
) -> EpisodeRaw:
    """Walk compiled route/message/feedback; each non-verifier hop is a Tir/mock episode."""
    compiled = compile_spec(spec)
    current = compiled.entry_agent
    hops = 0
    max_hops = max(4, int(spec.hub.max_feedback_hops or 1) * 3 + max(1, len(compiled.agents)))
    last_raw: Optional[EpisodeRaw] = None
    handoff: Optional[str] = None
    combined_skills: List[Dict[str, Any]] = []
    n_search = 0
    n_python = 0
    all_messages: List[Dict[str, Any]] = []
    merged_events: List[Dict[str, Any]] = []
    merged_snapshots: List[Dict[str, Any]] = []

    while current and hops < max_hops:
        hops += 1
        node = compiled.agents.get(current)
        if node is None:
            break
        # agent-framework A3: remember which agent node this hop belongs to so
        # per-hop window_events/snapshots merge into the final EpisodeRaw with
        # real node ids (executor = tool-agent set via compiled.tools_for).
        current_agent_id = str(current)
        archive.append(
            ExecutionEvent(
                kind=EventKind.AGENT_MESSAGE,
                agent_id=current,
                payload={"phase": "enter", "role": node.role, "hop": hops},
            )
        )
        role = str(node.role or current).lower()
        task_i = dict(task)
        if handoff:
            task_i["question"] = (
                f"{task.get('question') or ''}\n\n[Previous agent output]\n{handoff}"
            )

        if role in ("verifier", "critic"):
            ctx: Dict[str, Any] = {
                "task": task_i,
                "final_answer": last_raw.final_answer if last_raw else None,
                "format_ok": last_raw.format_ok if last_raw else False,
                "error": last_raw.error if last_raw else None,
                "hop": hops,
            }
            row = invoke_skill("verifier", ctx)
            combined_skills.append(row)
            archive.append(
                ExecutionEvent(
                    kind=EventKind.FEEDBACK,
                    agent_id=current,
                    payload={"ok": row.get("ok"), "route": row.get("route"), "hop": hops},
                )
            )
            if last_raw is not None:
                last_raw.skill_results = list(last_raw.skill_results) + [row]
            if not row.get("ok"):
                current = compiled.feedback_to.get(current) or compiled.entry_agent
                handoff = str((row.get("output") or {}).get("reason") or "verify_failed")
                continue
            break

        tools = list(compiled.tools_for.get(current) or [])
        if not tools and current in {"hub", "orchestrator"}:
            tools = list(spec.tools)
        prompt = (node.system_prompt or "").strip() or (spec.hub.system_prompt or None)
        sub_spec = spec.model_copy(deep=True)
        sub_spec.tools = list(tools)
        if prompt:
            sub_spec.hub = spec.hub.model_copy(update={"system_prompt": prompt})

        raw = runner.run(task_i, llm, archive, sub_spec, memory, agent_id=current_agent_id)
        last_raw = raw
        n_search += int(raw.n_search)
        n_python += int(raw.n_python)
        combined_skills.extend(list(raw.skill_results))
        all_messages.extend(list(raw.messages))
        # agent-framework A3: merge per-hop window_events/snapshots (real ids).
        merged_events.extend(list(raw.window_events or []))
        merged_snapshots.extend(list(raw.window_snapshots or []))
        nxt = next_agent(compiled, current)
        if nxt:
            handoff = raw.final_answer or ""
            current = nxt
            continue
        break

    if last_raw is None:
        last_raw = EpisodeRaw(messages=[], error="compiled graph produced no episode")
    last_raw.n_search = n_search
    last_raw.n_python = n_python
    last_raw.skill_results = combined_skills or last_raw.skill_results
    if all_messages:
        last_raw.messages = all_messages
    # agent-framework A3: real per-agent events produced along the graph path
    # (no longer relying on the single-agent fallback derivation).
    if merged_events:
        last_raw.window_events = merged_events
    if merged_snapshots:
        last_raw.window_snapshots = merged_snapshots
    return last_raw


class ExecutionService:
    """MAS construct facade: run / fork / collect. Does not compute reward."""

    def __init__(
        self,
        *,
        mock: bool = False,
        spec: Optional[MASSpec] = None,
        spec_path: Optional[str] = None,
        llm: Optional[LLMConfig] = None,
        archive_root: Optional[str] = None,
        memory: Optional[MemoryStore] = None,
        runner: Optional[EpisodeRunner] = None,
    ) -> None:
        self.mock = mock
        self.spec = spec or load_spec(spec_path)
        self.llm = llm
        self.archive_root = archive_root
        self.memory = memory or MemoryStore()
        self.runner: EpisodeRunner = runner or (MockRunner() if mock else TirRunner())

    def run(self, task: Dict[str, Any]) -> Trajectory:
        arch = register_archive(Archive(root_dir=self.archive_root))
        task_run = dict(task)
        resume_raw = task_run.get("resume_from")
        if resume_raw:
            bp = resume_raw if isinstance(resume_raw, BranchPoint) else BranchPoint.model_validate(resume_raw)
            task_run.update(branch_point_to_resume_task_fields(bp))
        compiled = compile_spec(self.spec)
        if compiled.ok and compiled.multi_agent:
            raw = run_compiled_episode(
                task_run, self.llm, arch, self.spec, self.memory, self.runner
            )
        else:
            raw = self.runner.run(task_run, self.llm, arch, self.spec, self.memory)
            raw = apply_verifier_feedback(
                task_run, raw, self.llm, arch, self.spec, self.memory, self.runner
            )
        collector = "mock" if self.mock else "tir_agent"
        traj = episode_to_trajectory(
            task_run, raw, arch, collector=collector, spec=self.spec, memory=self.memory
        )
        fb = [e for e in traj.events if e.kind == EventKind.FEEDBACK]
        traj.meta["feedback_hops"] = sum(1 for e in fb if (e.payload or {}).get("rerun"))
        traj.meta["verify"] = str(self.spec.hub.verify or "") or None
        traj.meta["entry_agent"] = compiled.entry_agent
        traj.meta["compiled"] = compiled.reason
        return traj

    def fork(self, branch_point: BranchPoint, task: Optional[Dict[str, Any]] = None) -> Trajectory:
        base = dict(task or {})
        base["resume_from"] = branch_point
        return self.run(base)

    def collect(self, tasks: Sequence[Dict[str, Any]]) -> List[Trajectory]:
        return [self.run(t) for t in tasks]
