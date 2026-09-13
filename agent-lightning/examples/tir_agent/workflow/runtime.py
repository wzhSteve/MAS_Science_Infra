"""One MAS execution path for Control, Collector and training (no AGL dependency)."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable
from uuid import uuid4

from .archive import Archive, ArchiveWriteError, branch_point_to_resume_task_fields, register_archive
from .compiler import KNOWN_ROLES, KNOWN_TOOLS, compile_spec, next_agent
from .contracts import AgentExecutionContext, BranchPoint, EventKind, ExecutionEvent, MemoryItem, ModelIdentity, Trajectory
from .memory import MemoryStore
from .plugins import REGISTRY, invoke_skill
from .recorder import BoundaryRecorder, Redactor, message_records
from .rewards import has_answer_format
from .spec import AgentNodeSpec, MASSpec, load_spec


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
    # None retains CLI environment lookup; "" explicitly declines an environment key.
    api_key: Optional[str] = field(default=None, repr=False)
    source: str = "api"
    policy_version: Optional[str] = None
    tokenizer_id: Optional[str] = None
    sampling_parameters: Dict[str, Any] = field(default_factory=dict)
    resource_id: Optional[str] = None
    resource_revision: Optional[int] = None
    resource_name: Optional[str] = None

    def identity(self) -> ModelIdentity:
        return ModelIdentity(
            source=self.source, model=self.model,
            policy_version=self.policy_version, tokenizer_id=self.tokenizer_id,
            resource_id=self.resource_id, resource_revision=self.resource_revision, resource_name=self.resource_name,
        )

    def public_error(self, error: BaseException) -> str:
        return Redactor([self.api_key] if self.api_key else [])(str(error))


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
    branch_event_id: Optional[str] = None
    turn_records: List[Dict[str, Any]] = field(default_factory=list)
    obs_hashes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    lc_messages: Any = None
    skill_results: List[Dict[str, Any]] = field(default_factory=list)
    termination_reason: str = "answer"
    error_stage: Optional[str] = None
    error_code: Optional[str] = None
    uncertainty_evidence: str = "unavailable_neutral_zero"


def _recorder(archive: Archive, agent_id: str = "hub") -> BoundaryRecorder:
    if archive.active_recorder is not None:
        return archive.active_recorder
    archive.run_id = archive.run_id or uuid4().hex
    archive.trajectory_id = archive.trajectory_id or uuid4().hex
    return BoundaryRecorder(
        archive, run_id=archive.run_id, trajectory_id=archive.trajectory_id,
        agent_id=agent_id, agent_execution_id=uuid4().hex,
    )


def run_mock_episode(
    task: Dict[str, Any], archive: Archive, *, spec: Optional[MASSpec] = None,
    memory: Optional[MemoryStore] = None,
) -> EpisodeRaw:
    """Synthetic behavior, never a model request and never derived from evaluation data."""
    spec = spec or load_spec()
    recorder = _recorder(archive)
    answer = str(task.get("_mock_answer", "mock answer"))
    messages = list(task.get("_initial_messages") or [
        {"role": "system", "content": spec.hub.system_prompt or "mock"},
        {"role": "user", "content": str(task.get("question") or "")},
    ])
    error = task.get("_mock_error")
    if task.get("_feedback_hop") and not task.get("_mock_error_persist"):
        error = None
    if error:
        recorder.emit(EventKind.ERROR, {"error": str(error), "stage": "mock", "mock": True})
        return EpisodeRaw(messages=messages, error=str(error), termination_reason="mock_error", error_stage="mock")
    n_python = n_search = 0
    branch = []
    branch_event_id = None
    if spec.tools:
        tool = "execute_python" if "execute_python" in spec.tools else spec.tools[0]
        args = {"code": "result = 1"} if tool == "execute_python" else {"query": task.get("question", "")}
        call_id = uuid4().hex
        recorder.emit(EventKind.TOOL_CALL, {"name": tool, "args": args, "mock": True}, tool_call_id=call_id)
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"name": tool, "args": args, "id": call_id, "type": "tool_call"},
        ]})
        result = "1" if tool == "execute_python" else "mock observation"
        messages.append({"role": "tool", "name": tool, "tool_call_id": call_id, "content": result})
        event = recorder.emit(EventKind.TOOL_RESULT, {
            "name": tool, "content": result, "status": "succeeded", "mock": True,
        }, tool_call_id=call_id)
        n_python = int(tool == "execute_python")
        n_search = int(tool in {"web_search", "wikipedia_search"})
        branch = list(messages)
        branch_event_id = event.event_id
    messages.append({"role": "assistant", "content": f"<answer>{answer}</answer>"})
    recorder.emit(EventKind.AGENT_MESSAGE, {"message": messages[-1], "mock": True})
    return EpisodeRaw(
        messages=messages, final_answer=answer, format_ok=True,
        n_search=n_search, n_python=n_python, branch_messages=branch,
        branch_event_id=branch_event_id,
        termination_reason="mock_answer",
    )


def run_episode(
    task: Dict[str, Any], llm: LLMConfig, archive: Archive, *,
    spec: Optional[MASSpec] = None, memory: Optional[MemoryStore] = None,
    agent_id: str = "hub", tools_override: Optional[Sequence[str]] = None,
    system_prompt: Optional[str] = None,
) -> EpisodeRaw:
    """TirAgent adapter. Capture partial state; persistence errors are never recovered."""
    spec = spec or load_spec()
    recorder = _recorder(archive, agent_id)
    agent = None
    initial: Dict[str, Any] = {}
    error = None
    stage = "model_setup"
    code = None
    try:
        tir = importlib.import_module("tir_agent")
        arpo = importlib.import_module("algos.arpo_rollout")
        lc = importlib.import_module("langchain_core.messages")
        tools = list(tools_override) if tools_override is not None else list(spec.tools)
        if llm.enabled_tools is not None:
            tools = [tool for tool in tools if tool in llm.enabled_tools]
        agent = tir.TirAgent.from_spec(
            spec, endpoint=llm.endpoint, model_name=llm.model, api_key=llm.api_key,
            temperature=llm.temperature, max_turns=llm.max_turns, max_tokens=llm.max_tokens,
            max_model_len=llm.max_model_len, request_logprobs=llm.request_logprobs,
            enabled_tools=tools, system_prompt=system_prompt,
            recorder=recorder, model_identity=llm.identity().model_dump(mode="json"),
            sampling_parameters=llm.sampling_parameters,
        )
        resume = tir._parse_resume_messages(task.get("resume_messages"))
        if task.get("resume_messages") and not resume:
            raise ValueError("resume_messages did not contain a valid message prefix")
        initial_messages = arpo.deserialize_messages(task.get("_initial_messages") or [])
        messages = resume or initial_messages or [
            lc.SystemMessage(content=agent.system_prompt), lc.HumanMessage(content=str(task["question"])),
        ]
        # Preserve the legacy prefix exactly. New context is only supplied to fresh entries.
        initial = {
            "question": str(task["question"]), "messages": messages,
            "num_turns": sum(isinstance(m, lc.AIMessage) for m in resume),
            "asked_finalize": False, "turn_records": [], "n_search": 0, "n_python": 0,
            "h_root": 0.0, "h_tool": 0.0, "consecutive_high": 0,
            "last_entropy": 0.0, "branch_messages": [],
        }
        cfg: Dict[str, Any] = {"recursion_limit": max(40, llm.max_turns * 4 + 4)}
        if llm.langchain_callbacks:
            cfg["callbacks"] = llm.langchain_callbacks
        stage = "model"
        if initial["num_turns"] >= llm.max_turns:
            result = {**initial, "termination_reason": "max_turns"}
        else:
            result = agent.graph().invoke(initial, cfg)
        result["termination_reason"] = agent.last_state.get("termination_reason", result.get("termination_reason"))
    except ArchiveWriteError:
        raise
    except Exception as exc:
        error = llm.public_error(exc)
        code = type(exc).__name__
        recorder.emit(EventKind.ERROR, {"error": error, "stage": stage, "error_code": code})
        result = (agent.last_state if agent else None) or initial
    messages = list(result.get("messages") or [])
    records = list(result.get("turn_records") or [])
    raw_text = tir.last_assistant_text(messages) if messages and agent is not None else None
    # Only the final assistant output can terminate an episode, not an older answer or tool text.
    prediction = tir.extract_answer_text(raw_text or "") if agent is not None else None
    reason = result.get("termination_reason") or ("model_error" if error else "missing_answer")
    if not error and (not prediction or reason not in {"answer"}):
        error = f"Episode terminated without a valid final answer ({reason})"
        stage = "termination"
        code = reason
    serialized = message_records(messages)
    return EpisodeRaw(
        messages=serialized, final_answer=prediction if not error else None,
        format_ok=bool(has_answer_format(raw_text)) and not error,
        n_search=int(result.get("n_search") or 0), n_python=int(result.get("n_python") or 0),
        h_root=float(result.get("h_root") or 0.0), h_tool=float(result.get("h_tool") or 0.0),
        consecutive_high=int(result.get("consecutive_high") or 0),
        branch_messages=list(result.get("branch_messages") or []),
        branch_event_id=result.get("branch_event_id"),
        turn_records=records, obs_hashes=[str(r.get("obs_hash") or "") for r in records],
        lc_messages=messages, error=error, error_stage=stage if error else None,
        error_code=code, termination_reason=str(reason),
        uncertainty_evidence=result.get("uncertainty_evidence", "unavailable_neutral_zero"),
    )


@runtime_checkable
class EpisodeRunner(Protocol):
    def run(
        self, task: Dict[str, Any], llm: Optional[LLMConfig], archive: Archive,
        spec: MASSpec, memory: MemoryStore,
    ) -> EpisodeRaw: ...


class MockRunner:
    def run(self, task, llm, archive, spec, memory) -> EpisodeRaw:
        return run_mock_episode(task, archive, spec=spec, memory=memory)


class TirRunner:
    def run(self, task, llm, archive, spec, memory) -> EpisodeRaw:
        if llm is None or not llm.endpoint:
            raise RuntimeError("Set LLMConfig.endpoint or pass mock=True")
        return run_episode(task, llm, archive, spec=spec, memory=memory,
                           system_prompt=spec.hub.system_prompt or None)


def _merge(previous: EpisodeRaw, current: EpisodeRaw) -> EpisodeRaw:
    """Keep all attempts while exposing the last attempt's outcome to legacy consumers."""
    if not current.messages:
        current.h_root = previous.h_root
        current.h_tool = previous.h_tool
        current.consecutive_high = previous.consecutive_high
        current.uncertainty_evidence = previous.uncertainty_evidence
    if not current.branch_messages:
        current.branch_messages = previous.branch_messages
        current.branch_event_id = previous.branch_event_id
    current.messages = previous.messages + current.messages
    current.n_search += previous.n_search
    current.n_python += previous.n_python
    current.turn_records = previous.turn_records + current.turn_records
    current.obs_hashes = previous.obs_hashes + current.obs_hashes
    current.skill_results = previous.skill_results + current.skill_results
    current.lc_messages = list(previous.lc_messages or []) + list(current.lc_messages or [])
    return current


def episode_to_trajectory(
    task: Dict[str, Any], raw: EpisodeRaw, archive: Archive, *, collector: str,
    spec: Optional[MASSpec] = None, memory: Optional[MemoryStore] = None,
) -> Trajectory:
    """Materialize the canonical trajectory without scoring it."""
    rid = str(task.get("_rollout_id") or archive.run_id or uuid4().hex)
    bps = []
    if raw.branch_messages or (raw.messages and not raw.error):
        source = next((e for e in archive._events if e.event_id == raw.branch_event_id), None)
        snap = archive.snapshot(
            messages=raw.branch_messages or raw.messages,
            event_id=raw.branch_event_id,
            meta={
                "reason": "post_tool" if raw.branch_messages else "episode_end",
                "h_root": raw.h_root, "h_tool": raw.h_tool,
                "consecutive_high": raw.consecutive_high,
                "uncertainty_evidence": raw.uncertainty_evidence,
                "arbitrary_graph_resume": False,
                "agent_id": source.agent_id if source else None,
                "agent_execution_id": source.agent_execution_id if source else None,
                "model_call_id": source.model_call_id if source else None,
            }, rollout_id=rid,
        )
        bps = [archive.to_branch_point(snap.snapshot_id, reason=str(snap.meta["reason"]))]
    meta = {
        "collector": collector, "run_id": archive.run_id, "rollout_id": rid,
        "h_root": raw.h_root, "h_tool": raw.h_tool, "consecutive_high": raw.consecutive_high,
        "uncertainty_evidence": raw.uncertainty_evidence,
        "memory": [item.model_dump(mode="json") for item in memory.items()] if memory else [],
        "tools": list((spec or load_spec()).tools),
        "skills": [row.get("skill") for row in raw.skill_results],
        "status": "failed" if raw.error else "succeeded",
        "termination_reason": raw.termination_reason,
        "trace_status": "partial" if raw.error else "complete",
        "model_call_count": sum(e.kind == EventKind.MODEL_CALL for e in archive._events),
        "tool_call_count": sum(e.kind == EventKind.TOOL_CALL for e in archive._events),
    }
    routes = [r["route"] for r in raw.skill_results if r.get("route")]
    if routes:
        meta["skill_route"] = routes[-1]
    if raw.error:
        meta.update(error=raw.error, error_stage=raw.error_stage or "execution",
                    error_code=raw.error_code or "execution_failed")
    traj = Trajectory(
        trajectory_id=archive.trajectory_id or uuid4().hex, task=archive.redact(task),
        messages=archive.redact(raw.messages), final_answer=archive.redact(raw.final_answer),
        format_ok=raw.format_ok, n_search=raw.n_search, n_python=raw.n_python,
        archive=archive.ref(), branch_points=bps, events=list(archive._events),
        branch_parent_id=str(task.get("resume_parent_id") or "") or None,
        meta=archive.redact(meta),
    )
    traj.sync_tir_meta()
    return traj


class ExecutionService:
    """Compile once, execute one task with isolated memory, expose partial failures."""

    def __init__(
        self, *, mock: bool = False, spec: Optional[MASSpec] = None,
        spec_path: Optional[str] = None, llm: Optional[LLMConfig] = None,
        archive_root: Optional[str] = None, memory: Optional[MemoryStore] = None,
        runner: Optional[EpisodeRunner] = None, run_id: Optional[str] = None,
    ) -> None:
        self.mock = mock
        self.spec = (spec or load_spec(spec_path)).model_copy(deep=True)
        self.llm = llm
        self.archive_root = archive_root
        self.memory = memory or MemoryStore()
        self.runner = runner or (MockRunner() if mock else TirRunner())
        self.run_id = run_id or uuid4().hex
        self._requested_run_id = run_id
        self.last_raw: Optional[EpisodeRaw] = None
        self.last_archive: Optional[Archive] = None
        self.compiled = compile_spec(self.spec.model_copy(deep=True))
        self.mode = "hub_react" if self.spec.topology in {"hub_react", "single", ""} else (
            "graph_hub_subset" if self.compiled.hub_subset else "graph_compiled"
        )
        self._validate()

    def _validate(self) -> None:
        spec, compiled = self.spec, self.compiled
        if not compiled.ok:
            raise ValueError(f"Workflow is not executable: {compiled.reason}")
        if spec.memory.agent not in {"messages", "none"} or spec.memory.system not in {"none", "kv"}:
            raise ValueError("Supported memory backends: agent=messages|none, system=none|kv")
        ids = [node.id for node in spec.agents]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate Agent IDs are not executable")
        if self.mode != "graph_compiled":
            if spec.entry_agent != "hub" or any(a.id not in {"hub", "verifier"} for a in spec.agents):
                raise ValueError("Hub compatibility mode only executes hub and optional verifier; use graph topology")
            if any(
                (edge.source, edge.target, edge.kind) not in {
                    ("hub", "verifier", "message"), ("hub", "verifier", "route"),
                    ("verifier", "hub", "feedback"),
                }
                for edge in spec.edges if edge.kind != "tool_call"
            ):
                raise ValueError("Unsupported handoff in hub compatibility mode")
        outgoing: Dict[str, int] = {}
        for edge in spec.edges:
            if edge.kind in {"route", "message", "feedback"}:
                key = edge.source + (":feedback" if edge.kind == "feedback" else ":forward")
                outgoing[key] = outgoing.get(key, 0) + 1
                if outgoing[key] > 1:
                    raise ValueError(f"Multiple outgoing {key} edges are not supported")
        reachable = {compiled.entry_agent}
        while True:
            expanded = reachable | {
                edge.target for edge in spec.edges
                if edge.source in reachable and edge.kind != "tool_call"
            }
            if spec.hub.verify and "hub" in reachable and self.mode != "graph_compiled":
                expanded.add("verifier")
            if expanded == reachable:
                break
            reachable = expanded
        unused = set(ids) - reachable
        if unused:
            raise ValueError(f"Unreachable configured Agents: {sorted(unused)}")
        if set(spec.tools) - KNOWN_TOOLS:
            raise ValueError(f"Unknown tools: {sorted(set(spec.tools) - KNOWN_TOOLS)}")
        if self.llm and self.llm.enabled_tools is not None and set(self.llm.enabled_tools) - KNOWN_TOOLS:
            raise ValueError("Model binding contains unsupported tools")
        for node in compiled.agents.values():
            if node.role not in KNOWN_ROLES:
                raise ValueError(f"Unsupported Agent role: {node.role}")
            if node.model not in {"", "inherit"}:
                raise ValueError("Agent-specific model bindings are not implemented")
            if node.memory_scope not in {"agent", "system", "none"}:
                raise ValueError(f"Unsupported memory scope: {node.memory_scope}")
            if node.memory_scope == "system" and spec.memory.system == "none":
                raise ValueError("Agent requests system memory but system memory is disabled")
            for skill in node.skills:
                REGISTRY.get_skill(skill)
            unknown = set(node.tools) - KNOWN_TOOLS
            if unknown:
                raise ValueError(f"Unknown tools: {sorted(unknown)}")
        for skill in spec.hub.skills:
            REGISTRY.get_skill(skill)
        if spec.hub.verify:
            REGISTRY.get_skill(spec.hub.verify)

    def _tools(self, agent_id: str) -> List[str]:
        node = next((a for a in self.spec.agents if a.id == agent_id), None)
        # Explicit [] disables tools. Only an absent legacy hub node inherits global tools.
        names = list(node.tools) if node is not None else (list(self.spec.tools) if agent_id == "hub" else [])
        for edge in self.spec.edges:
            if edge.source == agent_id and edge.kind == "tool_call" and edge.target not in names:
                names.append(edge.target)
        if self.llm is not None and self.llm.enabled_tools is not None:
            names = [name for name in names if name in self.llm.enabled_tools]
        return names

    def _skills(
        self, recorder: BoundaryRecorder, names: List[str], ctx: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        rows = []
        for name in names:
            call = recorder.emit(EventKind.SKILL_CALL, {"skill": name, "context": {
                key: value for key, value in ctx.items() if key != "spec"
            }})
            try:
                row = invoke_skill(name, ctx)
            except ArchiveWriteError:
                raise
            except Exception as exc:
                recorder.emit(EventKind.SKILL_RESULT, {
                    "skill": name, "ok": False, "error": str(exc),
                }, parent_id=call.event_id)
                raise
            recorder.emit(EventKind.SKILL_RESULT, row, parent_id=call.event_id)
            rows.append(row)
        return rows

    def _execute(
        self, task: Dict[str, Any], archive: Archive, agent_id: str, *,
        previous: Optional[EpisodeRaw] = None, handoff_from: Optional[str] = None,
        feedback: Optional[Any] = None, hop: int = 0,
    ) -> tuple[EpisodeRaw, str]:
        node = self.compiled.agents[agent_id]
        recorder = BoundaryRecorder(
            archive, run_id=self.run_id, trajectory_id=archive.trajectory_id,
            agent_id=agent_id, agent_execution_id=uuid4().hex,
        )
        archive.active_recorder = recorder
        verifier = node.role in {"verifier", "critic"}
        names = list(node.skills)
        if agent_id == "hub" and not any(a.id == "hub" for a in self.spec.agents):
            names = list(self.spec.hub.skills)
        if verifier and not names:
            names = [self.spec.hub.verify or "verifier"]
        tools = self._tools(agent_id)
        prompt = node.system_prompt or (self.spec.hub.system_prompt if agent_id == "hub" else "")
        skill_only = verifier and not prompt and not tools
        if not prompt and not skill_only:
            if self.mock:
                prompt = "Mock tool-using assistant. Return <answer>...</answer>."
            else:
                prompt = importlib.import_module("tir_agent").default_system_prompt(tools)
        tool_definitions = []
        if not self.mock and not skill_only:
            tir = importlib.import_module("tir_agent")
            tool_definitions = [tir.convert_to_openai_tool(tir.TOOL_MAP[name]) for name in tools]
        context = AgentExecutionContext(
            run_id=self.run_id, trajectory_id=archive.trajectory_id,
            agent_id=agent_id, agent_execution_id=recorder.agent_execution_id,
            model=self.llm.identity() if self.llm and not self.mock and not skill_only else None,
            system_prompt=prompt, skills=names, handoff_from_execution_id=handoff_from,
            tool_definitions=tool_definitions,
            sampling_parameters={
                **(self.llm.sampling_parameters if self.llm else {}),
                **({"temperature": self.llm.temperature, "max_tokens": self.llm.max_tokens} if self.llm else {}),
            },
        )
        recorder.emit(EventKind.AGENT_ENTER, {
            "phase": "enter", "role": node.role, "hop": hop,
            "execution_kind": "skill_only" if skill_only else ("mock" if self.mock else "model"),
            "context": context.model_dump(mode="json"), "enabled_tools": tools,
            "context_stage": "entry_before_skills_and_context_fitting",
        })
        raw = EpisodeRaw(messages=[])
        stage = "memory"
        try:
            items = []
            if node.memory_scope == "agent" and self.spec.memory.agent != "none":
                items = self.memory.read("agent", agent_id)
            elif node.memory_scope == "system":
                items = self.memory.read("system", "mas")
            recorder.emit(EventKind.MEMORY_READ, {
                "scope": node.memory_scope, "owner": agent_id if node.memory_scope == "agent" else "mas",
                "n": len(items), "items": [item.model_dump(mode="json") for item in items],
            })
            ctx = {
                "task": {key: task[key] for key in ("id", "question", "source", "split") if key in task},
                "agent_id": agent_id, "agent_execution_id": recorder.agent_execution_id,
                "spec": self.spec, "memory_agent": [item.model_dump(mode="json") for item in items],
                "final_answer": previous.final_answer if previous else None,
                "format_ok": previous.format_ok if previous else False,
                "error": previous.error if previous else None, "hop": hop,
            }
            stage = "skill"
            rows = self._skills(recorder, names, ctx) if not verifier or skill_only else []
            raw.skill_results = rows
            if skill_only:
                raw.final_answer = previous.final_answer if previous else None
                raw.format_ok = previous.format_ok if previous else False
                rejected = any(not row["ok"] for row in rows)
                raw.error = "Verifier rejected the result" if rejected else None
                raw.termination_reason = "verifier_rejected" if rejected else "verified"
                raw.error_stage = "skill" if rejected else None
            else:
                if any(not row["ok"] for row in rows):
                    raise RuntimeError("Agent entry skill failed")
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": str(task.get("question") or "")},
                ]
                if previous and previous.messages:
                    messages.append({"role": "user", "content": "[Upstream Agent output]\n" + str(
                        previous.final_answer or previous.messages[-1].get("content") or ""
                    )})
                if feedback is not None:
                    messages.append({"role": "user", "content": "[Verifier feedback]\n" + json.dumps(feedback, ensure_ascii=False)})
                if items:
                    messages.append({"role": "user", "content": "[Run memory]\n" + json.dumps(
                        [item.model_dump(mode="json") for item in items], ensure_ascii=False,
                    )})
                outputs = [row for row in rows if row.get("output") is not None]
                if outputs:
                    messages.append({"role": "user", "content": "[Skill results]\n" + json.dumps(outputs, ensure_ascii=False)})
                sub_spec = self.spec.model_copy(deep=True)
                sub_spec.tools = tools
                sub_spec.hub = sub_spec.hub.model_copy(update={"system_prompt": prompt, "skills": names})
                task_i = dict(task, _initial_messages=messages, _feedback_hop=hop)
                if handoff_from:
                    task_i.pop("resume_messages", None)
                    task_i.pop("resume_from", None)
                stage = "execution"
                raw = self.runner.run(task_i, self.llm, archive, sub_spec, self.memory)
                raw.skill_results = rows + raw.skill_results
                if verifier and not raw.error:
                    stage = "skill"
                    ctx.update(final_answer=raw.final_answer, format_ok=raw.format_ok, error=raw.error)
                    rows = self._skills(recorder, names, ctx)
                    raw.skill_results += rows
                    if any(not row["ok"] for row in rows):
                        raw.error = "Verifier rejected the result"
                        raw.termination_reason = "verifier_rejected"
                        raw.error_stage = "skill"
            stage = "memory"
            if node.memory_scope == "agent" and self.spec.memory.agent != "none":
                item = self.memory.write(MemoryItem(
                    scope="agent", owner=agent_id, content={
                        "messages": raw.messages, "final_answer": raw.final_answer, "error": raw.error,
                    },
                ))
                recorder.emit(EventKind.MEMORY_WRITE, {"scope": "agent", "owner": agent_id, "item": item.model_dump(mode="json")})
            if self.spec.memory.system != "none" and not raw.error:
                item = self.memory.write(MemoryItem(
                    scope="system", owner="mas",
                    content={"agent_id": agent_id, "final_answer": raw.final_answer},
                ))
                recorder.emit(EventKind.MEMORY_WRITE, {"scope": "system", "owner": "mas", "item": item.model_dump(mode="json")})
        except ArchiveWriteError:
            raise
        except Exception as exc:
            raw.error = archive.redact(str(exc))
            raw.error_stage = stage
            raw.error_code = type(exc).__name__
            raw.termination_reason = f"{stage}_error"
            recorder.emit(EventKind.ERROR, {
                "error": raw.error, "stage": stage, "error_code": raw.error_code,
            })
        if raw.error:
            raw.error = archive.redact(raw.error)
        recorder.emit(EventKind.AGENT_EXIT, {
            "status": "failed" if raw.error else "succeeded", "error": raw.error,
            "termination_reason": raw.termination_reason, "final_answer": raw.final_answer,
        })
        return raw, recorder.agent_execution_id

    def run(self, task: Dict[str, Any]) -> Trajectory:
        self.run_id = self._requested_run_id or uuid4().hex
        self.memory.clear()
        self.last_raw = None
        archive = register_archive(Archive(root_dir=self.archive_root))
        self.last_archive = archive
        archive.run_id = self.run_id
        archive.trajectory_id = uuid4().hex
        archive.redact = Redactor([self.llm.api_key] if self.llm and self.llm.api_key else [])
        started = datetime.now(timezone.utc)
        task_run = dict(task)
        archive.append(ExecutionEvent(kind=EventKind.TASK_START, payload={
            "task": task_run, "execution": "mock" if self.mock else "live", "execution_mode": self.mode,
        }))
        aggregate = EpisodeRaw(messages=[])
        current = self.compiled.entry_agent
        last_execution = None
        feedback = None
        feedback_hops = 0
        resume_point = None
        max_steps = max(4, (len(self.compiled.agents) + 1) * (self.spec.hub.max_feedback_hops + 1))
        try:
            if task_run.get("resume_from") or task_run.get("resume_messages"):
                if self.mode == "graph_compiled":
                    raise ValueError("Message-only resume is supported only by the legacy hub runtime, not arbitrary graphs")
                if task_run.get("resume_from"):
                    bp = task_run["resume_from"]
                    bp = bp if isinstance(bp, BranchPoint) else BranchPoint.model_validate(bp)
                    task_run.update(branch_point_to_resume_task_fields(bp))
                    resume_point = bp
            for step in range(max_steps):
                raw, execution = self._execute(
                    task_run, archive, current, previous=aggregate if last_execution else None,
                    handoff_from=last_execution, feedback=feedback, hop=feedback_hops,
                )
                aggregate = _merge(aggregate, raw)
                self.last_raw = aggregate
                node = self.compiled.agents[current]
                verifier = node.role in {"verifier", "critic"}
                if verifier:
                    rows = raw.skill_results
                    rejected = raw.error is not None
                    target = self.compiled.feedback_to.get(current)
                    if self.mode != "graph_compiled":
                        target = "hub"
                    retry = rejected and raw.termination_reason == "verifier_rejected" and bool(target)
                    retry = retry and feedback_hops < max(0, self.spec.hub.max_feedback_hops)
                    archive.active_recorder.emit(EventKind.FEEDBACK, {
                        "skill": rows[-1].get("skill") if rows else None, "ok": not rejected,
                        "route": target if rejected else None, "hop": feedback_hops, "rerun": retry,
                    })
                    if retry:
                        feedback_hops += 1
                        feedback = rows[-1] if rows else {"error": raw.error}
                        nxt = target
                    elif rejected:
                        if target and feedback_hops >= self.spec.hub.max_feedback_hops:
                            aggregate.termination_reason = "feedback_limit"
                        break
                    else:
                        nxt = next_agent(self.compiled, current) if self.mode == "graph_compiled" else None
                elif raw.error:
                    # Verifiers may repair a missing answer, never conceal provider/runtime failures.
                    if self.mock and self.spec.hub.verify and self.mode != "graph_compiled":
                        nxt = "verifier"
                    else:
                        break
                elif self.mode == "graph_compiled":
                    nxt = next_agent(self.compiled, current)
                else:
                    nxt = "verifier" if self.spec.hub.verify or next_agent(self.compiled, current) == "verifier" else None
                if not nxt:
                    break
                archive.active_recorder.emit(EventKind.HANDOFF, {
                    "from_agent": current, "to_agent": nxt, "from_execution_id": execution,
                    "output": raw.final_answer, "feedback": feedback,
                })
                current, last_execution = nxt, execution
            else:
                aggregate.error = "Workflow reached its execution step limit"
                aggregate.error_stage = "orchestration"
                aggregate.termination_reason = "step_limit"
        except ArchiveWriteError:
            raise
        except Exception as exc:
            aggregate.error = archive.redact(str(exc))
            aggregate.error_stage = "orchestration"
            aggregate.error_code = type(exc).__name__
            aggregate.termination_reason = "orchestration_error"
            archive.append(ExecutionEvent(kind=EventKind.ERROR, payload={
                "stage": aggregate.error_stage, "error": aggregate.error, "error_code": aggregate.error_code,
            }))
        if not aggregate.error and not aggregate.final_answer:
            aggregate.error = "Workflow produced no final answer"
            aggregate.error_stage = "termination"
            aggregate.termination_reason = "missing_answer"
        if aggregate.error:
            aggregate.error_code = aggregate.error_code or aggregate.termination_reason
        if not aggregate.error:
            archive.append(ExecutionEvent(kind=EventKind.FINAL_ANSWER, payload={"text": aggregate.final_answer}))
        archive.append(ExecutionEvent(kind=EventKind.TERMINATION, payload={
            "status": "failed" if aggregate.error else "succeeded",
            "termination_reason": aggregate.termination_reason, "error": aggregate.error,
        }))
        self.last_raw = aggregate
        trajectory = episode_to_trajectory(
            task_run, aggregate, archive, collector="mock" if self.mock else "tir_agent",
            spec=self.spec, memory=self.memory,
        )
        if resume_point is not None:
            trajectory.resume_from = BranchPoint.model_validate(archive.redact(resume_point.model_dump(mode="json")))
        trajectory.meta.update(archive.redact({
            "workflow": self.spec.model_dump(mode="json"), "execution": "mock" if self.mock else "live",
            "execution_mode": self.mode, "compiled": self.compiled.reason,
            "entry_agent": self.compiled.entry_agent, "verify": self.spec.hub.verify,
            "feedback_hops": feedback_hops, "started_at": started.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "attempt_id": task.get("_attempt_id"),
            "snapshot_resume_scope": "messages_only", "arbitrary_graph_resume": False,
            "recording_limitations": {
                "transport_headers": False, "external_tool_state": False,
                "hidden_reasoning": False, "exact_tokenization": False,
            },
        }))
        if self.llm is not None and not self.mock:
            identity = self.llm.identity().model_dump(mode="json")
            results = [e.payload for e in trajectory.events if e.kind == EventKind.MODEL_RESULT]
            identity["capabilities"]["logprobs"] = "available" if any(r.get("logprobs") is not None for r in results) else "unknown"
            identity["capabilities"]["usage"] = "available" if any(r.get("usage") is not None for r in results) else "unknown"
            trajectory.meta["model_identity"] = archive.redact(identity)
            trajectory.meta["actual_models"] = list(dict.fromkeys(r["model"] for r in results if r.get("model")))
        trajectory.meta["redaction"] = {
            "applied": archive.redact.redacted, "scope": "credentials", "content_truncation": False,
        }
        resumed = bool(task_run.get("resume_messages") or task_run.get("resume_from"))
        model_entries = {
            event.agent_execution_id for event in trajectory.events
            if event.kind == EventKind.AGENT_ENTER and event.payload.get("execution_kind") == "model"
        }
        recorded_entries = {
            event.agent_execution_id for event in trajectory.events if event.kind == EventKind.MODEL_CALL
        }
        missing_boundaries = bool(model_entries - recorded_entries)
        trajectory.meta["trace_coverage"] = {
            "current_calls": "synthetic_mock" if self.mock else "boundary_recorded",
            "historical_prefix": "messages_only" if resumed else "not_applicable",
            "missing_model_boundaries": missing_boundaries,
        }
        if self.mock or resumed or archive.redact.redacted or missing_boundaries:
            trajectory.meta["trace_status"] = "partial"
        # Archive is the durable core output even when no HTTP/Collector adapter is present.
        if archive.root_dir:
            from pathlib import Path

            path = Path(archive.root_dir) / archive.archive_id / "trajectory.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            pending = path.with_suffix(".json.pending")
            pending.write_text(trajectory.model_dump_json(indent=2), encoding="utf-8")
            pending.replace(path)
        return trajectory

    def fork(self, branch_point: BranchPoint, task: Optional[Dict[str, Any]] = None) -> Trajectory:
        return self.run(dict(task or {}, resume_from=branch_point))

    def collect(self, tasks: Sequence[Dict[str, Any]]) -> List[Trajectory]:
        return [self.run(task) for task in tasks]


def validate_execution_spec(spec: MASSpec) -> None:
    """Validate shared runtime support without executing, creating archives or loading models."""
    ExecutionService(mock=True, spec=spec)
