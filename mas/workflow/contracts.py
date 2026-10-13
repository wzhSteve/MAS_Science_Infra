"""Strict schemas between workflow data layer and RL / Harness (no AGL)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventKind(str, Enum):
    TASK_START = "task_start"
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    SNAPSHOT = "snapshot"
    ERROR = "error"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    FEEDBACK = "feedback"
    ON_TOKEN = "on_token"


class ExecutionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    parent_id: Optional[str] = None
    agent_id: str = "hub"
    kind: EventKind
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=_utcnow)

    model_config = {"extra": "forbid"}


class Snapshot(BaseModel):
    """Barrier sufficient to resume TirAgent via resume_messages / token_prefix.

    ``schema_version`` (Phase 2, docs/refactor/phase0_contract_freeze.md §4):
    persisted-contract versioning. Old snapshots written before this field
    deserialize with the default ``1`` (Pydantic back-compat).
    """

    snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    archive_id: str
    event_id: str = ""
    schema_version: int = 1
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    token_ids: Optional[List[int]] = None
    meta: Dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=_utcnow)

    model_config = {"extra": "forbid"}


class ArchiveRef(BaseModel):
    archive_id: str
    root_dir: Optional[str] = None
    n_events: int = 0
    n_snapshots: int = 0
    # Map rollout_id -> latest snapshot_id for ARPO daemon
    rollout_snapshots: Dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class BranchPoint(BaseModel):
    archive_id: str
    snapshot_id: str
    beam_size: int = 1
    reason: str = ""
    parent_rollout_id: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class BranchAnchor(BaseModel):
    """Where on the MAS graph / episode a branch site may fire."""

    kind: str = "after_tool"  # after_tool | after_agent_turn | after_verifier | on_edge | on_token
    agent_id: Optional[str] = None
    tool_id: Optional[str] = None
    edge_id: Optional[str] = None
    skill_id: Optional[str] = None

    model_config = {"extra": "forbid"}


class BranchGate(BaseModel):
    """Whether to expand at a matched site."""

    type: str = "entropy_delta"
    # entropy_delta | dual_entropy | always | tool_ok | tool_error
    # | verifier_pass | verifier_fail | contradiction | failure_trigger | signal
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class BranchForkSpec(BaseModel):
    beam_size: Optional[int] = None
    share_observation: bool = True
    resume_mode: str = "messages"  # messages | token_prefix
    probe_max_tokens: int = 128

    model_config = {"extra": "forbid"}


class BranchSiteReward(BaseModel):
    scheme: str = "scalar_grpo"  # scalar_grpo | rae_adjudicate
    p_plus: float = 0.8
    k_min: int = 2
    epsilon_f: float = 1e-3
    dead_end_backprop: int = 1

    model_config = {"extra": "forbid"}


class BranchSite(BaseModel):
    """UI-declared branch rollout site (first-class SamplePolicy entry)."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    enabled: bool = True
    anchor: BranchAnchor = Field(default_factory=BranchAnchor)
    when: str = "first"  # first | every | nth
    nth: int = 1
    gate: BranchGate = Field(default_factory=BranchGate)
    fork: BranchForkSpec = Field(default_factory=BranchForkSpec)
    reward: BranchSiteReward = Field(default_factory=BranchSiteReward)
    priority: int = 10

    model_config = {"extra": "forbid"}


def barriers_to_default_sites(barriers: List[str]) -> List[BranchSite]:
    """Map legacy SamplePolicy.barriers strings into BranchSite list."""
    out: List[BranchSite] = []
    for i, b in enumerate(barriers or []):
        kind = str(b or "after_tool").strip() or "after_tool"
        out.append(
            BranchSite(
                id=f"legacy_{kind}_{i}",
                enabled=True,
                anchor=BranchAnchor(kind=kind),
                gate=BranchGate(type="entropy_delta"),
                priority=10 + i,
            )
        )
    return out


class SamplePolicy(BaseModel):
    """MAS-declared multi-sampling policy (executed by RL daemon / Collector)."""

    mode: str = "grpo_n"  # grpo_n | arpo | aepo | appo | rae
    group_n: int = 1
    beam_size: int = 1
    initial_rollouts: int = 1
    max_branch_depth: int = 2
    expand_in_runner: bool = True
    barriers: List[str] = Field(default_factory=lambda: ["after_tool"])
    sites: List[BranchSite] = Field(default_factory=list)
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def resolved_sites(self) -> List[BranchSite]:
        """Explicit sites supersede legacy barriers, including when all are disabled."""
        if self.sites:
            return sorted((s for s in self.sites if s.enabled), key=lambda s: (-int(s.priority), s.id))
        return barriers_to_default_sites(list(self.barriers))


class MemoryItem(BaseModel):
    """Two-layer memory contract. Phase 0: agent=messages, system unused."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    scope: str = "agent"  # agent | system
    owner: str = "hub"
    content: Any = None

    model_config = {"extra": "forbid"}


class Trajectory(BaseModel):
    trajectory_id: str = Field(default_factory=lambda: uuid4().hex)
    task: Dict[str, Any] = Field(default_factory=dict)
    events: List[ExecutionEvent] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    final_answer: Optional[str] = None
    final_reward: Optional[float] = None
    format_ok: bool = False
    n_search: int = 0
    n_python: int = 0
    archive: Optional[ArchiveRef] = None
    branch_points: List[BranchPoint] = Field(default_factory=list)
    branch_parent_id: Optional[str] = None
    resume_from: Optional[BranchPoint] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    def sync_tir_meta(self) -> None:
        """Mirror TIR counters into meta (stable extension surface)."""
        self.meta = dict(self.meta or {})
        self.meta["format_ok"] = bool(self.format_ok)
        self.meta["n_search"] = int(self.n_search)
        self.meta["n_python"] = int(self.n_python)


class TrajectoryBatch(BaseModel):
    trajectories: List[Trajectory] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


# --- RolloutTree contract (new_framework P0) ---------------------------------


class ExecutionFailure(BaseModel):
    stage: Literal["graph_build", "graph_execution", "result_processing"]
    error_type: str = Field(max_length=128)
    message: str = Field(max_length=1000)

    model_config = {"extra": "forbid"}


class RolloutTreeNode(BaseModel):
    """One node of a per-query rollout tree (root=query, leaves=outcomes)."""

    node_id: str  # rollout_id or synthetic f"{parent}:{i}"
    parent_id: Optional[str] = None  # None = root (query level)
    depth: int = 0
    role: str = "root"  # root | child | probe
    agent_path: List[str] = Field(default_factory=list)
    boundary_snapshot_ref: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    reward: Optional[float] = None  # node-level credit (P2)
    verdict: Optional[str] = None  # RAE validate | invalidate | abstain (P2)
    kind: Literal["query", "rollout", "execution", "outcome"] = "rollout"
    rollout_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_kind: Optional[Literal["agent", "tool"]] = None
    turn: Optional[int] = None
    summary: Optional[str] = None
    detail_ref: Optional[str] = None
    result_node_id: Optional[str] = None
    origin: Optional[Literal["initial", "independent_fill", "branch"]] = None
    attempt_id: Optional[str] = None
    status: Optional[Literal["enqueued", "running", "succeeded", "failed", "cancelled"]] = None
    created_at: Optional[datetime] = None
    plan_id: Optional[str] = None
    site_id: Optional[str] = None
    window_id: Optional[str] = None
    event_id: Optional[str] = None
    branch_depth: Optional[int] = None
    decision: Dict[str, Any] = Field(default_factory=dict)
    attempt_sequence: int = 0
    store_status: Optional[str] = None
    attempt_status: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    execution_error: Optional[str] = None
    training_status: Optional[Literal["adapted", "empty", "accepted", "rejected"]] = None
    record_issues: List[str] = Field(default_factory=list)
    judgments: List[Dict[str, Any]] = Field(default_factory=list)
    previous_attempts: List[Dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class RolloutTreePlan(BaseModel):
    plan_id: str
    parent_id: str
    site_id: Optional[str] = None
    window_id: Optional[str] = None
    event_id: Optional[str] = None
    boundary_snapshot_ref: Optional[str] = None
    branch_depth: Optional[int] = None
    role: str = "child"
    decision: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["planned", "enqueued", "skipped"] = "planned"
    reason: Optional[str] = None
    child_rollout_id: Optional[str] = None
    source_attempt_id: Optional[str] = None
    fork_node_ids: List[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}


class RolloutTreeEdge(BaseModel):
    source_node_id: str
    target_node_id: str
    kind: Literal["sequence", "branch"] = "sequence"
    window_id: Optional[str] = None
    snapshot_ref: Optional[str] = None
    site_id: Optional[str] = None

    model_config = {"extra": "forbid"}


class ExecutionTrace(BaseModel):
    """One runner-owned attempt index; message bodies live in node detail files."""

    rollout_id: str
    attempt_id: str
    nodes: List[RolloutTreeNode] = Field(default_factory=list)
    edges: List[RolloutTreeEdge] = Field(default_factory=list)
    entry_ids: List[str] = Field(default_factory=list)
    tail_ids: List[str] = Field(default_factory=list)
    windows: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    resume: Dict[str, Any] = Field(default_factory=dict)
    resume_applied: bool = False
    complete: bool = False
    errors: List[str] = Field(default_factory=list)
    revision: int = 0

    model_config = {"extra": "forbid"}


class RolloutTree(BaseModel):
    """Explicit per-query branch tree; supersedes scattered ForkPlan.meta fields."""

    tree_id: str  # data_id (+ group)
    query: str = ""
    nodes: List[RolloutTreeNode] = Field(default_factory=list)
    outcomes: Dict[str, Any] = Field(default_factory=dict)  # rollout_id -> result, including parents
    schema_version: int = 1
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    group_id: Optional[str] = None
    task_id: Optional[str] = None
    mode: Optional[Literal["train", "val"]] = None
    revision: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: Optional[datetime] = None
    plans: List[RolloutTreePlan] = Field(default_factory=list)
    pending_nodes: List[RolloutTreeNode] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)
    edges: List[RolloutTreeEdge] = Field(default_factory=list)
    rollouts: List[RolloutTreeNode] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_structure(self) -> "RolloutTree":
        if self.schema_version == 1:
            return self
        if self.schema_version not in (2, 3):
            raise ValueError("Unsupported rollout tree version")
        if not all((self.experiment_id, self.run_id, self.group_id, self.mode)):
            raise ValueError("Version 2 trees require run and group identity")
        by_id = {node.node_id: node for node in self.nodes}
        all_ids = [node.node_id for node in [*self.nodes, *self.pending_nodes]]
        if len(set(all_ids)) != len(all_ids) or any(not value for value in all_ids):
            raise ValueError("Rollout node IDs must be nonempty and unique")
        roots = [node for node in self.nodes if node.kind == "query"]
        if len(roots) != 1:
            raise ValueError("A rollout tree requires exactly one query root")
        root = roots[0]
        if root.parent_id is not None or root.depth != 0 or any(
            value is not None for value in (root.status, root.reward, root.attempt_id, root.origin)
        ):
            raise ValueError("Query root cannot carry execution state")
        if self.schema_version == 3:
            self._validate_executions(by_id)
            return self
        for node in self.nodes:
            if node is root:
                continue
            parent = by_id.get(node.parent_id)
            if parent is None or node.depth != parent.depth + 1:
                raise ValueError("Invalid parent or depth (including cyclic ancestry)")
            if node.origin == "branch":
                if parent.kind != "rollout":
                    raise ValueError("Branch parent must be a rollout")
            elif node.origin not in ("initial", "independent_fill") or parent is not root:
                raise ValueError("Independent rollout must belong to the query root")
        pending = {node.node_id: node for node in self.pending_nodes}
        for node in self.pending_nodes:
            if node.kind != "rollout" or node.origin != "branch" or not node.parent_id:
                raise ValueError("Only unresolved branches can be pending")
            seen = {node.node_id}
            current = node.parent_id
            while current in pending:
                if current in seen:
                    raise ValueError("Cyclic pending lineage")
                seen.add(current)
                current = pending[current].parent_id
        plans = {plan.plan_id: plan for plan in self.plans}
        if len(plans) != len(self.plans):
            raise ValueError("Duplicate rollout plan identity")
        all_nodes = {**by_id, **pending}
        for plan in self.plans:
            if plan.status == "enqueued":
                child = all_nodes.get(plan.child_rollout_id)
                if child is None or child.plan_id != plan.plan_id:
                    raise ValueError("Enqueued plan requires its actual rollout")
            elif plan.child_rollout_id is not None:
                raise ValueError("Unsubmitted plan cannot have an actual rollout")
        for node in [*self.nodes, *self.pending_nodes]:
            if node.plan_id:
                plan = plans.get(node.plan_id)
                if plan is None or plan.parent_id != node.parent_id or plan.child_rollout_id != node.node_id:
                    raise ValueError("Rollout does not match its submitted plan")
        return self

    def _validate_executions(self, by_id: Dict[str, RolloutTreeNode]) -> None:
        from collections import deque

        records = {record.node_id: record for record in self.rollouts}
        if len(records) != len(self.rollouts) or any(r.kind != "rollout" for r in self.rollouts):
            raise ValueError("Execution trees require unique rollout records")
        for node in self.nodes:
            if node.kind == "query":
                continue
            if node.kind not in ("execution", "outcome") or node.rollout_id not in records or not node.attempt_id:
                raise ValueError("Execution node is missing rollout/attempt identity")
        incoming = dict.fromkeys(by_id, 0)
        children: Dict[str, List[str]] = {key: [] for key in by_id}
        edges = set()
        for edge in self.edges:
            source, target = edge.source_node_id, edge.target_node_id
            if source not in by_id or target not in by_id or (source, target) in edges:
                raise ValueError("Invalid or duplicate execution edge")
            if by_id[target].kind == "query" or by_id[source].kind == "outcome":
                raise ValueError("Query and outcome must be path endpoints")
            if edge.kind == "branch" and (
                not edge.window_id or not edge.snapshot_ref or by_id[source].kind != "execution"
            ):
                raise ValueError("A branch requires a real execution window and snapshot")
            edges.add((source, target))
            children[source].append(target)
            incoming[target] += 1
        ready = deque(key for key, count in incoming.items() if count == 0)
        visited = 0
        while ready:
            visited += 1
            for child in children[ready.popleft()]:
                incoming[child] -= 1
                if incoming[child] == 0:
                    ready.append(child)
        if visited != len(by_id):
            raise ValueError("Cyclic execution graph")

    def leaves(self) -> List[str]:
        if self.schema_version == 3:
            return [n.node_id for n in self.nodes if n.kind == "outcome"]
        parents = {n.parent_id for n in self.nodes if n.parent_id}
        return [n.node_id for n in self.nodes if n.node_id not in parents]

    def path_to_root(self, node_id: str) -> List[str]:
        if self.schema_version == 3:
            parents: Dict[str, List[str]] = {}
            for edge in self.edges:
                parents.setdefault(edge.target_node_id, []).append(edge.source_node_id)
            out: List[str] = []
            stack = [node_id]
            seen: set[str] = set()
            while stack:
                current = stack.pop()
                if current not in seen:
                    seen.add(current)
                    out.append(current)
                    stack.extend(reversed(parents.get(current, [])))
            return out
        by_id = {n.node_id: n for n in self.nodes}
        out: List[str] = []
        cur: Optional[str] = node_id
        seen = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            out.append(cur)
            node = by_id.get(cur)
            cur = node.parent_id if node else None
        return out  # [node, ..., root]

    def add_node(self, node: "RolloutTreeNode") -> None:
        self.nodes.append(node)

    def rollout_records(self) -> List[RolloutTreeNode]:
        if self.schema_version == 3:
            return self.rollouts
        return [node for node in [*self.nodes, *self.pending_nodes] if node.kind == "rollout"]


class RolloutTreeEvent(BaseModel):
    """Streamed tree update for real-time Harness / UI (new_framework P3)."""

    event: str = "node_added"  # node_added | outcome | reward | loss
    tree_id: str
    node_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1
    experiment_id: Optional[str] = None
    run_id: Optional[str] = None
    revision: Optional[int] = None

    model_config = {"extra": "forbid"}


class WindowEndEvent(BaseModel):
    """Emitted when an agent window closes (agent turn / tool boundary / verifier).

    new_framework P2: the event stream sites match against. Kept as a plain
    dict in EpisodeRaw/TirAgent state (must stay JSON-serializable); this
    contract documents/validates the shape.
    """

    event_id: Optional[str] = None
    agent_id: str = "hub"
    kind: str = "after_agent_turn"  # after_agent_turn | after_tool | after_verifier | on_edge
    tool_id: Optional[str] = None
    edge_id: Optional[str] = None
    turn: int = 0
    snapshot_ref: Optional[str] = None  # window snapshot key
    metrics: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}
