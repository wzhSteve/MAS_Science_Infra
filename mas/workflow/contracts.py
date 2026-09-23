"""Strict schemas between workflow data layer and RL / Harness (no AGL)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


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
    """Barrier sufficient to resume TirAgent via resume_messages / token_prefix."""

    snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    archive_id: str
    event_id: str = ""
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

    model_config = {"extra": "forbid"}


class RolloutTree(BaseModel):
    """Explicit per-query branch tree; supersedes scattered ForkPlan.meta fields."""

    tree_id: str  # data_id (+ group)
    query: str = ""
    nodes: List[RolloutTreeNode] = Field(default_factory=list)
    outcomes: Dict[str, Any] = Field(default_factory=dict)  # leaf node_id -> {answer, reward}

    model_config = {"extra": "forbid"}

    def leaves(self) -> List[str]:
        parents = {n.parent_id for n in self.nodes if n.parent_id}
        return [n.node_id for n in self.nodes if n.node_id not in parents]

    def path_to_root(self, node_id: str) -> List[str]:
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


class RolloutTreeEvent(BaseModel):
    """Streamed tree update for real-time Harness / UI (new_framework P3)."""

    event: str = "node_added"  # node_added | outcome | reward | loss
    tree_id: str
    node_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

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
