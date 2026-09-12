"""Strict schemas between workflow data layer and RL / Harness (no AGL)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EventKind(str, Enum):
    TASK_START = "task_start"
    AGENT_ENTER = "agent_enter"
    AGENT_EXIT = "agent_exit"
    HANDOFF = "handoff"
    MODEL_CALL = "model_call"
    MODEL_RESULT = "model_result"
    SKILL_CALL = "skill_call"
    SKILL_RESULT = "skill_result"
    TERMINATION = "termination"
    AGENT_MESSAGE = "agent_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINAL_ANSWER = "final_answer"
    SNAPSHOT = "snapshot"
    ERROR = "error"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    FEEDBACK = "feedback"


class ExecutionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    parent_id: Optional[str] = None
    agent_id: str = "hub"
    agent_execution_id: Optional[str] = None
    model_call_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    occurred_at: Optional[datetime] = None
    kind: EventKind
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=_utcnow)

    model_config = {"extra": "forbid"}


class ModelCapabilities(BaseModel):
    """Evidence-based capabilities; missing provider evidence stays unknown."""

    token_ids: Literal["available", "unsupported", "unknown"] = "unknown"
    logprobs: Literal["available", "unsupported", "unknown"] = "unknown"
    tool_calling: Literal["available", "unsupported", "unknown"] = "unknown"
    usage: Literal["available", "unsupported", "unknown"] = "unknown"

    model_config = {"extra": "forbid"}


class ModelIdentity(BaseModel):
    """Public model identity, separate from the Agent using it; never credentials."""

    source: str = "api"
    model: str
    policy_version: Optional[str] = None
    tokenizer_id: Optional[str] = None
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)

    model_config = {"extra": "forbid"}


class AgentExecutionContext(BaseModel):
    """Minimal effective input for one Agent entry, not an entire agent platform.

    Stage two wires this contract at execution boundaries. ``messages`` contains
    model-visible input after context processing, not evaluation answers.
    Memory items are explicitly scoped; repeated entries get distinct IDs.
    """

    contract_version: Literal["1"] = "1"
    run_id: str
    trajectory_id: str
    agent_id: str
    agent_execution_id: str = Field(default_factory=lambda: uuid4().hex)
    model: Optional[ModelIdentity] = None
    system_prompt: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    tool_definitions: List[Dict[str, Any]] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    memory_items: List["MemoryItem"] = Field(default_factory=list)
    handoff_from_execution_id: Optional[str] = None
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ExecutionRecorder(Protocol):
    """Archive-compatible sink; no dependency on HTTP, UI or a training framework."""

    def append(self, event: ExecutionEvent) -> ExecutionEvent:
        ...


class SnapshotCoverage(BaseModel):
    """What was actually captured, not a promise of full-system resumability."""

    messages: bool = True
    execution_cursor: bool = False
    agent_memory: bool = False
    shared_memory: bool = False
    external_tool_state: bool = False
    policy_identity: bool = False

    model_config = {"extra": "forbid"}


class Snapshot(BaseModel):
    """Barrier sufficient to resume TirAgent via resume_messages."""

    snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    archive_id: str
    event_id: str = ""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)
    coverage: SnapshotCoverage = Field(default_factory=SnapshotCoverage)
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


class MemoryItem(BaseModel):
    """Two-layer memory contract. Phase 0: agent=messages, system unused."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    scope: str = "agent"  # agent | system
    owner: str = "hub"
    content: Any = None

    model_config = {"extra": "forbid"}


class Trajectory(BaseModel):
    schema_version: str = "2"
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


AgentExecutionContext.model_rebuild()
