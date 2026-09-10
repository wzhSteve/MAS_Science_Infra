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


class ExecutionEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: uuid4().hex)
    parent_id: Optional[str] = None
    agent_id: str = "hub"
    kind: EventKind
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: datetime = Field(default_factory=_utcnow)

    model_config = {"extra": "forbid"}


class Snapshot(BaseModel):
    """Barrier sufficient to resume TirAgent via resume_messages."""

    snapshot_id: str = Field(default_factory=lambda: uuid4().hex)
    archive_id: str
    event_id: str = ""
    messages: List[Dict[str, Any]] = Field(default_factory=list)
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
