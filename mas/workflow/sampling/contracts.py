"""Stable Sampling Framework contracts shared across runtime and adapters."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class WindowKind(str, Enum):
    TOOL_RESULT = "tool_result"
    AGENT_COMPLETE = "agent_complete"
    ROUTER_DECISION = "router_decision"
    VERIFICATION_COMPLETE = "verification_complete"
    EDGE = "edge"
    TOKEN = "token"


class WindowSelector(BaseModel):
    kind: WindowKind
    owner_agent_id: Optional[str] = None
    interaction: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class SamplingWindow(BaseModel):
    window_id: str = Field(min_length=1)
    owner_agent_id: str
    kind: WindowKind
    sequence: int = 0
    snapshot_ref: str = Field(min_length=1)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    interaction: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class SamplingOpportunity(BaseModel):
    selector: WindowSelector
    resumable: bool = True
    allowed_gates: List[str] = Field(default_factory=list)
    support: str = "native"
    message: str = ""

    model_config = {"extra": "forbid"}


class SamplingDecision(BaseModel):
    window_id: str
    site_id: str
    passed: bool
    score: float = 0.0
    reason: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class ExpansionPlan(BaseModel):
    parent_rollout_id: str
    window_id: str
    snapshot_ref: str
    site_id: str
    count: int
    depth: int
    decision: SamplingDecision
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}
