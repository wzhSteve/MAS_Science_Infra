"""Shared solver dataclasses (no logic)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    analysis: str
    step_conclusion: str
    info_flag: bool
    obtained_info: str
    task_conclusion: Optional[str]
    diagnostic_signal: Optional[Dict[str, Any]]
    subgoal_complete: bool
    slot_updates: List[Dict[str, Any]] = field(default_factory=list)
    evidence_type: str = "DIRECT"
    tool_appropriate: bool = True
    had_slot_delta: bool = False


@dataclass
class StepContext:
    step_key: str
    target_information: str
    context: str
    sub_goal: str
    tool_name: str
    command: str
    result_executor: Any
    first_attempt_command: str
