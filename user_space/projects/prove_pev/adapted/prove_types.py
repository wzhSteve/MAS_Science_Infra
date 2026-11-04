"""Types and defaults for MAS-ProVe-style process verification on PEV."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


PROVE_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "n": 3,
    # plan | plan_and_command | iteration  (iteration == plan for this PEV host)
    "granularity": "plan",
    "candidate_temperature": 0.7,
    "judge_backend": "inline",  # inline | tcp
    "judge_model": None,
    "context_mode": "full_memory",  # full_memory | step_only
    "task_type": "qa",
    "tcp_host": "127.0.0.1",
    "tcp_port": 5555,
}


@dataclass
class ProveCandidate:
    index: int
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProveDecision:
    enabled: bool
    n: int
    best_idx: int
    rankings: List[int]
    judge_feedback: str = ""
    context: str = ""
    candidates: List[str] = field(default_factory=list)
    skipped_reason: str = ""
    site: str = ""  # plan | command

    def to_trace(self) -> Dict[str, Any]:
        if not self.enabled:
            decision = "disabled"
        elif self.skipped_reason:
            decision = "skipped"
        else:
            decision = "selected"
        out: Dict[str, Any] = {
            "prove_decision": decision,
            "prove_n": int(self.n),
            "prove_best_idx": int(self.best_idx),
            "prove_rankings": list(self.rankings),
            "prove_site": self.site,
        }
        if self.skipped_reason:
            out["prove_reason"] = self.skipped_reason
        if self.judge_feedback:
            out["prove_judge_head"] = self.judge_feedback[:300]
        return out
