"""Counterfactual probe helpers for dual-entropy site selection (RAE-lite)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from workflow.gates import branch_outcome_entropy


@dataclass
class ProbeResult:
    success: bool
    reward: float = 0.0
    truncated: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeAggregate:
    n: int
    n_success: int
    n_fail: int
    hat_p: float
    h_branch: float
    u: float
    explore: bool


def aggregate_probes(
    probes: Sequence[ProbeResult],
    *,
    h_pi: float,
    u_threshold: float = 0.05,
) -> ProbeAggregate:
    n = len(probes)
    n_ok = sum(1 for p in probes if p.success)
    n_fail = n - n_ok
    hat_p = (n_ok + 1.0) / (n + 2.0) if n else 0.5
    h_b = branch_outcome_entropy(n_ok, n_fail)
    u = float(abs(h_pi)) * float(h_b)
    return ProbeAggregate(
        n=n,
        n_success=n_ok,
        n_fail=n_fail,
        hat_p=float(hat_p),
        h_branch=float(h_b),
        u=float(u),
        explore=bool(u >= float(u_threshold)),
    )


def select_top_sites_by_u(
    scored: Sequence[Dict[str, Any]],
    *,
    top_m: int = 2,
) -> List[Dict[str, Any]]:
    """Pick top-m site instances by U (dicts must contain 'u')."""
    ordered = sorted(scored, key=lambda d: float(d.get("u") or 0.0), reverse=True)
    m = max(0, int(top_m))
    return list(ordered[:m]) if m else []


def action_key_from_messages(messages: Optional[List[Dict[str, Any]]]) -> str:
    """Coarse semantic key for tool call / last assistant action (RAE support set)."""
    if not messages:
        return "empty"
    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        if role != "assistant":
            continue
        tcs = msg.get("tool_calls") or msg.get("tool_call")
        if tcs:
            if isinstance(tcs, list) and tcs:
                tc = tcs[0] if isinstance(tcs[0], dict) else {}
                name = (tc.get("function") or {}).get("name") if isinstance(tc.get("function"), dict) else tc.get("name")
                return f"tool:{(name or 'unknown')}"
            return "tool:unknown"
        content = str(msg.get("content") or "").strip()
        if content:
            return f"text:{content[:64]}"
    return "unknown"
