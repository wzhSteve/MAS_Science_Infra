"""Branch-site gate evaluation (no AGL). Used by ActiveSet / BarrierEmitter."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from rl.hooks.branch_policy import arpo_should_fork, branch_probability, should_branch
from workflow.contracts import BranchGate, BranchSite


@dataclass
class GateContext:
    h_root: float = 0.0
    h_tool: float = 0.0
    consecutive_high: int = 0
    tool_ok: Optional[bool] = None
    verifier_ok: Optional[bool] = None
    signals: Optional[Dict[str, Any]] = None
    final_failed: bool = False
    h_branch: Optional[float] = None  # outcome entropy from probes (dual_entropy)
    rng: Any = None


@dataclass
class GateDecision:
    passed: bool
    score: float = 0.0
    reason: str = ""
    meta: Optional[Dict[str, Any]] = None


def evaluate_gate(gate: BranchGate, ctx: GateContext) -> GateDecision:
    gtype = str(gate.type or "entropy_delta").lower().strip()
    params = dict(gate.params or {})
    signals = dict(ctx.signals or {})

    if gtype == "always":
        return GateDecision(True, 1.0, "always")

    if gtype in ("entropy_delta", "arpo"):
        use_official = bool(params.get("use_official_arpo_gate", True))
        bp = float(params.get("branch_probability", params.get("alpha", 0.5)))
        ew = float(params.get("entropy_weight", params.get("gamma", 0.5)))
        tau = float(params.get("entropy_threshold", params.get("threshold", 0.15)))
        if use_official:
            ok = arpo_should_fork(
                ctx.h_tool, ctx.h_root, branch_probability=bp, entropy_weight=ew, rng=ctx.rng
            )
            return GateDecision(ok, float(ctx.h_tool - ctx.h_root), "arpo_official")
        p = branch_probability(ctx.h_tool - ctx.h_root, alpha=bp, gamma=ew, consecutive_high=ctx.consecutive_high)
        ok = should_branch(p, tau)
        return GateDecision(ok, float(p), "entropy_delta")

    if gtype == "dual_entropy":
        # U = H_pi_norm * H_B; if H_B unknown, fall back to entropy_delta
        h_pi = abs(float(ctx.h_tool - ctx.h_root))
        h_b = ctx.h_branch
        if h_b is None:
            return evaluate_gate(
                BranchGate(type="entropy_delta", params=params),
                ctx,
            )
        u = float(h_pi) * float(h_b)
        thr = float(params.get("u_threshold", 0.05))
        return GateDecision(u >= thr, u, "dual_entropy", {"h_pi": h_pi, "h_b": h_b})

    if gtype == "tool_ok":
        ok = ctx.tool_ok is True
        return GateDecision(ok, 1.0 if ok else 0.0, "tool_ok")

    if gtype == "tool_error":
        ok = ctx.tool_ok is False
        return GateDecision(ok, 1.0 if ok else 0.0, "tool_error")

    if gtype == "verifier_pass":
        ok = ctx.verifier_ok is True
        return GateDecision(ok, 1.0 if ok else 0.0, "verifier_pass")

    if gtype in ("verifier_fail", "contradiction"):
        ok = ctx.verifier_ok is False
        if gtype == "contradiction" and not ok:
            # also treat tool_error / explicit signal
            ok = ctx.tool_ok is False or bool(signals.get("contradiction"))
        return GateDecision(ok, 1.0 if ok else 0.0, gtype)

    if gtype == "failure_trigger":
        ok = bool(ctx.final_failed)
        return GateDecision(ok, 1.0 if ok else 0.0, "failure_trigger")

    if gtype == "signal":
        key = str(params.get("signal_key") or params.get("signal") or "")
        if not key:
            return GateDecision(False, 0.0, "signal_missing_key")
        val = signals.get(key, getattr(ctx, key, None))
        if "equals" in params:
            ok = val == params.get("equals")
        elif "truthy" in params:
            ok = bool(val) == bool(params.get("truthy"))
        else:
            ok = bool(val)
        return GateDecision(ok, 1.0 if ok else 0.0, f"signal:{key}")

    return GateDecision(False, 0.0, f"unknown_gate:{gtype}")


def site_matches_anchor(
    site: BranchSite,
    *,
    event_kind: str,
    agent_id: Optional[str] = None,
    tool_id: Optional[str] = None,
    edge_id: Optional[str] = None,
) -> bool:
    if not site.enabled:
        return False
    anchor = site.anchor
    kind = str(anchor.kind or "").lower()
    ek = str(event_kind or "").lower()

    if kind == "after_tool" and ek not in ("after_tool", "tool_result"):
        return False
    if kind == "after_verifier" and ek not in ("after_verifier", "feedback"):
        return False
    if kind == "after_agent_turn" and ek not in ("after_agent_turn", "agent_message"):
        return False
    if kind == "on_edge" and ek not in ("on_edge", "sample_barrier"):
        return False
    if kind == "on_token" and ek != "on_token":
        return False

    if anchor.agent_id and str(anchor.agent_id) != str(agent_id):
        if not (kind == "after_tool" and anchor.agent_id == anchor.tool_id == tool_id):
            return False
    if anchor.tool_id and str(anchor.tool_id) != str(tool_id):
        return False
    if anchor.edge_id and str(anchor.edge_id) != str(edge_id):
        return False
    return True


def site_matches_event(
    site: BranchSite,
    *,
    event_kind: str,
    agent_id: Optional[str] = None,
    tool_id: Optional[str] = None,
    edge_id: Optional[str] = None,
    hit_count: int = 0,
    window_events: Optional[list] = None,
) -> bool:
    """Match a window boundary, its identity, and the site's hit policy."""
    if window_events:
        matches = (
            event for event in window_events
            if isinstance(event, dict) and site_matches_anchor(
                site,
                event_kind=str(event.get("kind") or ""),
                agent_id=event.get("agent_id"),
                tool_id=event.get("tool_id"),
                edge_id=event.get("edge_id"),
            )
        )
        return any(
            site_matches_event(
                site,
                event_kind=str(event.get("kind") or ""),
                agent_id=event.get("agent_id"),
                tool_id=event.get("tool_id"),
                edge_id=event.get("edge_id"),
                hit_count=hit_count + index,
            )
            for index, event in enumerate(matches)
        )
    if not site_matches_anchor(
        site, event_kind=event_kind, agent_id=agent_id, tool_id=tool_id, edge_id=edge_id
    ):
        return False
    when = str(site.when or "first").lower()
    if when == "first":
        return hit_count == 0
    if when == "every":
        return True
    if when == "nth":
        return hit_count + 1 == max(1, int(site.nth or 1))
    return hit_count == 0


def branch_outcome_entropy(successes: int, failures: int) -> float:
    """Normalized Bernoulli entropy in [0, 1] with Laplace smoothing."""
    a = float(successes) + 1.0
    b = float(failures) + 1.0
    p = a / (a + b)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    h = -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)
    return float(max(0.0, min(1.0, h)))
