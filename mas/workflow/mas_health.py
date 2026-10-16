"""Separate MAS infrastructure faults from a model that simply answered wrong.

A fault means the reward must not be treated as a valid RL update: the tool
backend failed, a window raised, or the actor batch contains a frozen role.
A wrong answer with successful tool calls is not a fault.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

INFRA_MARKERS = (
    "tir_offline_search",
    "search_unavailable_in_cache",
    "http 5",
    "connectionerror",
    "connecttimeout",
    "readtimeout",
    "user tool error",
    "user agent error",
    "planner window failed",
    "verifier window failed",
    "agentflow build failed",
    "unknown tool-agent",
    "tool error:",
    "blank agent error",
)

_DEGENERATE = {0.0, -1.0}


class MasHealthError(RuntimeError):
    """The training step saw MAS faults, so its reward must not update the policy."""


def _texts_from_messages(messages: Any) -> list[str]:
    out: list[str] = []
    if not isinstance(messages, list):
        return out
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
        kind = str(msg.get("kind") or "")
        if kind == "error":
            out.append(str(payload.get("error") or msg))
        output = payload.get("output")
        if output:
            out.append(str(output))
        if payload.get("ok") is False and payload.get("evidence_type") == "ERROR":
            out.append(str(output or payload))
    return out


def _marker_hit(text: str) -> str:
    low = text.lower()
    # An empty result set means the backend answered. A down backend says HTTP,
    # offline, or the cache-gap sentinel.
    if "0 hits" in low:
        return ""
    for marker in INFRA_MARKERS:
        if marker in low:
            return marker
    return ""


def episode_faults(
    raw: Any = None,
    traj: Any = None,
    *,
    messages: Any = None,
    error: str = "",
    final_answer: str = "",
    events: Sequence[Any] | None = None,
    span_names: Sequence[str] | None = None,
    reward: float | None = None,
) -> list[str]:
    """Return human-readable MAS faults. Empty means the reward reflects the answer."""
    faults: list[str] = []
    if reward is not None:
        try:
            value = float(reward)
        except (TypeError, ValueError):
            value = float("nan")
        if math.isnan(value) or math.isinf(value):
            faults.append("reward is not finite")
    err = error or str(getattr(raw, "error", None) or "")
    if err and "用户停止" not in err:
        hit = _marker_hit(err) or "episode error"
        faults.append(f"{hit}: {err[:240]}")
    msgs = messages if messages is not None else getattr(raw, "messages", None)
    for text in _texts_from_messages(msgs):
        hit = _marker_hit(text)
        if hit:
            faults.append(f"{hit}: {text[:240]}")
            break
    evs = list(events or [])
    if traj is not None and not evs:
        evs = list(getattr(traj, "events", None) or [])
    for ev in evs:
        kind = str(getattr(ev, "kind", None) or (ev.get("kind") if isinstance(ev, dict) else "") or "")
        if kind.lower() in ("error", "eventkind.error"):
            payload = getattr(ev, "payload", None) or (ev.get("payload") if isinstance(ev, dict) else {}) or {}
            faults.append(f"error event: {payload.get('error') or payload}"[:240])
            break
    answer = final_answer or str(getattr(raw, "final_answer", None) or getattr(traj, "final_answer", None) or "")
    if not answer.strip() and faults:
        faults.append("empty final_answer alongside a MAS fault")
    for name in span_names or []:
        low = str(name).lower()
        if "agent__frozen" in low or "agent__u_agentflow__executor" in low or low in ("agent__verifier", "agent.verifier", "agent__executor"):
            faults.append(f"frozen span in actor batch: {name}")
            break
    # Deduplicate while keeping order.
    seen: set[str] = set()
    unique: list[str] = []
    for item in faults:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def assess_step(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Group check. Degenerate rewards without an infra fault are model error, not MAS."""
    faults: list[str] = []
    rewards: list[float] = []
    for row in rows:
        row_faults = list(row.get("faults") or [])
        text = str(row.get("mas_fault") or "")
        if text:
            row_faults.append(text)
        faults.extend(row_faults)
        if "reward" in row and row.get("reward") is not None:
            try:
                rewards.append(float(row["reward"]))
            except (TypeError, ValueError):
                faults.append("reward is not finite")
    degenerate = bool(rewards) and len(set(rewards)) == 1 and rewards[0] in _DEGENERATE
    if degenerate and faults:
        faults.append(f"degenerate reward {rewards[0]} accompanied by MAS faults")
    return {"ok": not faults, "faults": faults, "degenerate_reward": degenerate and not faults}
