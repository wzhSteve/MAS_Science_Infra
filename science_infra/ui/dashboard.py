"""Shared view-model for the Science MAS HTML dashboard (no agentlightning)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence


def build_dashboard_model(
    batch: Any,
    hypotheses: Sequence[Any],
    *,
    train_signal: Optional[Dict[str, Any]] = None,
    dashboard: str = "",
) -> Dict[str, Any]:
    trajs = list(getattr(batch, "trajectories", None) or [])
    rewards: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    n_error = 0
    n_feedback = 0
    for i, traj in enumerate(trajs):
        rew = traj.final_reward
        if rew is not None:
            rewards.append({"index": i + 1, "id": str(traj.trajectory_id), "reward": float(rew)})
        kinds: List[str] = []
        events: List[Dict[str, Any]] = []
        for ev in traj.events or []:
            kind = ev.kind.value if hasattr(ev.kind, "value") else str(ev.kind)
            event_counts[kind] += 1
            kinds.append(kind)
            if kind == "error":
                n_error += 1
            if kind == "feedback":
                n_feedback += 1
            payload = ev.payload if isinstance(getattr(ev, "payload", None), dict) else {}
            events.append(
                {
                    "event_id": getattr(ev, "event_id", ""),
                    "kind": kind,
                    "agent_id": getattr(ev, "agent_id", "hub"),
                    "payload": payload,
                }
            )
        rows.append(
            {
                "trajectory_id": str(traj.trajectory_id),
                "reward": rew,
                "answer": traj.final_answer,
                "format_ok": bool(traj.format_ok),
                "n_search": int(traj.n_search or 0),
                "n_python": int(traj.n_python or 0),
                "kinds": kinds,
                "events": events,
            }
        )
    meta = getattr(batch, "meta", None) or {}
    hyps: List[Dict[str, Any]] = []
    for hyp in hypotheses:
        hyps.append(
            {
                "plugin": getattr(hyp, "plugin", ""),
                "event_id": getattr(hyp, "event_id", None),
                "message": getattr(hyp, "message", str(hyp)),
                "meta": getattr(hyp, "meta", None) or {},
            }
        )
    signal = train_signal if isinstance(train_signal, dict) else {}
    advantage = signal.get("advantage") if isinstance(signal.get("advantage"), dict) else {}
    loss = signal.get("loss") if isinstance(signal.get("loss"), dict) else {}
    return {
        "n": len(trajs),
        "mean_reward": meta.get("mean_reward"),
        "rewards": rewards,
        "event_counts": dict(event_counts),
        "n_error": n_error,
        "n_feedback": n_feedback,
        "n_hypotheses": len(hyps),
        "trajectories": rows,
        "hypotheses": hyps,
        "train_signal": {
            "advantage": advantage.get("name") or "",
            "loss": loss.get("name") or "",
            "meta": signal.get("meta") if isinstance(signal.get("meta"), dict) else {},
        },
        "dashboard": dashboard or "",
    }


def svg_reward_chart(rewards: Sequence[Dict[str, Any]], *, width: int = 720, height: int = 200) -> str:
    if not rewards:
        return '<p class="muted" id="reward-chart">No rewarded trajectories.</p>'
    pad_l, pad_r, pad_t, pad_b = 40, 16, 16, 28
    inner_w = width - pad_l - pad_r
    inner_h = height - pad_t - pad_b
    vals = [float(p["reward"]) for p in rewards]
    n = len(vals)
    ymin, ymax = 0.0, 1.0

    def x_at(i: int) -> float:
        if n == 1:
            return pad_l + inner_w / 2.0
        return pad_l + i * inner_w / (n - 1)

    def y_at(v: float) -> float:
        t = (v - ymin) / (ymax - ymin) if ymax > ymin else 0.0
        return pad_t + inner_h * (1.0 - t)

    coords = " ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(vals))
    dots = []
    for i, v in enumerate(vals):
        tid = str(rewards[i].get("id") or "")[:16]
        dots.append(
            f'<circle cx="{x_at(i):.1f}" cy="{y_at(v):.1f}" r="3.2" fill="#228be6">'
            f"<title>#{i + 1} {tid} reward={v}</title></circle>"
        )
    return (
        f'<svg id="reward-chart" viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="episode reward">'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}" stroke="#dee2e6"/>'
        f'<line x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}" stroke="#dee2e6"/>'
        f'<text x="8" y="{pad_t + 4}" font-size="11" fill="#868e96">1</text>'
        f'<text x="8" y="{height - pad_b}" font-size="11" fill="#868e96">0</text>'
        f'<polyline fill="none" stroke="#228be6" stroke-width="2" points="{coords}"/>'
        f"{''.join(dots)}"
        f'<text x="{pad_l}" y="{height - 8}" font-size="11" fill="#868e96">episode</text>'
        f"</svg>"
    )
