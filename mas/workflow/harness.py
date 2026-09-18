"""Harness diagnoser registry. Must not import TirAgent / agentlightning."""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from pydantic import BaseModel, Field

from .contracts import EventKind, Trajectory, TrajectoryBatch

logger = logging.getLogger(__name__)


class Hypothesis(BaseModel):
    plugin: str
    event_id: Optional[str] = None
    message: str
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


@runtime_checkable
class Diagnoser(Protocol):
    name: str

    def diagnose(self, ctx: Dict[str, Any]) -> List[Hypothesis]:
        ...

    def consume(self, event: Dict[str, Any]) -> Optional[Hypothesis]:
        """P3 real-time harness: consume a streamed event (default no-op)."""
        ...


class LogErrorDiagnoser:
    name = "log_error"

    def diagnose(self, ctx: Dict[str, Any]) -> List[Hypothesis]:
        out: List[Hypothesis] = []
        for traj in _trajectories(ctx):
            for ev in traj.events:
                if ev.kind == EventKind.ERROR:
                    out.append(
                        Hypothesis(
                            plugin=self.name,
                            event_id=ev.event_id,
                            message=str((ev.payload or {}).get("error") or ev.payload),
                        )
                    )
        return out

    def consume(self, event: Dict[str, Any]) -> Optional[Hypothesis]:
        """Real-time: flag ERROR-kind events from the stream immediately."""
        if str(event.get("kind") or "") != "error":
            return None
        msg = str((event.get("payload") or {}).get("error") or event.get("payload") or "error event")
        return Hypothesis(plugin=self.name, message=f"[stream] {msg}"[:400])


class LossVolatilityDiagnoser:
    name = "loss_volatility"

    def diagnose(self, ctx: Dict[str, Any]) -> List[Hypothesis]:
        path = ctx.get("metrics_path")
        rewards = _reward_series(ctx)
        if path:
            series = _load_metric_series(str(path), key_substr="reward")
            if series:
                rewards = series
        if not rewards:
            if path:
                logger.warning("loss_volatility: no series in %s", path)
            else:
                logger.warning("loss_volatility: no metrics_path / mean_reward; skip")
            return []
        if len(rewards) < 2:
            return []
        first, last = float(rewards[0]), float(rewards[-1])
        out: List[Hypothesis] = []
        if last <= first:
            out.append(
                Hypothesis(
                    plugin=self.name,
                    message=f"reward not rising: first={first:.4f} last={last:.4f}",
                    meta={"n": len(rewards)},
                )
            )
        diffs = [abs(rewards[i] - rewards[i - 1]) for i in range(1, len(rewards))]
        mean_abs = sum(diffs) / len(diffs)
        if mean_abs > 0.5:
            out.append(
                Hypothesis(
                    plugin=self.name,
                    message=f"reward volatility mean_abs_delta={mean_abs:.4f}",
                    meta={"n": len(rewards)},
                )
            )
        return out

    def consume(self, event: Dict[str, Any]) -> Optional[Hypothesis]:
        """Real-time: watch loss/reward metric frames in the stream."""
        if str(event.get("event") or "") != "loss":
            return None
        metrics = dict((event.get("payload") or {}).get("metrics") or {})
        vals = [float(v) for v in metrics.values() if isinstance(v, (int, float))]
        if not vals:
            return None
        self._recent = getattr(self, "_recent", [])
        self._recent.append(float(vals[0]))
        self._recent = self._recent[-20:]
        if len(self._recent) >= 3:
            diffs = [abs(self._recent[i] - self._recent[i - 1]) for i in range(1, len(self._recent))]
            mean_abs = sum(diffs) / len(diffs)
            if mean_abs > 0.5:
                return Hypothesis(
                    plugin=self.name,
                    message=f"[stream] reward volatility mean_abs_delta={mean_abs:.4f}",
                    meta={"n": len(self._recent)},
                )
        return None


class CognitiveConvergenceDiagnoser:
    """Thin MAS harness: same error class repeating = high deviation; dying out = aligned.

    Not EPC-AW and not an LLM judge. Signature is a normalized ERROR payload.
    """

    name = "cognitive_convergence"

    def diagnose(self, ctx: Dict[str, Any]) -> List[Hypothesis]:
        trajs = _trajectories(ctx)
        if not trajs:
            return []
        hits: Dict[str, List[Tuple[int, str, str]]] = defaultdict(list)
        for i, traj in enumerate(trajs):
            for ev in traj.events:
                if ev.kind != EventKind.ERROR:
                    continue
                raw = str((ev.payload or {}).get("error") or ev.payload)
                sig = _error_signature(raw)
                hits[sig].append((i, ev.event_id, raw))
        if not hits:
            return []
        n = len(trajs)
        split = max(1, n // 2)
        out: List[Hypothesis] = []
        for sig, rows in hits.items():
            traj_idx = sorted({i for i, _, _ in rows})
            if len(traj_idx) < 2:
                continue
            in_late = [i for i in traj_idx if i >= split]
            if not in_late and n > split:
                # first-half only: later rollouts did not repeat the class
                continue
            last_eid, last_raw = rows[-1][1], rows[-1][2]
            out.append(
                Hypothesis(
                    plugin=self.name,
                    event_id=last_eid,
                    message=f"cognitive high deviation: repeated error {sig!r} on {len(traj_idx)} trajectories",
                    meta={"signature": sig, "n_traj": len(traj_idx), "example": last_raw[:200]},
                )
            )
        return out


class RewardHackingDiagnoser:
    """Thin RL harness: high reward vs broken format / ERROR. No external Judge LLM."""

    name = "reward_hacking"

    def diagnose(self, ctx: Dict[str, Any]) -> List[Hypothesis]:
        out: List[Hypothesis] = []
        for traj in _trajectories(ctx):
            reward = traj.final_reward
            if reward is None or float(reward) < 0.9:
                continue
            err_ev = next((e for e in traj.events if e.kind == EventKind.ERROR), None)
            empty = not str(traj.final_answer or "").strip()
            if err_ev is not None:
                out.append(
                    Hypothesis(
                        plugin=self.name,
                        event_id=err_ev.event_id,
                        message=f"reward hacking: reward={float(reward):.3f} but trajectory has ERROR",
                        meta={"reward": float(reward)},
                    )
                )
            elif empty or not traj.format_ok:
                out.append(
                    Hypothesis(
                        plugin=self.name,
                        message=f"reward hacking: reward={float(reward):.3f} but format/answer missing",
                        meta={"reward": float(reward), "format_ok": traj.format_ok},
                    )
                )
        return out


class RewardHackingMonitor:
    """P3 real-time: flag sibling reward z-score anomalies on the RolloutTree.

    Same-parent siblings should share the query distribution; one node's reward
    far above its siblings (z > threshold) suggests reward hacking at that
    branch window. Works on both dict and model trees.
    """

    name = "reward_hacking_monitor"

    def __init__(self, z_threshold: float = 2.0, min_siblings: int = 2) -> None:
        self.z_threshold = float(z_threshold)
        self.min_siblings = int(min_siblings)

    def diagnose(self, ctx: Dict[str, Any]) -> List[Hypothesis]:
        tree = ctx.get("tree") or ctx.get("rollout_tree")
        if tree is None:
            return []
        return self.check_tree(tree)

    def consume(self, event: Dict[str, Any]) -> Optional[Hypothesis]:
        """Real-time: re-check the tree carried in node_added/outcome events."""
        ev = str(event.get("event") or "")
        if ev not in ("node_added", "outcome"):
            return None
        tree = (event.get("payload") or {}).get("tree")
        if tree is None:
            return None
        hits = self.check_tree(tree)
        return hits[0] if hits else None

    def check_tree(self, tree: Any) -> List[Hypothesis]:
        nodes = tree.get("nodes") if isinstance(tree, dict) else getattr(tree, "nodes", None)
        if not nodes:
            return []
        siblings: Dict[str, List[Any]] = defaultdict(list)
        for n in nodes:
            pid = n.get("parent_id") if isinstance(n, dict) else getattr(n, "parent_id", None)
            if pid:
                siblings[str(pid)].append(n)
        out: List[Hypothesis] = []
        for pid, group in siblings.items():
            def _reward(n: Any) -> Optional[float]:
                r = n.get("reward") if isinstance(n, dict) else getattr(n, "reward", None)
                if r is None:
                    return None
                try:
                    return float(r)
                except (TypeError, ValueError):
                    return None

            rewards = [r for r in (_reward(n) for n in group) if r is not None]
            if len(rewards) < max(2, self.min_siblings):
                continue
            for n in group:
                nid = n.get("node_id") if isinstance(n, dict) else getattr(n, "node_id", None)
                r = _reward(n)
                if r is None:
                    continue
                # leave-one-out: compare against siblings EXCLUDING this node,
                # so a single inflated child is detectable even with n=3
                # (population z over all siblings caps |z| at (n-1)/√n).
                others = [x for m in group if m is not n for x in [_reward(m)] if x is not None]
                if len(others) < 1:
                    continue
                mean_o = sum(others) / len(others)
                var_o = sum((x - mean_o) ** 2 for x in others) / len(others)
                std_o = var_o ** 0.5
                if std_o < 1e-9:
                    # siblings constant: any material deviation is anomalous
                    if abs(r - mean_o) <= 0.05:
                        continue
                    z = float("inf") if r > mean_o else float("-inf")
                else:
                    z = (r - mean_o) / std_o
                # one-sided: reward hacking = suspiciously HIGH reward vs siblings
                if z > self.z_threshold:
                    out.append(
                        Hypothesis(
                            plugin=self.name,
                            message=(
                                f"reward hacking suspect: node {nid} z={z:+.2f} "
                                f"(reward={r:.3f} vs sibling mean={mean_o:.3f}±{std_o:.3f})"
                            ),
                            meta={"node_id": nid, "z": z, "parent_id": pid},
                        )
                    )
        return out


class StubDiagnoser:
    def __init__(self, name: str) -> None:
        self.name = name

    def diagnose(self, ctx: Dict[str, Any]) -> List[Hypothesis]:
        del ctx
        return []


class DiagnoserRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, Diagnoser] = {}

    def register(self, plugin: Diagnoser) -> None:
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> Diagnoser:
        if name not in self._plugins:
            raise KeyError(f"unknown diagnoser {name!r}; registered={sorted(self._plugins)}")
        return self._plugins[name]

    def diagnose(self, ctx: Dict[str, Any], names: Optional[List[str]] = None) -> List[Hypothesis]:
        keys = names or list(self._plugins)
        out: List[Hypothesis] = []
        for name in keys:
            out.extend(self.get(name).diagnose(ctx))
        return out


def default_harness() -> DiagnoserRegistry:
    reg = DiagnoserRegistry()
    reg.register(LogErrorDiagnoser())
    reg.register(LossVolatilityDiagnoser())
    reg.register(CognitiveConvergenceDiagnoser())
    reg.register(RewardHackingDiagnoser())
    reg.register(RewardHackingMonitor())
    reg.register(StubDiagnoser("epc_aw_consensus"))
    return reg


HARNESS = default_harness()


def _error_signature(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[0-9a-f]{8,}", "<id>", s)
    s = re.sub(r"\d+", "<n>", s)
    s = re.sub(r"[^\w\s<>]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:160] or "<empty>"


def _trajectories(ctx: Dict[str, Any]) -> List[Trajectory]:
    if "batch" in ctx and isinstance(ctx["batch"], TrajectoryBatch):
        return list(ctx["batch"].trajectories)
    if "trajectory" in ctx and isinstance(ctx["trajectory"], Trajectory):
        return [ctx["trajectory"]]
    if "trajectories" in ctx:
        items = ctx["trajectories"]
        out = []
        for t in items:
            if isinstance(t, Trajectory):
                out.append(t)
            elif isinstance(t, dict):
                out.append(Trajectory.model_validate(t))
        return out
    return []


def _reward_series(ctx: Dict[str, Any]) -> List[float]:
    if ctx.get("rewards"):
        return [float(x) for x in ctx["rewards"]]
    trajs = _trajectories(ctx)
    vals = [float(t.final_reward) for t in trajs if t.final_reward is not None]
    if vals:
        return vals
    batch = ctx.get("batch")
    if isinstance(batch, TrajectoryBatch) and batch.meta.get("mean_reward") is not None:
        return [float(batch.meta["mean_reward"])]
    return []


def _load_metric_series(path: str, key_substr: str) -> List[float]:
    p = Path(path)
    if not p.is_file():
        logger.warning("loss_volatility: metrics file missing: %s", path)
        return []
    series: List[float] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if key_substr in str(k).lower():
                try:
                    series.append(float(v))
                except (TypeError, ValueError):
                    pass
                break
    return series
