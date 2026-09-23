"""Active-set branch expansion (ARPO-style scheduling, messages-level prefix).

Does not import agentlightning. Used by LitTirAgent / ExecutionService / tests.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from rl.hooks.branch_policy import allocate_forks
from workflow.contracts import (
    BranchAnchor,
    BranchGate,
    BranchSite,
    RolloutTree,
    RolloutTreeNode,
)
from workflow.gates import GateContext, evaluate_gate, site_matches_anchor, site_matches_event
from workflow.probes import ProbeResult, action_key_from_messages, aggregate_probes

RunEpisodeFn = Callable[[Dict[str, Any]], Any]
RewardFn = Callable[[Any, Dict[str, Any]], float]


def resume_boundary_from_messages(messages: Optional[Sequence[Any]]) -> int:
    """Proxy shared-prefix length for R0 (message count when token ids unavailable)."""
    if not messages:
        return 0
    return int(len(list(messages)))

EXPANSION_DIR = os.getenv(
    "TIR_LOCAL_EXPANSION_DIR",
    os.path.join(os.path.dirname(__file__), "..", ".local_expansion"),
)


@dataclass
class ForkPlan:
    """One planned resume sibling (LLM still run via Store wave-2 or local run)."""

    resume_messages: List[Dict[str, Any]]
    parent_id: str
    depth: int
    reason: str = "arpo_branch"
    meta: Dict[str, Any] = field(default_factory=dict)
    site_id: str = ""
    role: str = "child"  # child | probe


@dataclass
class CompletedLeaf:
    traj_id: str
    parent_id: Optional[str]
    depth: int
    is_branch: bool
    raw: Any
    task: Dict[str, Any]
    reward: Optional[float] = None


@dataclass
class ActiveSetResult:
    completed: List[CompletedLeaf] = field(default_factory=list)
    plans: List[ForkPlan] = field(default_factory=list)
    branch_local_count: int = 0
    global_fill_count: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveSetConfig:
    group_budget: int = 1
    beam_size: int = 2
    max_branch_depth: int = 2
    branch_probability: float = 0.5
    entropy_weight: float = 0.5
    entropy_threshold: float = 0.15
    use_official_arpo_gate: bool = True
    alpha: float = 0.5
    gamma: float = 0.2
    execute_local: bool = False
    """If True, run resume/global siblings inside the session (Collect/mock).

    Training hot path keeps execute_local=False: only plan forks, Daemon enqueues
    resume so each sibling still gets AGL LLM spans.
    """
    parallel_local: bool = False
    """When execute_local, run sibling episodes concurrently (ready-batch-ish)."""
    sites: Optional[List[BranchSite]] = None
    event_kind: str = "after_tool"
    agent_id: Optional[str] = None
    tool_id: Optional[str] = None
    verifier_ok: Optional[bool] = None
    tool_ok: Optional[bool] = None
    final_failed: bool = False
    h_branch: Optional[float] = None
    probe_k: int = 2
    run_probes: bool = True
    """When True, run local probes before dual_entropy / rae_adjudicate gates."""


class ActiveSetSession:
    """ARPO/RAE-like active-set controller at the episode / messages barrier."""

    def __init__(self, config: ActiveSetConfig, *, rng: Any = None) -> None:
        self.config = config
        self._rng = rng
        self._site_hits: Dict[str, int] = {}
        self._probe_metrics: Dict[str, Any] = {}

    def _sites(self) -> List[BranchSite]:
        if self.config.sites is not None:
            resolved: List[BranchSite] = []
            for site in self.config.sites:
                if not site.enabled:
                    continue
                params = dict(site.gate.params or {})
                if site.gate.type in ("entropy_delta", "arpo"):
                    params = {
                        "use_official_arpo_gate": self.config.use_official_arpo_gate,
                        "branch_probability": self.config.branch_probability,
                        "entropy_weight": self.config.entropy_weight,
                        "entropy_threshold": self.config.entropy_threshold,
                        **params,
                    }
                elif site.gate.type == "dual_entropy":
                    params = {"probe_k": self.config.probe_k, **params}
                resolved.append(site.model_copy(update={
                    "gate": site.gate.model_copy(update={"params": params}),
                }))
            return resolved
        # legacy single entropy gate as one synthetic site
        return [
            BranchSite(
                id="default_after_tool",
                anchor=BranchAnchor(kind="after_tool"),
                gate=BranchGate(
                    type="entropy_delta",
                    params={
                        "use_official_arpo_gate": self.config.use_official_arpo_gate,
                        "branch_probability": self.config.branch_probability,
                        "entropy_weight": self.config.entropy_weight,
                        "entropy_threshold": self.config.entropy_threshold,
                        "alpha": self.config.alpha,
                        "gamma": self.config.gamma,
                    },
                ),
            )
        ]

    def _site_needs_probes(self, site: BranchSite) -> bool:
        gtype = str(site.gate.type or "").lower()
        scheme = str(site.reward.scheme or "").lower()
        return gtype == "dual_entropy" or scheme == "rae_adjudicate"

    def collect_h_branch(
        self,
        raw: Any,
        root_task: Dict[str, Any],
        run_episode: RunEpisodeFn,
        *,
        site: Optional[BranchSite] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[float]:
        """Run K probes from the selected window and return outcome entropy."""
        if self.config.h_branch is not None:
            return float(self.config.h_branch)
        msgs = list(messages if messages is not None else getattr(raw, "branch_messages", None) or [])
        if not msgs:
            return None
        sites = [site] if site is not None else [s for s in self._sites() if self._site_needs_probes(s)]
        if not sites:
            return None
        target = sites[0]
        k = int((target.gate.params or {}).get("probe_k", self.config.probe_k) or self.config.probe_k)
        k = max(1, k)
        max_tokens = int(target.fork.probe_max_tokens or 128)
        probes: List[ProbeResult] = []
        for i in range(k):
            probe_task = dict(root_task)
            probe_task.pop("expand_in_runner", None)
            probe_task.pop("sampling_budget", None)
            probe_task["resume_messages"] = deepcopy(msgs)
            probe_task["resume_parent_id"] = str(root_task.get("_rollout_id") or "")
            probe_task["is_probe"] = True
            probe_task["rollout_role"] = "probe"
            probe_task["probe_max_tokens"] = max_tokens
            probe_task["_rollout_id"] = f"probe_{uuid4().hex[:8]}_{i}"
            try:
                probe_raw = run_episode(probe_task)
            except Exception as e:
                probes.append(ProbeResult(success=False, reward=0.0, truncated=True, meta={"error": str(e)}))
                continue
            ok = bool(getattr(probe_raw, "format_ok", False)) and not getattr(probe_raw, "error", None)
            ans = str(getattr(probe_raw, "final_answer", "") or "").strip()
            if ans and ans.lower() not in ("none", "null", "n/a"):
                ok = True
            probes.append(
                ProbeResult(
                    success=bool(ok),
                    reward=1.0 if ok else 0.0,
                    truncated=False,
                    meta={"final_answer": ans},
                )
            )
        h_root = float(getattr(raw, "h_root", 0.0) or 0.0)
        h_tool = float(getattr(raw, "h_tool", 0.0) or 0.0)
        h_pi = abs(h_tool - h_root)
        thr = float((target.gate.params or {}).get("u_threshold", 0.05))
        agg = aggregate_probes(probes, h_pi=h_pi, u_threshold=thr)
        return float(agg.h_branch)

    def plan_forks_from_raw(
        self,
        raw: Any,
        *,
        parent_id: str,
        remaining: int,
        depth: int = 0,
        event_kind: Optional[str] = None,
        root_task: Optional[Dict[str, Any]] = None,
        run_episode: Optional[RunEpisodeFn] = None,
    ) -> List[ForkPlan]:
        """After a completed (or barrier-ready) episode, decide resume siblings."""
        cfg = self.config
        if remaining <= 0 or depth >= cfg.max_branch_depth:
            return []
        if getattr(raw, "error", None):
            return []
        ek = str(event_kind or cfg.event_kind or "after_tool")
        plans: List[ForkPlan] = []
        budget_left = remaining
        window_events = list(getattr(raw, "window_events", None) or [])
        windows = {
            window.get("event_id"): window
            for window in getattr(raw, "window_snapshots", None) or []
            if isinstance(window, dict) and window.get("event_id")
        }
        if not window_events and cfg.sites is None:
            msgs = list(getattr(raw, "branch_messages", None) or getattr(raw, "messages", None) or [])
            if msgs:
                window_events = [{
                    "kind": ek, "agent_id": cfg.agent_id, "tool_id": cfg.tool_id,
                    "messages": msgs, "metrics": {
                        "h_root": getattr(raw, "h_root", 0.0),
                        "h_tool": getattr(raw, "h_tool", 0.0),
                        "consecutive_high": getattr(raw, "consecutive_high", 0),
                    },
                }]
        for site in self._sites():
            if budget_left <= 0:
                break
            for event in window_events:
                if budget_left <= 0:
                    break
                if not site_matches_anchor(
                    site, event_kind=str(event.get("kind") or ""),
                    agent_id=event.get("agent_id"), tool_id=event.get("tool_id"),
                    edge_id=event.get("edge_id"),
                ):
                    continue
                hits = self._site_hits.get(site.id, 0)
                self._site_hits[site.id] = hits + 1
                if not site_matches_event(
                    site, event_kind=str(event.get("kind") or ""),
                    agent_id=event.get("agent_id"), tool_id=event.get("tool_id"),
                    edge_id=event.get("edge_id"), hit_count=hits,
                ):
                    continue
                window = windows.get(event.get("event_id"))
                msgs = list((window or {}).get("messages") or event.get("messages") or [])
                if not msgs:
                    logging.getLogger(__name__).warning(
                        "Branch site %s matched an event without a resumable window", site.id
                    )
                    continue
                metrics = event.get("metrics") or {}
                if site.gate.type in ("entropy_delta", "arpo", "dual_entropy") and "h_tool" not in metrics:
                    logging.getLogger(__name__).warning(
                        "Branch site %s has no entropy measurement at its window", site.id
                    )
                    continue
                h_root = float(metrics.get("h_root", getattr(raw, "h_root", 0.0)) or 0.0)
                h_tool = float(metrics.get("h_tool", getattr(raw, "h_tool", 0.0)) or 0.0)
                h_branch = cfg.h_branch
                if cfg.run_probes and self._site_needs_probes(site) and h_branch is None and run_episode and root_task:
                    h_branch = self.collect_h_branch(
                        raw, root_task, run_episode, site=site, messages=msgs
                    )
                    self._probe_metrics["probes_ran"] = h_branch is not None
                    self._probe_metrics["h_branch"] = h_branch
                    if h_branch is None:
                        logging.getLogger(__name__).warning(
                            "Branch site %s could not compute probe outcome entropy", site.id
                        )
                if site.gate.type == "dual_entropy" and h_branch is None:
                    logging.getLogger(__name__).warning(
                        "Branch site %s requires outcome entropy before training can fork", site.id
                    )
                    continue
                ctx = GateContext(
                    h_root=h_root, h_tool=h_tool,
                    consecutive_high=int(metrics.get("consecutive_high", 0) or 0),
                    tool_ok=metrics.get("tool_ok", cfg.tool_ok),
                    verifier_ok=metrics.get("verifier_ok", cfg.verifier_ok),
                    final_failed=bool(metrics.get("final_failed", cfg.final_failed)),
                    h_branch=h_branch, rng=self._rng,
                )
                decision = evaluate_gate(site.gate, ctx)
                if not decision.passed:
                    continue
                beam = int(site.fork.beam_size if site.fork.beam_size is not None else cfg.beam_size)
                counts = allocate_forks(remaining=budget_left, beam_size=beam, n_sources=1)
                n = counts[0] if counts else 0
                for _ in range(n):
                    plans.append(ForkPlan(
                        resume_messages=deepcopy(msgs),
                        parent_id=parent_id, depth=depth + 1,
                        reason=f"{site.gate.type}_branch", site_id=site.id, role="child",
                        meta={
                            "h_root": h_root, "h_tool": h_tool,
                            "h_branch": h_branch,
                            "consecutive_high": ctx.consecutive_high,
                            "gate": decision.reason, "gate_score": decision.score,
                            "site_id": site.id, "event_kind": event.get("kind"),
                            "event_id": event.get("event_id"),
                            "snapshot_ref": event.get("snapshot_ref"),
                            "reward_scheme": site.reward.scheme,
                            "share_observation": site.fork.share_observation,
                            "resume_mode": site.fork.resume_mode,
                            "action_key": action_key_from_messages(msgs),
                            "role": "child",
                            "resume_boundary": resume_boundary_from_messages(msgs),
                            "boundary_unit": "messages",
                            "p_plus": site.reward.p_plus,
                            "k_min": site.reward.k_min,
                            "epsilon_f": site.reward.epsilon_f,
                            "dead_end_backprop": site.reward.dead_end_backprop,
                        },
                    ))
                if n:
                    logging.getLogger(__name__).info(
                        "Branch site=%s event=%s snapshot=%s parent=%s prefix_messages=%s children=%s",
                        site.id, event.get("event_id"), event.get("snapshot_ref"),
                        parent_id, len(msgs), n,
                    )
                budget_left -= n
        return plans

    def run(
        self,
        root_task: Dict[str, Any],
        run_episode: RunEpisodeFn,
        *,
        reward_fn: Optional[RewardFn] = None,
        parent_rollout_id: Optional[str] = None,
    ) -> ActiveSetResult:
        """Run root; optionally execute local siblings; always fill ``plans`` for Daemon."""
        cfg = self.config
        budget = max(1, int(cfg.group_budget))
        result = ActiveSetResult()
        root_id = str(parent_rollout_id or root_task.get("_rollout_id") or uuid4().hex)
        root_task = dict(root_task)
        root_task["_rollout_id"] = root_id

        raw = run_episode(root_task)
        reward = float(reward_fn(raw, root_task)) if reward_fn else None
        result.completed.append(
            CompletedLeaf(
                traj_id=root_id,
                parent_id=None,
                depth=0,
                is_branch=False,
                raw=raw,
                task=dict(root_task),
                reward=reward,
            )
        )

        remaining = budget - len(result.completed)
        plans = self.plan_forks_from_raw(
            raw, parent_id=root_id, remaining=remaining, depth=0,
            root_task=root_task, run_episode=run_episode,
        )
        result.plans.extend(plans)
        result.branch_local_count = len(plans)

        if cfg.execute_local and plans:
            def _run_child(plan: ForkPlan):
                child_id = uuid4().hex
                child_task = dict(root_task)
                child_task.pop("expand_replay", None)
                child_task["resume_messages"] = plan.resume_messages
                child_task["resume_parent_id"] = plan.parent_id
                child_task["_rollout_id"] = child_id
                child_task["is_branch"] = True
                child_task["rollout_role"] = plan.role or "child"
                child_task["site_id"] = plan.site_id
                child_task["action_key"] = (plan.meta or {}).get("action_key")
                child_task["reward_scheme"] = (plan.meta or {}).get("reward_scheme")
                child_task["resume_boundary"] = (plan.meta or {}).get("resume_boundary")
                child_raw = run_episode(child_task)
                child_reward = float(reward_fn(child_raw, child_task)) if reward_fn else None
                return plan, child_id, child_task, child_raw, child_reward

            child_results = []
            to_run = plans[: max(0, budget - len(result.completed))]
            if cfg.parallel_local and len(to_run) > 1:
                from concurrent.futures import ThreadPoolExecutor, as_completed

                with ThreadPoolExecutor(max_workers=min(8, len(to_run))) as pool:
                    futs = [pool.submit(_run_child, p) for p in to_run]
                    for fut in as_completed(futs):
                        child_results.append(fut.result())
            else:
                child_results = [_run_child(p) for p in to_run]

            for plan, child_id, child_task, child_raw, child_reward in child_results:
                if len(result.completed) >= budget:
                    break
                result.completed.append(
                    CompletedLeaf(
                        traj_id=child_id,
                        parent_id=plan.parent_id,
                        depth=plan.depth,
                        is_branch=True,
                        raw=child_raw,
                        task=dict(child_task),
                        reward=child_reward,
                    )
                )
                nested_rem = budget - len(result.completed)
                nested = self.plan_forks_from_raw(
                    child_raw, parent_id=child_id, remaining=nested_rem, depth=plan.depth,
                    root_task=child_task, run_episode=run_episode,
                )
                result.plans.extend(nested)
                result.branch_local_count += len(nested)

        while cfg.execute_local and len(result.completed) < budget:
            fill_id = uuid4().hex
            fill_task = dict(root_task)
            fill_task.pop("resume_messages", None)
            fill_task.pop("resume_parent_id", None)
            fill_task.pop("resume_from", None)
            fill_task.pop("expand_replay", None)
            fill_task["_rollout_id"] = fill_id
            fill_task["is_branch"] = False
            fill_raw = run_episode(fill_task)
            fill_reward = float(reward_fn(fill_raw, fill_task)) if reward_fn else None
            result.completed.append(
                CompletedLeaf(
                    traj_id=fill_id,
                    parent_id=None,
                    depth=0,
                    is_branch=False,
                    raw=fill_raw,
                    task=dict(fill_task),
                    reward=fill_reward,
                )
            )
            result.global_fill_count += 1

        if not cfg.execute_local:
            # Training path: plans consume remaining slots conceptually; global fill
            # left to Daemon after planned resumes are counted.
            pass

        result.metrics = {
            "branch_local_count": int(result.branch_local_count),
            "global_fill_count": int(result.global_fill_count),
            "n_completed": len(result.completed),
            "n_plans": len(result.plans),
            "group_budget": budget,
            "execute_local": bool(cfg.execute_local),
            **self._probe_metrics,
        }
        return result


def dump_local_expansion(parent_rollout_id: str, payload: Dict[str, Any]) -> str:
    path = Path(EXPANSION_DIR)
    path.mkdir(parents=True, exist_ok=True)
    dest = path / f"{parent_rollout_id}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(dest)


def load_local_expansion(parent_rollout_id: str) -> Optional[Dict[str, Any]]:
    dest = Path(EXPANSION_DIR) / f"{parent_rollout_id}.json"
    if not dest.is_file():
        return None
    try:
        return json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        return None


def tree_from_plans(
    parent_id: str,
    plans: List[ForkPlan],
    *,
    task: Optional[Dict[str, Any]] = None,
) -> RolloutTree:
    """Build an explicit RolloutTree from planned siblings (root=parent rollout)."""
    root = RolloutTreeNode(node_id=str(parent_id), role="root", depth=0)
    nodes = [root]
    for i, p in enumerate(plans):
        m = p.meta or {}
        metrics = {
            k: m[k]
            for k in ("h_root", "h_tool", "event_kind", "gate", "site_id", "reward_scheme")
            if k in m
        }
        nodes.append(
            RolloutTreeNode(
                node_id=f"{parent_id}:{i}",
                parent_id=str(parent_id),
                depth=int(p.depth),
                role=str(p.role or "child"),
                metrics=metrics,
                boundary_snapshot_ref=m.get("snapshot_ref") or m.get("action_key"),
            )
        )
    query = str((task or {}).get("question") or (task or {}).get("query") or "")
    return RolloutTree(tree_id=str(parent_id), query=query, nodes=nodes)


def expansion_payload_from_result(result: ActiveSetResult) -> Dict[str, Any]:
    """Serializable plans for Daemon wave-2 enqueue (dual format: tree + flat plans)."""
    parent_ids = [str(p.parent_id) for p in result.plans]
    parent_id = parent_ids[0] if parent_ids else ""
    return {
        "branch_local_count": int(result.branch_local_count),
        "global_fill_count": int(result.global_fill_count),
        "metrics": dict(result.metrics),
        "tree": tree_from_plans(parent_id, list(result.plans)).model_dump(),
        "plans": [
            {
                "resume_messages": p.resume_messages,
                "parent_id": p.parent_id,
                "depth": p.depth,
                "reason": p.reason,
                "meta": p.meta,
                "site_id": p.site_id,
                "role": p.role,
            }
            for p in result.plans
        ],
    }
