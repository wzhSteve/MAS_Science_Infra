"""Same-runner ActiveSet scheduler: async tools + ready-batch LLM.

Messages-level (not vLLM Worker token ActiveSet). Used when
``algorithm.tir.ready_batch=True``.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import uuid4

from workflow.active_set import (
    ActiveSetConfig,
    ActiveSetResult,
    ActiveSetSession,
    CompletedLeaf,
    ForkPlan,
    resume_boundary_from_messages,
)

LlmStepFn = Callable[[Dict[str, Any]], Dict[str, Any]]
"""Return dict with keys: messages, tool_calls (list), done (bool), raw_partial (optional)."""

ToolStepFn = Callable[[Dict[str, Any], List[Any]], Dict[str, Any]]
"""(leaf_state, tool_calls) -> updated state with messages / done / error."""


class LeafState(str, Enum):
    READY = "ready"
    WAITING_LLM = "waiting_llm"
    WAITING_TOOL = "waiting_tool"
    DONE = "done"


@dataclass
class Leaf:
    leaf_id: str
    task: Dict[str, Any]
    messages: List[Dict[str, Any]] = field(default_factory=list)
    state: LeafState = LeafState.READY
    parent_id: Optional[str] = None
    depth: int = 0
    is_branch: bool = False
    role: str = "parent"
    pending_tools: List[Any] = field(default_factory=list)
    tool_started_at: float = 0.0
    tool_finished_at: float = 0.0
    raw: Any = None
    reward: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SchedulerMetrics:
    ready_batch_sizes: List[int] = field(default_factory=list)
    tool_wait_skews: List[float] = field(default_factory=list)
    llm_batch_calls: int = 0
    tool_pool_jobs: int = 0
    local_leaf_count: int = 0

    def as_dict(self) -> Dict[str, Any]:
        skews = self.tool_wait_skews
        batches = self.ready_batch_sizes
        return {
            "ready_batch_size": float(sum(batches) / len(batches)) if batches else 0.0,
            "ready_batch_size_max": float(max(batches) if batches else 0),
            "tool_wait_skew": float(sum(skews) / len(skews)) if skews else 0.0,
            "tool_wait_skew_max": float(max(skews) if skews else 0),
            "llm_batch_calls": int(self.llm_batch_calls),
            "tool_pool_jobs": int(self.tool_pool_jobs),
            "local_leaf_count": int(self.local_leaf_count),
        }


class ActiveSetScheduler:
    """Drain READY leaves → batch LLM → async tools → fork into active."""

    def __init__(
        self,
        config: ActiveSetConfig,
        *,
        llm_step: LlmStepFn,
        tool_step: ToolStepFn,
        finalize: Optional[Callable[[Leaf], Any]] = None,
        reward_fn: Optional[Callable[[Any, Dict[str, Any]], float]] = None,
        max_workers: int = 8,
        max_rounds: int = 64,
        rng: Any = None,
    ) -> None:
        self.config = config
        self.llm_step = llm_step
        self.tool_step = tool_step
        self.finalize = finalize
        self.reward_fn = reward_fn
        self.max_workers = max(1, int(max_workers))
        self.max_rounds = max(1, int(max_rounds))
        self._session = ActiveSetSession(config, rng=rng)
        self.metrics = SchedulerMetrics()

    def run(self, root_task: Dict[str, Any], *, parent_rollout_id: Optional[str] = None) -> ActiveSetResult:
        budget = max(1, int(self.config.group_budget))
        root_id = str(parent_rollout_id or root_task.get("_rollout_id") or uuid4().hex)
        root_task = dict(root_task)
        root_task["_rollout_id"] = root_id
        leaves: List[Leaf] = [
            Leaf(
                leaf_id=root_id,
                task=root_task,
                messages=list(root_task.get("resume_messages") or []),
                state=LeafState.READY,
                role="parent",
            )
        ]
        result = ActiveSetResult()
        plans: List[ForkPlan] = []
        rounds = 0

        while rounds < self.max_rounds:
            rounds += 1
            ready = [L for L in leaves if L.state == LeafState.READY]
            if not ready:
                waiting_tool = [L for L in leaves if L.state == LeafState.WAITING_TOOL]
                if waiting_tool:
                    self._drain_tools(waiting_tool)
                    continue
                break

            self.metrics.ready_batch_sizes.append(len(ready))
            self.metrics.llm_batch_calls += 1
            for L in ready:
                L.state = LeafState.WAITING_LLM
            # Pseudo-batch: parallel LLM via thread pool
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                futs = {pool.submit(self._llm_one, L): L for L in ready}
                for fut in as_completed(futs):
                    L = futs[fut]
                    try:
                        out = fut.result()
                    except Exception as e:
                        L.state = LeafState.DONE
                        L.meta["error"] = str(e)
                        continue
                    L.messages = list(out.get("messages") or L.messages)
                    tool_calls = list(out.get("tool_calls") or [])
                    if out.get("done") or not tool_calls:
                        L.state = LeafState.DONE
                        L.raw = out.get("raw_partial")
                        if self.finalize:
                            L.raw = self.finalize(L)
                        if self.reward_fn and L.raw is not None:
                            L.reward = float(self.reward_fn(L.raw, L.task))
                    else:
                        L.pending_tools = tool_calls
                        L.state = LeafState.WAITING_TOOL
                        L.tool_started_at = time.time()

            waiting = [L for L in leaves if L.state == LeafState.WAITING_TOOL]
            if waiting:
                self._drain_tools(waiting)

            # Fork from newly completed leaves that hit a barrier
            for L in list(leaves):
                if L.state != LeafState.DONE or L.raw is None:
                    continue
                if any(c.traj_id == L.leaf_id for c in result.completed):
                    continue
                result.completed.append(
                    CompletedLeaf(
                        traj_id=L.leaf_id,
                        parent_id=L.parent_id,
                        depth=L.depth,
                        is_branch=L.is_branch,
                        raw=L.raw,
                        task=dict(L.task),
                        reward=L.reward,
                    )
                )
                remaining = budget - len(result.completed) - sum(
                    1 for x in leaves if x.state != LeafState.DONE
                )
                remaining = max(0, budget - len(result.completed) - len(plans))
                new_plans = self._session.plan_forks_from_raw(
                    L.raw, parent_id=L.leaf_id, remaining=max(0, remaining), depth=L.depth
                )
                for plan in new_plans:
                    if len(leaves) + len([p for p in plans if True]) >= budget and len(
                        [x for x in leaves if x.state != LeafState.DONE]
                    ) + len(result.completed) >= budget:
                        break
                    if len(result.completed) + sum(1 for x in leaves if x.state != LeafState.DONE) >= budget:
                        plans.append(plan)
                        continue
                    child_id = uuid4().hex
                    child = Leaf(
                        leaf_id=child_id,
                        task={
                            **dict(root_task),
                            "resume_messages": plan.resume_messages,
                            "resume_parent_id": plan.parent_id,
                            "_rollout_id": child_id,
                            "is_branch": True,
                            "rollout_role": plan.role or "child",
                            "site_id": plan.site_id,
                            "action_key": (plan.meta or {}).get("action_key"),
                            "resume_boundary": (plan.meta or {}).get("resume_boundary")
                            or resume_boundary_from_messages(plan.resume_messages),
                        },
                        messages=deepcopy(plan.resume_messages),
                        state=LeafState.READY,
                        parent_id=plan.parent_id,
                        depth=plan.depth,
                        is_branch=True,
                        role=plan.role or "child",
                        meta=dict(plan.meta or {}),
                    )
                    leaves.append(child)
                    plans.append(plan)

            if len(result.completed) >= budget and all(L.state == LeafState.DONE for L in leaves):
                break
            if all(L.state == LeafState.DONE for L in leaves) and len(result.completed) >= min(
                budget, len(leaves)
            ):
                # global fill
                while len(result.completed) < budget:
                    fill_id = uuid4().hex
                    fill_task = dict(root_task)
                    fill_task.pop("resume_messages", None)
                    fill_task.pop("resume_parent_id", None)
                    fill_task["_rollout_id"] = fill_id
                    fill = Leaf(
                        leaf_id=fill_id,
                        task=fill_task,
                        messages=[],
                        state=LeafState.READY,
                        role="parent",
                    )
                    leaves.append(fill)
                    result.global_fill_count += 1
                if all(L.state == LeafState.DONE for L in leaves):
                    break

        # Capture any leftover DONE leaves
        for L in leaves:
            if L.state == LeafState.DONE and not any(c.traj_id == L.leaf_id for c in result.completed):
                if L.raw is None and self.finalize:
                    L.raw = self.finalize(L)
                if L.raw is not None:
                    result.completed.append(
                        CompletedLeaf(
                            traj_id=L.leaf_id,
                            parent_id=L.parent_id,
                            depth=L.depth,
                            is_branch=L.is_branch,
                            raw=L.raw,
                            task=dict(L.task),
                            reward=L.reward,
                        )
                    )

        result.plans = plans
        result.branch_local_count = sum(1 for p in plans if (p.role or "child") == "child")
        self.metrics.local_leaf_count = len(result.completed)
        result.metrics = {
            **self.metrics.as_dict(),
            "n_completed": len(result.completed),
            "n_plans": len(result.plans),
            "group_budget": budget,
            "execute_local": True,
            "ready_batch": True,
            "scheduler_rounds": rounds,
        }
        return result

    def _llm_one(self, leaf: Leaf) -> Dict[str, Any]:
        payload = {
            "messages": list(leaf.messages),
            "task": dict(leaf.task),
            "leaf_id": leaf.leaf_id,
        }
        return self.llm_step(payload)

    def _drain_tools(self, waiting: List[Leaf]) -> None:
        if not waiting:
            return
        started = [L.tool_started_at or time.time() for L in waiting]
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(waiting))) as pool:
            futs = {
                pool.submit(self.tool_step, {"messages": list(L.messages), "task": L.task}, list(L.pending_tools)): L
                for L in waiting
            }
            self.metrics.tool_pool_jobs += len(futs)
            finish_times: List[float] = []
            for fut in as_completed(futs):
                L = futs[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    L.state = LeafState.DONE
                    L.meta["error"] = str(e)
                    L.tool_finished_at = time.time()
                    finish_times.append(L.tool_finished_at)
                    continue
                L.messages = list(out.get("messages") or L.messages)
                L.pending_tools = []
                L.tool_finished_at = time.time()
                finish_times.append(L.tool_finished_at)
                if out.get("done"):
                    L.state = LeafState.DONE
                    L.raw = out.get("raw_partial")
                    if self.finalize:
                        L.raw = self.finalize(L)
                else:
                    L.state = LeafState.READY
        if finish_times and started:
            skew = max(finish_times) - min(started)
            # better: max finish - min finish among this batch
            skew = max(finish_times) - min(finish_times)
            self.metrics.tool_wait_skews.append(float(skew))
