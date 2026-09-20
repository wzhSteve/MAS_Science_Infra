"""TIR daemon: extra non_tensor fields + ARPO/AEPO tree expansion enqueue."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from agentlightning.types import EnqueueRolloutRequest, Rollout, RolloutConfig, RolloutLegacy, Task
from agentlightning.verl.daemon import AgentModeDaemon, _to_native

from .arpo_rollout import (
    aepo_global_budget,
    branch_probability,
    should_branch,
)
from .branch_policy import distribute_root_budgets
from workflow.active_set import load_local_expansion, resume_boundary_from_messages
from workflow.archive import load_resume_messages
from rl.hooks.rae_advantage import adjudicate_action_group, apply_dead_end_backprop_verdicts


def _persist_rollout_tree(tree_id: str, tree: Dict[str, Any]) -> None:
    """Persist one RolloutTree to mas/.local_expansion/ so the WebUI
    /api/mas/rollout-trees endpoint (which scans that dir) can serve it.

    Written next to dump_local_expansion()'s expansion payloads; filename
    is prefixed with "tree_" to distinguish UI trees from expansion plans.
    Failures are swallowed: persistence must never break training.
    """
    import json as _json
    import os as _os

    try:
        from workflow.active_set import EXPANSION_DIR
    except Exception:
        return
    try:
        root = _os.environ.get("TIR_LOCAL_EXPANSION_DIR") or EXPANSION_DIR
        _os.makedirs(root, exist_ok=True)
        dest = _os.path.join(root, f"tree_{tree_id}.json")
        with open(dest, "w", encoding="utf-8") as f:
            f.write(_json.dumps({"tree_id": tree_id, **tree}, ensure_ascii=False))
    except Exception:
        pass


def emit_rollout_tree_event(event: Dict[str, Any]) -> None:
    """P3 real-time harness: emit a marked JSONL frame to stdout.

    Frame shape: {"__rollout_tree_event__": {...}} — one per line; the Control
    SSE loop parses these marker frames and fans them out as `rollout_tree`
    events to the WebUI. Failures are swallowed (stream is best-effort).
    """
    import json
    import sys

    try:
        sys.stdout.write(json.dumps({"__rollout_tree_event__": event}, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def resume_boundary_proxy(messages: Any) -> int:
    return resume_boundary_from_messages(messages if isinstance(messages, list) else None)


def _span_attrs(span: Any) -> Dict[str, Any]:
    attrs = getattr(span, "attributes", None) or {}
    if hasattr(attrs, "items"):
        return dict(attrs)
    return {}


def extract_tir_meta_from_spans(spans: List[Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for span in spans or []:
        attrs = _span_attrs(span)
        for key, val in attrs.items():
            k = str(key)
            if k.startswith("tir.") or k.startswith("tir_"):
                meta[k.split(".", 1)[-1].replace("tir_", "")] = val
    return meta


class TirAgentModeDaemon(AgentModeDaemon):
    """Extends AgentModeDaemon with tir_meta on the batch and optional tree enqueue."""

    def __init__(self, *args: Any, tir_algo: str = "grpo", tir_config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.tir_algo = (tir_algo or "grpo").lower()
        self.tir_config = tir_config or {}
        self._full_group_n = int(getattr(self, "train_rollout_n", 1) or 1)
        self._pending_original_by_data_id: Dict[str, Dict[str, Any]] = {}
        self._rollout_meta: Dict[str, Dict[str, Any]] = {}
        self._store_enqueue_branch_count = 0
        self._branch_local_count_total = 0
        self._enqueued_expansion_parents: set = set()
        self._incremental_branch_total = 0
        self._rollout_trees: Dict[str, Any] = {}  # data_id -> RolloutTree dict (P0 bypass write)

    def _expand_in_runner(self) -> bool:
        return bool(self.tir_config.get("expand_in_runner", True))

    def _ready_batch(self) -> bool:
        return bool(self.tir_config.get("ready_batch", False))

    async def _validate_data_v1(self, rollout: Rollout) -> RolloutLegacy:
        spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")
        # --- debug probe: dump spans/triplets for empty-triplet diagnosis ---
        import json as _json
        import os as _os

        _dbg = _os.environ.get("TIR_DEBUG_SPANS", "").strip()
        if _dbg:
            try:
                _names = [str(getattr(s, "name", "?")) for s in spans]
                _payload = {
                    "rollout_id": rollout.rollout_id,
                    "n_spans": len(spans),
                    "span_names": _names[:80],
                    "adapter": type(self.adapter).__name__,
                    "agent_match": getattr(self.adapter, "agent_match", None),
                    "llm_call_match": getattr(self.adapter, "llm_call_match", None),
                    "attempts": [
                        {
                            "span": i,
                            "name": str(getattr(s, "name", "?")),
                            "attrs": {k: v for k, v in list(dict(getattr(s, "attributes", None) or {}).items()) if k in (
                                "langchain.chain.type", "agent.name", "operation.name", "agentops.span.kind",
                                "lc_name", "langchain.llm.model", "gen_ai.request.model",
                            )},
                            "parent_id": getattr(s, "parent_id", None),
                            "span_id": getattr(s, "span_id", None),
                        }
                        for i, s in enumerate(spans[:80])
                    ],
                    "n_triplets": -1,  # filled below
                }
                _t = self.adapter.adapt(spans)
                _payload["n_triplets"] = len(_t)
                with open(_dbg, "a", encoding="utf-8") as f:
                    f.write(_json.dumps(_payload, ensure_ascii=False) + "\n")
            except Exception:
                pass
        # --- end debug probe ---
        if not spans:
            triplets = []
        else:
            triplets = self.adapter.adapt(spans)
        tir_meta = extract_tir_meta_from_spans(spans)
        self._rollout_meta[rollout.rollout_id] = tir_meta
        try:
            self._branch_local_count_total += int(tir_meta.get("branch_local_count") or 0)
        except (TypeError, ValueError):
            pass

        final_reward: Optional[float] = None
        if triplets:
            for triplet in reversed(triplets):
                if triplet.reward is not None:
                    final_reward = triplet.reward
                    break

        task = Task(
            rollout_id=rollout.rollout_id,
            input=rollout.input,
            mode=rollout.mode,
            resources_id=rollout.resources_id,
            metadata=rollout.metadata or {},
        )
        merged_meta = dict(rollout.metadata or {})
        merged_meta["tir"] = tir_meta
        result_rollout = RolloutLegacy(
            rollout_id=rollout.rollout_id,
            task=task,
            final_reward=final_reward,
            triplets=triplets,
            metadata=merged_meta,
        )
        self._validate_data(result_rollout)
        if (
            self.is_train
            and self.tir_algo in ("arpo", "aepo", "rae")
            and self._expand_in_runner()
            and self._ready_batch()
        ):
            try:
                n_inc = await self._enqueue_expansion_for_parent(rollout.rollout_id)
                self._incremental_branch_total += int(n_inc or 0)
            except Exception:
                pass
        return result_rollout

    async def _async_set_up(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True):
        orig_n = int(self.train_rollout_n)
        self._full_group_n = orig_n
        if is_train and self.tir_algo in ("arpo", "aepo", "rae"):
            if self.tir_algo == "aepo":
                self.train_rollout_n = 1
            else:
                n_init = int(self.tir_config.get("initial_rollouts", 2))
                self.train_rollout_n = max(1, min(n_init, orig_n))
        # Pre-inject expand fields into the raw data BEFORE super() enqueues tasks.
        # super() snapshots each sample via _to_native() at enqueue time, so
        # post-hoc mutation of daemon-side dicts (the old approach) never reached
        # the agent and ARPO branching silently never happened (branch_local=0).
        if is_train and self.tir_algo in ("arpo", "aepo", "rae") and self._expand_in_runner():
            self._preinject_expand_fields(data, n_init=int(self.train_rollout_n) if self.tir_algo != "aepo" else 1)
        try:
            await super()._async_set_up(data, server_addresses, is_train=is_train)
        finally:
            self.train_rollout_n = orig_n
        self._pending_original_by_data_id = {}
        for _rid, sample in self._task_id_to_original_sample.items():
            did = str(sample.get("data_id", ""))
            if did:
                self._pending_original_by_data_id[did] = sample

        if is_train and self.tir_algo in ("arpo", "aepo", "rae") and self._expand_in_runner():
            self._stamp_expand_budgets()
            # Pass declared sites into wave-1 samples for ActiveSetSession
            sites = list(self.tir_config.get("sites") or [])
            for sample in self._task_id_to_original_sample.values():
                if sites and "branch_sites" not in sample:
                    sample["branch_sites"] = sites

    def _preinject_expand_fields(self, data: Dict[str, Any], *, n_init: int) -> None:
        """Attach expand fields onto raw batch columns before super() enqueues.

        super() builds each task input as {key: data[key][i]} then snapshots it
        via _to_native(); anything stamped only after super() returns is lost.
        Here each sample i gets one column-level scalar: expand_in_runner,
        sampling_budget (per-root budget = full group_n split across the
        n_init roots of that sample), plus gate/fork params and branch sites.
        All wave-1 rollouts of one sample share the same fields — correct,
        because each root independently plans up to its own budget.
        """
        try:
            keys = list(data.keys())
            if not keys:
                return
            num_samples = len(data[keys[0]])
        except Exception:
            return
        n_target = int(self._full_group_n or self.train_rollout_n or 1)
        # Per-root budget: group_n split across the wave-1 roots of one sample.
        budget = max(1, (n_target + n_init - 1) // max(1, n_init))
        fields: Dict[str, Any] = {
            "expand_in_runner": True,
            "sampling_budget": budget,
            "max_branch_depth": int(self.tir_config.get("max_branch_depth", 2)),
            "tir_beam_size": int(self.tir_config.get("beam_size", 2)),
            "tir_branch_probability": float(self.tir_config.get("branch_probability", 0.5)),
            "tir_entropy_weight": float(self.tir_config.get("entropy_weight", 0.5)),
            "tir_use_official_arpo_gate": bool(self.tir_config.get("use_official_arpo_gate", True)),
            "ready_batch": bool(self.tir_config.get("ready_batch", False)),
            # NOTE: no force_local_expand — training hot path keeps
            # execute_local=False (plan-only) so the Daemon materializes resume
            # rollouts through the store (full span/GRPO observability).
            # local_expand (collect/mock modes) is a collect-path concern.
        }
        sites = list(self.tir_config.get("sites") or [])
        for key, value in fields.items():
            if key not in data:
                data[key] = [value] * num_samples
        if sites and "branch_sites" not in data:
            data["branch_sites"] = [sites] * num_samples

    def _stamp_expand_budgets(self) -> None:
        """Attach expand_in_runner + per-root sampling_budget onto wave-1 samples."""
        n_target = int(getattr(self, "_full_group_n", None) or self.train_rollout_n)
        by_data: Dict[str, List[str]] = {}
        for rid, sample in self._task_id_to_original_sample.items():
            by_data.setdefault(str(sample.get("data_id")), []).append(rid)
        for data_id, rids in by_data.items():
            budgets = distribute_root_budgets(n_target, len(rids))
            for rid, budget in zip(rids, budgets):
                sample = self._task_id_to_original_sample[rid]
                sample["expand_in_runner"] = True
                sample["sampling_budget"] = int(budget)
                sample["max_branch_depth"] = int(self.tir_config.get("max_branch_depth", 2))
                sample["tir_beam_size"] = int(self.tir_config.get("beam_size", 2))
                sample["tir_branch_probability"] = float(self.tir_config.get("branch_probability", 0.5))
                sample["tir_entropy_weight"] = float(self.tir_config.get("entropy_weight", 0.5))
                sample["tir_use_official_arpo_gate"] = bool(self.tir_config.get("use_official_arpo_gate", True))
                sample["ready_batch"] = bool(self.tir_config.get("ready_batch", False))

    async def _enqueue_expansion_for_parent(self, parent_rid: str) -> int:
        """Incremental enqueue: materialize one root's plans as soon as it finishes."""
        if parent_rid in self._enqueued_expansion_parents:
            return 0
        expansion = load_local_expansion(parent_rid) or {}
        plans = list(expansion.get("plans") or [])
        if not plans:
            return 0
        sample_parent = self._task_id_to_original_sample.get(parent_rid) or {}
        data_id = str(sample_parent.get("data_id") or "")
        if not data_id:
            return 0
        n_target = int(getattr(self, "_full_group_n", None) or self.train_rollout_n)
        existing = [rid for rid, s in self._task_id_to_original_sample.items() if str(s.get("data_id")) == data_id]
        remaining = max(0, n_target - len(existing))
        if remaining <= 0:
            self._enqueued_expansion_parents.add(parent_rid)
            return 0
        original = dict(self._pending_original_by_data_id.get(data_id) or sample_parent)
        resources_id = None
        parent_rollout = self._completed_rollouts_v0.get(parent_rid)
        if parent_rollout and getattr(parent_rollout, "task", None):
            resources_id = parent_rollout.task.resources_id
        llm_timeout = self.llm_timeout_seconds
        requests: List[EnqueueRolloutRequest] = []
        samples_for_requests: List[Dict[str, Any]] = []
        for plan in plans:
            if remaining <= 0:
                break
            messages = plan.get("resume_messages")
            if not messages:
                continue
            sample = dict(original)
            sample["resume_messages"] = messages
            sample["resume_parent_id"] = str(plan.get("parent_id") or parent_rid)
            sample["is_branch"] = True
            plan_meta = dict(plan.get("meta") or {})
            sample["rollout_role"] = str(plan.get("role") or plan_meta.get("role") or "child")
            sample["site_id"] = str(plan.get("site_id") or plan_meta.get("site_id") or "")
            if plan_meta.get("action_key") is not None:
                sample["action_key"] = plan_meta.get("action_key")
            if plan_meta.get("reward_scheme") is not None:
                sample["reward_scheme"] = plan_meta.get("reward_scheme")
            try:
                sample["resume_boundary"] = int(
                    plan_meta.get("resume_boundary") or resume_boundary_proxy(messages)
                )
            except (TypeError, ValueError):
                sample["resume_boundary"] = resume_boundary_proxy(messages)
            sample.pop("expand_in_runner", None)
            sample.pop("sampling_budget", None)
            sample.pop("resume_from", None)
            samples_for_requests.append(sample)
            requests.append(
                EnqueueRolloutRequest(
                    input=_to_native(sample),
                    mode="train",
                    resources_id=resources_id,
                    config=RolloutConfig(
                        unresponsive_seconds=llm_timeout,
                        timeout_seconds=llm_timeout,
                    ),
                    metadata={"data_id": data_id, "is_train": True, "tir_branch": True},
                )
            )
            remaining -= 1
            self._store_enqueue_branch_count += 1
        if not requests:
            return 0
        rollouts = await self.store.enqueue_many_rollouts(requests)
        for rollout, sample in zip(rollouts, samples_for_requests):
            packed = dict(sample)
            did = str((rollout.metadata or {}).get("data_id") or packed.get("data_id") or "")
            packed["data_id"] = did
            self._task_id_to_original_sample[rollout.rollout_id] = packed
        self._total_tasks_queued += len(rollouts)
        self._enqueued_expansion_parents.add(parent_rid)
        # Incremental path: persist tree + emit node_added (same as the batch
        # path in _enqueue_from_runner_expansions) so the RolloutTree UI panel
        # updates live under ready_batch=True, which routes here.
        tree = dict(expansion.get("tree") or {}) or None
        if tree is not None:
            branch_rollouts = [r for r, s in zip(rollouts, samples_for_requests) if s.get("is_branch")]
            child_nodes = [n for n in (tree.get("nodes") or []) if n.get("role") != "root"]
            for node, rollout in zip(child_nodes, branch_rollouts):
                node["node_id"] = str(rollout.rollout_id)
            tree_id = str(tree.get("tree_id") or parent_rid)
            existing = self._rollout_trees.get(tree_id)
            if existing:
                seen = {n.get("node_id") for n in (existing.get("nodes") or [])}
                new_nodes = [n for n in tree.get("nodes") or [] if n.get("node_id") not in seen]
                existing.setdefault("nodes", []).extend(new_nodes)
            else:
                self._rollout_trees[tree_id] = tree
            _persist_rollout_tree(tree_id, self._rollout_trees[tree_id])
            emit_rollout_tree_event(
                {
                    "event": "node_added",
                    "tree_id": tree_id,
                    "node_id": None,
                    "payload": {
                        "n_nodes": len(self._rollout_trees[tree_id].get("nodes") or []),
                        "n_new": len(child_nodes),
                    },
                }
            )
        return len(rollouts)

    async def _async_run_until_finished(self, verbose: bool = True):
        await super()._async_run_until_finished(verbose=verbose)
        if not self.is_train or self.tir_algo not in ("arpo", "aepo", "rae"):
            return
        if self._expand_in_runner():
            extra = await self._enqueue_from_runner_expansions()
        else:
            extra = await self._enqueue_tree_branches_legacy()
        if extra <= 0:
            return
        if verbose:
            print(f"[TIR {self.tir_algo}] enqueued {extra} branch/resume rollouts; waiting...")
        await super()._async_run_until_finished(verbose=verbose)

    async def _enqueue_from_runner_expansions(self) -> int:
        """Materialize ActiveSetSession plans + global fill to reach group_n.

        Sibling resume tasks still run LLM once (AGL span / GRPO tokens).
        Daemon no longer recomputes entropy gates when expand_in_runner=True.
        """
        if self.mode != "v1":
            return 0
        n_target = int(getattr(self, "_full_group_n", None) or self.train_rollout_n)
        by_data: Dict[str, List[str]] = {}
        for rid, sample in self._task_id_to_original_sample.items():
            by_data.setdefault(str(sample.get("data_id")), []).append(rid)

        requests: List[EnqueueRolloutRequest] = []
        samples_for_requests: List[Dict[str, Any]] = []

        for data_id, rids in by_data.items():
            remaining = max(0, n_target - len(rids))
            if remaining <= 0:
                continue
            original = dict(
                self._pending_original_by_data_id.get(data_id) or self._task_id_to_original_sample[rids[0]]
            )
            llm_timeout = self.llm_timeout_seconds
            resources_id = None
            parent_rollout = self._completed_rollouts_v0.get(rids[0])
            if parent_rollout and getattr(parent_rollout, "task", None):
                resources_id = parent_rollout.task.resources_id

            def _mk_req(sample: Dict[str, Any], *, is_branch: bool) -> EnqueueRolloutRequest:
                return EnqueueRolloutRequest(
                    input=_to_native(sample),
                    mode="train",
                    resources_id=resources_id,
                    config=RolloutConfig(
                        unresponsive_seconds=llm_timeout,
                        timeout_seconds=llm_timeout,
                    ),
                    metadata={
                        "data_id": data_id,
                        "is_train": True,
                        "tir_branch": bool(is_branch),
                    },
                )

            for rid in rids:
                if remaining <= 0:
                    break
                if rid in self._enqueued_expansion_parents:
                    continue
                expansion = load_local_expansion(rid) or {}
                plans = list(expansion.get("plans") or [])
                tree = dict(expansion.get("tree") or {}) or None  # P0: bypass tree
                meta = self._rollout_meta.get(rid) or {}
                try:
                    self._branch_local_count_total += int(
                        expansion.get("branch_local_count") or meta.get("branch_local_count") or 0
                    )
                except (TypeError, ValueError):
                    pass
                for plan in plans:
                    if remaining <= 0:
                        break
                    messages = plan.get("resume_messages")
                    if not messages:
                        continue
                    sample = dict(original)
                    sample["resume_messages"] = messages
                    sample["resume_parent_id"] = str(plan.get("parent_id") or rid)
                    sample["is_branch"] = True
                    plan_meta = dict(plan.get("meta") or {})
                    role = str(plan.get("role") or plan_meta.get("role") or "child")
                    sample["rollout_role"] = role
                    sample["site_id"] = str(plan.get("site_id") or plan_meta.get("site_id") or "")
                    if plan_meta.get("action_key") is not None:
                        sample["action_key"] = plan_meta.get("action_key")
                    if plan_meta.get("reward_scheme") is not None:
                        sample["reward_scheme"] = plan_meta.get("reward_scheme")
                    try:
                        sample["resume_boundary"] = int(
                            plan_meta.get("resume_boundary")
                            or resume_boundary_proxy(messages)
                        )
                    except (TypeError, ValueError):
                        sample["resume_boundary"] = resume_boundary_proxy(messages)
                    sample["boundary_unit"] = str(plan_meta.get("boundary_unit") or "messages")
                    for k in ("p_plus", "k_min", "epsilon_f", "dead_end_backprop", "resume_mode"):
                        if k in plan_meta:
                            sample[k] = plan_meta[k]
                    if str(sample.get("resume_mode") or "messages").lower() == "token_prefix":
                        # Engine may lack token resume; keep messages and mark downgrade.
                        sample["resume_mode"] = "messages"
                        sample["resume_mode_requested"] = "token_prefix"
                        sample["resume_mode_downgraded"] = True
                    sample.pop("expand_in_runner", None)
                    sample.pop("sampling_budget", None)
                    sample.pop("resume_from", None)
                    samples_for_requests.append(sample)
                    requests.append(_mk_req(sample, is_branch=True))
                    remaining -= 1
                    self._store_enqueue_branch_count += 1
                self._enqueued_expansion_parents.add(rid)

            for _ in range(remaining):
                sample = dict(original)
                sample.pop("resume_messages", None)
                sample.pop("resume_parent_id", None)
                sample.pop("resume_from", None)
                sample.pop("expand_in_runner", None)
                sample.pop("sampling_budget", None)
                sample["is_branch"] = False
                samples_for_requests.append(sample)
                requests.append(_mk_req(sample, is_branch=False))

        if not requests:
            return 0
        rollouts = await self.store.enqueue_many_rollouts(requests)
        for rollout, sample in zip(rollouts, samples_for_requests):
            packed = dict(sample)
            did = str((rollout.metadata or {}).get("data_id") or packed.get("data_id") or "")
            packed["data_id"] = did
            self._task_id_to_original_sample[rollout.rollout_id] = packed
        self._total_tasks_queued += len(rollouts)

        # P0 bypass write: rewrite synthetic tree node ids with real store rollout ids.
        if tree is not None:
            branch_rollouts = [r for r, s in zip(rollouts, samples_for_requests) if s.get("is_branch")]
            child_nodes = [n for n in (tree.get("nodes") or []) if n.get("role") != "root"]
            for node, rollout in zip(child_nodes, branch_rollouts):
                node["node_id"] = str(rollout.rollout_id)
            tree_id = str(tree.get("tree_id") or "")
            if tree_id:
                existing = self._rollout_trees.get(tree_id)
                if existing:
                    seen = {n.get("node_id") for n in (existing.get("nodes") or [])}
                    new_nodes = [n for n in tree.get("nodes") or [] if n.get("node_id") not in seen]
                    existing.setdefault("nodes", []).extend(new_nodes)
                else:
                    self._rollout_trees[tree_id] = tree
                _persist_rollout_tree(tree_id, self._rollout_trees[tree_id])
                emit_rollout_tree_event(
                    {
                        "event": "node_added",
                        "tree_id": tree_id,
                        "node_id": None,
                        "payload": {
                            "n_nodes": len(self._rollout_trees[tree_id].get("nodes") or []),
                            "n_new": len(
                                [n for n in (tree.get("nodes") or []) if n.get("role") != "root"]
                            ),
                        },
                    }
                )
        return len(rollouts)

    async def _enqueue_tree_branches_legacy(self) -> int:
        """Legacy Daemon-side entropy gate (expand_in_runner=False)."""
        if self.mode != "v1":
            return 0
        n_target = int(getattr(self, "_full_group_n", None) or self.train_rollout_n)
        beam = int(self.tir_config.get("beam_size", 2))
        tau = float(self.tir_config.get("entropy_threshold", 0.15))
        alpha = float(self.tir_config.get("branch_probability", 0.5))
        gamma = float(self.tir_config.get("entropy_weight", 0.2))
        slope = float(self.tir_config.get("branch_penalty_slope", 0.25))
        beta = float(self.tir_config.get("aepo_beta", 1.0))

        by_data: Dict[str, List[str]] = {}
        for rid, sample in self._task_id_to_original_sample.items():
            by_data.setdefault(str(sample.get("data_id")), []).append(rid)

        requests: List[EnqueueRolloutRequest] = []
        samples_for_requests: List[Dict[str, Any]] = []

        for data_id, rids in by_data.items():
            remaining = max(0, n_target - len(rids))
            if remaining <= 0:
                continue
            parent = rids[0]
            resume = load_resume_messages(parent) or {}
            meta = self._rollout_meta.get(parent) or {}
            h_root = float(resume.get("h_root", meta.get("h_root", 0.0)) or 0.0)
            h_tool = float(resume.get("h_tool", meta.get("h_tool", 0.0)) or 0.0)
            delta_h = h_tool - h_root
            consecutive = int(resume.get("consecutive_high", 0) or 0)
            if self.tir_algo == "aepo":
                m = aepo_global_budget(n_target, h_root, h_tool, beta=beta)
                extra_global = max(0, m - len(rids))
                remaining = n_target - len(rids)
                branch_budget = max(0, remaining - extra_global)
                p = branch_probability(
                    delta_h, alpha=alpha, gamma=gamma, consecutive_high=consecutive, penalty_slope=slope
                )
            else:
                branch_budget = remaining
                p = alpha + gamma * delta_h

            original = dict(self._pending_original_by_data_id.get(data_id) or self._task_id_to_original_sample[parent])
            messages = resume.get("messages")
            do_branch = should_branch(p, tau) and messages
            n_branch = min(branch_budget, beam) if do_branch else 0
            n_global = remaining - n_branch

            llm_timeout = self.llm_timeout_seconds
            resources_id = None
            parent_rollout = self._completed_rollouts_v0.get(parent)
            if parent_rollout and getattr(parent_rollout, "task", None):
                resources_id = parent_rollout.task.resources_id

            def _mk_req(sample: Dict[str, Any]) -> EnqueueRolloutRequest:
                return EnqueueRolloutRequest(
                    input=_to_native(sample),
                    mode="train",
                    resources_id=resources_id,
                    config=RolloutConfig(
                        unresponsive_seconds=llm_timeout,
                        timeout_seconds=llm_timeout,
                    ),
                    metadata={"data_id": data_id, "is_train": True, "tir_branch": True},
                )

            for _ in range(n_branch):
                sample = dict(original)
                sample["resume_messages"] = messages
                sample["resume_parent_id"] = parent
                if resume.get("archive_id") and resume.get("snapshot_id"):
                    sample["resume_from"] = {
                        "archive_id": resume["archive_id"],
                        "snapshot_id": resume["snapshot_id"],
                        "parent_rollout_id": parent,
                        "reason": "arpo_branch",
                        "meta": {
                            "h_root": h_root,
                            "h_tool": h_tool,
                            "consecutive_high": consecutive,
                        },
                    }
                samples_for_requests.append(sample)
                requests.append(_mk_req(sample))
                self._store_enqueue_branch_count += 1
            for _ in range(n_global):
                sample = dict(original)
                sample.pop("resume_messages", None)
                sample.pop("resume_parent_id", None)
                samples_for_requests.append(sample)
                requests.append(_mk_req(sample))

        if not requests:
            return 0
        rollouts = await self.store.enqueue_many_rollouts(requests)
        for rollout, sample in zip(rollouts, samples_for_requests):
            packed = dict(sample)
            did = str((rollout.metadata or {}).get("data_id") or packed.get("data_id") or "")
            packed["data_id"] = did
            self._task_id_to_original_sample[rollout.rollout_id] = packed
        self._total_tasks_queued += len(rollouts)
        return len(rollouts)

    # Back-compat alias for callers / tests that still patch this name.
    async def _enqueue_tree_branches(self) -> int:
        if self._expand_in_runner():
            return await self._enqueue_from_runner_expansions()
        return await self._enqueue_tree_branches_legacy()

    def get_train_data_batch(self, max_prompt_length: int, max_response_length: int, device, global_steps: int):
        data_proto, data_metrics = super().get_train_data_batch(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            device=device,
            global_steps=global_steps,
        )

        def _as_list(key: str) -> List[Any]:
            raw = data_proto.non_tensor_batch.get(key, None)
            if raw is None:
                return []
            if isinstance(raw, np.ndarray):
                return raw.tolist()
            return list(raw)

        rollout_ids = _as_list("rollout_id_list")
        n = len(rollout_ids)
        turn_raw = _as_list("turn_index_list")
        turn_indices = turn_raw if turn_raw else [0] * n
        anchors: List[str] = []
        ig_deltas: List[float] = []
        h_roots: List[float] = []
        for i, rid in enumerate(rollout_ids):
            meta = self._rollout_meta.get(str(rid)) or {}
            hashes = str(meta.get("obs_hashes") or "")
            parts = [p for p in hashes.split("|") if p]
            turn = int(turn_indices[i]) if i < len(turn_indices) else 0
            if parts:
                anchors.append(parts[min(turn, len(parts) - 1)])
            else:
                anchors.append("")
            ig_blob = str(meta.get("ig_deltas") or "")
            ig_parts = [float(x) for x in ig_blob.split("|") if x not in ("", None)]
            if ig_parts:
                ig_deltas.append(ig_parts[min(turn, len(ig_parts) - 1)])
            else:
                ig_deltas.append(0.0)
            try:
                h_roots.append(float(meta.get("h_root") or 0.0))
            except (TypeError, ValueError):
                h_roots.append(0.0)

        if n:
            data_proto.non_tensor_batch["anchor_obs"] = np.array(anchors, dtype=object)
            data_proto.non_tensor_batch["ig_logprob_delta"] = np.array(ig_deltas, dtype=np.float64)
            data_proto.non_tensor_batch["h_root"] = np.array(h_roots, dtype=np.float64)
            roles: List[str] = []
            site_ids: List[str] = []
            verdicts: List[str] = []
            bounds: List[int] = []
            action_keys: List[str] = []
            schemes: List[str] = []
            successes: List[bool] = []
            data_ids_nt: List[str] = []
            parent_ids: List[str] = []
            for rid in rollout_ids:
                meta = self._rollout_meta.get(str(rid)) or {}
                sample = self._task_id_to_original_sample.get(str(rid)) or {}
                role = str(
                    meta.get("role")
                    or sample.get("rollout_role")
                    or ("child" if sample.get("resume_messages") or sample.get("is_branch") else "parent")
                )
                roles.append(role)
                site_ids.append(str(meta.get("site_id") or sample.get("site_id") or ""))
                action_keys.append(str(meta.get("action_key") or sample.get("action_key") or ""))
                schemes.append(str(meta.get("reward_scheme") or sample.get("reward_scheme") or "scalar_grpo"))
                data_ids_nt.append(str(sample.get("data_id") or meta.get("data_id") or ""))
                parent_ids.append(str(sample.get("resume_parent_id") or meta.get("resume_parent_id") or ""))
                try:
                    rewards_row = float(meta.get("reward") if meta.get("reward") is not None else -1.0)
                except (TypeError, ValueError):
                    rewards_row = -1.0
                fmt_ok = meta.get("format_ok")
                if fmt_ok is None:
                    fmt_ok = rewards_row >= 0.5
                successes.append(bool(fmt_ok) if rewards_row < 0 else bool(rewards_row > 0))
                try:
                    bounds.append(int(meta.get("resume_boundary") or sample.get("resume_boundary") or 0))
                except (TypeError, ValueError):
                    bounds.append(0)

            # Fill verdicts: prefer explicit meta, else adjudicate rae_adjudicate groups.
            explicit = []
            for rid in rollout_ids:
                meta = self._rollout_meta.get(str(rid)) or {}
                sample = self._task_id_to_original_sample.get(str(rid)) or {}
                explicit.append(str(meta.get("verdict") or sample.get("verdict") or ""))
            from collections import defaultdict

            groups: Dict[tuple, List[int]] = defaultdict(list)
            for i, (did, ak, scheme) in enumerate(zip(data_ids_nt, action_keys, schemes)):
                if str(scheme).lower() != "rae_adjudicate":
                    continue
                if not ak:
                    continue
                groups[(did, ak)].append(i)
            computed = ["none"] * len(rollout_ids)
            p_plus = float(self.tir_config.get("rae_p_plus", 0.8))
            k_min = int(self.tir_config.get("rae_k_min", 2))
            for (_did, _ak), idxs in groups.items():
                outcomes = [successes[j] for j in idxs]
                # pull per-sample overrides if present
                sample0 = self._task_id_to_original_sample.get(str(rollout_ids[idxs[0]])) or {}
                pp = float(sample0.get("p_plus", p_plus))
                km = int(sample0.get("k_min", k_min))
                v = adjudicate_action_group(outcomes, p_plus=pp, k_min=km)
                for j in idxs:
                    computed[j] = v
            for i, ex in enumerate(explicit):
                if ex and ex.lower() not in ("", "none"):
                    verdicts.append(ex)
                else:
                    verdicts.append(computed[i] if i < len(computed) else "none")
            if bool(self.tir_config.get("rae_dead_end_backprop", True)):
                verdicts = apply_dead_end_backprop_verdicts(
                    verdicts,
                    roles=roles,
                    parent_ids=parent_ids,
                    data_ids=data_ids_nt,
                    depth=int(self.tir_config.get("rae_dead_end_depth", 1)),
                )
            data_proto.non_tensor_batch["rollout_role"] = np.array(roles, dtype=object)
            data_proto.non_tensor_batch["site_id_list"] = np.array(site_ids, dtype=object)
            data_proto.non_tensor_batch["verdict_list"] = np.array(verdicts, dtype=object)
            data_proto.non_tensor_batch["resume_boundary"] = np.array(bounds, dtype=object)
            data_proto.non_tensor_batch["action_key_list"] = np.array(action_keys, dtype=object)
        algo_id = {"grpo": 0.0, "arpo": 1.0, "aepo": 2.0, "igpo": 3.0, "gigpo": 4.0, "rae": 5.0}
        data_metrics["training/tir_algo"] = algo_id.get(self.tir_algo, 0.0)
        data_metrics["training/n_with_anchor"] = float(sum(1 for a in anchors if a))
        data_metrics["training/branch_local_count"] = float(self._branch_local_count_total)
        data_metrics["training/store_enqueue_branch_count"] = float(self._store_enqueue_branch_count)
        data_metrics["training/incremental_branch_count"] = float(self._incremental_branch_total)
        # P3: loss/batch event for real-time harness (marker frame on stdout)
        try:
            emit_rollout_tree_event(
                {
                    "event": "loss",
                    "tree_id": "global",
                    "node_id": None,
                    "payload": {
                        "metrics": {k: float(v) for k, v in dict(data_metrics).items() if isinstance(v, (int, float))},
                    },
                }
            )
        except Exception:
            pass
        return data_proto, data_metrics

    def clear_data_and_server(self):
        super().clear_data_and_server()
        self._pending_original_by_data_id.clear()
        self._rollout_meta.clear()
        self._store_enqueue_branch_count = 0
        self._branch_local_count_total = 0
        self._enqueued_expansion_parents.clear()
        self._incremental_branch_total = 0
