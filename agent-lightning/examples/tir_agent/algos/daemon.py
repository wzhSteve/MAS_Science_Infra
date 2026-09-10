"""TIR daemon: extra non_tensor fields + ARPO/AEPO two-phase enqueue."""

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
from workflow.archive import load_resume_messages


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

    async def _validate_data_v1(self, rollout: Rollout) -> RolloutLegacy:
        spans = await self.store.query_spans(rollout.rollout_id, attempt_id="latest")
        if not spans:
            triplets = []
        else:
            triplets = self.adapter.adapt(spans)
        tir_meta = extract_tir_meta_from_spans(spans)
        self._rollout_meta[rollout.rollout_id] = tir_meta

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
        return result_rollout

    async def _async_set_up(self, data: Dict[str, Any], server_addresses: List[str], is_train: bool = True):
        orig_n = int(self.train_rollout_n)
        self._full_group_n = orig_n
        if is_train and self.tir_algo in ("arpo", "aepo"):
            if self.tir_algo == "aepo":
                # One probe trajectory per prompt, then allocate remaining budget.
                self.train_rollout_n = 1
            else:
                n_init = int(self.tir_config.get("initial_rollouts", 2))
                self.train_rollout_n = max(1, min(n_init, orig_n))
        try:
            await super()._async_set_up(data, server_addresses, is_train=is_train)
        finally:
            self.train_rollout_n = orig_n
        self._pending_original_by_data_id = {}
        for _rid, sample in self._task_id_to_original_sample.items():
            did = str(sample.get("data_id", ""))
            if did:
                self._pending_original_by_data_id[did] = sample

    async def _async_run_until_finished(self, verbose: bool = True):
        await super()._async_run_until_finished(verbose=verbose)
        if not self.is_train or self.tir_algo not in ("arpo", "aepo"):
            return
        extra = await self._enqueue_tree_branches()
        if extra <= 0:
            return
        if verbose:
            print(f"[TIR {self.tir_algo}] enqueued {extra} branch/resume rollouts; waiting...")
        await super()._async_run_until_finished(verbose=verbose)

    async def _enqueue_tree_branches(self) -> int:
        """After the first wave, spawn resume rollouts from high-entropy tool prefixes."""
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
            # Use first completed rollout's resume snapshot / entropy.
            parent = rids[0]
            resume = load_resume_messages(parent) or {}
            meta = self._rollout_meta.get(parent) or {}
            h_root = float(resume.get("h_root", meta.get("h_root", 0.0)) or 0.0)
            h_tool = float(resume.get("h_tool", meta.get("h_tool", 0.0)) or 0.0)
            delta_h = h_tool - h_root
            consecutive = int(resume.get("consecutive_high", 0) or 0)
            if self.tir_algo == "aepo":
                m = aepo_global_budget(n_target, h_root, h_tool, beta=beta)
                # Remaining budget split: extra global vs branch.
                extra_global = max(0, m - len(rids))
                remaining = n_target - len(rids)
                branch_budget = max(0, remaining - extra_global)
                p = branch_probability(
                    delta_h, alpha=alpha, gamma=gamma, consecutive_high=consecutive, penalty_slope=slope
                )
            else:
                extra_global = 0
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

    def get_train_data_batch(self, max_prompt_length: int, max_response_length: int, device, global_steps: int):
        data_proto, data_metrics = super().get_train_data_batch(
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            device=device,
            global_steps=global_steps,
        )
        rollout_ids = list(data_proto.non_tensor_batch.get("rollout_id_list", []) or [])
        n = len(rollout_ids)
        turn_raw = list(data_proto.non_tensor_batch.get("turn_index_list", []) or [])
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
        algo_id = {"grpo": 0.0, "arpo": 1.0, "aepo": 2.0, "igpo": 3.0, "gigpo": 4.0}
        data_metrics["training/tir_algo"] = algo_id.get(self.tir_algo, 0.0)
        data_metrics["training/n_with_anchor"] = float(sum(1 for a in anchors if a))
        return data_proto, data_metrics

    def clear_data_and_server(self):
        super().clear_data_and_server()
        self._pending_original_by_data_id.clear()
        self._rollout_meta.clear()
