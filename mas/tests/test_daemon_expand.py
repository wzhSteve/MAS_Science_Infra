"""Daemon expand_in_runner materialization (no GPU / no live AGL store)."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


@contextmanager
def _agl_modules_quarantined():
    """Import rl.hooks.daemon (pulls agentlightning) without leaking it.

    Later stage tests assert ``"agentlightning" not in sys.modules`` (layer
    independence red line); scrub AGL modules on exit so this suite can run
    in the same discover process.
    """
    import gc

    try:
        yield
    finally:
        agl = [k for k in list(sys.modules) if k == "agentlightning" or k.startswith("agentlightning.")]
        for k in agl:
            del sys.modules[k]
        gc.collect()


class TestDaemonExpandEnqueue(unittest.TestCase):
    def test_enqueue_from_runner_expansions_counts_and_fills(self):
        from workflow.active_set import dump_local_expansion

        with _agl_modules_quarantined(), tempfile.TemporaryDirectory() as td:
            # Import after path setup; daemon pulls agentlightning.
            from rl.hooks.daemon import TirAgentModeDaemon

            import workflow.active_set as active_set

            active_set.EXPANSION_DIR = td

            daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
            daemon.mode = "v1"
            daemon.tir_algo = "arpo"
            daemon.tir_config = {"expand_in_runner": True, "beam_size": 2}
            daemon._full_group_n = 4
            daemon.train_rollout_n = 4
            daemon.llm_timeout_seconds = 30
            daemon._store_enqueue_branch_count = 0
            daemon._branch_local_count_total = 0
            # RL6 D6: group-level counters start at zero (init-shape contract).
            daemon._group_requested_total = 0
            daemon._group_valid_total = 0
            daemon._group_expanded_total = 0
            daemon._group_duplicate_total = 0
            daemon._pending_original_by_data_id = {
                "q1": {"data_id": "q1", "question": "1+1", "answer": "2"},
            }
            daemon._task_id_to_original_sample = {
                "r0": {"data_id": "q1", "question": "1+1", "answer": "2"},
            }
            daemon._rollout_meta = {"r0": {"branch_local_count": 2}}
            daemon._completed_rollouts_v0 = {}
            daemon._enqueued_expansion_parents = set()
            daemon._incremental_branch_total = 0
            daemon._rollout_trees = {}

            dump_local_expansion(
                "r0",
                {
                    "branch_local_count": 2,
                    "plans": [
                        {
                            "resume_messages": [{"role": "user", "content": "x"}],
                            "parent_id": "r0",
                            "depth": 1,
                            "reason": "arpo_branch",
                            "meta": {},
                        },
                        {
                            "resume_messages": [{"role": "user", "content": "y"}],
                            "parent_id": "r0",
                            "depth": 1,
                            "reason": "arpo_branch",
                            "meta": {},
                        },
                    ],
                },
            )

            created: List[Any] = []

            async def _enqueue_many(requests):
                out = []
                for i, req in enumerate(requests):
                    rid = f"new{i}"
                    r = MagicMock()
                    r.rollout_id = rid
                    r.metadata = {"data_id": "q1"}
                    out.append(r)
                    created.append(req)
                return out

            daemon.store = MagicMock()
            daemon.store.enqueue_many_rollouts = AsyncMock(side_effect=_enqueue_many)
            daemon._total_tasks_queued = 1

            n = asyncio.get_event_loop().run_until_complete(daemon._enqueue_from_runner_expansions())
            # remaining = 4 - 1 = 3 → 2 branch + 1 global
            self.assertEqual(n, 3)
            # RL6 D6 group-level ledger (MAS v2 §9.3):
            # requested = 4 (n_target) × 1 group = 4 (卷面值)
            self.assertEqual(daemon._group_requested_total, 4)
            # valid = 1 existing (wave1) + 3 enqueued = 4 (有效样本账面)
            self.assertEqual(daemon._group_valid_total, 4)
            # expanded = 2 runner-plan branches (global fill 不算 expanded)
            self.assertEqual(daemon._group_expanded_total, 2)
            # duplicate = 0（未有增量路径提前入队）
            self.assertEqual(daemon._group_duplicate_total, 0)
            # 对账：requested == valid（本用例补齐无缺口）
            self.assertEqual(daemon._group_requested_total, daemon._group_valid_total)
            self.assertEqual(daemon._store_enqueue_branch_count, 2)
            branch_reqs = [c for c in created if (c.metadata or {}).get("tir_branch")]
            self.assertEqual(len(branch_reqs), 2)
            self.assertTrue(created[0].input.get("resume_messages"))
            self.assertIsNone(created[2].input.get("resume_messages"))

    def test_group_metrics_counts_gap_and_duplicate(self):
        """RL6 D6 (MAS v2 §9.3): 组级五计数的缺口与重复递交账目。

        场景：n_target=4、单组已有 1 条 wave1 样本，runner 只产出 1 个
        plan（分支预算不足）→ 全局补齐 2 条后仍有 1 缺口？不——后续全局
        补齐把 remaining 清到 0。真正的缺口场景：store.enqueue_many 或
        plans 为空。本用例锁定：requested=卷面值恒定、valid=账面值、
        expanded 只计 plans 分支、duplicate 计增量路径已入队父级的重复
        递交、五键进 get_train_data_batch 的 metrics dict。
        """
        from workflow.active_set import dump_local_expansion

        with _agl_modules_quarantined(), tempfile.TemporaryDirectory() as td:
            from rl.hooks.daemon import TirAgentModeDaemon

            import workflow.active_set as active_set

            active_set.EXPANSION_DIR = td

            daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
            daemon.mode = "v1"
            daemon.tir_algo = "arpo"
            daemon.tir_config = {"expand_in_runner": True, "beam_size": 2}
            daemon._full_group_n = 4
            daemon.train_rollout_n = 4
            daemon.llm_timeout_seconds = 30
            daemon._store_enqueue_branch_count = 0
            daemon._branch_local_count_total = 0
            daemon._incremental_branch_total = 0
            daemon._expansion_enqueue_failures = 0
            daemon._group_requested_total = 0
            daemon._group_valid_total = 0
            daemon._group_expanded_total = 0
            daemon._group_duplicate_total = 0
            daemon._pending_original_by_data_id = {
                "q1": {"data_id": "q1", "question": "1+1", "answer": "2"},
            }
            daemon._task_id_to_original_sample = {
                # r0 已被增量路径入队过（模拟 ready_batch 先行）
                "r0": {"data_id": "q1", "question": "1+1", "answer": "2"},
                "r1": {"data_id": "q1", "question": "1+1", "answer": "2"},
            }
            daemon._rollout_meta = {"r0": {"branch_local_count": 1}}
            daemon._completed_rollouts_v0 = {}
            # 增量路径已处理 r0 → 波次2 兜底将命中 duplicate 计数
            daemon._enqueued_expansion_parents = {"r0"}
            daemon._rollout_trees = {}

            dump_local_expansion(
                "r0",
                {
                    "branch_local_count": 1,
                    "plans": [
                        {
                            "resume_messages": [{"role": "user", "content": "x"}],
                            "parent_id": "r0",
                            "depth": 1,
                            "reason": "arpo_branch",
                            "meta": {},
                        },
                    ],
                },
            )

            async def _enqueue_many(requests):
                out = []
                for i, _req in enumerate(requests):
                    rid = f"new{i}"
                    r = MagicMock()
                    r.rollout_id = rid
                    r.metadata = {"data_id": "q1"}
                    out.append(r)
                return out

            daemon.store = MagicMock()
            daemon.store.enqueue_many_rollouts = AsyncMock(side_effect=_enqueue_many)
            daemon._total_tasks_queued = 2

            n = asyncio.get_event_loop().run_until_complete(daemon._enqueue_from_runner_expansions())
            # remaining = 4 - 2 = 2 → r0 duplicate（跳过），r1 无 plans → 2 全局补齐
            self.assertEqual(n, 2)
            # requested = 4 × 1 组 = 4
            self.assertEqual(daemon._group_requested_total, 4)
            # valid = 2 既有 + 2 全局补齐 = 4
            self.assertEqual(daemon._group_valid_total, 4)
            # expanded = 0（唯一 plan 在 duplicate 父级上，未入队）
            self.assertEqual(daemon._group_expanded_total, 0)
            # duplicate = 1（r0 已被增量入队）
            self.assertEqual(daemon._group_duplicate_total, 1)
            # 对账：requested == valid（全局补齐掩盖了 plan 缺口——expanded=0
            # 就是 §9.3 要暴露的"plan 不足但组大小已补齐"信号）
            self.assertEqual(daemon._group_requested_total, daemon._group_valid_total)

    def test_group_metrics_five_keys_contract(self):
        """RL6 D6: get_train_data_batch 的 metrics dict 带 training/group_*
        五键（与既有 6 键同 dict），且数值来自 daemon 实例计数器；
        clear_data_and_server 期末清零（per-wave ledger）。"""
        with _agl_modules_quarantined():
            import inspect

            from rl.hooks.daemon import TirAgentModeDaemon

            source = inspect.getsource(TirAgentModeDaemon.get_train_data_batch)
            for key in (
                "training/group_requested",
                "training/group_valid",
                "training/group_expanded",
                "training/group_duplicate",
                "training/group_failed",
            ):
                self.assertIn(key, source)
            init_source = inspect.getsource(TirAgentModeDaemon.__init__)
            for attr in (
                "_group_requested_total",
                "_group_valid_total",
                "_group_expanded_total",
                "_group_duplicate_total",
            ):
                self.assertIn(attr, init_source)
            clear_source = inspect.getsource(TirAgentModeDaemon.clear_data_and_server)
            self.assertIn("_group_requested_total = 0", clear_source)

    def test_preinject_expand_fields_reaches_task_inputs(self):
        """Fields must land in raw batch columns BEFORE super() snapshots them."""
        with _agl_modules_quarantined():
            from rl.hooks.daemon import TirAgentModeDaemon

            daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
            daemon.tir_config = {
                "max_branch_depth": 2,
                "beam_size": 2,
                "ready_batch": True,
                "sites": [{"anchor": {"kind": "after_tool"}, "when": "first"}],
            }
            daemon._full_group_n = 4
            daemon.train_rollout_n = 2

            # Simulate the raw batch dict AGL passes to _async_set_up.
            data = {"question": ["q1", "q2"], "answer": ["a1", "a2"]}
            daemon._preinject_expand_fields(data, n_init=2)

            self.assertEqual(len(data["expand_in_runner"]), 2)
            self.assertTrue(all(data["expand_in_runner"]))
            # per-root budget = ceil(group_n / n_init) = ceil(4/2) = 2
            self.assertEqual(data["sampling_budget"], [2, 2])
            # super() builds inputs as {key: data[key][i]} — verify a materialized
            # task input would carry the fields.
            task_input = {k: data[k][0] for k in data}
            self.assertTrue(task_input["expand_in_runner"])
            self.assertEqual(task_input["sampling_budget"], 2)
            self.assertEqual(len(task_input["branch_sites"]), 1)
            # hot path must NOT force local execution (plan-only in runner)
            self.assertNotIn("force_local_expand", task_input)

    def test_stamp_expand_budgets_distributes(self):
        with _agl_modules_quarantined():
            from rl.hooks.daemon import TirAgentModeDaemon

        daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
        daemon.tir_config = {
            "max_branch_depth": 2,
            "beam_size": 2,
            "branch_probability": 0.5,
            "entropy_weight": 0.5,
            "use_official_arpo_gate": True,
        }
        daemon._full_group_n = 4
        daemon._task_id_to_original_sample = {
            "a": {"data_id": "d", "question": "q"},
            "b": {"data_id": "d", "question": "q"},
        }
        daemon._stamp_expand_budgets()
        budgets = [
            daemon._task_id_to_original_sample["a"]["sampling_budget"],
            daemon._task_id_to_original_sample["b"]["sampling_budget"],
        ]
        self.assertEqual(sorted(budgets), [2, 2])
        self.assertTrue(daemon._task_id_to_original_sample["a"]["expand_in_runner"])


def _fake_data_proto(rows):
    """Minimal DataProto-like object with the columns super() would produce."""
    import numpy as np

    class _Proto:
        def __init__(self, n):
            self.non_tensor_batch = {
                "rollout_id_list": np.array([f"r{i}" for i in range(n)], dtype=object),
                "turn_index_list": np.array([0] * n, dtype=object),
            }
            self.batch = {}

    return _Proto(rows)


class TestGetTrainDataBatchColumns(unittest.TestCase):
    """RL2: column injection + conditional grpo early return (daemon.py:696+)."""

    def _daemon(self, algo, tir_config):
        from rl.hooks.daemon import TirAgentModeDaemon

        daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
        daemon.mode = "v1"
        daemon.tir_algo = algo
        daemon.tir_config = dict(tir_config)
        daemon._tree_recorder = None
        daemon._rollout_meta = {
            "r0": {"role": "parent", "resume_parent_id": ""},
            "r1": {"role": "child", "resume_parent_id": "r0", "resume_boundary": 3},
        }
        daemon._task_id_to_original_sample = {
            "r0": {"data_id": "t1"},
            "r1": {"data_id": "t1"},
        }
        daemon._completed_rollouts_v0 = {}
        daemon.adapter = None
        daemon._branch_local_count_total = 0
        daemon._store_enqueue_branch_count = 0
        daemon._incremental_branch_total = 0
        daemon._expansion_enqueue_failures = 0
        daemon._group_requested_total = 8
        daemon._group_valid_total = 8
        daemon._group_expanded_total = 3
        daemon._group_duplicate_total = 1
        return daemon

    def _run_get_batch(self, daemon, n_rows=2):
        """Call get_train_data_batch with super() and require_training_samples patched."""
        with _agl_modules_quarantined():
            import rl.hooks.daemon as daemon_mod

            proto = _fake_data_proto(n_rows)

            def _fake_super_get(self, max_prompt_length, max_response_length, device, global_steps):
                return proto, {}

            saved_require = daemon_mod.require_training_samples
            # Patch the parent-class method with an unbound function so the
            # super() call inside get_train_data_batch gets our fake batch.
            mro_cls = daemon_mod.TirAgentModeDaemon.__mro__[1]
            saved_super = mro_cls.get_train_data_batch
            mro_cls.get_train_data_batch = _fake_super_get
            daemon_mod.require_training_samples = lambda *a, **k: None
            try:
                out_proto, metrics = daemon.get_train_data_batch(
                    max_prompt_length=16, max_response_length=16, device="cpu", global_steps=3
                )
            finally:
                mro_cls.get_train_data_batch = saved_super
                daemon_mod.require_training_samples = saved_require
            return out_proto, metrics

    def test_resume_parent_id_injected_for_branched_algos(self):
        """RL2 (a): the 11th injected column resume_parent_id reaches
        non_tensor_batch for branching algos — apply_appo_credit
        (advantage.py:132) reads ntb.get("resume_parent_id") and previously
        always fell back to whole-group grouping because the column was
        never injected."""
        daemon = self._daemon("appo", {"zero_prefix_adv_for_children": True})
        proto, _ = self._run_get_batch(daemon)
        self.assertIn("resume_parent_id", proto.non_tensor_batch)
        parents = list(proto.non_tensor_batch["resume_parent_id"])
        self.assertEqual(parents, ["", "r0"])

    def test_grpo_with_zero_prefix_still_injects_rollout_role(self):
        """RL2 (b): grpo + zero_prefix_adv_for_children active (default True /
        DEFAULT_TIR) must NOT take the early return — rows now carry
        rollout_role / resume_boundary / resume_parent_id so
        _maybe_zero_prefix (advantage.py:82) actually zeroes shared
        prefixes instead of silently no-opping."""
        daemon = self._daemon("grpo", {})  # default → zero_prefix on
        proto, metrics = self._run_get_batch(daemon)
        self.assertIn("rollout_role", proto.non_tensor_batch)
        self.assertIn("resume_boundary", proto.non_tensor_batch)
        self.assertIn("resume_parent_id", proto.non_tensor_batch)
        self.assertEqual(list(proto.non_tensor_batch["rollout_role"]), ["parent", "child"])
        # RL2 (c): noop counter placeholder always present in metrics
        self.assertIn("training/pipeline_noop", metrics)
        self.assertEqual(metrics["training/pipeline_noop"], 0.0)

    def test_grpo_without_zero_prefix_keeps_early_return(self):
        """RL2 (b) negative: zero_prefix explicitly off → grpo keeps the exact
        legacy early-return shape (no column injection, no pipeline_noop key
        since metrics come from the fake super())."""
        daemon = self._daemon("grpo", {"zero_prefix_adv_for_children": False})
        proto, metrics = self._run_get_batch(daemon)
        self.assertNotIn("rollout_role", proto.non_tensor_batch)
        self.assertNotIn("resume_parent_id", proto.non_tensor_batch)
        self.assertNotIn("training/pipeline_noop", metrics)

    def test_group_metrics_emitted_with_six_key_family(self):
        """RL6 D6: 分支算法路径的 metrics 里 training/group_* 五键与既有 6 键
        同 dict 出现，数值取自实例计数器（requested=卷面值 vs valid 账面值
        在此用例中相等——缺口的暴露语义由 enqueue 用例锁定）。"""
        daemon = self._daemon("arpo", {})
        proto, metrics = self._run_get_batch(daemon)
        self.assertEqual(metrics["training/group_requested"], 8.0)
        self.assertEqual(metrics["training/group_valid"], 8.0)
        self.assertEqual(metrics["training/group_expanded"], 3.0)
        self.assertEqual(metrics["training/group_duplicate"], 1.0)
        self.assertEqual(metrics["training/group_failed"], 0.0)
        # 与既有 6 键同 dict（相邻键共存断言）
        for key in (
            "training/tir_algo",
            "training/n_with_anchor",
            "training/branch_local_count",
            "training/store_enqueue_branch_count",
            "training/incremental_branch_count",
            "training/expansion_enqueue_failures",
        ):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
