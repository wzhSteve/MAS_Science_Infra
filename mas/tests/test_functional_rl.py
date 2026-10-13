"""Functional contracts: reward tiers, sample policy, and expansion failures."""

from __future__ import annotations

import asyncio
import gc
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


@contextmanager
def _agl_modules_quarantined():
    try:
        yield
    finally:
        for key in [k for k in list(sys.modules) if k == "agentlightning" or k.startswith("agentlightning.")]:
            del sys.modules[key]
        gc.collect()


class TestRewardTiers(unittest.TestCase):
    def test_outcome_ladder_and_hash_format_contract(self):
        from workflow.collector import default_reward_fn
        from workflow.contracts import Trajectory
        from rl.rewards.outcome import compute_outcome_reward

        self.assertEqual(
            compute_outcome_reward(None, "2", source="gsm8k", format_ok=False, n_search=0, n_python=0),
            -1.0,
        )
        self.assertEqual(
            compute_outcome_reward("9", "2", source="gsm8k", format_ok=True, n_search=0, n_python=0),
            0.0,
        )
        self.assertEqual(
            compute_outcome_reward("2", "2", source="gsm8k", format_ok=True, n_search=0, n_python=0),
            1.0,
        )
        self.assertAlmostEqual(
            compute_outcome_reward("2", "2", source="gsm8k", format_ok=True, n_search=1, n_python=1),
            1.1,
        )
        empty = Trajectory(final_answer=None, format_ok=False)
        self.assertEqual(default_reward_fn(empty, {"answer": "2", "source": "gsm8k"}), -1.0)
        hashed = Trajectory(final_answer="2", format_ok=False)
        self.assertEqual(default_reward_fn(hashed, {"answer": "2", "source": "gsm8k"}), 1.0)


class TestSamplePolicyOverlay(unittest.TestCase):
    def test_apply_sample_policy_writes_rollout_and_tir(self):
        from rl.hooks.overlay import apply_sample_policy
        from workflow.contracts import SamplePolicy

        policy = SamplePolicy(mode="arpo", group_n=4, beam_size=2, initial_rollouts=2, expand_in_runner=True)
        cfg = apply_sample_policy({"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 1}}}, policy)
        self.assertEqual(cfg["actor_rollout_ref"]["rollout"]["n"], 4)
        self.assertEqual(cfg["algorithm"]["tir_algo"], "arpo")
        self.assertEqual(cfg["algorithm"]["tir"]["beam_size"], 2)
        self.assertEqual(cfg["algorithm"]["tir"]["initial_rollouts"], 2)
        self.assertTrue(cfg["algorithm"]["tir"]["expand_in_runner"])
        self.assertTrue(cfg["algorithm"]["tir"]["sites"])


class TestDaemonExpansion(unittest.TestCase):
    def test_preinject_fields_before_enqueue_snapshot(self):
        with _agl_modules_quarantined():
            from rl.hooks.daemon import TirAgentModeDaemon

            daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
            daemon.tir_config = {"max_branch_depth": 2, "beam_size": 2, "ready_batch": True, "sites": [{"anchor": {"kind": "after_tool"}}]}
            daemon._full_group_n = 4
            daemon.train_rollout_n = 2
            daemon.tir_algo = "arpo"
            data = {"question": ["q1"], "answer": ["a1"]}
            daemon._preinject_expand_fields(data, n_init=2)
            task_input = {key: data[key][0] for key in data}
            self.assertTrue(task_input["expand_in_runner"])
            self.assertEqual(task_input["sampling_budget"], 2)
            self.assertEqual(len(task_input["branch_sites"]), 1)

    def test_expansion_failure_is_counted(self):
        with _agl_modules_quarantined():
            from rl.hooks.daemon import TirAgentModeDaemon

            daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
            daemon._incremental_branch_total = 3

            async def boom(rollout_id: str) -> int:
                del rollout_id
                raise RuntimeError("enqueue down")

            daemon._enqueue_expansion_for_parent = boom
            with self.assertLogs("rl.hooks.daemon", level="ERROR"):
                asyncio.run(daemon._enqueue_expansion_safely("parent"))
            self.assertEqual(daemon._expansion_enqueue_failures, 1)
            self.assertEqual(daemon._incremental_branch_total, 3)
