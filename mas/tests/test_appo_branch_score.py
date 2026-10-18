"""APPO token score, reward-scale M, and prefix write-back. CPU only."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from verl import DataProto

ROOT = Path(__file__).resolve().parents[2]
MAS = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(MAS)):
    if path not in sys.path:
        sys.path.insert(0, path)


class TestBranchScore(unittest.TestCase):
    def test_positive_z_product_is_preferred(self):
        from rl.hooks.appo_branching import select_branch_tokens

        entropy = [0.1, 0.2, 0.9]
        future = [0.1, 0.2, 0.9]
        picked = select_branch_tokens(entropy, future, k=1)
        self.assertEqual(picked, [2])

    def test_shortfall_repeats_the_chosen_token(self):
        from rl.hooks.appo_branching import select_branch_tokens

        picked = select_branch_tokens([1.0], None, k=3)
        self.assertEqual(picked, [0, 0, 0])

    def test_reward_scale_stays_one_after_the_split(self):
        from rl.hooks.appo_branching import reward_scale_row

        row = reward_scale_row(4, split=1, branch_score=1.0, max_branch_score=1.0, discount=0.5)
        self.assertAlmostEqual(row[1], 2.0)
        self.assertAlmostEqual(row[0], 1.0 + 0.5)
        self.assertEqual(row[2:], [1.0, 1.0])


class TestPrefixCredit(unittest.TestCase):
    def test_branch_scalar_stops_at_the_split_token(self):
        from rl.hooks.advantage import apply_appo_credit

        scores = torch.zeros(2, 4)
        scores[0, -1] = 1.0
        scores[1, -1] = 3.0
        batch = DataProto.from_dict(tensors={
            "advantages": torch.ones(2, 4),
            "token_level_scores": scores,
            "token_level_rewards": scores.clone(),
            "response_mask": torch.ones(2, 4),
            "returns": torch.ones(2, 4),
            "loss_mask": torch.ones(2, 4),
        })
        batch.non_tensor_batch["rollout_role"] = np.array(["parent", "branch"], dtype=object)
        batch.non_tensor_batch["data_id_list"] = np.array(["t", "t"], dtype=object)
        batch.non_tensor_batch["rollout_id_list"] = np.array(["p", "c"], dtype=object)
        batch.non_tensor_batch["resume_parent_id"] = np.array(["", "p"], dtype=object)
        batch.non_tensor_batch["appo_branch_token"] = np.array(["", 1], dtype=object)
        out = apply_appo_credit(batch, discount=0.9)
        parent = 1.0 / (1 + 1e-6)
        branch = 3.0 / (1 + 1e-6)
        torch.testing.assert_close(
            out.batch["advantages"][0, :2], torch.full((2,), parent + branch), rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            out.batch["advantages"][0, 2:], torch.full((2,), parent), rtol=1e-5, atol=1e-6,
        )
        self.assertTrue(torch.equal(out.batch["advantages"][1], torch.zeros(4)))
        self.assertTrue(torch.equal(out.batch["loss_mask"][1], torch.zeros(4)))

    def test_reward_scale_m_scales_the_reward_token_inside_the_split(self):
        from rl.hooks.advantage import apply_appo_credit

        scores = torch.zeros(2, 4)
        scores[0, -1] = 1.0
        scores[1, -1] = 0.0
        batch = DataProto.from_dict(tensors={
            "advantages": torch.ones(2, 4),
            "token_level_scores": scores,
            "token_level_rewards": scores.clone(),
            "response_mask": torch.ones(2, 4),
            "returns": torch.ones(2, 4),
        })
        batch.non_tensor_batch["rollout_role"] = np.array(["parent", "branch"], dtype=object)
        batch.non_tensor_batch["data_id_list"] = np.array(["t", "t"], dtype=object)
        batch.non_tensor_batch["rollout_id_list"] = np.array(["p", "c"], dtype=object)
        batch.non_tensor_batch["resume_parent_id"] = np.array(["", "p"], dtype=object)
        batch.non_tensor_batch["appo_branch_token"] = np.array(["", 3], dtype=object)
        batch.non_tensor_batch["appo_branch_score"] = np.array([0.0, 1.0], dtype=object)
        out = apply_appo_credit(batch, discount=0.5)
        # M at the reward token is 2, so the init singleton scalar is 2 / (1+eps).
        torch.testing.assert_close(
            out.batch["token_level_rewards"][0, -1], torch.tensor(2.0), rtol=1e-5, atol=1e-6,
        )
        torch.testing.assert_close(
            out.batch["advantages"][0, -1], torch.tensor(2.0 / (1 + 1e-6)), rtol=1e-5, atol=1e-5,
        )


class TestPlannerEntropyTrace(unittest.TestCase):
    def test_trace_reads_the_planner_call(self):
        from workflow.token_entropy import distribution_entropy, entropy_from_trace

        self.assertGreater(distribution_entropy([0.0, -1.0]), 0.0)
        value = entropy_from_trace({"llm_calls": {"calls": [
            {"role": "executor", "entropy": 9.0},
            {"role": "planner", "entropy": 1.5},
        ]}})
        self.assertEqual(value, 1.5)


if __name__ == "__main__":
    unittest.main()
