"""Flow-GRPO registration and EPC-AW workflow wiring. No GPU."""

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

ALPHA_RL = ROOT / "experiments" / "alpha" / "rl.yaml"
ALPHA_WORKFLOW = ROOT / "experiments" / "alpha" / "workflow.yaml"


def _credit_batch() -> DataProto:
    advantages = torch.ones(2, 4)
    scores = torch.zeros(2, 4)
    scores[0, -1] = 1.0
    scores[1, -1] = 2.0
    batch = DataProto.from_dict(tensors={
        "advantages": advantages,
        "token_level_scores": scores,
        "response_mask": torch.ones(2, 4),
        "returns": advantages.clone(),
    })
    batch.non_tensor_batch["rollout_role"] = np.array(["parent", "child"], dtype=object)
    batch.non_tensor_batch["data_id_list"] = np.array(["task", "task"], dtype=object)
    batch.non_tensor_batch["rollout_id_list"] = np.array(["parent-1", "child-1"], dtype=object)
    batch.non_tensor_batch["resume_parent_id"] = np.array(["", "parent-1"], dtype=object)
    batch.non_tensor_batch["resume_boundary"] = np.array([0, 3], dtype=object)
    return batch


class TestFlowGrpoRegistration(unittest.TestCase):
    def test_allowlists_match_and_do_not_branch(self):
        from rl.hooks.overlay import BRANCHING_ALGOS, VALID_ALGOS
        from science_infra.control.experiments import VALID_ALGOS as CONTROL_ALGOS

        self.assertEqual(CONTROL_ALGOS, VALID_ALGOS)
        self.assertIn("flow_grpo", VALID_ALGOS)
        self.assertNotIn("flow_grpo", BRANCHING_ALGOS)

    def test_overlay_keeps_group_size_and_agentflow_defaults(self):
        from rl.hooks.overlay import apply_algo_overlay

        base = {
            "algorithm": {},
            "actor_rollout_ref": {"rollout": {"n": 8}, "actor": {}},
            "trainer": {},
        }
        cfg = apply_algo_overlay(base, "flow_grpo")
        actor = cfg["actor_rollout_ref"]["actor"]
        self.assertEqual(cfg["algorithm"]["tir_algo"], "flow_grpo")
        self.assertEqual(cfg["algorithm"]["adv_estimator"], "grpo")
        self.assertFalse(cfg["algorithm"]["use_kl_in_reward"])
        self.assertEqual(cfg["actor_rollout_ref"]["rollout"]["n"], 8)
        self.assertFalse(cfg["algorithm"]["tir"]["ready_batch"])
        self.assertTrue(actor["use_kl_loss"])
        self.assertEqual(actor["kl_loss_coef"], 0.001)
        self.assertEqual(actor["entropy_coeff"], 0.0)
        self.assertEqual(actor["clip_ratio_low"], 0.2)
        self.assertEqual(actor["clip_ratio_high"], 0.3)


class TestFlowGrpoCredit(unittest.TestCase):
    def test_does_not_zero_children_or_apply_appo_discount(self):
        from rl.hooks import advantage as advantage_mod

        def _keep(batch, **_kwargs):
            return batch

        original = advantage_mod.compute_advantage
        advantage_mod.compute_advantage = _keep
        try:
            out = advantage_mod.apply_tir_advantages(
                _credit_batch(),
                adv_estimator="grpo",
                gamma=1.0,
                lam=1.0,
                num_repeat=1,
                norm_adv_by_std_in_grpo=True,
                config={"tir_algo": "flow_grpo", "tir": {"zero_prefix_adv_for_children": True}},
            )
        finally:
            advantage_mod.compute_advantage = original
        self.assertTrue(torch.equal(out.batch["advantages"], torch.ones(2, 4)))


class TestEpcAwWorkflowWiring(unittest.TestCase):
    def test_alpha_workflow_trains_planner_windows(self):
        from workflow.compiler import trainable_agents
        from workflow.runtime import agent_window_rollout
        from workflow.spec import load_spec

        spec = load_spec(str(ALPHA_WORKFLOW))
        self.assertTrue(agent_window_rollout(spec))
        self.assertEqual(trainable_agents(spec), ["planner"])

    def test_rl_yaml_resolves_sibling_workflow(self):
        from train_tir_agent import resolve_workflow_spec_path, trainable_ids_from_spec

        spec_path = resolve_workflow_spec_path(str(ALPHA_RL))
        self.assertEqual(Path(spec_path), ALPHA_WORKFLOW.resolve())
        self.assertEqual(trainable_ids_from_spec(spec_path), ["planner"])
        missing = resolve_workflow_spec_path(None, "/tmp/does-not-exist-flow-grpo.yaml")
        self.assertIsNone(missing)
