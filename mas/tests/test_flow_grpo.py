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

    def test_overlay_respects_explicit_kl_override(self):
        """Single-GPU rl-yaml sets use_kl_loss: false / use_kl_in_reward: true
        to skip the ref-model forward; both overlay passes must keep it."""
        from rl.hooks.overlay import apply_algo_overlay, apply_train_signal
        from rl.train_signal import AdvantageSpec, LossSpec, TrainSignal

        base = {
            "algorithm": {"use_kl_in_reward": True},
            "actor_rollout_ref": {
                "rollout": {"n": 4},
                "actor": {"use_kl_loss": False, "kl_loss_coef": 0.0},
            },
            "trainer": {},
        }
        cfg = apply_algo_overlay(base, "flow_grpo")
        self.assertTrue(cfg["algorithm"]["use_kl_in_reward"])  # explicit kept
        self.assertFalse(cfg["actor_rollout_ref"]["actor"]["use_kl_loss"])
        signal = TrainSignal(
            advantage=AdvantageSpec(name="flow_grpo", use_critic=False),
            loss=LossSpec(name="flow_grpo"),
            meta={},
        )
        cfg = apply_train_signal(cfg, signal)
        self.assertFalse(cfg["actor_rollout_ref"]["actor"]["use_kl_loss"])
        self.assertTrue(cfg["algorithm"]["use_kl_in_reward"])
        # coef stays 0: no ref forward is planned, so no kl term needed
        self.assertEqual(cfg["actor_rollout_ref"]["actor"]["kl_loss_coef"], 0.0)


class TestFlowGrpoSamplingPrecedence(unittest.TestCase):
    """rl.yaml algo wins over workflow.sampling.mode (AgentFlow 4algo bug)."""

    _ARPO_SAMPLING = {
        "mode": "arpo",
        "group_n": 4,
        "beam_size": 2,
        "sites": [{"id": "s1", "anchor": {"kind": "after_agent_turn", "agent_id": "planner"}}],
    }

    def _base(self):
        return {"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 4}, "actor": {}}, "trainer": {}}

    def test_explicit_flow_grpo_survives_arpo_sampling(self):
        from rl.hooks.overlay import apply_sample_policy

        cfg = apply_sample_policy(self._base(), self._ARPO_SAMPLING, algo="flow_grpo")
        self.assertEqual(cfg["algorithm"]["tir_algo"], "flow_grpo")
        self.assertEqual(cfg["algorithm"]["tir"]["sites"], [])
        self.assertFalse(cfg["algorithm"]["tir"]["ready_batch"])
        self.assertEqual(cfg["actor_rollout_ref"]["rollout"]["n"], 4)

    def test_mode_still_decides_without_explicit_algo(self):
        from rl.hooks.overlay import apply_sample_policy

        cfg = apply_sample_policy(self._base(), self._ARPO_SAMPLING)
        self.assertEqual(cfg["algorithm"]["tir_algo"], "arpo")
        self.assertTrue(cfg["algorithm"]["tir"]["sites"])

    def test_igpo_and_gigpo_not_collapsed_to_grpo(self):
        from rl.hooks.overlay import apply_sample_policy

        for algo in ("igpo", "gigpo", "flow_grpo"):
            cfg = apply_sample_policy(self._base(), {"mode": "grpo_n", "group_n": 4}, algo=algo)
            self.assertEqual(cfg["algorithm"]["tir_algo"], algo)

    def test_flow_grpo_resolves_to_independent_adapter(self):
        from workflow.sampling.registry import sampling_adapters

        self.assertEqual(sampling_adapters.resolve("flow_grpo").id, "independent")

    def test_control_sync_keeps_rl_algo(self):
        from science_infra.control.services import _sync_workflow_sampling_into_rl

        rl = {"algo": "flow_grpo", **self._base()}
        synced = _sync_workflow_sampling_into_rl(rl, {"sampling": self._ARPO_SAMPLING})
        self.assertEqual(synced["algo"], "flow_grpo")
        self.assertEqual(synced["algorithm"]["tir_algo"], "flow_grpo")


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


class TestFlowGrpoBinaryOutcome(unittest.TestCase):
    def test_binary_correct(self):
        from rl.rewards import binary_correct

        self.assertEqual(binary_correct("<answer>16</answer>", "16", source="gsm8k"), 1.0)
        self.assertEqual(binary_correct("<answer>15</answer>", "16", source="gsm8k"), 0.0)
        self.assertEqual(binary_correct("<answer>The British</answer>", "British", source="hotpot"), 1.0)
        self.assertEqual(binary_correct("<answer>He was English.</answer>", "British", source="hotpot",
                                         aliases=["English"]), 1.0)
        self.assertEqual(binary_correct("<answer>Britishness</answer>", "British", source="hotpot"), 0.0)
        self.assertEqual(binary_correct(None, "x", source="hotpot"), 0.0)

    def test_binary_spec_ignores_format_and_bonus(self):
        from rl.rewards import compute_outcome_reward
        from rl.rewards.assembly import RewardSpec

        spec = RewardSpec(outcome="binary")
        kw = dict(source="hotpot", n_search=2, n_python=1, reward_spec=spec)
        self.assertEqual(compute_outcome_reward("<answer>Paris</answer>", "Paris", format_ok=True, **kw), 1.0)
        self.assertEqual(compute_outcome_reward("Paris", "Paris", format_ok=False, **kw), 1.0)
        self.assertEqual(compute_outcome_reward("<answer>Lyon city</answer>", "Paris", format_ok=True, **kw), 0.0)

    def test_overlay_defaults_flow_grpo_to_binary(self):
        from rl.hooks.overlay import apply_algo_overlay

        base = {"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 4}, "actor": {}}, "trainer": {}}
        flow = apply_algo_overlay(base, "flow_grpo")
        self.assertEqual(flow["algorithm"]["reward"]["outcome"], "binary")
        self.assertEqual(apply_algo_overlay(flow, "flow_grpo")["algorithm"]["reward"]["outcome"], "binary")
        self.assertEqual(apply_algo_overlay(flow, "grpo")["algorithm"]["reward"]["outcome"], "hierarchical")
        self.assertEqual(apply_algo_overlay(base, "grpo")["algorithm"]["reward"]["outcome"], "hierarchical")

    def test_user_outcome_survives_overlay(self):
        from rl.hooks.overlay import apply_algo_overlay

        base = {"algorithm": {"reward": {"outcome": "hierarchical"}},
                "actor_rollout_ref": {"rollout": {"n": 4}, "actor": {}}, "trainer": {}}
        cfg = apply_algo_overlay(base, "flow_grpo")
        self.assertEqual(cfg["algorithm"]["reward"]["outcome"], "hierarchical")
        base["algorithm"]["reward"]["outcome"] = "binary"
        self.assertEqual(apply_algo_overlay(base, "grpo")["algorithm"]["reward"]["outcome"], "binary")

    def test_trainer_exports_outcome_and_collector_reads_it(self):
        import os
        from unittest import mock

        from train_tir_agent import reward_outcome_for
        from workflow.collector import default_reward_fn
        from workflow.contracts import Trajectory

        self.assertEqual(reward_outcome_for({"algorithm": {}}, "flow_grpo"), "binary")
        self.assertEqual(reward_outcome_for({"algorithm": {}}, "grpo"), "hierarchical")
        self.assertEqual(reward_outcome_for({"algorithm": {"reward": {"outcome": "hierarchical"}}}, "flow_grpo"),
                         "hierarchical")
        traj = Trajectory.model_construct(final_answer="<answer>Paris France</answer>", format_ok=True,
                                          n_search=1, n_python=1)
        task = {"answer": "Paris", "source": "hotpot"}
        with mock.patch.dict(os.environ, {"TIR_REWARD_OUTCOME": "binary"}):
            self.assertEqual(default_reward_fn(traj, task), 1.0)
        with mock.patch.dict(os.environ, {"TIR_REWARD_OUTCOME": ""}):
            self.assertNotIn(default_reward_fn(traj, task), (0.0, 1.0))


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
