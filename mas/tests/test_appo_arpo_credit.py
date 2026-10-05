"""Single-agent five-tool template plus ARPO/APPO credit, no GPU."""

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


FIVE_TOOLS = ["wikipedia_search", "bing_search", "web_fetch", "python_coder", "think"]


def _credit_batch() -> DataProto:
    advantages = torch.ones(2, 4)
    scores = torch.zeros(2, 4)
    scores[0, -1] = 1.0
    scores[1, -1] = 2.0
    mask = torch.ones(2, 4)
    batch = DataProto.from_dict(tensors={
        "advantages": advantages,
        "token_level_scores": scores,
        "response_mask": mask,
        "returns": advantages.clone(),
    })
    batch.non_tensor_batch["rollout_role"] = np.array(["parent", "child"], dtype=object)
    batch.non_tensor_batch["data_id_list"] = np.array(["task", "task"], dtype=object)
    batch.non_tensor_batch["rollout_id_list"] = np.array(["parent-1", "child-1"], dtype=object)
    batch.non_tensor_batch["resume_parent_id"] = np.array(["", "parent-1"], dtype=object)
    batch.non_tensor_batch["resume_boundary"] = np.array([0, 3], dtype=object)
    return batch


class TestFiveToolTemplate(unittest.TestCase):
    def test_compiles_one_trainable_planner(self):
        from workflow.compiler import compile_spec
        from workflow.spec import MASSpec
        from workflow.templates import load_template_workflow

        raw = load_template_workflow("tir_five_tools")
        spec = MASSpec.model_validate(raw)
        compiled = compile_spec(spec)
        self.assertTrue(compiled.ok, compiled.issues)
        self.assertEqual(compiled.tools_for["planner"], FIVE_TOOLS)
        self.assertNotIn("verifier", compiled.agents)
        self.assertTrue(compiled.agents["planner"].trainable)
        for tool_id in FIVE_TOOLS:
            self.assertFalse(compiled.agents[tool_id].trainable)
        self.assertEqual(compiled.routers["route_exec"].candidates, FIVE_TOOLS)
        site_tools = [site.anchor.tool_id for site in spec.sampling.sites]
        self.assertEqual(site_tools, FIVE_TOOLS)


class TestAlgoIdentity(unittest.TestCase):
    def test_appo_and_arpo_stay_distinct(self):
        from rl.hooks.overlay import VALID_ALGOS, apply_algo_overlay, apply_sample_policy
        from science_infra.control.experiments import VALID_ALGOS as CONTROL_ALGOS

        self.assertEqual(CONTROL_ALGOS, VALID_ALGOS)
        self.assertIn("appo", VALID_ALGOS)
        base = {"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 4}}, "trainer": {}}
        arpo = apply_sample_policy(base, {"mode": "arpo", "group_n": 4})
        appo = apply_algo_overlay(base, "appo")
        self.assertEqual(arpo["algorithm"]["tir_algo"], "arpo")
        self.assertEqual(arpo["algorithm"]["tir"]["loss_mode"], "vanilla")
        self.assertEqual(appo["algorithm"]["tir_algo"], "appo")
        self.assertEqual(appo["algorithm"]["tir"]["loss_mode"], "future_kl")
        self.assertEqual(appo["algorithm"]["tir"]["reward_scale_discount"], 0.9)


class TestCredit(unittest.TestCase):
    def test_arpo_keeps_child_prefix(self):
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
                config={"tir_algo": "arpo", "tir": {"zero_prefix_adv_for_children": True}},
            )
        finally:
            advantage_mod.compute_advantage = original
        self.assertTrue(torch.equal(out.batch["advantages"][1, :3], torch.ones(3)))

    def test_appo_zeros_branch_and_writes_discount_to_parent(self):
        from rl.hooks.advantage import apply_appo_credit

        out = apply_appo_credit(_credit_batch(), discount=0.9)
        self.assertTrue(torch.equal(out.batch["advantages"][1], torch.zeros(4)))
        self.assertTrue(torch.allclose(out.batch["advantages"][0], torch.full((4,), 2.8)))
        self.assertTrue(torch.equal(out.batch["returns"][1], torch.zeros(4)))


def _batch_from(roles, scores, *, data_ids=None, rollout_ids=None, parents=None, mask=None, advantages=None):
    n = len(roles)
    width = 4
    adv = advantages if advantages is not None else torch.ones(n, width)
    score = torch.zeros(n, width)
    for i, value in enumerate(scores):
        score[i, -1] = float(value)
    tensors = {
        "advantages": adv,
        "token_level_scores": score,
        "returns": adv.clone(),
    }
    if mask is not None:
        tensors["response_mask"] = mask
    batch = DataProto.from_dict(tensors=tensors)
    batch.non_tensor_batch["rollout_role"] = np.array(roles, dtype=object)
    batch.non_tensor_batch["data_id_list"] = np.array(data_ids or ["task"] * n, dtype=object)
    batch.non_tensor_batch["rollout_id_list"] = np.array(rollout_ids or [f"r{i}" for i in range(n)], dtype=object)
    batch.non_tensor_batch["resume_parent_id"] = np.array(parents or [""] * n, dtype=object)
    batch.non_tensor_batch["resume_boundary"] = np.array([3 if role != "parent" else 0 for role in roles], dtype=object)
    return batch


class TestFiveToolEpisode(unittest.TestCase):
    def test_each_tool_completes_with_synthetic_verify(self):
        from tests.test_template_combinations import (
            NETWORK_TOOL_IDS,
            _mock_tool_invokes,
            _plan,
            _run,
            _spec,
        )

        calls = {
            "wikipedia_search": {"query": "q"},
            "bing_search": {"query": "q"},
            "web_fetch": {"url": "https://example.com"},
            "python_coder": {"code": "result=42"},
            "think": {"thought": "note"},
        }
        spec = _spec("tir_five_tools")
        for tool_id, args in calls.items():
            from workflow.centralized_runtime import MockWindowLLM

            win = MockWindowLLM({"planner": [_plan(tool_id, args, done=True, sub_goal=tool_id)]})
            with _mock_tool_invokes(NETWORK_TOOL_IDS):
                raw, _compiled, kinds = _run(spec, win, task_id=f"five-{tool_id}")
            self.assertFalse(raw.error, f"{tool_id}: {raw.error}")
            self.assertIn("verify", kinds, tool_id)
            self.assertIn(tool_id, [m.get("src") for m in raw.messages], tool_id)
        from workflow.compiler import trainable_agents

        self.assertEqual(trainable_agents(spec), ["planner"])


class TestPresets(unittest.TestCase):
    def test_user_override_and_train_signal_keep_future_kl_off_actor(self):
        from rl.hooks.overlay import BRANCHING_ALGOS, apply_algo_overlay, apply_sample_policy, apply_train_signal
        from rl.loss import AdvantageSpec, LossSpec
        from rl.train_signal import TrainSignal

        base = {"algorithm": {"tir": {"loss_mode": "vanilla", "reward_scale_discount": 0.4}}, "actor_rollout_ref": {"rollout": {"n": 4}}, "trainer": {}}
        overridden = apply_algo_overlay(base, "appo")
        self.assertEqual(overridden["algorithm"]["tir"]["loss_mode"], "vanilla")
        self.assertEqual(overridden["algorithm"]["tir"]["reward_scale_discount"], 0.4)
        self.assertEqual(overridden["algorithm"]["adv_estimator"], "grpo")

        sampled = apply_sample_policy(
            {"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 4}}, "trainer": {}},
            {"mode": "appo", "group_n": 4, "sites": []},
        )
        self.assertEqual(sampled["algorithm"]["tir_algo"], "appo")
        self.assertNotEqual(sampled["algorithm"]["tir_algo"], "arpo")

        signal = TrainSignal(advantage=AdvantageSpec(name="appo"), loss=LossSpec(name="appo"))
        trained = apply_train_signal(
            {"algorithm": {}, "actor_rollout_ref": {"rollout": {"n": 4}, "actor": {}}, "trainer": {"total_training_steps": 1}},
            signal,
        )
        self.assertEqual(trained["algorithm"]["tir_algo"], "appo")
        self.assertEqual(trained["algorithm"]["tir"]["loss_mode"], "future_kl")
        self.assertFalse(trained["algorithm"]["tir"]["loss_applied_to_actor"])
        self.assertEqual(trained["algorithm"]["adv_estimator"], "grpo")
        self.assertIn("appo", BRANCHING_ALGOS)
        self.assertNotIn("grpo", BRANCHING_ALGOS)

    def test_branch_expansion_includes_appo(self):
        from pathlib import Path

        daemon = Path("/root/autodl-tmp/MAS_Science_Infra/rl/hooks/daemon.py").read_text(encoding="utf-8")
        training = Path("/root/autodl-tmp/MAS_Science_Infra/science_infra/control/training.py").read_text(encoding="utf-8")
        ui = Path("/root/autodl-tmp/MAS_Science_Infra/scripts/branch_rollout_ui_test.py").read_text(encoding="utf-8")
        self.assertNotIn('("arpo", "aepo", "rae")', daemon)
        self.assertNotIn('("arpo", "aepo", "rae")', training)
        self.assertIn("BRANCHING_ALGOS", daemon)
        self.assertIn("BRANCHING_ALGOS", training)
        self.assertNotIn('if algo == "appo"', ui)

    def test_template_is_on_the_palette(self):
        from science_infra.control.services import mas_palette
        from workflow.spec import MASSpec
        from workflow.templates import load_template_workflow

        pal = mas_palette()
        row = next(item for item in pal["templates"] if item["id"] == "tir_five_tools")
        self.assertIn("APPO", row["label"])
        spec = MASSpec.model_validate(load_template_workflow("tir_five_tools"))
        self.assertEqual(spec.sampling.mode, "grpo_n")
        self.assertEqual(len(spec.sampling.sites), 5)
        for site in spec.sampling.sites:
            self.assertEqual(site.anchor.kind, "after_tool")
            self.assertEqual(site.gate.params["entropy_weight"], 0.2)
            self.assertEqual(site.gate.params["branch_probability"], 0.5)
            self.assertTrue(site.gate.params["use_official_arpo_gate"])


class TestCreditEdges(unittest.TestCase):
    def _through(self, algo, batch, **tir):
        from rl.hooks import advantage as advantage_mod

        def _keep(data, **_kwargs):
            return data

        original = advantage_mod.compute_advantage
        advantage_mod.compute_advantage = _keep
        try:
            return advantage_mod.apply_tir_advantages(
                batch,
                adv_estimator="grpo",
                gamma=1.0,
                lam=1.0,
                num_repeat=1,
                norm_adv_by_std_in_grpo=True,
                config={"tir_algo": algo, "tir": tir},
            )
        finally:
            advantage_mod.compute_advantage = original

    def test_appo_path_zeros_entire_branch_row(self):
        batch = _batch_from(["parent", "probe", "branch"], [1.0, 2.0, 4.0], parents=["", "r0", "r0"], rollout_ids=["r0", "c1", "c2"])
        out = self._through("appo", batch, reward_scale_discount=0.9)
        self.assertTrue(torch.equal(out.batch["advantages"][1], torch.zeros(4)))
        self.assertTrue(torch.equal(out.batch["advantages"][2], torch.zeros(4)))
        self.assertTrue(torch.allclose(out.batch["advantages"][0], torch.full((4,), 3.7)))

    def test_unrelated_task_is_not_credited(self):
        batch = _batch_from(
            ["parent", "child"],
            [1.0, 9.0],
            data_ids=["task-a", "task-b"],
            rollout_ids=["r0", "c1"],
            parents=["", "r0"],
        )
        out = self._through("appo", batch)
        self.assertTrue(torch.equal(out.batch["advantages"][0], torch.ones(4)))

    def test_mask_limits_writeback(self):
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
        batch = _batch_from(["parent", "child"], [0.0, 2.0], parents=["", "r0"], rollout_ids=["r0", "c1"], mask=mask)
        out = self._through("appo", batch, reward_scale_discount=0.9)
        self.assertTrue(torch.allclose(out.batch["advantages"][0, :2], torch.full((2,), 2.8)))
        self.assertTrue(torch.equal(out.batch["advantages"][0, 2:], torch.ones(2)))

    def test_grpo_still_zeros_only_the_prefix(self):
        batch = _credit_batch()
        out = self._through("grpo", batch, zero_prefix_adv_for_children=True)
        self.assertTrue(torch.equal(out.batch["advantages"][1, :3], torch.zeros(3)))
        self.assertEqual(float(out.batch["advantages"][1, 3]), 1.0)

    def test_missing_advantages_is_unchanged(self):
        from rl.hooks.advantage import apply_appo_credit

        batch = DataProto.from_dict(tensors={"token_level_scores": torch.ones(1, 2)})
        out = apply_appo_credit(batch)
        self.assertTrue(torch.equal(out.batch["token_level_scores"], batch.batch["token_level_scores"]))


if __name__ == "__main__":
    unittest.main()
