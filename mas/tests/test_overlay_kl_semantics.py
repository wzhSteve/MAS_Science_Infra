"""RL0 收编：overlay.py KL 显式存活语义的回归测试。

对应未提交改动（已收编 commit）：
1. ``apply_algo_overlay`` 对 ``algorithm.use_kl_in_reward``：仅键缺失时缺省 False，
   rl-yaml 显式键存活不被覆盖。
2. flow_grpo 的 ``actor.use_kl_loss``：显式 False 存活、缺省时仍强制 True + kl_loss_coef=0.001，
   且 ``apply_train_signal`` 再施加（re-apply）不得复活该标志。
"""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.dirname(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl.hooks.overlay import apply_algo_overlay, apply_train_signal  # noqa: E402
from rl.loss import AdvantageSpec, LossSpec  # noqa: E402
from rl.train_signal import TrainSignal  # noqa: E402


def _base_config() -> dict:
    return {
        "algorithm": {"tir_algo": "grpo"},
        "actor_rollout_ref": {"rollout": {"n": 4}},
        "trainer": {"total_training_steps": 3, "total_epochs": 1},
    }


class TestKlInRewardExplicitSurvival(unittest.TestCase):
    """use_kl_in_reward：显式键存活、缺省落 False。"""

    def test_missing_key_defaults_to_false(self) -> None:
        cfg = apply_algo_overlay(_base_config(), "grpo")
        self.assertFalse(cfg["algorithm"]["use_kl_in_reward"])

    def test_explicit_use_kl_in_reward_survives(self) -> None:
        cfg = _base_config()
        cfg["algorithm"]["use_kl_in_reward"] = True
        out = apply_algo_overlay(cfg, "grpo")
        self.assertTrue(out["algorithm"]["use_kl_in_reward"])
        # 输入 dict 不被原地污染（deepcopy 语义保持）
        self.assertEqual(cfg["algorithm"]["use_kl_in_reward"], True)


class TestFlowGrpoKlLossExplicitSurvival(unittest.TestCase):
    """flow_grpo use_kl_loss：显式 False 存活；缺省强制 True + coef=0.001。"""

    def test_default_forces_use_kl_loss_true(self) -> None:
        cfg = apply_algo_overlay(_base_config(), "flow_grpo")
        actor = cfg["actor_rollout_ref"]["actor"]
        self.assertTrue(actor["use_kl_loss"])
        self.assertEqual(actor["kl_loss_coef"], 0.001)

    def test_explicit_use_kl_loss_false_survives(self) -> None:
        cfg = _base_config()
        cfg["actor_rollout_ref"] = {"rollout": {"n": 4}, "actor": {"use_kl_loss": False}}
        # kl_loss_coef 取现值（flow_grpo Actor 默认），仅验证 use_kl_loss 存活
        out = apply_algo_overlay(cfg, "flow_grpo")
        self.assertFalse(out["actor_rollout_ref"]["actor"]["use_kl_loss"])
        self.assertNotIn("kl_loss_coef", out["actor_rollout_ref"]["actor"])

    def test_explicit_use_kl_loss_false_survives_train_signal(self) -> None:
        """apply_train_signal 再施加不得复活已显式关闭的 use_kl_loss。"""
        cfg = _base_config()
        cfg["actor_rollout_ref"] = {"rollout": {"n": 4}, "actor": {"use_kl_loss": False}}
        signal = TrainSignal(
            advantage=AdvantageSpec(name="flow_grpo"),
            loss=LossSpec(loss_mode="vanilla"),
        )
        out = apply_train_signal(cfg, signal)
        self.assertFalse(out["actor_rollout_ref"]["actor"]["use_kl_loss"])

    def test_train_signal_default_still_forces_true_and_coef(self) -> None:
        """缺省路径：re-apply 后仍强制 use_kl_loss=True，kl_loss_coef==0 时兜底 0.001。"""
        signal = TrainSignal(
            advantage=AdvantageSpec(name="flow_grpo"),
            loss=LossSpec(loss_mode="vanilla"),
        )
        out = apply_train_signal(_base_config(), signal)
        actor = out["actor_rollout_ref"]["actor"]
        self.assertTrue(actor["use_kl_loss"])
        self.assertEqual(actor["kl_loss_coef"], 0.001)


if __name__ == "__main__":
    unittest.main()
