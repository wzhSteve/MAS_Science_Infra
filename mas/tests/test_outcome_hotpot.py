"""HotpotQA outcome-reward behavior: alias F1, think stripping, format gate."""

from __future__ import annotations

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.dirname(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl.rewards.outcome import (  # noqa: E402
    compute_outcome_reward,
    extract_answer_text,
    f1_score,
    numeric_match,
    strip_thinking,
)


class TestStripThinking(unittest.TestCase):
    def test_closed_think_removed(self) -> None:
        self.assertEqual(
            strip_thinking("<answer><think>\nreasoning...\n</think>final</answer>"),
            "<answer>final</answer>",
        )

    def test_unclosed_think_blanked(self) -> None:
        """Truncated reasoning (no closing tag) leaves no scoreable answer.

        The `<answer>` open tag survives stripping, but it is a placeholder
        that ``extract_answer_text`` rejects, so accuracy stays 0.
        """
        self.assertEqual(strip_thinking("<answer><think>\nokay partial"), "<answer>")
        self.assertIsNone(extract_answer_text("<answer><think>\nokay partial"))

    def test_plain_text_untouched(self) -> None:
        self.assertEqual(strip_thinking("The answer is 42"), "The answer is 42")


class TestExtractAnswer(unittest.TestCase):
    def test_answer_tag_after_think(self) -> None:
        text = "<answer><think>chain</think>Prussian</answer>"
        self.assertEqual(extract_answer_text(text), "Prussian")

    def test_truncated_think_returns_none(self) -> None:
        self.assertIsNone(extract_answer_text("<answer><think>cut off mid"))


class TestHotpotF1(unittest.TestCase):
    def test_exact_alias_hit(self) -> None:
        self.assertEqual(f1_score("Prussian", ["Prussian"]), 1.0)

    def test_alias_list_second_entry(self) -> None:
        self.assertGreater(f1_score("royal", ["Prussian", "royal"]), 0.0)

    def test_partial_overlap(self) -> None:
        self.assertGreater(f1_score("the Prussian soldier", ["Prussian"]), 0.0)
        self.assertLess(f1_score("totally wrong", ["Prussian"]), 0.1)


class TestNumericMatchMultiCandidate(unittest.TestCase):
    """RL0 收编补充：numeric_match CoT 包裹答案的多候选语义。"""

    def test_cot_wrapped_answer_body_extracted(self) -> None:
        # <answer> 体内含多个数字时，直接比对整个候选串（首个数字锚定）
        self.assertTrue(numeric_match("<answer>Jason has 16 cousins.</answer>", "16"))

    def test_last_number_candidate(self) -> None:
        # CoT 末尾惯例 "Answer: 16"，取最后一个数字作为候选
        self.assertTrue(numeric_match("After much thought, the total is: 16", "16"))

    def test_first_number_candidate(self) -> None:
        # "16 cousins" 惯用模式：取第一个数字兜底
        self.assertTrue(numeric_match("16 cousins attended.", "16"))

    def test_answer_body_beats_plain_pred(self) -> None:
        # 整串首数字(文本区)不命中时应经 answer 体或末数字命中
        self.assertTrue(
            numeric_match(
                "Q: In 1492, how many ships? <answer>3</answer>",
                "3",
            )
        )

    def test_no_match_returns_false(self) -> None:
        self.assertFalse(numeric_match("The sky is blue", "16"))
        self.assertFalse(numeric_match("there were 17 apples", "16"))

    def test_literal_fallback_unchanged(self) -> None:
        # 数字解析失败时回落字面比较（大小写不敏感）——沿旧语义
        self.assertTrue(numeric_match("sixteen", "sixteen"))
        self.assertFalse(numeric_match("sixteen", "17"))


class TestOutcomeReward(unittest.TestCase):
    def _kwargs(self) -> dict:
        return {"source": "hotpot", "aliases": ["Prussian"], "format_ok": True, "n_search": 0, "n_python": 0}

    def test_correct_answer_scores_f1(self) -> None:
        r = compute_outcome_reward("<answer>Prussian</answer>", "Prussian", **self._kwargs())
        self.assertAlmostEqual(r, 1.0)

    def test_truncated_think_scores_zero_not_negative(self) -> None:
        """Format tag present but reasoning-only content -> acc 0 -> reward 0."""
        r = compute_outcome_reward("<answer><think>cut", "Prussian", **self._kwargs())
        self.assertEqual(r, 0.0)

    def test_format_fail_negative(self) -> None:
        kw = self._kwargs()
        kw["format_ok"] = False
        r = compute_outcome_reward("no tags at all", "Prussian", **kw)
        self.assertEqual(r, -1.0)

    def test_empty_prediction_zero(self) -> None:
        self.assertEqual(compute_outcome_reward("", "Prussian", **self._kwargs()), 0.0)
        self.assertEqual(compute_outcome_reward(None, "Prussian", **self._kwargs()), 0.0)


class TestRewardSpecNonePathByteEqual(unittest.TestCase):
    """RL4 等价门：reward_spec=None vs 默认字面量路径逐值一致；默认 spec 与字面量逐值一致。"""

    CASES = [
        # (pred, gold, source, aliases, format_ok, n_search, n_python, kwargs_bonus)
        ("<answer>Prussian</answer>", "Prussian", "hotpot", ["Prussian"], True, 0, 0, 0.1),
        ("<answer>the Prussian soldier</answer>", "Prussian", "hotpot", ["Prussian"], True, 2, 1, 0.1),
        ("no tags", "Prussian", "hotpot", [], False, 0, 0, 0.1),
        ("", "Prussian", "hotpot", [], True, 0, 0, 0.1),
        (None, "Prussian", "hotpot", [], True, 1, 1, 0.1),
        ("#### 16", "16", "gsm8k", [], True, 1, 1, 0.1),
        ("#### 17", "16", "gsm8k", [], True, 0, 0, 0.1),
    ]

    def test_reward_spec_none_path_byte_equal(self):
        """compute_outcome_reward(..., reward_spec=None) must equal the不带 spec 调用逐值，
        i.e. the None path IS the literal path (byte-level compat for v1 callers)."""
        for pred, gold, source, aliases, fmt, ns, np_, kw_bonus in self.CASES:
            for bonus in (0.1, 0.25, 0.0):
                literal = compute_outcome_reward(
                    pred, gold, source=source, aliases=aliases,
                    format_ok=fmt, n_search=ns, n_python=np_, multi_tool_bonus=bonus,
                )
                via_none = compute_outcome_reward(
                    pred, gold, source=source, aliases=aliases,
                    format_ok=fmt, n_search=ns, n_python=np_,
                    multi_tool_bonus=bonus, reward_spec=None,
                )
                self.assertEqual(
                    literal, via_none,
                    f"None path diverged for case {(pred, gold, source, fmt, ns, np_, bonus)}",
                )

    def test_default_spec_equals_literal_path_value_for_value(self):
        """The built-in default RewardSpec reproduces the literal path exactly
        (format -1 / acc / +0.1 bonus), across the same case grid."""
        from rl.rewards.assembly import default_reward_spec

        spec = default_reward_spec()
        for pred, gold, source, aliases, fmt, ns, np_, _ in self.CASES:
            literal = compute_outcome_reward(
                pred, gold, source=source, aliases=aliases,
                format_ok=fmt, n_search=ns, n_python=np_,
            )
            via_spec = compute_outcome_reward(
                pred, gold, source=source, aliases=aliases,
                format_ok=fmt, n_search=ns, n_python=np_, reward_spec=spec,
            )
            self.assertEqual(
                literal, via_spec,
                f"default spec diverged for case {(pred, gold, source, fmt, ns, np_)}",
            )


class TestRewardSpecComponents(unittest.TestCase):
    """Component switches: off disables bonus; format penalty configurable; cost slot inert."""

    def test_component_off_disables_bonus(self):
        """multi_tool_bonus enabled=False → the双工具 bonus 不再加成。"""
        from rl.rewards.assembly import RewardComponent, RewardSpec

        spec = RewardSpec(
            components=[
                RewardComponent(id="format_penalty", weight=1.0, enabled=True),
                RewardComponent(id="accuracy_reward", weight=1.0, enabled=True),
                RewardComponent(id="multi_tool_bonus", weight=0.1, enabled=False),
                RewardComponent(id="cost_penalty", weight=0.0, enabled=False),
            ]
        )
        r = compute_outcome_reward(
            "<answer>Prussian</answer>", "Prussian",
            source="hotpot", format_ok=True, n_search=2, n_python=1,
            reward_spec=spec,
        )
        self.assertEqual(r, 1.0)  # bonus off → 无 0.1 加成
        # 对照：默认 spec 下同输入得 1.1
        self.assertEqual(
            compute_outcome_reward(
                "<answer>Prussian</answer>", "Prussian",
                source="hotpot", format_ok=True, n_search=2, n_python=1,
            ),
            1.1,
        )

    def test_component_weight_scales_bonus(self):
        """weight 配权：multi_tool_bonus weight=0.3 → +0.3。"""
        from rl.rewards.assembly import RewardComponent, RewardSpec

        spec = RewardSpec(
            components=[
                RewardComponent(id="format_penalty", weight=1.0, enabled=True),
                RewardComponent(id="accuracy_reward", weight=1.0, enabled=True),
                RewardComponent(id="multi_tool_bonus", weight=0.3, enabled=True),
                RewardComponent(id="cost_penalty", weight=0.0, enabled=False),
            ]
        )
        r = compute_outcome_reward(
            "<answer>Prussian</answer>", "Prussian",
            source="hotpot", format_ok=True, n_search=1, n_python=1,
            reward_spec=spec,
        )
        self.assertAlmostEqual(r, 1.3, places=9)

    def test_format_penalty_zero_is_allowed(self):
        """format_penalty weight=0（合法配权）→ 格式失败罚 0 而非 -1；
        显式 enabled=False → 完全关：格式失败落到 accuracy 层评估。"""
        from rl.rewards.assembly import RewardComponent, RewardSpec

        zero_w = RewardSpec(
            components=[
                RewardComponent(id="format_penalty", weight=0.0, enabled=True),
                RewardComponent(id="accuracy_reward", weight=1.0, enabled=True),
                RewardComponent(id="multi_tool_bonus", weight=0.1, enabled=True),
                RewardComponent(id="cost_penalty", weight=0.0, enabled=False),
            ]
        )
        r = compute_outcome_reward(
            "no tags at all", "Prussian",
            source="hotpot", format_ok=False, n_search=0, n_python=0,
            reward_spec=zero_w,
        )
        self.assertEqual(r, 0.0)  # -1.0 * 0.0 = 0.0 罚值
        off = RewardSpec(
            components=[
                RewardComponent(id="format_penalty", weight=1.0, enabled=False),
                RewardComponent(id="accuracy_reward", weight=1.0, enabled=True),
                RewardComponent(id="multi_tool_bonus", weight=0.1, enabled=True),
                RewardComponent(id="cost_penalty", weight=0.0, enabled=False),
            ]
        )
        r2 = compute_outcome_reward(
            "no tags at all", "Prussian",
            source="hotpot", format_ok=False, n_search=0, n_python=0,
            reward_spec=off,
        )
        # 关罚后掉到 accuracy 层：答案文本 'no tags at all' vs 'Prussian' F1=0 → 0.0
        self.assertEqual(r2, 0.0)
        # 但若关闭罚件且答案确实在文本里（非 <answer> 包裹也可 F1 命中）:
        r3 = compute_outcome_reward(
            "Prussian", "Prussian",
            source="hotpot", format_ok=False, n_search=0, n_python=0,
            reward_spec=off,
        )
        self.assertEqual(r3, 1.0)

    def test_format_penalty_weight_scales_penalty(self):
        """format penalty 可配权：weight=2.5 → 格式失败罚 -2.5。"""
        from rl.rewards.assembly import RewardComponent, RewardSpec

        spec = RewardSpec(
            components=[
                RewardComponent(id="format_penalty", weight=2.5, enabled=True),
                RewardComponent(id="accuracy_reward", weight=1.0, enabled=True),
                RewardComponent(id="multi_tool_bonus", weight=0.1, enabled=True),
                RewardComponent(id="cost_penalty", weight=0.0, enabled=False),
            ]
        )
        r = compute_outcome_reward(
            "no tags", "Prussian", source="hotpot",
            format_ok=False, n_search=0, n_python=0, reward_spec=spec,
        )
        self.assertEqual(r, -2.5)

    def test_cost_penalty_placeholder_disabled(self):
        """cost_penalty 预留位：默认 spec 中存在但 disabled，参与计算即冗余 no-op；
        尝试 enabled=True 创建 spec 直接被 pydantic 校验拒绝。"""
        from rl.rewards.assembly import (
            COST_PENALTY,
            RewardComponent,
            RewardSpec,
            default_reward_spec,
        )

        spec = default_reward_spec()
        comp = spec.component(COST_PENALTY)
        self.assertFalse(comp.enabled, "cost_penalty 预留位必须 born disabled")
        self.assertEqual(comp.weight, 0.0)
        # 启用预留位 → spec 校验拒绝（fail-fast，不留半开状态）
        with self.assertRaises(ValueError):
            RewardSpec(
                components=[
                    RewardComponent(id="format_penalty"),
                    RewardComponent(id="accuracy_reward"),
                    RewardComponent(id="multi_tool_bonus"),
                    RewardComponent(id="cost_penalty", weight=0.1, enabled=True),
                ]
            )

    def test_accuracy_reward_cannot_be_disabled(self):
        from rl.rewards.assembly import RewardComponent, RewardSpec

        with self.assertRaises(ValueError):
            RewardSpec(
                components=[
                    RewardComponent(id="format_penalty"),
                    RewardComponent(id="accuracy_reward", enabled=False),
                    RewardComponent(id="multi_tool_bonus"),
                ]
            )
        with self.assertRaises(ValueError):
            RewardSpec(components=[RewardComponent(id="format_penalty")])

    def test_spec_validation_rejects_duplicates_and_unknowns(self):
        from rl.rewards.assembly import RewardComponent, RewardSpec

        with self.assertRaises(ValueError):
            RewardSpec(
                components=[
                    RewardComponent(id="format_penalty"),
                    RewardComponent(id="accuracy_reward"),
                    RewardComponent(id="format_penalty"),
                ]
            )
        with self.assertRaises(ValueError):
            RewardSpec(
                components=[
                    RewardComponent(id="format_penalty"),
                    RewardComponent(id="accuracy_reward"),
                    RewardComponent(id="visibility_bonus"),
                ]
            )


class TestCollectorRewardSpecPassThrough(unittest.TestCase):
    """RL4 collector 透传链：reward_spec=None 字节级不变；提供时经 default_reward_fn 生效。"""

    def _traj_and_task(self):
        from workflow.contracts import Trajectory

        traj = Trajectory(
            messages=[{"role": "assistant", "content": "<answer>Prussian</answer>"}],
            final_answer="<answer>Prussian</answer>",
            format_ok=True,
            n_search=2,
            n_python=1,
        )
        task = {"answer": "Prussian", "source": "hotpot"}
        return traj, task

    def test_collector_none_path_equals_legacy_call(self):
        from workflow.collector import default_reward_fn

        traj, task = self._traj_and_task()
        self.assertEqual(default_reward_fn(traj, task), 1.1)  # 现状（F1 1.0 + 0.1 双工具）
        # None 显式传参与缺省一致
        self.assertEqual(default_reward_fn(traj, task, reward_spec=None), 1.1)

    def test_collector_spec_disables_bonus(self):
        from rl.rewards.assembly import RewardComponent, RewardSpec
        from workflow.collector import default_reward_fn

        traj, task = self._traj_and_task()
        spec = RewardSpec(
            components=[
                RewardComponent(id="format_penalty"),
                RewardComponent(id="accuracy_reward"),
                RewardComponent(id="multi_tool_bonus", weight=0.1, enabled=False),
                RewardComponent(id="cost_penalty", enabled=False),
            ]
        )
        self.assertEqual(default_reward_fn(traj, task, reward_spec=spec), 1.0)


class TestRewardSectionWiring(unittest.TestCase):
    """RL4 死配置接通：overlay 把 tir.multi_tool_bonus 镜像进 algorithm.reward 段。"""

    def _base(self):
        return {
            "algorithm": {},
            "actor_rollout_ref": {"rollout": {"n": 4}},
            "trainer": {},
        }

    def test_overlay_writes_reward_section(self):
        from rl.hooks.overlay import apply_algo_overlay

        cfg = apply_algo_overlay(self._base(), "appo")
        section = cfg["algorithm"]["reward"]
        self.assertEqual(section["outcome"], "hierarchical")
        by_id = {c["id"]: c for c in section["components"]}
        self.assertTrue(by_id["format_penalty"]["enabled"])
        self.assertTrue(by_id["accuracy_reward"]["enabled"])
        self.assertTrue(by_id["multi_tool_bonus"]["enabled"])
        self.assertEqual(by_id["multi_tool_bonus"]["weight"], 0.1)  # DEFAULT_TIR 值接通
        self.assertFalse(by_id["cost_penalty"]["enabled"])  # 预留位 disabled

    def test_tir_multi_tool_bonus_survives_dual_read(self):
        """双读兼容：algorithm.tir.multi_tool_bonus 原路径保留，同值可读。"""
        from rl.hooks.overlay import apply_algo_overlay

        cfg = apply_algo_overlay(self._base(), "grpo")
        self.assertEqual(cfg["algorithm"]["tir"]["multi_tool_bonus"], 0.1)
        self.assertEqual(cfg["algorithm"]["reward"]["components"][2]["weight"], 0.1)

    def test_user_yaml_override_flows_into_reward_section(self):
        """用户 yaml 的 tir.multi_tool_bonus 覆写值成为 spec 的 weight。"""
        from rl.hooks.overlay import apply_algo_overlay

        base = self._base()
        base["algorithm"]["tir"] = {"multi_tool_bonus": 0.3}
        cfg = apply_algo_overlay(base, "grpo")
        by_id = {c["id"]: c for c in cfg["algorithm"]["reward"]["components"]}
        self.assertEqual(by_id["multi_tool_bonus"]["weight"], 0.3)
        self.assertEqual(by_id["multi_tool_bonus"]["enabled"], True)

    def test_zero_bonus_weight_disables_component(self):
        from rl.hooks.overlay import apply_algo_overlay

        base = self._base()
        base["algorithm"]["tir"] = {"multi_tool_bonus": 0.0}
        cfg = apply_algo_overlay(base, "grpo")
        by_id = {c["id"]: c for c in cfg["algorithm"]["reward"]["components"]}
        self.assertEqual(by_id["multi_tool_bonus"]["weight"], 0.0)
        self.assertFalse(by_id["multi_tool_bonus"]["enabled"])

    def test_reward_spec_from_config_reads_v2_and_v1(self):
        """reward_spec_from_config：v2 algorithm.reward 段优先，v1 tir 键 fallback。"""
        from rl.rewards.assembly import reward_spec_from_config

        # v1 tir 路径
        spec = reward_spec_from_config(
            {"algorithm": {"tir": {"multi_tool_bonus": 0.25}}}
        )
        self.assertIsNotNone(spec)
        self.assertEqual(spec.weight_of("multi_tool_bonus"), 0.25)
        # v2 段路径
        spec2 = reward_spec_from_config(
            {
                "algorithm": {
                    "reward": {
                        "outcome": "hierarchical",
                        "components": [
                            {"id": "accuracy_reward", "weight": 1.0, "enabled": True},
                            {"id": "multi_tool_bonus", "weight": 0.5, "enabled": True},
                            {"id": "format_penalty", "weight": 2.0, "enabled": True},
                        ],
                    }
                }
            }
        )
        self.assertEqual(spec2.weight_of("multi_tool_bonus"), 0.5)
        self.assertEqual(spec2.weight_of("format_penalty"), 2.0)
        # v2 段不含 accuracy_reward → 校验拒绝（主奖励必在）
        with self.assertRaises(Exception):
            reward_spec_from_config(
                {
                    "algorithm": {
                        "reward": {
                            "components": [{"id": "multi_tool_bonus", "weight": 0.5}],
                        }
                    }
                }
            )
        # 无任何 reward 面 → None（保持字面量路径）
        self.assertIsNone(reward_spec_from_config({"algorithm": {"tir": {}}}))

    def test_train_signal_path_keeps_reward_section(self):
        """apply_train_signal 两趟 re-apply 后 reward 段仍在（幂等）。"""
        from rl.hooks.overlay import apply_train_signal
        from rl.loss import AdvantageSpec, LossSpec
        from rl.train_signal import TrainSignal

        signal = TrainSignal(
            advantage=AdvantageSpec(name="appo"),
            loss=LossSpec(name="appo"),
        )
        cfg = apply_train_signal(self._base(), signal)
        # re-apply (simulating the merge flow's second pass)
        cfg = apply_train_signal(cfg, signal)
        section = cfg["algorithm"]["reward"]
        by_id = {c["id"]: c for c in section["components"]}
        self.assertEqual(by_id["multi_tool_bonus"]["weight"], 0.1)
        self.assertTrue(by_id["accuracy_reward"]["enabled"])


if __name__ == "__main__":
    unittest.main()
