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


if __name__ == "__main__":
    unittest.main()
