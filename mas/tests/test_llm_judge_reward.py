"""Unit tests for LLM judge + deep_research_llm outcome (mocked API)."""

from __future__ import annotations

import unittest
from unittest import mock


class TestParseTrueFalse(unittest.TestCase):
    def test_tagged(self):
        from rl.rewards.llm_judge import parse_true_false

        self.assertTrue(parse_true_false("<true_false>True</true_false>\n<analysis>ok</analysis>"))
        self.assertFalse(parse_true_false("<true_false>False</true_false>"))

    def test_truncated_open_tag(self):
        from rl.rewards.llm_judge import parse_true_false

        self.assertTrue(parse_true_false("<true_false>True\nmore analysis cut"))


class TestOutcomeModes(unittest.TestCase):
    def test_llm_binary(self):
        from rl.rewards.assembly import RewardSpec
        from rl.rewards.outcome import compute_outcome_reward

        spec = RewardSpec(outcome="llm_binary")
        with mock.patch("rl.rewards.llm_judge.llm_judge_correct", return_value=1.0) as judge:
            score = compute_outcome_reward(
                "<answer>Paris</answer>",
                "Paris",
                source="hotpot",
                format_ok=False,
                n_search=2,
                n_python=2,
                reward_spec=spec,
                question="capital?",
            )
        self.assertEqual(score, 1.0)
        judge.assert_called()

    def test_deep_research_unparseable(self):
        from rl.rewards.assembly import RewardSpec
        from rl.rewards.outcome import compute_outcome_reward

        spec = RewardSpec(outcome="deep_research_llm")
        score = compute_outcome_reward(
            "no answer tags",
            "Paris",
            source="hotpot",
            format_ok=False,
            n_search=0,
            n_python=0,
            reward_spec=spec,
            question="q",
        )
        self.assertEqual(score, -0.3)

    def test_deep_research_multi_tool(self):
        from rl.rewards.assembly import RewardSpec
        from rl.rewards.outcome import compute_outcome_reward

        spec = RewardSpec(outcome="deep_research_llm")
        with mock.patch("rl.rewards.llm_judge.llm_judge_correct", return_value=1.0):
            score = compute_outcome_reward(
                "<answer>\\boxed{Paris}</answer>",
                "Paris",
                source="hotpot",
                format_ok=True,
                n_search=1,
                n_python=1,
                reward_spec=spec,
                question="capital?",
            )
        self.assertAlmostEqual(score, 1.1)


if __name__ == "__main__":
    unittest.main()
