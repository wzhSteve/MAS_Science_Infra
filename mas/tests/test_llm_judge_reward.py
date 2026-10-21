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

    def test_both_verdicts_are_not_a_decision(self):
        from rl.rewards.llm_judge import build_judge_prompt, parse_true_false

        copied = "<true_false>True</true_false> or <true_false>False</true_false>"
        self.assertIsNone(parse_true_false(copied))
        prompt = build_judge_prompt("q", "gold", "pred")
        self.assertNotIn("<true_false>True</true_false>", prompt)
        self.assertIn("<true_false>VERDICT</true_false>", prompt)


class TestOutcomeModes(unittest.TestCase):
    def _score(self, prediction, gold, question="", judge_value=1.0):
        from rl.rewards.assembly import RewardSpec
        from rl.rewards.outcome import compute_outcome_reward

        spec = RewardSpec(outcome="llm_binary")
        with mock.patch("rl.rewards.llm_judge.llm_judge_correct", return_value=judge_value) as judge:
            score = compute_outcome_reward(
                prediction,
                gold,
                source="hotpot",
                format_ok=False,
                n_search=0,
                n_python=0,
                reward_spec=spec,
                question=question,
            )
        return score, judge

    def test_llm_binary_exact_does_not_call_judge(self):
        score, judge = self._score("<answer>Paris</answer>", "Paris", question="capital?")
        self.assertEqual(score, 1.0)
        judge.assert_not_called()

    def test_placeholder_formula_and_partial_name_stay_zero(self):
        short, short_judge = self._score("<answer>short answer</answer>", "1185")
        self.assertEqual(short, 0.0)
        short_judge.assert_not_called()
        formula, formula_judge = self._score(
            "<answer>distance_between_centers + radius_A + radius_B</answer>",
            "137.0",
        )
        self.assertEqual(formula, 0.0)
        formula_judge.assert_not_called()
        partial, partial_judge = self._score("<answer>Mats Hummel</answer>", "Mats Julian Hummels")
        self.assertEqual(partial, 0.0)
        partial_judge.assert_not_called()

    def test_shared_long_name_can_use_the_judge(self):
        score, judge = self._score("<answer>Mats Hummels</answer>", "Mats Julian Hummels")
        self.assertEqual(score, 1.0)
        judge.assert_called()

    def test_option_letter_matches_locally(self):
        question = (
            r"The largest divisor of $2,014,000,000$ is itself. "
            r"$\textbf{(A) } 125,875,000 \qquad \textbf{(B) } 201,400,000 "
            r"\qquad \textbf{(C) } 251,750,000$"
        )
        score, judge = self._score("<answer>C</answer>", "251,750,000", question=question)
        self.assertEqual(score, 1.0)
        judge.assert_not_called()
        wrong, wrong_judge = self._score("<answer>B</answer>", "251,750,000", question=question)
        self.assertEqual(wrong, 0.0)
        wrong_judge.assert_not_called()

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
