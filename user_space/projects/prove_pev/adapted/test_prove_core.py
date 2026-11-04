"""Offline tests for MAS-ProVe Judge protocol and sync Best-of-N (no LLM)."""

from __future__ import annotations

import unittest

from prove_controller import (
    build_prove_context,
    command_site_enabled,
    load_prove_settings,
    plan_site_enabled,
)
from prove_judge import build_messages_for_judge, feedback2rankings, judge_inline
from prove_search import sample_candidates, select_best
from prove_types import ProveCandidate


class JudgeProtocolTests(unittest.TestCase):
    def test_ranking_parse(self) -> None:
        feedback = "explanation...\n<ranking>2,1,3</ranking>"
        self.assertEqual(feedback2rankings(feedback, 3), [1, 0, 2])

    def test_ranking_fallback(self) -> None:
        self.assertEqual(feedback2rankings("no tags", 3), [0, 1, 2])
        self.assertEqual(feedback2rankings("<ranking>1,2</ranking>", 3), [0, 1, 2])

    def test_messages_qa(self) -> None:
        messages = build_messages_for_judge(["a", "b"], "qa", "Q?")
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("<ranking>", messages[0]["content"])
        self.assertIn("Candidate 1", messages[1]["content"])
        self.assertIn("Q?", messages[1]["content"])

    def test_judge_inline_uses_llm_call(self) -> None:
        result = judge_inline(
            [
                {"context": "ctx", "current-step": "step-a"},
                {"context": "ctx", "current-step": "step-b"},
            ],
            question="Q",
            task_type="qa",
            llm_call=lambda _s, _u: "reason\n<ranking>2,1</ranking>",
        )
        self.assertEqual(result["rankings"], [1, 0])


class SearchTests(unittest.TestCase):
    def test_select_best_picks_rankings_zero(self) -> None:
        cands = [
            ProveCandidate(0, "bad"),
            ProveCandidate(1, "good"),
            ProveCandidate(2, "mid"),
        ]
        decision = select_best(
            cands,
            question="Q",
            context="ctx",
            task_type="qa",
            site="plan",
            judge_backend="inline",
            llm_call=lambda _s, _u: "<ranking>2,3,1</ranking>",
        )
        self.assertEqual(decision.best_idx, 1)
        self.assertEqual(decision.candidates[decision.best_idx], "good")

    def test_n_eq_1_skips_judge(self) -> None:
        calls = []
        decision = select_best(
            [ProveCandidate(0, "only")],
            question="Q",
            context="ctx",
            task_type="qa",
            site="plan",
            judge_backend="inline",
            llm_call=lambda s, u: calls.append((s, u)) or "",
        )
        self.assertEqual(decision.skipped_reason, "n_eq_1")
        self.assertEqual(calls, [])

    def test_sample_candidates_order(self) -> None:
        cands = sample_candidates(lambda i: f"c{i}", 3, parallel=False)
        self.assertEqual([c.text for c in cands], ["c0", "c1", "c2"])


class SettingsTests(unittest.TestCase):
    def test_defaults_and_granularity(self) -> None:
        settings = load_prove_settings({"prove": {"enabled": True, "granularity": "iteration", "n": 3}})
        self.assertEqual(settings["granularity"], "plan")
        self.assertTrue(plan_site_enabled(settings))
        self.assertFalse(command_site_enabled(settings))

    def test_plan_and_command(self) -> None:
        settings = load_prove_settings({"prove": {"enabled": True, "granularity": "plan_and_command"}})
        self.assertTrue(plan_site_enabled(settings))
        self.assertTrue(command_site_enabled(settings))

    def test_disabled(self) -> None:
        settings = load_prove_settings({"prove": {"enabled": False}})
        self.assertFalse(plan_site_enabled(settings))

    def test_context_modes(self) -> None:
        class Mem:
            outline = ["a"]
            obtained_information = ["b"]
            actions = {"Action Step 1": {"tool_name": "Bing", "sub_goal": "s", "result": "r"}}

        full = build_prove_context(question="Q", memory=Mem(), query_analysis="qa", mode="full_memory")
        step = build_prove_context(question="Q", memory=Mem(), mode="step_only")
        self.assertIn("Executed steps", full)
        self.assertEqual(step, "Question: Q")


if __name__ == "__main__":
    unittest.main()
