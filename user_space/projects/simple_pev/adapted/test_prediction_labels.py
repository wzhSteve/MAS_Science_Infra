"""Planner labels are 满足 / 不满足, and fetch URLs must come from Bing."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prediction_probs import (  # noqa: E402
    binary_softmax,
    prediction_record,
    sum_suffix_logprobs,
)
from sft_prompts import (  # noqa: E402
    listed_fetch_command,
    normalize_prediction,
    tags_to_execute,
)


class _Memory:
    def __init__(self, actions):
        self.actions = actions

    def get_actions(self):
        return self.actions


class TestPredictionLabels(unittest.TestCase):
    def test_partial_is_not_a_prediction(self) -> None:
        # canonical_prediction maps every accepted spelling onto YES / NO.
        self.assertEqual(normalize_prediction("满足"), "YES")
        self.assertEqual(normalize_prediction("不满足"), "NO")
        self.assertEqual(normalize_prediction("YES"), "YES")
        self.assertEqual(normalize_prediction("no"), "NO")
        self.assertEqual(normalize_prediction("partial"), "")
        self.assertEqual(normalize_prediction("部分满足"), "")
        self.assertEqual(normalize_prediction("success"), "")

    def test_lenient_anchor_bare_and_quoted(self) -> None:
        from prediction_probs import prediction_anchor, prediction_prefix, written_prediction

        quoted = 'Some prefix "prediction": "NO", "prediction_reason": "x"'
        bare = 'Some prefix "prediction": NO, "prediction_reason": "x"'
        spaced = '"prediction":   YES'
        for text, want in ((quoted, "NO"), (bare, "NO"), (spaced, "YES")):
            self.assertEqual(written_prediction(text), want)
            self.assertIsNotNone(prediction_anchor(text))
            prefix = prediction_prefix(text)
            self.assertIsNotNone(prefix)
            self.assertTrue(text.startswith(prefix))
            self.assertFalse(prefix.endswith("NO"))
            self.assertFalse(prefix.endswith("YES"))
        # Truncated before the value: nothing to anchor, must report None.
        self.assertIsNone(prediction_anchor('{"prediction": '))
        self.assertEqual(written_prediction('{"prediction": partial'), "")

    def test_two_token_label_softmax(self) -> None:
        # Echo row: [BOS=None, prefix..., 不, 满足]
        echoed = [None, -0.2, -0.5, -0.4]
        unsatisfy = sum_suffix_logprobs(echoed, n_prefix=2, n_label=2)
        satisfy = sum_suffix_logprobs(echoed, n_prefix=2, n_label=1)
        self.assertAlmostEqual(unsatisfy, -0.9)
        self.assertAlmostEqual(satisfy, -0.5)
        p_yes, p_no = binary_softmax(score_satisfy=-0.2, score_unsatisfy=-0.9)
        self.assertAlmostEqual(p_yes + p_no, 1.0)
        self.assertGreater(p_yes, 0.5)
        record = prediction_record("不满足", score_satisfy=-0.2, score_unsatisfy=-0.9)
        self.assertTrue(record["prediction_mismatch"])
        agreed = prediction_record("满足", score_satisfy=-0.2, score_unsatisfy=-0.9)
        self.assertFalse(agreed["prediction_mismatch"])
        self.assertTrue(math.isfinite(record["p_satisfy"]))


class TestListedFetch(unittest.TestCase):
    def test_copied_urls_skip_rewrite_and_invented_urls_do_not(self) -> None:
        memory = _Memory({
            "Action Step 1": {
                "tool_name": "Bing_Search_Tool",
                "result": (
                    "bing results for: q\n"
                    "1. A\n   URL: https://example.com/a\n"
                    "2. B\n   URL: https://example.com/b\n"
                    "3. C\n   URL: https://example.com/c\n"
                    "4. D\n   URL: https://example.com/d\n"
                ),
            }
        })
        command = listed_fetch_command(
            "Web_Fetch_Tool",
            "https://example.com/a\nhttps://example.com/c",
            memory,
        )
        self.assertIn("https://example.com/a", command)
        self.assertIn("https://example.com/c", command)
        self.assertNotIn("https://example.com/d", command)
        invented = listed_fetch_command(
            "Web_Fetch_Tool",
            "https://en.wikipedia.org/wiki/Other",
            memory,
        )
        self.assertEqual(invented, "")

    def test_search_tag_keeps_three_urls(self) -> None:
        raw = (
            "<search>\n"
            "https://example.com/a\n"
            "https://example.com/b\n"
            "https://example.com/c\n"
            "https://example.com/d\n"
            "</search>"
        )
        command = tags_to_execute("Web_Fetch_Tool", raw, "No command found.")
        self.assertIn("https://example.com/a", command)
        self.assertIn("https://example.com/c", command)
        self.assertNotIn("https://example.com/d", command)


    def test_planner_step_scoreable_gate(self) -> None:
        from sft_prompts import planner_step_scoreable

        good = '{"selected_tool": "Bing_Search_Tool", "tool_input": "quantum error", "prediction": "YES"}'
        self.assertTrue(planner_step_scoreable(good))
        # Canonicalized Chinese label counts too.
        self.assertTrue(
            planner_step_scoreable(
                '{"selected_tool": "Bing_Search_Tool", "tool_input": "q", "prediction": "\\u4e0d\\u6ee1\\u8db3"}'
            )
        )
        # Bare NO is not valid JSON: not parseable -> not scoreable (Rf pushes
        # the policy back to canonical JSON, the lenient anchor keeps it scoreable
        # by logprobs in the meantime).
        bare = '{"selected_tool": "Bing_Search_Tool", "tool_input": "q", "prediction": NO}'
        self.assertFalse(planner_step_scoreable(bare))
        # Unknown tool, empty input, missing/partial prediction all fail.
        self.assertFalse(
            planner_step_scoreable(
                '{"selected_tool": "NotATool", "tool_input": "q", "prediction": "YES"}'
            )
        )
        self.assertFalse(
            planner_step_scoreable(
                '{"selected_tool": "Bing_Search_Tool", "tool_input": "", "prediction": "YES"}'
            )
        )
        self.assertFalse(
            planner_step_scoreable(
                '{"selected_tool": "Bing_Search_Tool", "tool_input": "q", "prediction": "partial"}'
            )
        )
        self.assertFalse(planner_step_scoreable(""))
        self.assertFalse(planner_step_scoreable(None))

    def test_score_failures_are_counted_not_swallowed(self) -> None:
        import engine

        engine.reset_score_failure_counts()
        engine.note_score_failure("anchor_token_out_of_range")
        engine.note_score_failure("anchor_token_out_of_range")
        engine.note_score_failure("logprobs_missing")
        counts = engine.score_failure_counts()
        self.assertEqual(counts.get("anchor_token_out_of_range"), 2)
        self.assertEqual(counts.get("logprobs_missing"), 1)
        engine.reset_score_failure_counts()
        self.assertEqual(engine.score_failure_counts(), {})


if __name__ == "__main__":
    unittest.main()
