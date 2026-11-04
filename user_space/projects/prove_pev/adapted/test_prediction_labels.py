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
        self.assertEqual(normalize_prediction("满足"), "满足")
        self.assertEqual(normalize_prediction("不满足"), "不满足")
        self.assertEqual(normalize_prediction("partial"), "")
        self.assertEqual(normalize_prediction("部分满足"), "")
        self.assertEqual(normalize_prediction("success"), "")

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


if __name__ == "__main__":
    unittest.main()
