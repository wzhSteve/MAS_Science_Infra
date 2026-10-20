"""10k parquet rows expose question and answer before a rollout starts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lit_tir_agent import hydrate_rl_task  # noqa: E402


class TestHydrateRlTask(unittest.TestCase):
    def test_maps_10k_columns(self):
        task = {
            "data_source": "DR_grpo_mix",
            "extra_info": {"question": "Who is older?", "index": 0},
            "reward_model": {"ground_truth": "Eleanor Of Lancaster", "style": "rule"},
        }
        hydrate_rl_task(task)
        self.assertEqual(task["question"], "Who is older?")
        self.assertEqual(task["answer"], "Eleanor Of Lancaster")
        self.assertEqual(task["source"], "DR_grpo_mix")

    def test_parses_stringified_structs(self):
        task = {
            "data_source": "math",
            "extra_info": "{'question': 'Triangle ABC?', 'index': 1}",
            "reward_model": "{'ground_truth': '3', 'style': 'rule'}",
        }
        hydrate_rl_task(task)
        self.assertEqual(task["question"], "Triangle ABC?")
        self.assertEqual(task["answer"], "3")

    def test_keeps_existing_question_and_answer(self):
        task = {
            "question": "already set",
            "answer": "gold",
            "source": "gsm8k",
            "extra_info": {"question": "other"},
            "reward_model": {"ground_truth": "other-gold"},
            "data_source": "DR_grpo_mix",
        }
        hydrate_rl_task(task)
        self.assertEqual(task["question"], "already set")
        self.assertEqual(task["answer"], "gold")
        self.assertEqual(task["source"], "gsm8k")


if __name__ == "__main__":
    unittest.main()
