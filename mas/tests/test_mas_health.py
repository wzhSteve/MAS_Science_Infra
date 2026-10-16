"""MAS faults are not model-wrong answers, and they block a training update."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAS = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(MAS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from workflow.mas_health import assess_step, episode_faults  # noqa: E402


class TestMasHealth(unittest.TestCase):
    def test_wrong_answer_with_a_successful_tool_is_not_a_fault(self):
        faults = episode_faults(
            messages=[{
                "kind": "tool_result",
                "payload": {"output": "bing results for: citibank", "ok": True, "evidence_type": "DIRECT"},
            }],
            final_answer="<answer>Thomas Jefferson</answer>",
            reward=0.0,
        )
        self.assertEqual(faults, [])

    def test_empty_hit_list_is_not_a_backend_fault(self):
        faults = episode_faults(
            messages=[{
                "kind": "tool_result",
                "payload": {
                    "output": '["search_unavailable: direct: 0 hits"]',
                    "ok": False,
                    "evidence_type": "ERROR",
                },
            }],
            final_answer="<answer>no</answer>",
            reward=0.0,
        )
        self.assertEqual(faults, [])

    def test_missing_page_is_not_a_backend_fault(self):
        faults = episode_faults(
            messages=[{
                "kind": "tool_result",
                "payload": {"output": '["fetch_unavailable: GET 404; GET 404"]', "ok": False, "evidence_type": "ERROR"},
            }],
            final_answer="<answer>no</answer>",
            reward=0.0,
        )
        self.assertEqual(faults, [])

    def test_search_unavailable_is_a_fault(self):
        faults = episode_faults(
            messages=[{
                "kind": "tool_result",
                "payload": {"output": "search_unavailable: TIR_OFFLINE_SEARCH=1", "ok": False, "evidence_type": "ERROR"},
            }],
            final_answer="",
            reward=0.0,
        )
        self.assertTrue(any("tir_offline_search" in item for item in faults))

    def test_degenerate_reward_without_a_fault_stays_a_model_error(self):
        report = assess_step([
            {"reward": 0.0, "mas_fault": ""},
            {"reward": 0.0, "mas_fault": ""},
        ])
        self.assertTrue(report["ok"])
        self.assertTrue(report["degenerate_reward"])

    def test_degenerate_reward_with_a_fault_blocks_the_step(self):
        report = assess_step([
            {"reward": -1.0, "mas_fault": "planner window failed: boom"},
            {"reward": -1.0, "mas_fault": ""},
        ])
        self.assertFalse(report["ok"])
        self.assertTrue(any("degenerate" in item for item in report["faults"]))

    def test_frozen_executor_span_is_a_fault(self):
        faults = episode_faults(span_names=["agent__planner", "agent__u_agentflow__executor"], reward=1.0)
        self.assertTrue(any("frozen span" in item for item in faults))

    def test_nan_reward_is_a_fault(self):
        faults = episode_faults(reward=float("nan"))
        self.assertTrue(any("not finite" in item for item in faults))
