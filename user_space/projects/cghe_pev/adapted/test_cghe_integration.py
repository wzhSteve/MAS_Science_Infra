"""Mocked planner → executor → verifier → recovery loop. No network and no model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_cghe_example import candidate_json, replay_audit, run_window_case  # noqa: E402


class MockRecoveryLoopTests(unittest.TestCase):
    def test_switch_off_does_not_enter_recovery(self) -> None:
        case = run_window_case(enabled=False, candidate_text=candidate_json(), question="Where is the capital of France?")
        self.assertFalse(case["controller_present"])
        self.assertEqual(case["decisions"], [])
        self.assertEqual(case["candidate_engine_calls"], 0)
        self.assertEqual(case["generate_next_step_calls"], 2)
        self.assertEqual(case["final_decision"], "continue_pev")
        self.assertEqual(case["final_tool"], "Bing_Search_Tool")
        self.assertEqual(case["executed_tools"], ["Bing_Search_Tool"])
        self.assertFalse(case["graph_open"])

    def test_explicit_no_prunes_then_continues_on_the_selected_node(self) -> None:
        case = run_window_case(enabled=True, candidate_text=candidate_json(), question="Where is the capital of France?")
        self.assertEqual(case["decisions"], ["need_universe", "continue_execution"])
        self.assertEqual(case["task_impossible_flags"], [False, False])
        self.assertEqual(case["candidate_engine_calls"], 3)
        self.assertEqual(case["generate_next_step_calls"], 1)
        self.assertEqual(case["final_decision"], "continue_execution")
        self.assertEqual(case["final_tool"], "Web_Fetch_Tool")
        self.assertEqual(case["final_subgoal"], "open the page")
        self.assertEqual(set(case["active_plan_ids"]), {"fetch_a", "fetch_b", "code"})
        self.assertNotIn("bing", case["active_plan_ids"])
        self.assertEqual(case["executed_tools"], ["Bing_Search_Tool", "Web_Fetch_Tool"])

    def test_unparsed_candidates_fall_back_to_single_step_planning(self) -> None:
        case = run_window_case(
            enabled=True,
            candidate_text="I cannot write JSON plans",
            question="Where is the capital of France?",
        )
        self.assertEqual(case["decisions"], ["need_universe"])
        self.assertEqual(case["task_impossible_flags"], [False])
        self.assertEqual(case["candidate_engine_calls"], 3)
        self.assertEqual(case["generate_next_step_calls"], 2)
        self.assertEqual(case["final_tool"], "Bing_Search_Tool")
        self.assertFalse(case["graph_open"])
        self.assertEqual(case["executed_tools"], ["Bing_Search_Tool"])

    def test_replay_audit_refuses_python_and_generalist(self) -> None:
        audit = replay_audit()
        self.assertFalse(audit["python_replay_allowed"])
        self.assertIn("not proven idempotent", audit["python_reason"])
        self.assertFalse(audit["generalist_replay_allowed"])
        self.assertTrue(audit["bing_replay_allowed"])
        self.assertTrue(audit["fetch_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
