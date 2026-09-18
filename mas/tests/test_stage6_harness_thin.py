"""v1.2: thin cognitive_convergence + reward_hacking (no GPU / no Judge LLM)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _err_task(tid: str, err: str) -> dict:
    return {
        "id": tid,
        "question": "fail",
        "answer": "0",
        "source": "gsm8k",
        "_mock_error": err,
    }


def _ok_task(tid: str, q: str = "1+1", a: str = "2") -> dict:
    return {"id": tid, "question": q, "answer": a, "source": "gsm8k", "_mock_answer": a}


class TestCognitiveConvergence(unittest.TestCase):
    def test_repeated_same_error_is_high_deviation(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        c = Collector(mock=True)
        trajs = [c.collect_one(_err_task("e1", "boom")), c.collect_one(_err_task("e2", "boom"))]
        hyps = HARNESS.diagnose({"trajectories": trajs}, names=["cognitive_convergence"])
        self.assertTrue(hyps)
        self.assertEqual(hyps[0].plugin, "cognitive_convergence")
        self.assertIn("high deviation", hyps[0].message)
        self.assertIn("boom", hyps[0].message)

    def test_normalized_ids_same_class(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        c = Collector(mock=True)
        trajs = [
            c.collect_one(_err_task("e1", "timeout after 12s")),
            c.collect_one(_err_task("e2", "timeout after 99s")),
        ]
        hyps = HARNESS.diagnose({"trajectories": trajs}, names=["cognitive_convergence"])
        self.assertTrue(hyps)
        self.assertEqual(hyps[0].meta.get("n_traj"), 2)

    def test_single_error_not_enough(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        bad = Collector(mock=True).collect_one(_err_task("e1", "boom"))
        self.assertEqual(HARNESS.diagnose({"trajectory": bad}, names=["cognitive_convergence"]), [])

    def test_distinct_errors_once_each(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        c = Collector(mock=True)
        trajs = [c.collect_one(_err_task("e1", "boom")), c.collect_one(_err_task("e2", "timeout"))]
        self.assertEqual(HARNESS.diagnose({"trajectories": trajs}, names=["cognitive_convergence"]), [])

    def test_error_then_success_counts_as_aligned(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        c = Collector(mock=True)
        trajs = [
            c.collect_one(_err_task("e1", "boom")),
            c.collect_one(_err_task("e2", "boom")),
            c.collect_one(_ok_task("ok1")),
            c.collect_one(_ok_task("ok2")),
        ]
        self.assertEqual(HARNESS.diagnose({"trajectories": trajs}, names=["cognitive_convergence"]), [])

    def test_clean_batch_empty(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        clean = Collector(mock=True).collect_one(_ok_task("ok"))
        self.assertEqual(HARNESS.diagnose({"trajectory": clean}, names=["cognitive_convergence"]), [])


class TestRewardHacking(unittest.TestCase):
    def test_high_reward_missing_format(self):
        from workflow.contracts import Trajectory
        from workflow.harness import HARNESS

        t = Trajectory(final_reward=1.0, format_ok=False, final_answer=None)
        hyps = HARNESS.diagnose({"trajectory": t}, names=["reward_hacking"])
        self.assertTrue(hyps)
        self.assertEqual(hyps[0].plugin, "reward_hacking")
        self.assertIn("format", hyps[0].message)

    def test_high_reward_with_error(self):
        from workflow.contracts import EventKind, ExecutionEvent, Trajectory
        from workflow.harness import HARNESS

        t = Trajectory(
            final_reward=1.0,
            format_ok=True,
            final_answer="2",
            events=[ExecutionEvent(kind=EventKind.ERROR, payload={"error": "tool crashed"})],
        )
        hyps = HARNESS.diagnose({"trajectory": t}, names=["reward_hacking"])
        self.assertTrue(hyps)
        self.assertIn("ERROR", hyps[0].message)

    def test_honest_success_empty(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        ok = Collector(mock=True).collect_one(_ok_task("ok"))
        self.assertEqual(HARNESS.diagnose({"trajectory": ok}, names=["reward_hacking"]), [])

    def test_low_reward_not_flagged(self):
        from workflow.contracts import Trajectory
        from workflow.harness import HARNESS

        t = Trajectory(final_reward=0.0, format_ok=False, final_answer=None)
        self.assertEqual(HARNESS.diagnose({"trajectory": t}, names=["reward_hacking"]), [])


if __name__ == "__main__":
    unittest.main()
