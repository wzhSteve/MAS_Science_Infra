"""Stage 3: Harness plugins + fork (no GPU / no agentlightning)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestLogError(unittest.TestCase):
    def test_clean_vs_error_traj(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        clean = Collector(mock=True).collect_one(
            {"id": "ok", "question": "1+1", "answer": "2", "source": "gsm8k", "_mock_answer": "2"}
        )
        self.assertEqual(HARNESS.diagnose({"trajectory": clean}, names=["log_error"]), [])
        bad = Collector(mock=True).collect_one(
            {
                "id": "bad",
                "question": "fail",
                "answer": "0",
                "source": "gsm8k",
                "_mock_error": "boom",
            }
        )
        hyps = HARNESS.diagnose({"trajectory": bad}, names=["log_error"])
        self.assertTrue(hyps)
        self.assertEqual(hyps[0].plugin, "log_error")
        self.assertIn("boom", hyps[0].message)
        self.assertTrue(hyps[0].event_id)


class TestLossVolatility(unittest.TestCase):
    def test_missing_metrics_empty_with_warning(self):
        from workflow.harness import HARNESS

        with self.assertLogs("workflow.harness", level="WARNING") as cm:
            hyps = HARNESS.diagnose({"metrics_path": "/tmp/no-such-metrics.jsonl"}, names=["loss_volatility"])
        self.assertEqual(hyps, [])
        self.assertTrue(any("missing" in m.lower() or "no series" in m.lower() for m in cm.output))

    def test_flat_rewards(self):
        from workflow.harness import HARNESS

        hyps = HARNESS.diagnose({"rewards": [0.5, 0.4, 0.3]}, names=["loss_volatility"])
        self.assertTrue(any("not rising" in h.message for h in hyps))

    def test_volatility_and_jsonl(self):
        from workflow.harness import HARNESS

        hyps = HARNESS.diagnose({"rewards": [0.0, 1.0, 0.0, 1.0]}, names=["loss_volatility"])
        self.assertTrue(any("volatility" in h.message for h in hyps))

        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"train/reward": 0.9}) + "\n")
            f.write(json.dumps({"train/reward": 0.2}) + "\n")
            path = f.name
        hyps = HARNESS.diagnose({"metrics_path": path}, names=["loss_volatility"])
        self.assertTrue(any("not rising" in h.message for h in hyps))


class TestStubsAndRegistry(unittest.TestCase):
    def test_stubs_empty(self):
        from workflow.harness import HARNESS

        self.assertEqual(HARNESS.diagnose({"rewards": [0.1, 0.0]}, names=["epc_aw_consensus"]), [])
        self.assertEqual(HARNESS.diagnose({"rewards": [0.1, 0.0]}, names=["cognitive_convergence"]), [])
        self.assertEqual(HARNESS.diagnose({"rewards": [0.1, 0.0]}, names=["reward_hacking"]), [])

    def test_unknown_plugin(self):
        from workflow.harness import HARNESS

        with self.assertRaises(KeyError):
            HARNESS.get("not_a_plugin")


class TestFork(unittest.TestCase):
    def test_fork_restore(self):
        from workflow.runtime import ExecutionService

        svc = ExecutionService(mock=True)
        traj = svc.run({"id": "f", "question": "1+1", "answer": "2", "_mock_answer": "2"})
        self.assertTrue(traj.branch_points)
        bp = traj.branch_points[0]
        self.assertTrue(bp.archive_id)
        self.assertTrue(bp.snapshot_id)
        child = svc.fork(bp, {"id": "f2", "question": "1+1", "answer": "2", "_mock_answer": "2"})
        self.assertTrue(child.messages)
        self.assertNotIn("agentlightning", sys.modules)


if __name__ == "__main__":
    unittest.main()
