"""P3 real-time harness tests: JSONL frames → SSE parse → Diagnoser.consume."""

from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestStdoutJsonlFrames(unittest.TestCase):
    def test_emit_frame_round_trip(self):
        from rl.hooks.daemon import emit_rollout_tree_event

        buf = io.StringIO()
        with redirect_stdout(buf):
            emit_rollout_tree_event({"event": "node_added", "tree_id": "q1", "payload": {"n_new": 2}})
        line = buf.getvalue().strip()
        self.assertTrue(line.startswith('{"__rollout_tree_event__"'))
        frame = json.loads(line)
        payload = frame["__rollout_tree_event__"]
        self.assertEqual(payload["event"], "node_added")
        self.assertEqual(payload["tree_id"], "q1")

    def test_control_sse_line_filter(self):
        """The exact line filter used by /api/events drain."""
        lines = [
            json.dumps({"__rollout_tree_event__": {"event": "loss", "tree_id": "global"}}),
            "INFO some other log line",
            json.dumps({"unrelated": True}),
        ]
        kept = []
        for line in lines:
            line = line.strip()
            if not line.startswith('{"__rollout_tree_event__"'):
                continue
            kept.append(json.loads(line)["__rollout_tree_event__"])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["event"], "loss")


class TestDiagnoserConsume(unittest.TestCase):
    def test_log_error_consume(self):
        from workflow.harness import LogErrorDiagnoser

        d = LogErrorDiagnoser()
        self.assertIsNone(d.consume({"kind": "tool_result"}))
        h = d.consume({"kind": "error", "payload": {"error": "boom"}})
        self.assertIsNotNone(h)
        self.assertIn("boom", h.message)

    def test_loss_volatility_consume(self):
        from workflow.harness import LossVolatilityDiagnoser

        d = LossVolatilityDiagnoser()
        # escalating jitter → volatility hypothesis
        vals = [0.1, 0.9, 0.1, 0.95]
        hyp = None
        for i, v in enumerate(vals):
            hyp = d.consume({"event": "loss", "payload": {"metrics": {"r": v}}})
        self.assertIsNotNone(hyp)
        self.assertIn("volatility", hyp.message)

    def test_reward_hacking_monitor_sibling_zscore(self):
        from workflow.harness import RewardHackingMonitor

        mon = RewardHackingMonitor(z_threshold=2.0)
        # 3 siblings: two normal, one inflated → z ≈ 1.73 with n=3; use tighter threshold
        tree = {
            "nodes": [
                {"node_id": "root", "parent_id": None, "role": "root", "reward": None},
                {"node_id": "a", "parent_id": "root", "role": "child", "reward": 0.2},
                {"node_id": "b", "parent_id": "root", "role": "child", "reward": 0.2},
                {"node_id": "c", "parent_id": "root", "role": "child", "reward": 1.0},
            ]
        }
        hits = mon.check_tree(tree)
        self.assertTrue(hits)
        self.assertEqual(hits[0].meta["node_id"], "c")
        self.assertGreater(abs(hits[0].meta["z"]), 1.0)

        # uniform siblings → no hypothesis
        tree_ok = {
            "nodes": [
                {"node_id": "root", "parent_id": None, "role": "root", "reward": None},
                {"node_id": "a", "parent_id": "root", "role": "child", "reward": 0.5},
                {"node_id": "b", "parent_id": "root", "role": "child", "reward": 0.5},
                {"node_id": "c", "parent_id": "root", "role": "child", "reward": 0.5},
            ]
        }
        self.assertEqual(mon.check_tree(tree_ok), [])

        # consume() with tree in event payload
        hyp = mon.consume({"event": "outcome", "payload": {"tree": tree}})
        self.assertIsNotNone(hyp)

    def test_monitor_registered_in_default_harness(self):
        from workflow.harness import HARNESS

        self.assertIn("reward_hacking_monitor", HARNESS._plugins)


class TestEndToEndFrameToHypothesis(unittest.TestCase):
    def test_synthetic_stdout_stream_end_to_end(self):
        """stdout JSONL → line filter → consume → hypothesis (P3 acceptance)."""
        from rl.hooks.daemon import emit_rollout_tree_event
        from workflow.harness import LossVolatilityDiagnoser, RewardHackingMonitor

        # 1) emit frames to a captured stdout (as the train daemon would)
        buf = io.StringIO()
        with redirect_stdout(buf):
            emit_rollout_tree_event({"event": "loss", "tree_id": "global", "payload": {"metrics": {"r": 0.1}}})
            emit_rollout_tree_event({"event": "loss", "tree_id": "global", "payload": {"metrics": {"r": 0.95}}})
            emit_rollout_tree_event({"event": "loss", "tree_id": "global", "payload": {"metrics": {"r": 0.1}}})
            emit_rollout_tree_event(
                {
                    "event": "outcome",
                    "tree_id": "q9",
                    "payload": {
                        "tree": {
                            "nodes": [
                                {"node_id": "root", "parent_id": None, "role": "root", "reward": None},
                                {"node_id": "k0", "parent_id": "root", "role": "child", "reward": 0.1},
                                {"node_id": "k1", "parent_id": "root", "role": "child", "reward": 0.12},
                                {"node_id": "k2", "parent_id": "root", "role": "child", "reward": 0.98},
                            ]
                        }
                    },
                }
            )

        # 2) parse with the control-side filter
        events = []
        for line in buf.getvalue().splitlines():
            line = line.strip()
            if not line.startswith('{"__rollout_tree_event__"'):
                continue
            events.append(json.loads(line)["__rollout_tree_event__"])
        self.assertEqual(len(events), 4)

        # 3) diagnosers consume the stream
        vol = LossVolatilityDiagnoser()
        mon = RewardHackingMonitor()
        hyps = []
        for ev in events:
            for d in (vol, mon):
                h = d.consume(ev)
                if h is not None:
                    hyps.append(h)
        self.assertTrue(any("volatility" in h.message for h in hyps))
        self.assertTrue(any("reward hacking suspect" in h.message for h in hyps))


if __name__ == "__main__":
    unittest.main()
