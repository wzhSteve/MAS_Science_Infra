"""VERL tqdm tails become a remaining-time estimate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from science_infra.control.train_progress import parse_train_progress


class TestTrainProgress(unittest.TestCase):
    def test_latest_numeric_update(self):
        text = (
            "\x1b[36m(TaskRunner pid=1)\x1b[0m \r"
            "Training Progress:   0%|          | 0/5 [00:00<?, ?it/s]\n"
            "Training Progress:  40%|████      | 2/5 [31:39<45:44, 914.77s/it]"
        )
        got = parse_train_progress(text)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got["step"], 2)
        self.assertEqual(got["total"], 5)
        self.assertEqual(got["elapsed_seconds"], 31 * 60 + 39)
        self.assertEqual(got["eta_seconds"], 45 * 60 + 44)
        self.assertAlmostEqual(got["seconds_per_step"], 914.77)

    def test_unknown_rate_has_no_estimate(self):
        got = parse_train_progress("Training Progress:   0%|          | 0/5 [00:00<?, ?it/s]")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got["step"], 1)
        self.assertEqual(got["total"], 5)
        self.assertIsNone(got["eta_seconds"])
        self.assertIsNone(got["seconds_per_step"])

    def test_checkpoint_shards_are_not_training_progress(self):
        text = (
            "Training Progress:  40%|████      | 2/5 [31:39<45:44, 914.77s/it]\n"
            "Loading checkpoint shards: 100%|██████████| 3/3 [00:01<00:00,  1.63it/s]"
        )
        got = parse_train_progress(text)
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual((got["step"], got["total"]), (2, 5))
        self.assertEqual(got["eta_seconds"], 45 * 60 + 44)
        self.assertIsNone(parse_train_progress(
            "Loading checkpoint shards: 100%|██████████| 3/3 [00:01<00:00,  1.63it/s]"
        ))

    def test_resume_inflight_bar_is_the_current_step(self):
        got = parse_train_progress("Training Progress:  50%|█████     | 1/2 [00:00<?, ?it/s]")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual((got["step"], got["total"]), (2, 2))
        self.assertIsNone(got["eta_seconds"])

    def test_iterations_per_second(self):
        got = parse_train_progress("Training Progress: 100%|##| 1/5 [00:02<00:08, 2.0it/s]")
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got["eta_seconds"], 8)
        self.assertAlmostEqual(got["seconds_per_step"], 0.5)


if __name__ == "__main__":
    unittest.main()
