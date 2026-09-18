"""Stage 2: TrainSignal overlay + Archive-only resume (no GPU)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _base_cfg():
    return {
        "algorithm": {"adv_estimator": "grpo"},
        "actor_rollout_ref": {
            "rollout": {"n": 4},
            "actor": {
                "clip_ratio_low": 0.2,
                "clip_ratio_high": 0.3,
                "entropy_coeff": 0,
                "kl_loss_coef": 0.0,
            },
        },
        "trainer": {"experiment_name": "tir_agent"},
    }


class TestTrainSignalOverlay(unittest.TestCase):
    def test_igpo_overlay(self):
        from algos.overlay import apply_train_signal
        from workflow.train_signal import AdvantageSpec, LossSpec, TrainSignal

        sig = TrainSignal(
            advantage=AdvantageSpec(name="igpo", gamma=0.9),
            loss=LossSpec(name="igpo", clip_ratio_low=0.1, entropy_coeff=0.01),
        )
        cfg = apply_train_signal(_base_cfg(), sig)
        self.assertEqual(cfg["algorithm"]["tir_algo"], "igpo")
        self.assertEqual(cfg["algorithm"]["adv_estimator"], "grpo")
        self.assertEqual(cfg["actor_rollout_ref"]["actor"]["clip_ratio_low"], 0.1)
        self.assertEqual(cfg["actor_rollout_ref"]["actor"]["entropy_coeff"], 0.01)
        self.assertEqual(cfg["algorithm"]["tir"]["gamma"], 0.9)

    def test_cli_algo_matches_collect_signal_fields(self):
        """train_tir_agent.py --algo igpo uses the same TrainSignal field set as collect JSON."""
        from algos.overlay import apply_train_signal
        from workflow import Collector, batch_to_train_signal
        from workflow.train_signal import AdvantageSpec, LossSpec, TrainSignal

        cli_sig = TrainSignal(
            advantage=AdvantageSpec(name="igpo", use_critic=False),
            loss=LossSpec(name="igpo"),
            meta={"algo": "igpo", "source": "cli"},
        )
        cli_cfg = apply_train_signal(_base_cfg(), cli_sig)

        batch = Collector(mock=True).collect(
            [{"id": "t", "question": "1+1", "answer": "2", "source": "gsm8k", "_mock_answer": "2"}]
        )
        collect_sig = batch_to_train_signal(batch, algo="igpo")
        collect_cfg = apply_train_signal(_base_cfg(), collect_sig)

        self.assertEqual(cli_cfg["algorithm"]["tir_algo"], "igpo")
        self.assertEqual(collect_cfg["algorithm"]["tir_algo"], cli_cfg["algorithm"]["tir_algo"])
        self.assertEqual(collect_sig.advantage.name, cli_sig.advantage.name)
        self.assertEqual(collect_sig.loss.name, cli_sig.loss.name)

    def test_unknown_algo_raises(self):
        from algos.overlay import apply_algo_overlay

        with self.assertRaises(ValueError):
            apply_algo_overlay(_base_cfg(), "ppo")


class TestArchiveResume(unittest.TestCase):
    def test_archive_without_resume_cache(self):
        from workflow.archive import Archive, dump_resume_with_archive, load_resume_messages

        rid = f"roll_{uuid4().hex[:12]}"
        with tempfile.TemporaryDirectory() as td:
            arch = Archive(root_dir=td)
            dumped = dump_resume_with_archive(
                rid,
                {"messages": [{"role": "user", "content": "hi"}], "h_root": 0.2, "h_tool": 0.9},
                archive=arch,
            )
            self.assertTrue(dumped["archive_id"])
            loaded = load_resume_messages(rid)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["messages"][0]["content"], "hi")
            self.assertEqual(loaded["archive_id"], dumped["archive_id"])
            self.assertEqual(loaded["snapshot_id"], dumped["snapshot_id"])
            cache = Path(ROOT) / ".resume_cache" / f"{rid}.json"
            self.assertFalse(cache.is_file(), "dump must not write legacy .resume_cache")
            snap_path = Path(td) / dumped["archive_id"] / "snapshots" / f"{dumped['snapshot_id']}.json"
            self.assertTrue(snap_path.is_file())

    def test_legacy_resume_cache_read_fallback(self):
        import json

        from workflow.archive import load_resume_messages

        rid = f"legacy_{uuid4().hex[:12]}"
        cache_dir = Path(ROOT) / ".resume_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / f"{rid}.json"
        dest.write_text(json.dumps({"messages": [{"role": "user", "content": "legacy"}]}), encoding="utf-8")
        try:
            loaded = load_resume_messages(rid)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded["messages"][0]["content"], "legacy")
        finally:
            dest.unlink(missing_ok=True)

    def test_trajectory_sync_tir_meta(self):
        from workflow.contracts import Trajectory

        t = Trajectory(n_search=2, n_python=1, format_ok=True)
        t.sync_tir_meta()
        self.assertEqual(t.meta["n_search"], 2)
        self.assertEqual(t.meta["n_python"], 1)
        self.assertTrue(t.meta["format_ok"])


if __name__ == "__main__":
    unittest.main()
