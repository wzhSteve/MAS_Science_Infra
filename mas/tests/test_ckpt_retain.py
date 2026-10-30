"""Latest + best-val checkpoint retention (no GPU)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERL = ROOT / "rl" / "verl_backend"
for p in (str(VERL), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from verl.utils.checkpoint.retain import (  # noqa: E402
    best_from_metrics_jsonl,
    dirs_to_keep,
    drop_unkept_global_steps,
    resolve_best_step,
)


class TestCkptRetain(unittest.TestCase):
    def test_keep_latest_and_better_best(self):
        steps = [(i, f"/ckpt/global_step_{i}") for i in (18, 19, 20)]
        keep = dirs_to_keep(steps, latest_step=20, best_step=18, max_keep=1)
        self.assertEqual(keep, {"/ckpt/global_step_18", "/ckpt/global_step_20"})

    def test_best_equals_latest_drops_rest(self):
        steps = [(i, f"/ckpt/global_step_{i}") for i in (19, 20)]
        keep = dirs_to_keep(steps, latest_step=20, best_step=20, max_keep=1)
        self.assertEqual(keep, {"/ckpt/global_step_20"})

    def test_max_keep_plus_best_outside_window(self):
        steps = [(i, f"/ckpt/global_step_{i}") for i in range(10, 16)]
        keep = dirs_to_keep(steps, latest_step=15, best_step=10, max_keep=3)
        self.assertEqual(
            keep,
            {
                "/ckpt/global_step_10",
                "/ckpt/global_step_13",
                "/ckpt/global_step_14",
                "/ckpt/global_step_15",
            },
        )

    def test_metrics_seed_and_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (2, 6, 20):
                (root / f"global_step_{step}").mkdir()
            (root / "metrics.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"training/global_step": 2, "val/reward": 0.283}),
                        json.dumps({"training/global_step": 6, "val/reward": 0.278}),
                        json.dumps({"training/global_step": 20, "val/reward": 0.194}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            seeded = best_from_metrics_jsonl(str(root / "metrics.jsonl"), {2, 6, 20})
            self.assertEqual(seeded, (2, 0.283))
            best = resolve_best_step(str(root), 21, 0.20)
            self.assertEqual(best, 2)
            (root / "global_step_21").mkdir()
            keep = dirs_to_keep(
                [(2, str(root / "global_step_2")), (6, str(root / "global_step_6")),
                 (20, str(root / "global_step_20")), (21, str(root / "global_step_21"))],
                latest_step=21,
                best_step=best,
                max_keep=1,
            )
            drop_unkept_global_steps(str(root), keep)
            names = sorted(p.name for p in root.iterdir() if p.name.startswith("global_step_"))
            self.assertEqual(names, ["global_step_2", "global_step_21"])


if __name__ == "__main__":
    unittest.main()
