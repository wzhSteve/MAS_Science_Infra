"""Monitor reward/loss series stay on the fullest metrics.jsonl."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class _DownClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def request(self, *args, **kwargs):
        import httpx

        raise httpx.ConnectError("store down")


class TestMonitorMetrics(unittest.TestCase):
    def test_live_rollout_does_not_hide_jsonl_steps(self):
        from science_infra.control.services import _agl_reward_series

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "agentflow_demo_grpo" / "metrics.jsonl"
            _write_jsonl(path, [
                {"step": 0, "val/reward": 0.4},
                {"step": 1, "training/global_step": 1, "training/reward": 0.6,
                 "actor/pg_loss": 0.0, "actor/entropy_loss": 0.11},
                {"step": 2, "training/global_step": 2, "training/reward": 0.0,
                 "actor/pg_loss": 0.0, "actor/entropy_loss": 0.14},
                {"step": 3, "training/global_step": 3, "training/reward": 0.5,
                 "actor/pg_loss": 0.0, "actor/entropy_loss": 0.12, "actor/kl_loss": 0.01},
            ])
            pinned = str(path.resolve())
            with patch(
                "science_infra.control.services._metrics_candidate_paths",
                return_value=([pinned], {pinned}),
            ):
                series = _agl_reward_series(
                    {
                        "rollout_rewards": [{"rollout_id": "r1", "final_reward": 1.0}],
                        "training_steps": [{"step": 1, "training/reward": 0.1, "actor/pg_loss": 0.2}],
                    },
                    experiment_name="agentflow_demo_grpo",
                )
        self.assertEqual(len(series["train_rewards"]), 1)
        self.assertEqual([row["reward"] for row in series["step_rewards"]], [0.6, 0.0, 0.5])
        self.assertEqual(series["val_rewards"][0]["reward"], 0.4)
        self.assertEqual(len(series["step_losses"]), 3)
        self.assertEqual(series["step_losses"][0]["pg_loss"], 0.0)
        self.assertAlmostEqual(series["step_losses"][0]["entropy_loss"], 0.11)
        self.assertEqual(series["step_losses"][2]["kl_loss"], 0.01)
        self.assertEqual(series["agl_source"], "merged")
        self.assertEqual(series["metrics_path"], pinned)

    def test_named_experiment_ignores_a_longer_foreign_file(self):
        from science_infra.control.services import _agl_reward_series

        with tempfile.TemporaryDirectory() as tmp:
            foreign = Path(tmp) / "other_exp" / "metrics.jsonl"
            owned = Path(tmp) / "my_exp_grpo" / "metrics.jsonl"
            _write_jsonl(foreign, [{"step": i, "training/reward": 0.1} for i in range(1, 8)])
            _write_jsonl(owned, [
                {"step": 1, "training/reward": 0.2, "actor/pg_loss": 0.3},
                {"step": 2, "training/reward": 0.4, "actor/pg_loss": 0.5},
            ])
            owned_s = str(owned.resolve())
            foreign_s = str(foreign.resolve())
            with patch(
                "science_infra.control.services._metrics_candidate_paths",
                return_value=([foreign_s, owned_s], {owned_s}),
            ):
                series = _agl_reward_series({}, experiment_name="my_exp_grpo")
        self.assertEqual([row["reward"] for row in series["step_rewards"]], [0.2, 0.4])
        self.assertEqual(series["agl_source"], "offline_metrics_jsonl")

    def test_metrics_api_falls_back_when_store_is_down(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hist" / "metrics.jsonl"
            _write_jsonl(path, [
                {"step": 1, "training/global_step": 1, "training/reward": 0.61, "actor/pg_loss": 0.0},
                {"step": 2, "training/global_step": 2, "training/reward": 0.0, "actor/entropy_loss": 0.2},
                {"step": 3, "training/global_step": 3, "training/reward": 0.5, "actor/pg_loss": 0.0},
            ])
            pinned = str(path.resolve())
            with patch("httpx.AsyncClient", _DownClient), patch(
                "science_infra.control.services._metrics_candidate_paths",
                return_value=([pinned], set()),
            ):
                with TestClient(create_app()) as client:
                    metrics = client.get("/v1/agl/metrics")
                    page = client.get("/agl/metrics")
        self.assertEqual(metrics.status_code, 200, metrics.text)
        body = metrics.json()
        self.assertGreaterEqual(len(body["training_steps"]), 3)
        self.assertEqual(body["training_steps"][0]["step"], 1)
        self.assertIn("training/reward", body["training_steps"][0])
        self.assertTrue(body["source"]["exists"])
        self.assertEqual(page.status_code, 200, page.text[:200])
        self.assertIn("text/html", page.headers.get("content-type", ""))
        self.assertIn('id="root"', page.text)


if __name__ == "__main__":
    unittest.main()
