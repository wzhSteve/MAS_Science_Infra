"""Stage 9: Science MAS HTML dashboard (no GPU, no AGL import)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TIR = Path(__file__).resolve().parents[1]
INFRA = TIR.parent
if str(TIR) not in sys.path:
    sys.path.insert(0, str(TIR))
if str(INFRA) not in sys.path:
    sys.path.insert(0, str(INFRA))


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(INFRA), str(TIR), env.get("PYTHONPATH", "")])
    return env


class TestDashboardModel(unittest.TestCase):
    def test_html_has_chart_harness_and_train_signal(self):
        from science_infra.ui.dashboard import build_dashboard_model
        from science_infra.ui.status import render_status_html
        from workflow import Collector
        from workflow.harness import HARNESS
        from workflow.train_signal import batch_to_train_signal

        batch = Collector(mock=True).collect(
            [{"id": "s", "question": "1+1", "answer": "2", "source": "gsm8k", "_mock_answer": "2"}]
        )
        hyps = HARNESS.diagnose({"batch": batch})
        signal = batch_to_train_signal(batch, algo="grpo")
        ts = {
            "advantage": signal.advantage.model_dump(mode="json"),
            "loss": signal.loss.model_dump(mode="json"),
            "meta": signal.meta,
        }
        model = build_dashboard_model(batch, hyps, train_signal=ts, dashboard="file:///tmp/tb")
        self.assertGreaterEqual(model["n"], 1)
        self.assertIn("rewards", model)
        html = render_status_html(batch, hyps, dashboard="file:///tmp/tb", train_signal=ts)
        self.assertIn("science-infra status", html)
        self.assertIn('id="reward-chart"', html)
        self.assertIn('id="harness"', html)
        self.assertIn('id="train-signal"', html)
        self.assertIn("grpo", html)
        self.assertIn("file:///tmp/tb", html)
        self.assertIn("mean_reward", html)


class TestDashboardCli(unittest.TestCase):
    def test_dashboard_subcommand_writes_html(self):
        with tempfile.TemporaryDirectory() as td:
            traj = Path(td) / "traj.json"
            html = Path(td) / "dash.html"
            collect = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "science_infra.ui.cli",
                    "collect",
                    "--mock",
                    "--n",
                    "1",
                    "--out",
                    str(traj),
                ],
                cwd=str(INFRA),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(collect.returncode, 0, collect.stderr + collect.stdout)
            dash = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "science_infra.ui.cli",
                    "dashboard",
                    str(traj),
                    "--html",
                    str(html),
                    "--tensorboard",
                    "http://127.0.0.1:6006",
                ],
                cwd=str(INFRA),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(dash.returncode, 0, dash.stderr + dash.stdout)
            text = html.read_text(encoding="utf-8")
            self.assertIn('id="reward-chart"', text)
            self.assertIn("http://127.0.0.1:6006", text)
            payload = json.loads(traj.read_text(encoding="utf-8"))
            self.assertIn("train_signal", payload)


if __name__ == "__main__":
    unittest.main()
