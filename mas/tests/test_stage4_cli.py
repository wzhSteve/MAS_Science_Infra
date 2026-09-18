"""Stage 4: science-infra CLI + static HTML (no GPU)."""

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


class TestStatusHtml(unittest.TestCase):
    def test_render_contains_tables(self):
        from science_infra.ui.status import render_status_html
        from workflow import Collector
        from workflow.harness import HARNESS

        batch = Collector(mock=True).collect(
            [{"id": "s", "question": "1+1", "answer": "2", "source": "gsm8k", "_mock_answer": "2"}]
        )
        hyps = HARNESS.diagnose({"batch": batch})
        html = render_status_html(batch, hyps, dashboard="file:///tmp/tb")
        self.assertIn("science-infra status", html)
        self.assertIn("mean_reward", html)
        self.assertIn("file:///tmp/tb", html)
        self.assertIn("<table>", html)


class TestCliSubprocess(unittest.TestCase):
    """Acceptance: collect/diagnose/status/doctor; mock must not import agentlightning."""

    def test_collect_diagnose_status_doctor(self):
        with tempfile.TemporaryDirectory() as td:
            traj = Path(td) / "traj.json"
            html = Path(td) / "status.html"
            diag = Path(td) / "hyps.json"
            collect = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "science_infra.ui.cli",
                    "collect",
                    "--mock",
                    "--n",
                    "2",
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
            self.assertNotIn("WARNING: agentlightning", collect.stderr)
            payload = json.loads(traj.read_text(encoding="utf-8"))
            self.assertEqual(payload["batch"]["meta"]["n_trajectories"], 4)
            self.assertIn("train_signal", payload)

            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import json,sys; json.load(open(sys.argv[1])); import sys as s; "
                    "s.exit(1 if 'agentlightning' in s.modules else 0)",
                    str(traj),
                ],
                cwd=str(INFRA),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(probe.returncode, 0)

            isolate = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; sys.path[:0]=%r; from science_infra.ui.cli import main; "
                        "rc=main(['collect','--mock','--n','1','--out',%r]); "
                        "sys.exit(2 if 'agentlightning' in sys.modules else rc)"
                    )
                    % ([str(INFRA), str(TIR)], str(Path(td) / "traj2.json")),
                ],
                cwd=str(INFRA),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(isolate.returncode, 0, isolate.stderr + isolate.stdout)

            d = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "science_infra.ui.cli",
                    "diagnose",
                    str(traj),
                    "--out",
                    str(diag),
                ],
                cwd=str(INFRA),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(d.returncode, 0, d.stderr + d.stdout)
            self.assertTrue(diag.is_file())

            s = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "science_infra.ui.cli",
                    "status",
                    str(traj),
                    "--html",
                    str(html),
                    "--tensorboard",
                    "file:///tmp/tb",
                ],
                cwd=str(INFRA),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(s.returncode, 0, s.stderr + s.stdout)
            text = html.read_text(encoding="utf-8")
            self.assertIn("<html", text.lower())
            self.assertIn("file:///tmp/tb", text)

            doc = subprocess.run(
                [sys.executable, "-m", "science_infra.ui.cli", "doctor"],
                cwd=str(INFRA),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(doc.returncode, 0, doc.stderr + doc.stdout)
            self.assertIn("python=", doc.stdout)
            self.assertIn("hub_react", doc.stdout)

    def test_collect_rollouts_script(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "traj.json"
            r = subprocess.run(
                [
                    sys.executable,
                    str(TIR / "scripts" / "collect_rollouts.py"),
                    "--mock",
                    "--n",
                    "2",
                    "--out",
                    str(out),
                ],
                cwd=str(TIR),
                env=_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertNotIn("WARNING: agentlightning", r.stderr)
            data = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(data["batch"]["meta"]["n_trajectories"], 4)

    def test_check_workflow_deps_script(self):
        r = subprocess.run(
            [sys.executable, str(TIR / "scripts" / "check_workflow_deps.py")],
            cwd=str(TIR),
            env=_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("OK check_workflow_deps", r.stdout)


if __name__ == "__main__":
    unittest.main()
