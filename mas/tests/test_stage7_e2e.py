"""End-to-end cases: collect → diagnose → status → TrainSignal overlay.

Cases (no GPU / no agentlightning):

C1  成功采集：tool/final 事件、reward、memory、skills、JSON 往返
C2  错答案：format 好但 Acc=0 → reward 0
C3  重复同类 ERROR：diagnose 出 cognitive high deviation
C4  成功轨迹：不误报 reward_hacking / cognitive_convergence
C5  YAML 去掉 wikipedia：meta.tools 与事件不含该工具
C6  collect JSON 的 TrainSignal 能 overlay 出相同 tir_algo
C7  成功轨迹可 fork；status HTML 含 harness 结论
C8  CLI：--tasks / --spec / --algo 与 diagnose/status 闭环
"""

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
FIXTURES = Path(__file__).resolve().parent / "fixtures"

if str(TIR) not in sys.path:
    sys.path.insert(0, str(TIR))
if str(INFRA) not in sys.path:
    sys.path.insert(0, str(INFRA))


def _cli_env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(INFRA), str(TIR), env.get("PYTHONPATH", "")])
    return env


def _cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "science_infra.ui.cli", *args],
        cwd=str(INFRA),
        env=_cli_env(),
        capture_output=True,
        text=True,
        check=False,
    )


def _load_tasks(name: str) -> list:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class TestC1SuccessCollectAndRoundtrip(unittest.TestCase):
    def test_events_memory_skills_and_json_roundtrip(self):
        from workflow import Collector
        from workflow.contracts import EventKind, TrajectoryBatch

        batch = Collector(mock=True).collect(_load_tasks("tasks_ok.json"))
        self.assertEqual(batch.meta.get("n_trajectories"), 2)
        self.assertGreaterEqual(float(batch.meta.get("mean_reward") or 0), 1.0)
        t0 = batch.trajectories[0]
        kinds = {e.kind for e in t0.events}
        self.assertTrue({EventKind.TOOL_CALL, EventKind.TOOL_RESULT, EventKind.FINAL_ANSWER} <= kinds)
        self.assertIn(EventKind.MEMORY_READ, kinds)
        self.assertIn(EventKind.MEMORY_WRITE, kinds)
        self.assertIn("react_loop", t0.meta.get("skills") or [])
        self.assertIn("execute_python", t0.meta.get("tools") or [])
        dumped = json.loads(json.dumps({"batch": batch.model_dump(mode="json")}))
        restored = TrajectoryBatch.model_validate(dumped["batch"])
        self.assertEqual(len(restored.trajectories), 2)
        self.assertEqual(restored.trajectories[0].final_answer, t0.final_answer)
        self.assertNotIn("agentlightning", sys.modules)


class TestC2WrongAnswerReward(unittest.TestCase):
    def test_format_ok_wrong_acc_zero(self):
        from workflow import Collector

        traj = Collector(mock=True).collect_one(_load_tasks("tasks_wrong_answer.json")[0])
        self.assertTrue(traj.format_ok)
        self.assertEqual(traj.final_answer, "9")
        self.assertIsNotNone(traj.final_reward)
        self.assertEqual(float(traj.final_reward), 0.0)


class TestC3C4Diagnose(unittest.TestCase):
    def test_repeat_error_triggers_cognitive(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        batch = Collector(mock=True).collect(_load_tasks("tasks_repeat_error.json"))
        hyps = HARNESS.diagnose({"batch": batch}, names=["cognitive_convergence", "reward_hacking"])
        plugins = {h.plugin for h in hyps}
        self.assertIn("cognitive_convergence", plugins)
        self.assertNotIn("reward_hacking", plugins)
        cog = [h for h in hyps if h.plugin == "cognitive_convergence"][0]
        self.assertIn("high deviation", cog.message)
        self.assertEqual(cog.meta.get("n_traj"), 2)

    def test_success_does_not_false_positive(self):
        from workflow import Collector
        from workflow.harness import HARNESS

        batch = Collector(mock=True).collect(_load_tasks("tasks_ok.json"))
        hyps = HARNESS.diagnose(
            {"batch": batch}, names=["cognitive_convergence", "reward_hacking", "log_error"]
        )
        self.assertEqual(hyps, [])


class TestC5YamlTools(unittest.TestCase):
    def test_spec_drops_wikipedia(self):
        from workflow import Collector

        batch = Collector(mock=True, spec_path=str(FIXTURES / "hub_no_wiki.yaml")).collect(
            _load_tasks("tasks_ok.json")
        )
        tools = batch.trajectories[0].meta.get("tools") or []
        self.assertEqual(tools, ["web_search", "execute_python"])
        names = [e.payload.get("name") for e in batch.trajectories[0].events]
        self.assertNotIn("wikipedia_search", names)


class TestC6TrainSignalOverlay(unittest.TestCase):
    def test_collect_json_drives_hydra_tir_algo(self):
        from algos.overlay import apply_train_signal
        from workflow import Collector, batch_to_train_signal
        from workflow.train_signal import AdvantageSpec, LossSpec, TrainSignal

        batch = Collector(mock=True).collect(_load_tasks("tasks_ok.json"))
        sig = batch_to_train_signal(batch, algo="igpo")
        cfg = apply_train_signal(
            {
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
            },
            sig,
        )
        self.assertEqual(cfg["algorithm"]["tir_algo"], "igpo")
        self.assertEqual(cfg["algorithm"]["adv_estimator"], "grpo")
        payload = {
            "advantage": sig.advantage.model_dump(mode="json"),
            "loss": sig.loss.model_dump(mode="json"),
        }
        revived = TrainSignal(
            advantage=AdvantageSpec.model_validate(payload["advantage"]),
            loss=LossSpec.model_validate(payload["loss"]),
        )
        self.assertEqual(revived.advantage.name, "igpo")


class TestC7ForkAndStatusHtml(unittest.TestCase):
    def test_fork_and_html_show_hypotheses(self):
        from science_infra.ui.status import render_status_html
        from workflow import Collector
        from workflow.harness import HARNESS
        from workflow.runtime import ExecutionService

        ok = Collector(mock=True).collect_one(_load_tasks("tasks_ok.json")[0])
        self.assertTrue(ok.branch_points)
        child = ExecutionService(mock=True).fork(
            ok.branch_points[0], _load_tasks("tasks_ok.json")[0]
        )
        self.assertTrue(child.messages)

        batch = Collector(mock=True).collect(_load_tasks("tasks_repeat_error.json"))
        hyps = HARNESS.diagnose({"batch": batch})
        html = render_status_html(batch, hyps, dashboard="file:///tmp/tb")
        self.assertIn("cognitive_convergence", html)
        self.assertIn("high deviation", html)
        self.assertIn("file:///tmp/tb", html)


class TestC8CliClosedLoop(unittest.TestCase):
    def test_cli_ok_collect_diagnose_status(self):
        with tempfile.TemporaryDirectory() as td:
            traj = Path(td) / "ok.json"
            hyps_path = Path(td) / "hyps.json"
            html = Path(td) / "status.html"
            r = _cli(
                [
                    "collect",
                    "--mock",
                    "--tasks",
                    str(FIXTURES / "tasks_ok.json"),
                    "--algo",
                    "arpo",
                    "--out",
                    str(traj),
                ]
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertNotIn("WARNING: agentlightning", r.stderr)
            payload = json.loads(traj.read_text(encoding="utf-8"))
            self.assertEqual(payload["train_signal"]["advantage"]["name"], "arpo")
            self.assertEqual(payload["batch"]["meta"]["n_trajectories"], 2)

            d = _cli(["diagnose", str(traj), "--out", str(hyps_path)])
            self.assertEqual(d.returncode, 0, d.stderr + d.stdout)
            hyps = json.loads(hyps_path.read_text(encoding="utf-8"))
            plugins = {h.get("plugin") for h in hyps}
            self.assertNotIn("cognitive_convergence", plugins)
            self.assertNotIn("reward_hacking", plugins)
            self.assertNotIn("log_error", plugins)

            s = _cli(["status", str(traj), "--html", str(html)])
            self.assertEqual(s.returncode, 0, s.stderr + s.stdout)
            self.assertIn("science-infra status", html.read_text(encoding="utf-8"))

    def test_cli_repeat_error_and_spec(self):
        with tempfile.TemporaryDirectory() as td:
            traj = Path(td) / "err.json"
            hyps_path = Path(td) / "hyps.json"
            r = _cli(
                [
                    "collect",
                    "--mock",
                    "--tasks",
                    str(FIXTURES / "tasks_repeat_error.json"),
                    "--spec",
                    str(FIXTURES / "hub_no_wiki.yaml"),
                    "--out",
                    str(traj),
                ]
            )
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            payload = json.loads(traj.read_text(encoding="utf-8"))
            tools = payload["batch"]["trajectories"][0]["meta"]["tools"]
            self.assertEqual(tools, ["web_search", "execute_python"])

            d = _cli(["diagnose", str(traj), "--out", str(hyps_path)])
            self.assertEqual(d.returncode, 0, d.stderr + d.stdout)
            hyps = json.loads(hyps_path.read_text(encoding="utf-8"))
            plugins = {h.get("plugin") for h in hyps}
            self.assertIn("cognitive_convergence", plugins)


if __name__ == "__main__":
    unittest.main()
