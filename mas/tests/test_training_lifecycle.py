from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from science_infra.control.experiments import create_experiment
from science_infra.control.process_manager import ProcessManager
from science_infra.control.training import TrainingError, stop_training_run


class TestTrainingLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.env = patch.dict(
            os.environ,
            {"SCIENCE_EXPERIMENTS_DIR": self.tempdir.name},
        )
        self.env.start()
        create_experiment("alpha")
        create_experiment("beta")
        self.manager = ProcessManager()

    def tearDown(self) -> None:
        active = self.manager.active("train")
        if active is not None:
            self.manager.stop_run(
                active.run_id,
                experiment_id=active.experiment_id,
                timeout=2,
            )
        self.env.stop()
        self.tempdir.cleanup()

    def _wait(self, run_id: str, states: set[str]) -> dict:
        deadline = time.time() + 5
        while time.time() < deadline:
            row = self.manager.status(run_id)
            if row and row["state"] in states:
                return row
            time.sleep(0.05)
        self.fail(f"run {run_id} did not reach {states}")

    def test_success_and_disk_recovery(self) -> None:
        process = self.manager.start(
            kind="train",
            experiment_id="alpha",
            argv=[sys.executable, "-c", "print('done')"],
            replace=False,
            meta={"request_id": "success"},
        )
        row = self._wait(process.run_id, {"succeeded"})
        self.assertEqual(row["returncode"], 0)
        self.assertFalse(row["running"])

        recovered = ProcessManager().disk_run(process.run_id)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["state"], "succeeded")
        self.assertEqual(recovered["meta"]["request_id"], "success")

    def test_conflict_and_run_specific_stop(self) -> None:
        process = self.manager.start(
            kind="train",
            experiment_id="alpha",
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            replace=False,
            meta={"request_id": "long"},
        )
        self._wait(process.run_id, {"running"})
        with self.assertRaisesRegex(RuntimeError, "already running"):
            self.manager.start(
                kind="train",
                experiment_id="beta",
                argv=[sys.executable, "-c", "print('never')"],
                replace=False,
            )
        with self.assertRaisesRegex(ValueError, "does not belong"):
            self.manager.stop_run(
                process.run_id,
                experiment_id="beta",
                timeout=2,
            )

        stopped = self.manager.stop_run(
            process.run_id,
            experiment_id="alpha",
            timeout=2,
        )
        self.assertIsNotNone(stopped)
        self.assertEqual(stopped["state"], "cancelled")
        self.assertEqual(stopped["stop_reason"], "user_requested")

    def test_preparing_record_is_recovered_as_interrupted(self) -> None:
        self.manager.write_preparing(
            run_id="prepared",
            kind="train",
            experiment_id="alpha",
            meta={"request_id": "prepared-request"},
        )
        recovered = ProcessManager().disk_run("prepared")
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["state"], "interrupted")
        self.assertFalse(recovered["running"])

    def test_process_start_failure_is_durable(self) -> None:
        with self.assertRaises(OSError):
            self.manager.start(
                kind="train",
                experiment_id="alpha",
                run_id="cannot-start",
                argv=[str(Path(self.tempdir.name) / "missing-executable")],
                replace=False,
            )
        row = self.manager.disk_run("cannot-start")
        self.assertIsNotNone(row)
        self.assertEqual(row["state"], "failed")
        self.assertEqual(row["failure_stage"], "process_start")


class TestTrainingStopContract(unittest.TestCase):
    def test_missing_run_has_stable_error(self) -> None:
        with patch(
            "science_infra.control.process_manager.PROCS.status",
            return_value=None,
        ):
            with self.assertRaises(TrainingError) as context:
                stop_training_run("alpha", "missing")
        self.assertEqual(context.exception.code, "run_not_found")

    def test_stop_api_returns_stable_error_code(self) -> None:
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        with TestClient(create_app()) as client:
            response = client.post(
                "/api/rl/runs/missing/stop?experiment_id=alpha"
            )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["code"], "run_not_found")


if __name__ == "__main__":
    unittest.main()
