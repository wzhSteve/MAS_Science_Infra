from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from science_infra.control.experiments import create_experiment, load_bundle, save_section
from science_infra.control.model_resources import create_resource, get_bindings, put_binding
from science_infra.control.training import build_training_plan, discover_local_models
from science_infra.control.process_manager import ProcessManager
from science_infra.control.training import launch_training
from science_infra.control.training import TrainingError


class TestTrainingPreflight(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.experiments = root / "experiments"
        self.models = root / "models"
        self.database = root / "resources.sqlite3"
        self.models.mkdir()
        self.model = self.models / "Qwen3-4B"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["Qwen3ForCausalLM"],
                    "model_type": "qwen3",
                    "torch_dtype": "bfloat16",
                }
            ),
            encoding="utf-8",
        )
        self.train = root / "train.parquet"
        self.val = root / "val.parquet"
        self.train.write_bytes(b"test")
        self.val.write_bytes(b"test")
        self.env = patch.dict(
            os.environ,
            {
                "SCIENCE_EXPERIMENTS_DIR": str(self.experiments),
                "SCIENCE_MODEL_RESOURCES_DB": str(self.database),
                "SCIENCE_MODEL_ROOTS": str(self.models),
            },
        )
        self.env.start()
        create_experiment("training")
        bundle = load_bundle("training")
        rl = {
            **bundle["rl"],
            "model_path": str(self.model),
            "data": {
                **bundle["rl"]["data"],
                "train_files": str(self.train),
                "val_files": str(self.val),
            },
        }
        save_section("training", "rl", rl)

    def tearDown(self) -> None:
        self.env.stop()
        self.tempdir.cleanup()

    def _plan(self):
        with (
            patch(
                "science_infra.control.services.list_gpus",
                return_value={
                    "count": 1,
                    "gpus": [
                        {
                            "id": 0,
                            "name": "A800",
                            "mem_total_mb": 81920,
                            "mem_used_mb": 0,
                            "util": 0,
                        }
                    ],
                },
            ),
            patch(
                "science_infra.control.training._dependency_status",
                return_value=(["all"], []),
            ),
        ):
            return build_training_plan("training")

    def test_legacy_model_path_is_ready(self) -> None:
        plan = self._plan()
        self.assertTrue(plan.ready, plan.blocking_issues)
        self.assertEqual(plan.source.source, "legacy")
        self.assertEqual(plan.source.model_path, str(self.model))

    def test_bound_training_resource_overrides_legacy_path(self) -> None:
        replacement = self.models / "Qwen3-8B"
        replacement.mkdir()
        (replacement / "config.json").write_text("{}", encoding="utf-8")
        resource = create_resource(
            {
                "name": "Qwen3-8B",
                "type": "training",
                "config": {"model_path": str(replacement)},
                "credential_mode": "none",
            }
        )
        bindings = get_bindings("training")
        put_binding(
            "training",
            revision=bindings["revision"],
            purpose="training",
            resource_id=resource["id"],
        )

        plan = self._plan()
        self.assertTrue(plan.ready, plan.blocking_issues)
        self.assertEqual(plan.source.source, "resource")
        self.assertEqual(plan.source.model_path, str(replacement))

    def test_local_model_discovery_reports_registered_resource(self) -> None:
        resource = create_resource(
            {
                "name": "Qwen3-4B",
                "type": "training",
                "config": {"model_path": str(self.model)},
                "credential_mode": "none",
            }
        )
        candidates = discover_local_models()
        candidate = next(
            item for item in candidates if Path(item["path"]) == self.model.resolve()
        )
        self.assertEqual(candidate["registered_resource_id"], resource["id"])
        self.assertEqual(candidate["model_type"], "qwen3")

    def test_launch_writes_snapshots_and_reuses_request(self) -> None:
        manager = ProcessManager()
        with (
            patch(
                "science_infra.control.process_manager.PROCS",
                manager,
            ),
            patch(
                "science_infra.control.services.list_gpus",
                return_value={
                    "count": 1,
                    "gpus": [
                        {
                            "id": 0,
                            "name": "A800",
                            "mem_total_mb": 81920,
                            "mem_used_mb": 0,
                            "util": 0,
                        }
                    ],
                },
            ),
            patch(
                "science_infra.control.training._dependency_status",
                return_value=(["all"], []),
            ),
            patch.object(
                manager,
                "start",
                side_effect=lambda **kwargs: SimpleNamespace(
                    run_id=kwargs["run_id"]
                ),
            ) as start,
        ):
            plan = build_training_plan("training")
            result = launch_training(
                "training",
                request_id="request-1",
                preflight_revision=plan.revision,
                stop_local_llm=True,
            )
            run_id = result["run_id"]
            run_dir = manager.run_dir("training", run_id)
            self.assertTrue((run_dir / "effective-rl.yaml").is_file())
            self.assertTrue((run_dir / "effective-workflow.yaml").is_file())
            launch = json.loads((run_dir / "launch.json").read_text(encoding="utf-8"))
            self.assertIn("--workflow-yaml", launch["argv"])
            self.assertEqual(launch["preflight_revision"], plan.revision)

            repeated = launch_training(
                "training",
                request_id="request-1",
                preflight_revision=plan.revision,
                stop_local_llm=True,
            )
            self.assertTrue(repeated["reused"])
            self.assertEqual(repeated["run_id"], run_id)
            start.assert_called_once()

    def test_stale_preflight_does_not_create_run(self) -> None:
        manager = ProcessManager()
        with (
            patch(
                "science_infra.control.process_manager.PROCS",
                manager,
            ),
            patch(
                "science_infra.control.services.list_gpus",
                return_value={
                    "count": 1,
                    "gpus": [
                        {
                            "id": 0,
                            "name": "A800",
                            "mem_total_mb": 81920,
                            "mem_used_mb": 0,
                            "util": 0,
                        }
                    ],
                },
            ),
            patch(
                "science_infra.control.training._dependency_status",
                return_value=(["all"], []),
            ),
            patch.object(manager, "start") as start,
        ):
            with self.assertRaises(TrainingError) as context:
                launch_training(
                    "training",
                    request_id="stale",
                    preflight_revision="old",
                    stop_local_llm=True,
                )
        self.assertEqual(context.exception.code, "preflight_stale")
        start.assert_not_called()

    def test_launch_preserves_process_start_failure_stage(self) -> None:
        manager = ProcessManager()
        def fail_start(**kwargs):
            manager.mark_failed(
                run_id=kwargs["run_id"],
                experiment_id=kwargs["experiment_id"],
                kind=kwargs["kind"],
                stage="process_start",
                message="cannot create process",
                meta=kwargs["meta"],
            )
            raise OSError("cannot create process")

        with (
            patch(
                "science_infra.control.process_manager.PROCS",
                manager,
            ),
            patch(
                "science_infra.control.services.list_gpus",
                return_value={
                    "count": 1,
                    "gpus": [
                        {
                            "id": 0,
                            "name": "A800",
                            "mem_total_mb": 81920,
                            "mem_used_mb": 0,
                            "util": 0,
                        }
                    ],
                },
            ),
            patch(
                "science_infra.control.training._dependency_status",
                return_value=(["all"], []),
            ),
            patch.object(
                manager,
                "start",
                side_effect=fail_start,
            ),
        ):
            plan = build_training_plan("training")
            with self.assertRaises(TrainingError) as context:
                launch_training(
                    "training",
                    request_id="process-error",
                    preflight_revision=plan.revision,
                    stop_local_llm=True,
                )
        row = manager.status(context.exception.data["run_id"])
        self.assertIsNotNone(row)
        self.assertEqual(row["failure_stage"], "process_start")
        self.assertEqual(context.exception.code, "process_start_failed")


if __name__ == "__main__":
    unittest.main()
