from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from science_infra.control.experiments import create_experiment, load_bundle, save_section
from science_infra.control.model_resources import create_resource, get_bindings, put_binding
from science_infra.control.training import build_training_plan, discover_local_models


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


if __name__ == "__main__":
    unittest.main()
