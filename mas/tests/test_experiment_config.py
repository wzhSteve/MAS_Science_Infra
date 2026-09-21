from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from science_infra.control.experiments import (
    create_experiment,
    load_bundle,
    save_section,
)


class TestExperimentConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"SCIENCE_EXPERIMENTS_DIR": self.tempdir.name})
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.tempdir.cleanup()

    def test_missing_experiment_is_not_created_by_read(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "experiment not found"):
            load_bundle("missing")
        self.assertFalse((Path(self.tempdir.name) / "missing").exists())

    def test_duplicate_create_is_rejected(self) -> None:
        create_experiment("existing", name="Existing")
        with self.assertRaisesRegex(ValueError, "already exists"):
            create_experiment("existing", name="Replacement")
        self.assertEqual(load_bundle("existing")["meta"]["name"], "Existing")

    def test_meta_update_preserves_extension_fields_and_document_sections(self) -> None:
        root = create_experiment("extended", name="Original")
        path = root / "experiment.yaml"
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        document["experiment"]["owner"] = {"team": "research"}
        document["workspace"] = {"retention": "keep"}
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

        bundle = load_bundle("extended")
        self.assertEqual(bundle["meta"]["owner"], {"team": "research"})
        save_section("extended", "meta", {"name": "Renamed", "seed": 7})

        saved = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["experiment"]["name"], "Renamed")
        self.assertEqual(saved["experiment"]["seed"], 7)
        self.assertEqual(saved["experiment"]["owner"], {"team": "research"})
        self.assertEqual(saved["workspace"], {"retention": "keep"})

    def test_api_create_read_and_update_contract(self) -> None:
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        with TestClient(create_app()) as client:
            created = client.post(
                "/api/experiments",
                json={"id": "api-test", "seed": 11, "name": "API Test"},
            )
            self.assertEqual(created.status_code, 200, created.text)
            self.assertEqual(created.json()["meta"]["seed"], 11)

            duplicate = client.post(
                "/api/experiments",
                json={"id": "api-test", "seed": 12, "name": "Duplicate"},
            )
            self.assertEqual(duplicate.status_code, 400, duplicate.text)

            updated = client.put(
                "/api/experiments/api-test",
                json={"data": {"name": "Updated"}},
            )
            self.assertEqual(updated.status_code, 200, updated.text)
            self.assertEqual(updated.json()["meta"]["name"], "Updated")
            self.assertEqual(updated.json()["meta"]["seed"], 11)

            missing = client.get("/api/experiments/not-found")
            self.assertEqual(missing.status_code, 404, missing.text)
            self.assertFalse((Path(self.tempdir.name) / "not-found").exists())


if __name__ == "__main__":
    unittest.main()
