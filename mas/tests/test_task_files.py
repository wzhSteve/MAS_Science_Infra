"""HIVE JSON datasets normalize to the same task rows as parquet."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from science_infra.control.dataset_resources import Dataset, _read  # noqa: E402
from science_infra.control.rollout_runs import registered_dataset  # noqa: E402
from science_infra.control.services import sample_parquet_tasks  # noqa: E402
from science_infra.control.task_files import load_task_rows  # noqa: E402

HIVE = REPO / "ref_Rep" / "HIVE" / "test"


class TestHiveTaskRows(unittest.TestCase):
    def test_hotpotqa_list_answer(self):
        row = load_task_rows(HIVE / "hotpotqa" / "data" / "data.json")[0]
        self.assertEqual(row["source"], "hotpotqa")
        self.assertIn("Oliver Reed", row["question"])
        self.assertEqual(row["answer"], "Prussian")
        self.assertEqual(row["answers"], ["Prussian"])

    def test_2wiki_uses_query(self):
        row = load_task_rows(HIVE / "2wiki" / "data" / "data.json")[0]
        self.assertEqual(row["source"], "2wiki")
        self.assertTrue(row["question"].startswith("Which film"))
        self.assertEqual(row["answer"], "Let'S Make Money")

    def test_gaia_string_answer(self):
        row = load_task_rows(HIVE / "gaia" / "data" / "data.json")[0]
        self.assertEqual(row["source"], "gaia")
        self.assertEqual(row["answer"], "egalitarian")
        self.assertTrue(row["question"])

    def test_catalog_lists_hive_json(self):
        names = {item.name for item in _read().items}
        self.assertTrue({"hotpotqa", "bamboogle", "musique", "2wiki", "gaia", "medqa"} <= names)
        Dataset(
            id="hotpotqa",
            name="hotpotqa",
            path=str(HIVE / "hotpotqa" / "data" / "data.json"),
        )

    def test_every_catalog_dataset_is_readable_for_eval_and_train(self):
        items = _read().items
        self.assertGreaterEqual(len(items), 8)
        for item in items:
            with self.subTest(dataset=item.name):
                self.assertTrue(Path(item.path).is_file(), item.path)
                trained = load_task_rows(item.path)
                self.assertTrue(trained)
                self.assertTrue(str(trained[0]["question"]).strip())
                self.assertIn("answer", trained[0])
                self.assertTrue(str(trained[0]["source"]).strip())
                if item.id in {"hotpotqa", "bamboogle", "musique", "2wiki", "gaia", "medqa"}:
                    self.assertEqual(trained[0]["source"], item.id)
                self.assertEqual(registered_dataset(item.path), item.path)
                sampled = sample_parquet_tasks(item.path, n=1, source="all")
                self.assertEqual(sampled[0]["question"], trained[0]["question"])

    def test_unregistered_eval_path_is_rejected(self):
        with self.assertRaises(ValueError):
            registered_dataset("/tmp/not-in-catalog.parquet")


if __name__ == "__main__":
    unittest.main()
