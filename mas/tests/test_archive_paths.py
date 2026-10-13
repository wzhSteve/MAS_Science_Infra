"""Phase 5 artifacts migration tests (mas/tests/test_archive_paths.py).

Covers:
- default write roots resolve under artifacts/mas/ (tir_archives,
  local_expansion) and never to the legacy in-tree locations
- load_resume_messages three-tier fallback: in-memory index → new
  artifacts dir → legacy mas/.tir_archives (read-only)
- local_expansion compat: new default + legacy directory JSON still readable
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import unittest.mock  # noqa: F401  (used below)
from pathlib import Path

MAS_TESTS_DIR = Path(__file__).resolve().parent
MAS_ROOT = MAS_TESTS_DIR.parent
REPO_ROOT = MAS_ROOT.parent
for p in (str(MAS_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestArchiveRootNewDefault(unittest.TestCase):
    def test_archive_root_default_is_artifacts_dir(self) -> None:
        from workflow.memory.archive import archive_root

        old = os.environ.get("TIR_ARCHIVE_DIR", "")
        try:
            os.environ.pop("TIR_ARCHIVE_DIR", None)
            root = Path(archive_root()).resolve()
            expected = (REPO_ROOT / "artifacts" / "mas" / "tir_archives").resolve()
            self.assertEqual(root, expected)
        finally:
            if old:
                os.environ["TIR_ARCHIVE_DIR"] = old

    def test_control_paths_helper_matches_archive_helper(self) -> None:
        """science_infra/control/paths.mas_artifacts_root() and the workflow
        archive helper must agree on the same artifacts root."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "control_paths_module",
            REPO_ROOT / "science_infra" / "control" / "paths.py",
        )
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # standalone module, no package deps
        from workflow.memory.archive import mas_artifacts_root

        self.assertEqual(mod.mas_artifacts_root(), mas_artifacts_root())


class TestResumeFallback(unittest.TestCase):
    """Three-tier fallback: in-memory _ROLLOUT_INDEX → new artifacts dir →
    legacy mas/.tir_archives (read-only, pre-existing data must survive)."""

    def test_disk_fallback_finds_snapshot_in_legacy_dir(self) -> None:
        """Legacy dir has a rollout snapshot, new dir empty/no index →
        disk-scan fallback resolves it via the legacy root (tier 2 → 3)."""
        from workflow.memory import archive as arch_mod
        from workflow.memory.archive import (
            Archive,
            dump_resume_with_archive,
            load_resume_messages,
        )

        with tempfile.TemporaryDirectory() as new_root, tempfile.TemporaryDirectory() as legacy_root:
            old_env = os.environ.get("TIR_ARCHIVE_DIR", "")
            try:
                # Simulate: at Phase-4 time the snapshot was written to the
                # (then) default root == what is now legacy root.
                os.environ["TIR_ARCHIVE_DIR"] = legacy_root
                arch_mod.DEFAULT_ARCHIVE_ROOT = legacy_root
                arch_mod._GLOBAL.clear()
                arch_mod._ROLLOUT_INDEX.clear()
                arch = Archive(root_dir=legacy_root)
                arch_mod.register_archive(arch)
                rid = "legacy-fallback-rollout"
                ref = dump_resume_with_archive(
                    rid,
                    {"messages": [{"role": "user", "content": "legacy-q"}],
                     "h_root": 0.0, "h_tool": 0.0, "consecutive_high": 0},
                    archive=arch,
                )
                # Simulate a fresh runner process: no in-memory index at all.
                arch_mod._GLOBAL.clear()
                arch_mod._ROLLOUT_INDEX.clear()

                # Patch roots: new artifacts dir exists (empty), legacy dir has data.
                with unittest.mock.patch.object(arch_mod, "archive_root", return_value=new_root), \
                     unittest.mock.patch.object(arch_mod, "legacy_archive_root", return_value=legacy_root):
                    # drop the default archive so the fresh-process scan runs
                    self.assertNotIn(rid, arch_mod._ROLLOUT_INDEX)
                    got = load_resume_messages(rid)
                self.assertIsNotNone(got)
                assert got is not None
                self.assertEqual(got["messages"][0]["content"], "legacy-q")
                self.assertEqual(got["archive_id"], ref["archive_id"])
                self.assertEqual(got["snapshot_id"], ref["snapshot_id"])
            finally:
                if old_env:
                    os.environ["TIR_ARCHIVE_DIR"] = old_env
                else:
                    os.environ.pop("TIR_ARCHIVE_DIR", None)
                arch_mod.DEFAULT_ARCHIVE_ROOT = arch_mod.archive_root()
                arch_mod._GLOBAL.clear()
                arch_mod._ROLLOUT_INDEX.clear()

    def test_new_dir_takes_priority_over_legacy(self) -> None:
        """Same rollout_id in both dirs → the new artifacts dir wins."""
        from workflow.memory import archive as arch_mod
        from workflow.memory.archive import (
            Archive,
            dump_resume_with_archive,
            load_resume_messages,
        )

        with tempfile.TemporaryDirectory() as new_root, tempfile.TemporaryDirectory() as legacy_root:
            old_env = os.environ.get("TIR_ARCHIVE_DIR", "")
            try:
                rid = "priority-rollout"
                # write legacy snapshot
                os.environ["TIR_ARCHIVE_DIR"] = legacy_root
                arch_mod.DEFAULT_ARCHIVE_ROOT = legacy_root
                arch_mod._GLOBAL.clear()
                arch_mod._ROLLOUT_INDEX.clear()
                legacy_arch = Archive(root_dir=legacy_root)
                arch_mod.register_archive(legacy_arch)
                dump_resume_with_archive(
                    rid,
                    {"messages": [{"role": "user", "content": "from-legacy"}],
                     "h_root": 0.0, "h_tool": 0.0, "consecutive_high": 0},
                    archive=legacy_arch,
                )
                # write new snapshot (simulate post-migration write)
                os.environ["TIR_ARCHIVE_DIR"] = new_root
                arch_mod.DEFAULT_ARCHIVE_ROOT = new_root
                arch_mod._GLOBAL.clear()
                arch_mod._ROLLOUT_INDEX.clear()
                new_arch = Archive(root_dir=new_root)
                arch_mod.register_archive(new_arch)
                dump_resume_with_archive(
                    rid,
                    {"messages": [{"role": "user", "content": "from-new"}],
                     "h_root": 0.0, "h_tool": 0.0, "consecutive_high": 0},
                    archive=new_arch,
                )
                # fresh runner: memory index empty
                arch_mod._GLOBAL.clear()
                arch_mod._ROLLOUT_INDEX.clear()
                with unittest.mock.patch.object(arch_mod, "archive_root", return_value=new_root), \
                     unittest.mock.patch.object(arch_mod, "legacy_archive_root", return_value=legacy_root):
                    got = load_resume_messages(rid)
                self.assertIsNotNone(got)
                assert got is not None
                self.assertEqual(got["messages"][0]["content"], "from-new")
            finally:
                if old_env:
                    os.environ["TIR_ARCHIVE_DIR"] = old_env
                else:
                    os.environ.pop("TIR_ARCHIVE_DIR", None)
                arch_mod.DEFAULT_ARCHIVE_ROOT = arch_mod.archive_root()
                arch_mod._GLOBAL.clear()
                arch_mod._ROLLOUT_INDEX.clear()


class TestExpansionDirCompat(unittest.TestCase):
    def test_expansion_dir_new_default(self) -> None:
        """EXPANSION_DIR default resolves to artifacts/mas/local_expansion
        (TIR_LOCAL_EXPANSION_DIR env still wins when set)."""
        from workflow import active_set

        old = os.environ.get("TIR_LOCAL_EXPANSION_DIR", "")
        try:
            os.environ.pop("TIR_LOCAL_EXPANSION_DIR", None)
            resolved = Path(active_set.EXPANSION_DIR).resolve()
            expected = (REPO_ROOT / "artifacts" / "mas" / "local_expansion").resolve()
            self.assertEqual(resolved, expected)
            # legacy stays distinct and is never the write target
            legacy = Path(active_set.LEGACY_EXPANSION_DIR).resolve()
            self.assertTrue(
                str(legacy).replace(os.sep, "/").endswith("mas/.local_expansion")
            )
            self.assertNotEqual(resolved, legacy)
        finally:
            if old:
                os.environ["TIR_LOCAL_EXPANSION_DIR"] = old

    def test_load_local_expansion_reads_legacy_dir(self) -> None:
        """Old-directory tree JSON must remain readable (read-only compat)."""
        from workflow import active_set

        with tempfile.TemporaryDirectory() as legacy_dir, tempfile.TemporaryDirectory() as new_dir:
            rid = "legacy-tree-rollout"
            payload = {"plans": [{"plan_id": "p1", "data_id": "d1"}], "branch_local_count": 2}
            Path(legacy_dir).mkdir(parents=True, exist_ok=True)
            (Path(legacy_dir) / f"{rid}.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            with unittest.mock.patch.object(active_set, "EXPANSION_DIR", new_dir), \
                 unittest.mock.patch.object(active_set, "LEGACY_EXPANSION_DIR", legacy_dir):
                got = active_set.load_local_expansion(rid)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertEqual(got["branch_local_count"], 2)
            self.assertEqual(got["plans"][0]["plan_id"], "p1")

    def test_new_dir_wins_when_both_exist(self) -> None:
        from workflow import active_set

        with tempfile.TemporaryDirectory() as legacy_dir, tempfile.TemporaryDirectory() as new_dir:
            rid = "both-dirs-rollout"
            (Path(new_dir) / f"{rid}.json").write_text(
                json.dumps({"plans": [{"plan_id": "new-p"}]}), encoding="utf-8"
            )
            (Path(legacy_dir) / f"{rid}.json").write_text(
                json.dumps({"plans": [{"plan_id": "legacy-p"}]}), encoding="utf-8"
            )
            with unittest.mock.patch.object(active_set, "EXPANSION_DIR", new_dir), \
                 unittest.mock.patch.object(active_set, "LEGACY_EXPANSION_DIR", legacy_dir):
                got = active_set.load_local_expansion(rid)
            self.assertIsNotNone(got)
            assert got is not None
            self.assertEqual(got["plans"][0]["plan_id"], "new-p")

    def test_dump_never_writes_legacy_dir(self) -> None:
        """Single-write guarantee: dump_local_expansion writes only under the
        new EXPANSION_DIR; the legacy dir is never touched."""
        from workflow import active_set

        with tempfile.TemporaryDirectory() as legacy_dir, tempfile.TemporaryDirectory() as new_dir:
            rid = "no-dual-write-rollout"
            before = sorted(p.name for p in Path(legacy_dir).iterdir())
            with unittest.mock.patch.object(active_set, "EXPANSION_DIR", new_dir), \
                 unittest.mock.patch.object(active_set, "LEGACY_EXPANSION_DIR", legacy_dir):
                dest = active_set.dump_local_expansion(rid, {"plans": [], "branch_local_count": 0})
                # dump must land in the new dir only
                self.assertTrue(str(dest).startswith(str(new_dir)))
                self.assertTrue((Path(new_dir) / f"{rid}.json").is_file())
                # legacy dir state is byte-identical
                after = sorted(p.name for p in Path(legacy_dir).iterdir())
            self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
