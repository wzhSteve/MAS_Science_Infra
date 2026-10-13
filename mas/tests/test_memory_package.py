"""Phase 2 memory-subpackage tests (docs/refactor/phase0_contract_freeze.md §3-4).

Covers:
  - import compatibility: ``workflow.memory`` package re-exports identical
    objects; the ``workflow.archive`` shim resolves to the moved module.
  - FactLedger matrix: the five classification rows of the frozen
    commit gate (COMPLETE+DIRECT+slots / no-slots / unfilled / INCOMPLETE /
    non-DIRECT), including the evidence_type fallback to tool_result.
  - bool bridge: ``protocol.commit_to_fact`` (frozen signature) stays
    consistent with the three-valued classification.
  - Archive concurrency: 2 threads x 20 appends -> events.jsonl has all
    lines intact (ActiveSetScheduler hardening, MAS_design §4.6).
  - Snapshot schema_version: new snapshots persist version 1; legacy JSON
    without the field still parses (back-compat).
  - archive_root: TIR_ARCHIVE_DIR env takes priority over the default.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path

MAS_ROOT = Path(__file__).resolve().parents[1]
if str(MAS_ROOT) not in sys.path:
    sys.path.insert(0, str(MAS_ROOT))


def _verify(payload: dict, evidence: str | None = None) -> dict:
    base = {"ok": True, "reason": "r", "step_conclusion": None, "slot_updates": []}
    base.update(payload)
    if evidence is not None:
        base["evidence_type"] = evidence
    return base


def _tool_result(evidence: str = "DIRECT") -> dict:
    return {"output": "o", "ok": True, "evidence_type": evidence}


class TestImportCompatibility(unittest.TestCase):
    def test_memory_package_reexport_identity(self) -> None:
        from workflow import memory as mem_pkg
        from workflow.memory import FactLedger, MemoryItem, MemoryStore
        from workflow.memory import facts as facts_mod
        from workflow.memory import store as store_mod

        self.assertIs(MemoryStore, store_mod.MemoryStore)
        self.assertIs(MemoryItem, store_mod.MemoryItem)
        self.assertIs(FactLedger, facts_mod.FactLedger)
        # module -> package transition keeps the canonical import working
        import workflow
        self.assertIs(workflow.MemoryStore, MemoryStore)

    def test_archive_shim_identity(self) -> None:
        from workflow import archive as shim
        from workflow.memory import archive as real

        self.assertIs(shim.Archive, real.Archive)
        self.assertIs(shim.register_archive, real.register_archive)
        self.assertIs(shim.load_resume_messages, real.load_resume_messages)
        self.assertIs(shim.dump_resume_with_archive, real.dump_resume_with_archive)
        self.assertIs(shim.branch_point_to_resume_task_fields, real.branch_point_to_resume_task_fields)


class TestFactLedgerMatrix(unittest.TestCase):
    """The 5-row frozen commit-gate matrix (phase0_contract_freeze.md §2)."""

    def setUp(self) -> None:
        from workflow.memory.facts import FactLedger

        self.cls = FactLedger.classify

    def test_row1_complete_direct_slots_filled_commits(self) -> None:
        v = _verify({"step_conclusion": "COMPLETE", "slot_updates": [{"slot": "s1", "filled": True}]}, "DIRECT")
        self.assertEqual(self.cls(v, _tool_result()), "COMMIT_FACT")

    def test_row2_complete_direct_no_slots_commits(self) -> None:
        v = _verify({"step_conclusion": "COMPLETE", "slot_updates": []}, "DIRECT")
        self.assertEqual(self.cls(v, _tool_result()), "COMMIT_FACT")

    def test_row3_unfilled_slot_keeps_window_only(self) -> None:
        v = _verify({"step_conclusion": "COMPLETE", "slot_updates": [{"slot": "s1", "filled": False}]}, "DIRECT")
        self.assertEqual(self.cls(v, _tool_result()), "KEEP_WINDOW_ONLY")

    def test_row4_non_direct_evidence_rejects(self) -> None:
        for ev in ("ABSENCE", "ERROR", "EMPTY"):
            v = _verify({"step_conclusion": "COMPLETE"}, ev)
            self.assertEqual(self.cls(v, _tool_result(ev)), "REJECT", ev)

    def test_row5_incomplete_keeps_window_only(self) -> None:
        v = _verify({"step_conclusion": "INCOMPLETE"}, "DIRECT")
        self.assertEqual(self.cls(v, _tool_result()), "KEEP_WINDOW_ONLY")

    def test_evidence_falls_back_to_tool_result(self) -> None:
        v = _verify({"step_conclusion": "COMPLETE"})  # verifier omits evidence
        self.assertEqual(self.cls(v, _tool_result("DIRECT")), "COMMIT_FACT")
        self.assertEqual(self.cls(v, _tool_result("ABSENCE")), "REJECT")

    def test_missing_evidence_entirely_rejects(self) -> None:
        v = _verify({"step_conclusion": "COMPLETE"})  # no evidence anywhere
        self.assertEqual(self.cls(v, {"output": "x", "ok": True}), "REJECT")


class TestCommitToFactBoolBridge(unittest.TestCase):
    def test_bridge_consistency(self) -> None:
        from workflow.memory.facts import FactLedger
        from workflow.protocol import commit_to_fact

        cases = [
            (_verify({"step_conclusion": "COMPLETE"}, "DIRECT"), _tool_result(), True),
            (_verify({"step_conclusion": "INCOMPLETE"}, "DIRECT"), _tool_result(), False),
            (_verify({"step_conclusion": "COMPLETE"}, "EMPTY"), _tool_result("EMPTY"), False),
        ]
        for v, t, expected in cases:
            decision = FactLedger.classify(v, t)
            self.assertEqual(commit_to_fact(v, t), expected)
            self.assertEqual(FactLedger.commits(decision), expected)


class TestArchiveConcurrency(unittest.TestCase):
    def test_concurrent_append_integrity(self) -> None:
        """2 threads x 20 appends -> events.jsonl has 40 clean JSON lines."""
        from workflow.contracts import ExecutionEvent
        from workflow.memory.archive import Archive

        with tempfile.TemporaryDirectory() as td:
            arch = Archive(root_dir=td, archive_id="conc")
            n_threads, per_thread = 2, 20

            def worker(tid: int) -> None:
                for i in range(per_thread):
                    arch.append(ExecutionEvent(kind="tool_call", payload={"t": tid, "i": i}))

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            path = Path(td) / "conc" / "events.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), n_threads * per_thread)
            parsed = [json.loads(ln) for ln in lines]  # every line valid, none interleaved
            self.assertEqual(len(parsed), 40)
            keys = {(p["payload"]["t"], p["payload"]["i"]) for p in parsed}
            self.assertEqual(len(keys), 40)  # no lost appends

    def test_concurrent_snapshot_and_append(self) -> None:
        """snapshot() inside a thread while others append — RLock prevents
        events.jsonl corruption and snapshot/index stay consistent."""
        from workflow.contracts import ExecutionEvent
        from workflow.memory.archive import Archive

        with tempfile.TemporaryDirectory() as td:
            arch = Archive(root_dir=td, archive_id="mix")
            errors: list[str] = []

            def appender() -> None:
                try:
                    for i in range(30):
                        arch.append(ExecutionEvent(kind="tool_call", payload={"i": i}))
                except Exception as e:  # pragma: no cover
                    errors.append(repr(e))

            def snapper() -> None:
                try:
                    for i in range(10):
                        arch.snapshot(messages=[{"role": "user", "content": str(i)}], rollout_id=f"rr{i}")
                except Exception as e:  # pragma: no cover
                    errors.append(repr(e))

            threads = [threading.Thread(target=appender), threading.Thread(target=snapper)]
            for th in threads:
                th.start()
            for th in threads:
                th.join()
            self.assertFalse(errors, errors)
            lines = (Path(td) / "mix" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            # 30 appends + 10 SNAPSHOT events (snapshot() emits its own event)
            self.assertEqual(len(lines), 40)
            for ln in lines:
                json.loads(ln)


class TestSnapshotSchemaVersion(unittest.TestCase):
    def test_new_snapshot_persists_version_1(self) -> None:
        from workflow.memory.archive import Archive

        with tempfile.TemporaryDirectory() as td:
            arch = Archive(root_dir=td, archive_id="sv")
            snap = arch.snapshot(messages=[], meta={})
            raw = json.loads((Path(td) / "sv" / "snapshots" / f"{snap.snapshot_id}.json").read_text())
        self.assertEqual(raw["schema_version"], 1)

    def test_legacy_json_without_version_still_parses(self) -> None:
        from workflow.contracts import Snapshot

        legacy = {
            "snapshot_id": "legacy1",
            "archive_id": "a",
            "event_id": "",
            "messages": [{"role": "user", "content": "q"}],
            "meta": {},
            # no schema_version key — pre-Phase-2 snapshot
        }
        snap = Snapshot.model_validate(legacy)
        self.assertEqual(snap.schema_version, 1)
        self.assertEqual(snap.messages[0]["content"], "q")


class TestArchiveRoot(unittest.TestCase):
    def test_env_takes_priority(self) -> None:
        from workflow.memory import archive as arch_mod

        old = os.environ.get("TIR_ARCHIVE_DIR")
        try:
            os.environ["TIR_ARCHIVE_DIR"] = "/tmp/custom_archive_root"
            # re-resolve (the module reads env lazily via archive_root())
            self.assertEqual(arch_mod.archive_root(), "/tmp/custom_archive_root")
        finally:
            if old is None:
                os.environ.pop("TIR_ARCHIVE_DIR", None)
            else:
                os.environ["TIR_ARCHIVE_DIR"] = old

    def test_default_points_at_new_artifacts_location(self) -> None:
        """Phase 5 artifacts migration: default root resolves to
        artifacts/mas/tir_archives (legacy mas/.tir_archives read-only)."""
        from workflow.memory.archive import archive_root, legacy_archive_root

        self.assertEqual(os.path.basename(archive_root()), "tir_archives")
        self.assertTrue(
            archive_root().replace(os.sep, "/").endswith("artifacts/mas/tir_archives")
        )
        # legacy location stays resolvable and distinct (read-only fallback)
        self.assertEqual(os.path.basename(legacy_archive_root()), ".tir_archives")
        self.assertNotEqual(archive_root(), legacy_archive_root())


if __name__ == "__main__":
    unittest.main()
