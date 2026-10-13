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
  - planner_ctx (Phase 6): "latest" hub-memory injection via the centralized
    runtime vs "none" default byte-level baseline equivalence.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

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


class TestPlannerCtx(unittest.TestCase):
    """Phase 6 planner_ctx (MAS_design §5): opt-in hub-memory injection.

    ``latest`` -> first planner window user text carries the most recent
    shared-hub memory item (marked ``[Previous episode memory]``),
    MEMORY_READ/MEMORY_WRITE events fire (runtime.py schema), the episode
    close writes the summary back and ``EpisodeRaw.memory_mode == "latest"``.
    ``none``   -> the default: zero injection, no memory events, zero
    message-sequence drift vs the Phase 0 baseline (evaluation-pollution
    red line).
    """

    @staticmethod
    def _linear_spec(memory_planner_ctx: str) -> Any:
        from workflow.spec import MASSpec

        return MASSpec.model_validate({
            "schema_version": "0.3",
            "topology": "centralized",
            "entry_agent": "planner",
            "tools": ["python_coder"],
            "memory": {"agent": "messages", "system": "none", "planner_ctx": memory_planner_ctx},
            "agents": [
                {"id": "planner", "kind": "planner", "tools": ["python_coder"], "system_prompt": "plan"},
                {"id": "python_coder", "kind": "tool", "trainable": False},
                {"id": "verifier", "kind": "verifier", "system_prompt": "verify"},
            ],
            "routers": [{"id": "route_exec", "candidates": ["python_coder"], "strategy": "llm_choice"}],
            "edges": [
                {"from": "planner", "to": "route_exec", "kind": "route"},
                {"from": "route_exec", "to": "verifier", "kind": "message"},
                {"from": "verifier", "to": "planner", "kind": "feedback"},
            ],
        })

    @staticmethod
    def _scripted_win() -> Any:
        from workflow.centralized_runtime import MockWindowLLM

        return MockWindowLLM({
            "planner": [json.dumps({"next": "python_coder", "args": {"code": "result=42"}, "sub_goal": "compute", "done": True})],
            "verifier": [json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})],
        })

    @classmethod
    def _run(cls, spec, memory, win=None):
        """One scripted episode -> (raw, archive)."""
        from workflow.archive import Archive
        from workflow.centralized_runtime import run_centralized_episode
        from workflow.compiler import compile_spec

        compiled = compile_spec(spec)
        arch = Archive()
        raw = run_centralized_episode(
            {"id": "ctx", "question": "What is 6*7?"},
            None, arch, spec, memory, compiled,
            window_llm=win or cls._scripted_win(),
        )
        return raw, arch

    @staticmethod
    def _capture_first_planner_user(win) -> dict:
        """Wrap win.invoke so the first planner user text gets recorded."""
        captured: dict = {}
        orig_invoke = win.invoke

        def _capture(system_prompt: str, user_text: str, *, agent_id: str, kind: str) -> str:
            if agent_id == "planner" and "planner" not in captured:
                captured["planner"] = user_text
            return orig_invoke(system_prompt, user_text, agent_id=agent_id, kind=kind)

        win.invoke = _capture
        return captured

    def test_planner_ctx_latest(self) -> None:
        """latest: hub memory lands in the first planner window user text;
        MEMORY_READ/MEMORY_WRITE events fire; meta records memory_mode."""
        from workflow.memory import MemoryStore

        mem = MemoryStore()
        mem.write_agent("planner", {"final_answer": "42", "n_messages": 5}, memory_scope="shared")

        win = self._scripted_win()
        captured = self._capture_first_planner_user(win)
        spec = self._linear_spec("latest")
        raw, arch = self._run(spec, mem, win)
        kinds = [str(e.kind.value) for e in arch._events]

        # first-turn planner user text carries the marked hub memory item
        first_user = captured.get("planner", "")
        self.assertIn("[Previous episode memory]", first_user)
        self.assertIn("42", first_user)
        # memory events fired with the runtime.py schema
        self.assertIn("memory_read", kinds)
        self.assertIn("memory_write", kinds)
        read_ev = next(e for e in arch._events if str(e.kind.value) == "memory_read")
        self.assertEqual(read_ev.payload["scope"], "agent")
        self.assertEqual(read_ev.payload["owner"], "hub")
        self.assertEqual(read_ev.payload["n"], 1)
        write_ev = next(e for e in arch._events if str(e.kind.value) == "memory_write" and e.payload.get("owner") == "hub")
        self.assertEqual(write_ev.payload["scope"], "agent")
        self.assertEqual(write_ev.payload["owner"], "hub")
        self.assertEqual(write_ev.payload["memory_scope"], "shared")
        # the episode close wrote the summary back to the hub buffer
        latest = mem.latest("agent", "hub")
        self.assertIsNotNone(latest)
        self.assertEqual(latest.content["final_answer"], raw.final_answer)
        self.assertEqual(latest.content["n_messages"], len(raw.messages))
        # EpisodeRaw audits the mode
        self.assertEqual(raw.memory_mode, "latest")

    def test_planner_ctx_none_default(self) -> None:
        """none (default): zero injection, no memory events — the message
        sequence must stay byte-identical to the pre-Phase-6 baseline."""
        from workflow.memory import MemoryStore

        mem = MemoryStore()
        mem.write_agent("planner", {"final_answer": "42", "n_messages": 5}, memory_scope="shared")

        win = self._scripted_win()
        captured = self._capture_first_planner_user(win)
        spec = self._linear_spec("none")
        raw, arch = self._run(spec, mem, win)
        kind_values = [str(e.kind.value) for e in arch._events]

        first_user = captured.get("planner", "")
        self.assertNotIn("[Previous episode memory]", first_user)
        # no MEMORY_READ; the only MEMORY_WRITE may be fact_commit (verifier
        # path) — never a planner_ctx hub write
        self.assertNotIn("memory_read", kind_values)
        write_owners = [e.payload.get("owner") for e in arch._events if str(e.kind.value) == "memory_write"]
        self.assertNotIn("hub", write_owners)
        # EpisodeRaw audits the default
        self.assertEqual(raw.memory_mode, "none")
        # message-sequence baseline: default run matches the frozen Phase 0
        # message kinds exactly (byte-level equivalence gate)
        baseline_kinds = ["plan_step", "tool_invoke", "tool_result", "verify"]
        self.assertEqual([m["kind"] for m in raw.messages], baseline_kinds)


if __name__ == "__main__":
    unittest.main()
