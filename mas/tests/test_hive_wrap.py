"""HIVE PEV wrap: assistant LLM env, scaffold, sampling sites, fake episode."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

HIVE_PROMPT = "阅读 ref_Rep/HIVE 的 Planner/Executor/Diagnoser，保持 PEV，封装到用户区，底层代码不要动。"


class HiveWrapTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        self._old_user = os.environ.get("SCIENCE_USER_SPACE_DIR")
        self._old_mock = os.environ.get("SCIENCE_HIVE_MOCK")
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)
        os.environ["SCIENCE_HIVE_MOCK"] = "1"
        from workflow.user_gateway.tools import clear_user_tools

        clear_user_tools()

    def tearDown(self) -> None:
        from workflow.user_gateway.tools import clear_user_tools

        clear_user_tools()
        if self._old_user is None:
            os.environ.pop("SCIENCE_USER_SPACE_DIR", None)
        else:
            os.environ["SCIENCE_USER_SPACE_DIR"] = self._old_user
        if self._old_mock is None:
            os.environ.pop("SCIENCE_HIVE_MOCK", None)
        else:
            os.environ["SCIENCE_HIVE_MOCK"] = self._old_mock
        self._tmp.cleanup()

    def _scaffold(self, project_id: str = "hive"):
        from workflow.user_gateway.hive import ensure_hive_project, scaffold_hive_pev

        ensure_hive_project(project_id)
        return scaffold_hive_pev(project_id)


class TestAssistantLlm(unittest.TestCase):
    def test_resolve_assistant_llm_reads_env(self) -> None:
        from science_infra.control.assistant import resolve_assistant_llm

        old = {
            "AI_ASSISTANT_API_KEY": os.environ.get("AI_ASSISTANT_API_KEY"),
            "AI_ASSISTANT_API_BASE": os.environ.get("AI_ASSISTANT_API_BASE"),
            "AI_ASSISTANT_MODEL": os.environ.get("AI_ASSISTANT_MODEL"),
        }
        os.environ["AI_ASSISTANT_API_KEY"] = "sk-test-not-for-logs"
        os.environ["AI_ASSISTANT_API_BASE"] = "https://example.test/v1"
        os.environ["AI_ASSISTANT_MODEL"] = "unit-model"
        try:
            cfg = resolve_assistant_llm()
            self.assertEqual(cfg.source, "assistant_env")
            self.assertEqual(cfg.model, "unit-model")
            self.assertEqual(cfg.base_url, "https://example.test/v1")
            self.assertNotIn("sk-test-not-for-logs", repr(cfg))
            self.assertNotIn("api_key", repr(cfg))
        finally:
            for key, val in old.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_resolve_assistant_llm_none_without_env(self) -> None:
        from unittest.mock import patch

        from science_infra.control.assistant import resolve_assistant_llm

        env = {
            "AI_ASSISTANT_API_KEY": "",
            "AI_ASSISTANT_API_BASE": "",
            "AI_ASSISTANT_BASE_URL": "",
            "AI_ASSISTANT_MODEL": "",
        }
        with patch("science_infra.control.assistant.load_science_env", lambda **k: None):
            with patch.dict(os.environ, env, clear=False):
                cfg = resolve_assistant_llm(None)
                self.assertEqual(cfg.source, "none")


class TestHiveScaffold(HiveWrapTestBase):
    def test_compile_and_sampling_sites(self) -> None:
        from workflow.sampling.adapters.agent_router import agent_router_opportunities
        from workflow.site_policy import sampling_preview
        from workflow.spec import MASSpec
        from workflow.user_gateway.validate import compile_project, validate_project

        built = self._scaffold("hive")
        report = validate_project("hive")
        self.assertTrue(report["ok"], report)
        spec, compiled = compile_project("hive")
        self.assertTrue(compiled.ok, compiled.reason)
        self.assertTrue(spec.is_executable()[0])
        sites = spec.sampling.resolved_sites()
        planner_sites = [s for s in sites if s.anchor.agent_id == "planner" and s.anchor.kind == "after_agent_turn"]
        verifier_sites = [s for s in sites if s.anchor.agent_id == "verifier" and s.anchor.kind == "after_verifier"]
        self.assertTrue(planner_sites)
        self.assertTrue(verifier_sites)
        self.assertTrue(all(s.fork.resume_mode == "messages" for s in sites))
        ids = [a.id for a in spec.agents]
        self.assertIn("planner", ids)
        self.assertIn("verifier", ids)
        self.assertIn("u_hive__executor", ids)
        self.assertEqual(built["tool_ids"], ["u_hive__executor"])

        preview = sampling_preview(built["workflow"])
        owners = {row.get("node_id") for row in preview.get("opportunities") or []}
        self.assertIn("planner", owners)
        self.assertIn("route_exec", owners)
        self.assertIn("verifier", owners)
        self.assertNotIn("u_hive__executor", owners)
        opps = list(agent_router_opportunities(MASSpec.model_validate(built["workflow"]), allowed_gates=("always",), message="x"))
        self.assertFalse(any(o.selector.owner_agent_id == "u_hive__executor" for o in opps))

    def test_adapted_no_science_infra_import(self) -> None:
        from workflow.user_gateway.sandbox import scan_user_imports

        self._scaffold("hive")
        adapted = self.user_root / "projects" / "hive" / "adapted"
        for path in adapted.rglob("*.py"):
            mods = scan_user_imports(path)
            self.assertNotIn("science_infra", [m.split(".")[0] for m in mods], path)

    def test_ref_hive_untouched(self) -> None:
        r = subprocess.run(
            ["git", "diff", "--", "ref_Rep/HIVE"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual((r.stdout or "").strip(), "", r.stdout)


class TestHiveEpisode(HiveWrapTestBase):
    def test_fake_context_pev_window_events(self) -> None:
        from workflow.archive import Archive, register_archive
        from workflow.centralized_runtime import MockWindowLLM, run_centralized_episode
        from workflow.memory import MemoryStore
        from workflow.user_gateway.loader import load_entry_module
        from workflow.user_gateway.validate import compile_project

        self._scaffold("hive")
        spec, compiled = compile_project("hive")
        sys.modules.pop("context", None)
        load_entry_module("hive", reload=True)
        ctx_mod = sys.modules.get("context")
        if ctx_mod is not None:
            ctx_mod.SOLVE_CALLS = 0
            ctx_mod.get_episode_context(reset=True)
        raw = run_centralized_episode(
            {"id": "hive-t", "question": "1+1"},
            None,
            register_archive(Archive()),
            spec,
            MemoryStore(),
            compiled,
            window_llm=MockWindowLLM({}),
        )
        self.assertIsNone(raw.error, raw.error)
        kinds = [m.get("kind") for m in (raw.messages or [])]
        self.assertIn("plan_step", kinds, kinds)
        self.assertIn("tool_result", kinds, kinds)
        self.assertIn("verify", kinds, kinds)
        events = list(raw.window_events or [])
        self.assertTrue(
            any(e.get("agent_id") == "planner" and e.get("kind") == "after_agent_turn" for e in events),
            events,
        )
        self.assertTrue(
            any(e.get("agent_id") == "verifier" and e.get("kind") == "after_verifier" for e in events),
            events,
        )
        self.assertTrue(any(e.get("agent_id") == "route_exec" for e in events), events)
        load_entry_module("hive")
        ctx_mod = sys.modules.get("context")
        self.assertIsNotNone(ctx_mod)
        self.assertEqual(int(getattr(ctx_mod, "SOLVE_CALLS", 0)), 0)
        with self.assertRaises(RuntimeError):
            ctx_mod.get_episode_context().solver.solve("1+1")
        self.assertGreater(int(ctx_mod.SOLVE_CALLS), 0)

    def test_activeset_matches_declared_sites(self) -> None:
        from workflow.active_set import ActiveSetConfig, ActiveSetSession
        from workflow.archive import Archive, register_archive
        from workflow.centralized_runtime import MockWindowLLM, run_centralized_episode
        from workflow.memory import MemoryStore
        from workflow.user_gateway.validate import compile_project

        self._scaffold("hive")
        spec, compiled = compile_project("hive")
        raw = run_centralized_episode(
            {"id": "hive-fork", "question": "1+1"},
            None,
            register_archive(Archive()),
            spec,
            MemoryStore(),
            compiled,
            window_llm=MockWindowLLM({}),
        )
        sess = ActiveSetSession(
            ActiveSetConfig(
                beam_size=2,
                sites=list(spec.sampling.resolved_sites()),
                run_probes=False,
                strategy="arpo",
            )
        )
        plans = sess.plan_forks_from_raw(raw, parent_id="p0", remaining=3)
        site_ids = {p.site_id for p in plans}
        self.assertTrue(site_ids.intersection({"after_planner", "after_route_exec", "after_verifier"}), site_ids)


class TestAssistantHiveTool(HiveWrapTestBase):
    def test_local_answer_scaffolds_hive(self) -> None:
        from science_infra.control.assistant import _local_answer

        text = _local_answer(HIVE_PROMPT, "hive")
        self.assertTrue((self.user_root / "projects" / "hive" / "adapted" / "context.py").is_file())
        self.assertIn("ok", text.lower())

    def test_write_management_paths_rejected(self) -> None:
        from science_infra.control.assistant import assistant_tools
        from workflow.user_gateway.paths import assert_writable_user_path

        self._scaffold("hive")
        tools = {t["name"]: t["fn"] for t in assistant_tools("hive")}
        with self.assertRaises((PermissionError, ValueError)):
            assert_writable_user_path(TIR / "tir_agent.py", project_id="hive")
        with self.assertRaises((PermissionError, ValueError)):
            assert_writable_user_path(ROOT / "ref_Rep" / "HIVE" / "MAS" / "hive" / "solver.py", project_id="hive")
        before = (TIR / "tir_agent.py").read_text(encoding="utf-8")[:80]
        tools["write_file"](path="mas/tir_agent.py", content="hack")
        self.assertEqual((TIR / "tir_agent.py").read_text(encoding="utf-8")[:80], before)
        solver = ROOT / "ref_Rep" / "HIVE" / "MAS" / "hive" / "solver.py"
        solver_head = solver.read_text(encoding="utf-8")[:80]
        tools["write_file"](path="ref_Rep/HIVE/MAS/hive/solver.py", content="hack")
        self.assertEqual(solver.read_text(encoding="utf-8")[:80], solver_head)

    def test_assistant_chat_deconstructs_hive(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from science_infra.control.assistant import resolve_assistant_llm, router

        self._scaffold("hive")
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        cfg = resolve_assistant_llm()
        body = {"message": HIVE_PROMPT, "project_id": "hive", "experiment_id": "demo"}
        if cfg.source != "assistant_env":
            resp = client.post("/api/assistant/chat", json=body)
            self.assertEqual(resp.status_code, 200)
            self.assertTrue((self.user_root / "projects" / "hive" / "adapted" / "entry.py").is_file())
            return
        try:
            resp = client.post("/api/assistant/chat", json=body, timeout=90.0)
        except Exception:
            self.assertTrue((self.user_root / "projects" / "hive" / "adapted" / "context.py").is_file())
            return
        self.assertEqual(resp.status_code, 200)
        text = resp.text
        self.assertFalse(any(tok in text for tok in ("sk-", "API_KEY")), "assistant must not echo secrets")
        self.assertTrue((self.user_root / "projects" / "hive" / "adapted" / "context.py").is_file())
        lowered = text.lower()
        self.assertTrue(
            any(token in lowered for token in ("planner", "executor", "verifier", "hiv", "包装", "封装")),
            text[:400],
        )


class TestHiveAstLiveOptional(unittest.TestCase):
    def test_construct_solver_optional(self) -> None:
        mas = ROOT / "ref_Rep" / "HIVE" / "MAS"
        if not mas.is_dir():
            self.skipTest("HIVE missing")
        if mas.as_posix() not in sys.path:
            sys.path.insert(0, str(mas))
        try:
            from MAS.hive.solver import construct_solver  # type: ignore
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"HIVE import skipped: {exc}")
        try:
            solver = construct_solver(verbose=False)
        except Exception as exc:  # noqa: BLE001
            self.skipTest(f"construct_solver skipped: {exc}")
        self.assertTrue(hasattr(solver, "planner"))
        self.assertTrue(hasattr(solver, "executor"))
        self.assertTrue(hasattr(solver, "diagnoser"))


if __name__ == "__main__":
    unittest.main()
