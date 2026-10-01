"""User-space isolation, gateway, Mode 1/2 scaffold, assistant write jail."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


class UserSpaceTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        self._old = os.environ.get("SCIENCE_USER_SPACE_DIR")
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)
        from workflow.user_gateway.tools import clear_user_tools

        clear_user_tools()

    def tearDown(self) -> None:
        from workflow.user_gateway.tools import clear_user_tools

        clear_user_tools()
        if self._old is None:
            os.environ.pop("SCIENCE_USER_SPACE_DIR", None)
        else:
            os.environ["SCIENCE_USER_SPACE_DIR"] = self._old
        self._tmp.cleanup()


class TestPathJail(UserSpaceTestBase):
    def test_write_management_rejected(self) -> None:
        from workflow.user_gateway.paths import assert_writable_user_path

        with self.assertRaises((PermissionError, ValueError)):
            assert_writable_user_path(TIR / "tir_agent.py")

    def test_write_upload_rejected(self) -> None:
        from workflow.user_gateway.paths import assert_writable_user_path, ensure_project_layout

        ensure_project_layout("demo")
        with self.assertRaises(PermissionError):
            assert_writable_user_path(self.user_root / "projects" / "demo" / "upload" / "x.py", project_id="demo")

    def test_write_adapted_ok(self) -> None:
        from workflow.user_gateway.paths import assert_writable_user_path, ensure_project_layout

        root = ensure_project_layout("demo")
        target = root / "adapted" / "entry.py"
        target.write_text("ok", encoding="utf-8")
        locked = assert_writable_user_path(target, project_id="demo")
        self.assertTrue(str(locked).endswith("adapted/entry.py"))

    def test_dotdot_escape_rejected(self) -> None:
        from workflow.user_gateway.paths import assert_in_user_space, ensure_project_layout

        ensure_project_layout("demo")
        with self.assertRaises((PermissionError, ValueError)):
            assert_in_user_space(self.user_root / "projects" / "demo" / "adapted" / ".." / ".." / ".." / ".." / "mas" / "x.py")


class TestGatewayMode1(UserSpaceTestBase):
    def test_io_module_run_window(self) -> None:
        from workflow.protocol import make_message
        from workflow.user_gateway.loader import invoke_user_window
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_io_module

        create_project("wrapme", title="Wrap", mode="io_module")
        built = scaffold_io_module("wrapme", title="Wrap")
        self.assertEqual(built["agent_ids"], ["u_wrapme__solver"])
        inbound = make_message(
            task_id="t", turn=1, src="router", dst="u_wrapme__solver",
            kind="tool_invoke", payload={"query": "hello"},
        )
        out = invoke_user_window("wrapme", inbound, agent_id="u_wrapme__solver", expected_kind="tool_result")
        self.assertEqual(out.kind, "tool_result")
        self.assertEqual(out.payload.get("output"), "hello")

    def test_centralized_blank_uses_gateway(self) -> None:
        from workflow.archive import Archive, register_archive
        from workflow.centralized_runtime import MockWindowLLM, run_centralized_episode
        from workflow.compiler import compile_spec
        from workflow.memory import MemoryStore
        from workflow.spec import MASSpec
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_io_module
        from workflow.user_gateway.validate import load_project_workflow

        create_project("live1", title="Live", mode="io_module")
        scaffold_io_module("live1")
        solver = "u_live1__solver"
        spec = MASSpec.model_validate({
            "schema_version": "0.3",
            "topology": "centralized",
            "entry_agent": "planner",
            "hub": {"role": "planner", "max_feedback_hops": 1},
            "agents": [
                {"id": "planner", "kind": "planner", "system_prompt": "plan"},
                *load_project_workflow("live1").get("agents", []),
                {"id": "verifier", "kind": "verifier", "trainable": False, "system_prompt": "verify"},
            ],
            "routers": [{"id": "route_exec", "candidates": [solver], "strategy": "from_plan"}],
            "edges": [
                {"from": "planner", "to": "route_exec", "kind": "route"},
                {"from": "route_exec", "to": "verifier", "kind": "message"},
                {"from": "verifier", "to": "planner", "kind": "feedback"},
            ],
        })
        compiled = compile_spec(spec)
        self.assertTrue(compiled.ok, compiled.reason)
        llm = MockWindowLLM({
            "planner": ['{"next": "%s", "args": {"query": "ping"}, "sub_goal": "x", "done": true}' % solver],
            "verifier": ['{"ok": true, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []}'],
        })
        raw = run_centralized_episode(
            {"id": "t1", "question": "q"}, None, register_archive(Archive()), spec, MemoryStore(), compiled,
            window_llm=llm,
        )
        kinds = [m.get("kind") for m in (raw.messages or [])]
        self.assertIn("tool_result", kinds, raw.error or kinds)
        tool = next(m for m in raw.messages if m.get("kind") == "tool_result")
        self.assertEqual(tool.get("payload", {}).get("output"), "ping")

    def test_validate_and_compile(self) -> None:
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_io_module
        from workflow.user_gateway.validate import validate_project

        create_project("okone", title="OK", mode="io_module")
        scaffold_io_module("okone")
        report = validate_project("okone")
        self.assertTrue(report["ok"], report)


class TestGatewayMode2(UserSpaceTestBase):
    def test_native_mas_compile_and_sites(self) -> None:
        from workflow.spec import MASSpec
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_native_mas
        from workflow.user_gateway.validate import compile_project, validate_project

        create_project("nativ", title="Native", mode="native_mas")
        built = scaffold_native_mas("nativ")
        report = validate_project("nativ")
        self.assertTrue(report["ok"], report)
        self.assertGreaterEqual(report["n_sites"], 2)
        spec, compiled = compile_project("nativ")
        self.assertTrue(compiled.ok, compiled.reason)
        self.assertTrue(spec.is_executable()[0])
        sites = spec.sampling.resolved_sites()
        kinds = {s.anchor.kind for s in sites}
        self.assertIn("after_agent_turn", kinds)
        self.assertTrue(any(s.fork.resume_mode == "messages" for s in sites))
        self.assertIn("u_nativ__tool", [a.id for a in spec.agents])
        self.assertEqual(MASSpec.model_validate(built["workflow"]).topology, "centralized")

    def test_user_tool_does_not_mutate_builtin(self) -> None:
        from tools.tool_agents import TOOL_AGENTS, get_tool_agent
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_native_mas

        before = set(TOOL_AGENTS)
        create_project("ovl", title="Ovl", mode="native_mas")
        scaffold_native_mas("ovl")
        ta = get_tool_agent("u_ovl__tool")
        self.assertIsNotNone(ta)
        self.assertEqual(set(TOOL_AGENTS), before)
        self.assertIn("wikipedia_search", TOOL_AGENTS)

    def test_agent_registry_user_bound(self) -> None:
        from workflow.agents import AgentRegistry
        from workflow.spec import MASSpec
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_io_module
        from workflow.user_gateway.validate import load_project_workflow

        create_project("regme", title="Reg", mode="io_module")
        scaffold_io_module("regme")
        wf = load_project_workflow("regme")
        spec = MASSpec.model_validate({
            "schema_version": "0.3",
            "topology": "centralized",
            "entry_agent": "planner",
            "agents": [
                {"id": "planner", "kind": "planner"},
                *wf.get("agents", []),
                {"id": "verifier", "kind": "verifier", "trainable": False},
            ],
            "routers": [{"id": "route_exec", "candidates": ["u_regme__solver"], "strategy": "from_plan"}],
            "edges": [
                {"from": "planner", "to": "route_exec", "kind": "route"},
                {"from": "route_exec", "to": "verifier", "kind": "message"},
            ],
        })
        reg = AgentRegistry.from_spec(spec)
        self.assertIn("u_regme__solver", reg.user_agents)
        self.assertEqual(reg.user_agents["u_regme__solver"].backend, "user_space")


class TestApplyMerge(UserSpaceTestBase):
    def test_mode1_merges_blank_into_router(self) -> None:
        from workflow.user_gateway.apply import merge_user_workflow
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_io_module

        create_project("addme", title="Add", mode="io_module")
        scaffold_io_module("addme")
        current = {
            "schema_version": "0.3",
            "topology": "centralized",
            "agents": [{"id": "planner", "kind": "planner"}],
            "routers": [{"id": "route_exec", "candidates": ["think"], "strategy": "from_plan"}],
        }
        merged = merge_user_workflow(current, "addme")
        ids = [a["id"] for a in merged["agents"]]
        self.assertIn("u_addme__solver", ids)
        self.assertIn("u_addme__solver", merged["routers"][0]["candidates"])


class TestIsolationScript(unittest.TestCase):
    def test_check_user_zone_ok(self) -> None:
        import subprocess

        r = subprocess.run(
            [sys.executable, str(TIR / "scripts" / "check_user_zone.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("OK check_user_zone", r.stdout)

    def test_check_workflow_deps_still_ok(self) -> None:
        import subprocess

        r = subprocess.run(
            [sys.executable, str(TIR / "scripts" / "check_workflow_deps.py")],
            cwd=TIR,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class TestAssistantJail(UserSpaceTestBase):
    def test_write_tool_rejects_mas(self) -> None:
        from science_infra.control.assistant import assistant_tools
        from workflow.user_gateway.registry import create_project

        create_project("jail", title="Jail", mode="io_module")
        tools = {t["name"]: t["fn"] for t in assistant_tools("jail")}
        with self.assertRaises((PermissionError, ValueError)):
            from workflow.user_gateway.paths import assert_writable_user_path

            assert_writable_user_path(TIR / "tir_agent.py", project_id="jail")
        # write_file remaps into adapted/, so writing mas path becomes adapted/tir_agent.py
        result = tools["write_file"](path="mas/tir_agent.py", content="hack")
        self.assertIn("adapted/tir_agent.py", result)
        self.assertFalse((TIR / "tir_agent.py").read_text(encoding="utf-8").startswith("hack"))

    def test_palette_lists_user_project(self) -> None:
        from science_infra.control.services import mas_palette
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_io_module

        create_project("palx", title="Palette", mode="io_module")
        scaffold_io_module("palx")
        pal = mas_palette()
        ids = [p["id"] for p in pal.get("user_projects") or []]
        self.assertIn("palx", ids)


if __name__ == "__main__":
    unittest.main()
