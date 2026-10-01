"""EPC-AW Mode 2: detect upload and scaffold Planner/Executor/Diagnoser windows."""

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


class EpcAwWrapTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        self._old_user = os.environ.get("SCIENCE_USER_SPACE_DIR")
        self._old_mock = os.environ.get("SCIENCE_EPC_AW_MOCK")
        self._old_n = os.environ.get("SCIENCE_EPC_AW_N")
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)
        os.environ["SCIENCE_EPC_AW_MOCK"] = "1"
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
            os.environ.pop("SCIENCE_EPC_AW_MOCK", None)
        else:
            os.environ["SCIENCE_EPC_AW_MOCK"] = self._old_mock
        if self._old_n is None:
            os.environ.pop("SCIENCE_EPC_AW_N", None)
        else:
            os.environ["SCIENCE_EPC_AW_N"] = self._old_n
        self._tmp.cleanup()

    def _drop_upload(self, project_id: str = "epc-aw-main") -> None:
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.paths import ensure_project_layout

        create_project(project_id, title="EPC-AW-main", mode="native_mas")
        root = ensure_project_layout(project_id)
        solver = root / "upload" / "EPC-AW-main" / "MAS" / "epc_aw" / "solver.py"
        solver.parent.mkdir(parents=True, exist_ok=True)
        solver.write_text("# marker\n", encoding="utf-8")
        (solver.parent / "models" / "planner.py").parent.mkdir(parents=True, exist_ok=True)
        (solver.parent / "models" / "planner.py").write_text("# planner\n", encoding="utf-8")
        (solver.parent / "models" / "executor.py").write_text("# executor\n", encoding="utf-8")
        (solver.parent / "models" / "diagnoser.py").write_text("# diagnoser\n", encoding="utf-8")


class TestEpcAwDetect(EpcAwWrapTestBase):
    def test_detect_from_upload_tree(self) -> None:
        from workflow.user_gateway.detect import detect_mas_family

        self._drop_upload("epc-aw-main")
        self.assertEqual(detect_mas_family("epc-aw-main", "EPC-AW-main"), "epc_aw")

    def test_detect_from_title(self) -> None:
        from workflow.user_gateway.detect import detect_mas_family
        from workflow.user_gateway.registry import create_project

        create_project("other", title="EPC-AW zip", mode="native_mas")
        self.assertEqual(detect_mas_family("other", "EPC-AW zip"), "epc_aw")


class TestEpcAwScaffold(EpcAwWrapTestBase):
    def test_native_mas_splits_pev(self) -> None:
        from workflow.user_gateway.scaffold import scaffold_native_mas
        from workflow.user_gateway.validate import compile_project, validate_project

        self._drop_upload("epc-aw-main")
        built = scaffold_native_mas("epc-aw-main", title="EPC-AW-main")
        ids = [a["id"] for a in built["workflow"]["agents"]]
        self.assertEqual(ids, ["planner", "u_epc-aw-main__executor", "verifier"])
        self.assertNotIn("u_epc-aw-main__solver", ids)
        wraps = [a.get("meta", {}).get("wraps") for a in built["workflow"]["agents"]]
        self.assertEqual(wraps, ["EPC-AW.Planner", "EPC-AW.Executor", "EPC-AW.Diagnoser"])
        report = validate_project("epc-aw-main")
        self.assertTrue(report["ok"], report)
        spec, compiled = compile_project("epc-aw-main")
        self.assertTrue(compiled.ok, compiled.reason)
        self.assertTrue(spec.is_executable()[0])

    def test_apply_upgrades_generic_stub(self) -> None:
        from science_infra.control.user_projects import apply_project
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.scaffold import scaffold_native_mas

        create_project("plain", title="generic", mode="native_mas")
        generic = scaffold_native_mas("plain", title="generic")
        self.assertIn("u_plain__solver", [a["id"] for a in generic["workflow"]["agents"]])

        self._drop_upload("epc-aw-main")
        out = apply_project("epc-aw-main", {"schema_version": "0.3", "agents": [{"id": "only"}]}, replace=True)
        ids = [a["id"] for a in out["workflow"]["agents"]]
        self.assertEqual(ids, ["planner", "u_epc-aw-main__executor", "verifier"])
        self.assertEqual(out["project"].get("wraps"), "EPC-AW.PEV")

    def test_fake_windows(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx_mod = sys.modules.get("context")
        self.assertIsNotNone(ctx_mod)
        ctx = ctx_mod.get_episode_context(reset=True)
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "1+1"}})
        self.assertEqual(plan["kind"], "plan_step")
        self.assertEqual(plan["payload"]["next"], "u_epc-aw-main__executor")
        self.assertFalse(plan["payload"]["done"])
        self.assertIn("Python_Coder_Tool", str(plan["payload"]["trace"].get("tool")))
        tool = ctx.exec_window({"kind": "tool_invoke", "dst": "u_epc-aw-main__executor", "payload": plan["payload"]["args"]})
        self.assertEqual(tool["kind"], "tool_result")
        self.assertIn("command", tool["payload"]["trace"])
        self.assertNotIn("No module named", tool["payload"]["output"])
        verify = ctx.verify_window({"kind": "verify", "dst": "verifier", "payload": {}})
        self.assertEqual(verify["kind"], "verify")
        self.assertFalse(verify["payload"]["ok"])
        self.assertFalse(verify["payload"].get("ready_to_stop"))
        self.assertIn("outline", verify["payload"]["trace"])
        plan2 = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "1+1"}})
        self.assertFalse(plan2["payload"]["done"])
        tool2 = ctx.exec_window({"kind": "tool_invoke", "dst": "u_epc-aw-main__executor", "payload": plan2["payload"]["args"]})
        self.assertTrue(tool2["payload"]["ok"])
        verify2 = ctx.verify_window({"kind": "verify", "dst": "verifier", "payload": {}})
        self.assertTrue(verify2["payload"].get("ready_to_stop"))
        self.assertEqual(verify2["payload"].get("answer"), "2")
        with self.assertRaises(RuntimeError):
            ctx.solver.solve("1+1")

    def test_illegal_tool_skips_import(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx = sys.modules["context"].get_episode_context(reset=True)
        tool = ctx.exec_window({
            "kind": "tool_invoke",
            "dst": "u_epc-aw-main__executor",
            "payload": {"question": "1+1", "tool_name": "No matched tool given: ", "sub_goal": "x"},
        })
        self.assertFalse(tool["payload"]["ok"])
        self.assertNotIn("tools.no matched", tool["payload"]["output"].lower())
        self.assertEqual(getattr(ctx.executor, "executed", []), [])

    def test_bts_branch_when_n_is_2(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        os.environ["SCIENCE_EPC_AW_N"] = "2"
        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx = sys.modules["context"].get_episode_context(reset=True)
        if hasattr(ctx.planner, "n"):
            ctx.planner.n = 2
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "1+1"}})
        bts = plan["payload"]["trace"].get("bts") or {}
        self.assertGreaterEqual(int(bts.get("n") or 0), 2)
        self.assertIn(str(bts.get("selected")), {"0", "1"})

    def test_engine_maps_qwen_to_server_model(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx_mod = sys.modules["context"]
        old_model = os.environ.get("MODEL")
        old_server = os.environ.get("SERVER_MODEL")
        old_asst = os.environ.get("AI_ASSISTANT_MODEL")
        try:
            os.environ["MODEL"] = "Qwen3-4B"
            os.environ["AI_ASSISTANT_MODEL"] = "Qwen3-4B"
            os.environ.pop("SERVER_MODEL", None)
            ctx_mod._map_llm_env()
            self.assertIn("Qwen3-4B", os.environ.get("SERVER_MODEL", ""))
            self.assertTrue(ctx_mod._is_llm_error({"error": "UnboundLocalError", "message": "response"}))
            merged = ctx_mod._merge_no_think({})
            self.assertEqual(merged["extra_body"]["enable_thinking"], False)
            self.assertEqual(merged["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        finally:
            if old_model is None:
                os.environ.pop("MODEL", None)
            else:
                os.environ["MODEL"] = old_model
            if old_server is None:
                os.environ.pop("SERVER_MODEL", None)
            else:
                os.environ["SERVER_MODEL"] = old_server
            if old_asst is None:
                os.environ.pop("AI_ASSISTANT_MODEL", None)
            else:
                os.environ["AI_ASSISTANT_MODEL"] = old_asst


if __name__ == "__main__":
    unittest.main()
