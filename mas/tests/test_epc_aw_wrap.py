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
        labels = {a["id"]: a.get("label") for a in built["workflow"]["agents"]}
        self.assertEqual(labels["planner"], "epc_aw_planner")
        self.assertEqual(labels["verifier"], "epc_aw_verifier")
        self.assertEqual(labels["u_epc-aw-main__executor"], "epc_aw_executor")
        wraps = [a.get("meta", {}).get("wraps") for a in built["workflow"]["agents"]]
        self.assertEqual(wraps, ["EPC-AW.Planner", "EPC-AW.Executor", "EPC-AW.Diagnoser"])
        report = validate_project("epc-aw-main")
        self.assertTrue(report["ok"], report)
        spec, compiled = compile_project("epc-aw-main")
        self.assertTrue(compiled.ok, compiled.reason)
        self.assertTrue(spec.is_executable()[0])
        self.assertEqual(list(spec.routers or []), [])
        self.assertNotIn("_implicit_route", compiled.routers)
        executor_id = "u_epc-aw-main__executor"
        self.assertEqual(compiled.message_out.get("planner"), executor_id)
        site_ids = [s.anchor.agent_id for s in (spec.sampling.sites or [])]
        self.assertEqual(site_ids, ["planner", executor_id, "verifier"])
        kinds = [a.kind for a in spec.agents if a.id == executor_id]
        self.assertEqual(kinds, ["blank"])

    def test_direct_executor_is_a_sampling_window(self) -> None:
        from workflow.archive import Archive, register_archive
        from workflow.centralized_runtime import MockWindowLLM, run_centralized_episode
        from workflow.memory import MemoryStore
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module
        from workflow.user_gateway.validate import compile_project

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        spec, compiled = compile_project("epc-aw-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        raw = run_centralized_episode(
            {"id": "epc-direct", "question": "1+1"},
            None,
            register_archive(Archive()),
            spec,
            MemoryStore(),
            compiled,
            window_llm=MockWindowLLM({}),
        )
        self.assertIsNone(raw.error, raw.error)
        events = list(raw.window_events or [])
        executor_id = "u_epc-aw-main__executor"
        self.assertTrue(
            any(e.get("agent_id") == "planner" and e.get("kind") == "after_agent_turn" for e in events),
            events,
        )
        self.assertTrue(
            any(e.get("agent_id") == executor_id and e.get("kind") == "after_agent_turn" for e in events),
            events,
        )
        self.assertTrue(
            any(e.get("agent_id") == "verifier" and e.get("kind") == "after_verifier" for e in events),
            events,
        )
        self.assertFalse(any(e.get("agent_id") == "route_exec" for e in events), events)
        from workflow.runtime import run_compiled_episode

        routed = run_compiled_episode(
            {"id": "epc-direct", "question": "1+1"},
            None,
            register_archive(Archive()),
            spec,
            MemoryStore(),
            runner=None,  # type: ignore[arg-type]
            window_llm=MockWindowLLM({}),
        )
        self.assertIsNone(routed.error, routed.error)
        routed_ids = {e.get("agent_id") for e in (routed.window_events or [])}
        self.assertIn(executor_id, routed_ids)
        self.assertNotIn("route_exec", routed_ids)

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

    def test_runtime_yaml_written_and_max_steps_applied(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module
        from workflow.user_gateway.paths import project_dir
        from science_infra.control.user_projects import get_project_runtime, put_project_runtime

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        runtime_path = project_dir("epc-aw-main") / "contracts" / "runtime.yaml"
        self.assertTrue(runtime_path.is_file())
        saved = put_project_runtime(
            "epc-aw-main",
            {"n": 3, "max_steps": 7, "max_time": 100, "max_tokens": 512, "temperature": 0.2},
        )
        self.assertEqual(saved["n"], 3)
        self.assertEqual(saved["max_steps"], 7)
        self.assertNotIn("Google_Search_Tool", saved["enabled_tools"])
        self.assertIn("Bing_Search_Tool", saved["enabled_tools"])
        self.assertIn("Web_Fetch_Tool", saved["enabled_tools"])
        loaded = get_project_runtime("epc-aw-main")
        self.assertEqual(loaded["max_steps"], 7)
        os.environ.pop("SCIENCE_EPC_AW_N", None)
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx = sys.modules["context"].get_episode_context(reset=True)
        self.assertEqual(ctx.max_steps, 7)
        solver = __import__("types").SimpleNamespace(
            planner=__import__("types").SimpleNamespace(
                available_tools=["Wikipedia_Search_Tool"],
                toolbox_metadata={},
            )
        )
        sys.modules["context"]._register_dispatch_tools(
            solver, ["Wikipedia_Search_Tool", "Bing_Search_Tool"],
        )
        self.assertIn("Bing_Search_Tool", solver.planner.available_tools)
        self.assertNotIn("Google_Search_Tool", solver.planner.available_tools)
        self.assertNotIn("Web_Fetch_Tool", solver.planner.available_tools)
        self.assertNotIn("Web_Search_Tool", solver.planner.available_tools)
        self.assertEqual(sys.modules["context"]._plan_n(), 3)

    def test_failed_tool_is_not_rewritten_to_base_generator(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx = sys.modules["context"].get_episode_context(reset=True)
        from types import SimpleNamespace

        def _one(*_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(context="c", sub_goal="s", tool_name="Python_Coder_Tool"), ""

        ctx.planner.generate_next_step = _one  # type: ignore[method-assign]
        ctx.failed_tools = ["Python_Coder_Tool"]
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "1+1"}})
        tool_name = plan["payload"]["args"]["tool_name"]
        self.assertEqual(tool_name, "Python_Coder_Tool")
        self.assertNotIn("failed_tool_skip", plan["payload"].get("trace") or {})

    def test_transport_failure_keeps_subgoal_and_switches_search_tool(self) -> None:
        from types import SimpleNamespace

        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx = sys.modules["context"].get_episode_context(reset=True)
        question = "Who was president when Citibank was founded?"
        ctx._active_question = question
        ctx.question = question
        ctx.last_tool_output = (
            "Wikipedia unreachable: HTTPSConnectionPool(host='en.wikipedia.org', port=443): Max retries exceeded"
        )
        ctx.last_sub_goal = "Find the founding year of Citibank"
        ctx.last_tool_name = "Wikipedia_Search_Tool"
        ctx.failed_tools = ["Wikipedia_Search_Tool"]
        ctx.planner.available_tools = [
            "Wikipedia_Search_Tool",
            "Bing_Search_Tool",
            "Web_Search_Tool",
            "Base_Generator_Tool",
            "Python_Coder_Tool",
        ]

        def _drift(*_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(
                context="c",
                sub_goal="Find the launch date",
                tool_name="Base_Generator_Tool",
            ), ""

        ctx.planner.generate_next_step = _drift  # type: ignore[method-assign]
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": question}})
        args = plan["payload"]["args"]
        self.assertEqual(args["tool_name"], "Bing_Search_Tool")
        self.assertEqual(args["sub_goal"], "Find the founding year of Citibank")
        called = {"n": 0}

        def _should_not_verify(*_args: object, **_kwargs: object) -> object:
            called["n"] += 1
            return ("direct", True, "1812", "STOP")

        ctx.diagnoser.verificate_context = _should_not_verify  # type: ignore[method-assign]
        ctx.last_tool_output = ctx.last_tool_output
        verify = ctx.verify_window({"kind": "verify", "dst": "verifier", "payload": {}})
        self.assertEqual(called["n"], 0)
        self.assertFalse(verify["payload"]["ready_to_stop"])
        self.assertEqual(verify["payload"]["trace"].get("conclusion"), "CONTINUE")

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

    def test_diagnosis_receives_dual_plans(self) -> None:
        from types import SimpleNamespace
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx = sys.modules["context"].get_episode_context(reset=True)
        planner_plan = SimpleNamespace(context="p", sub_goal="planner-goal", tool_name="Python_Coder_Tool")
        bts_plan = SimpleNamespace(context="b", sub_goal="bts-goal", tool_name="Base_Generator_Tool")
        ctx.planner_selected_index = "0"
        ctx.bts_selected_index = "1"
        ctx.planner_selected_plan = planner_plan
        ctx.bts_selected_plan = bts_plan
        ctx.last_plan = bts_plan
        ctx.last_tool_output = "mock-out"
        ctx.last_sub_goal = "target"
        ctx.step_count = 1
        ctx.system_memory.set_outline({"1": "target", "2": "later"})
        calls: list = []

        def _capture(*args, **kwargs):
            calls.append(args)
            return "diagnosis", "constraint"

        ctx.diagnoser.epc_aw_diagnosis = _capture
        verify = ctx.verify_window({"kind": "verify", "dst": "verifier", "payload": {}})
        self.assertFalse(verify["payload"].get("ready_to_stop"))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], planner_plan)
        self.assertIs(calls[0][1], bts_plan)

    def test_outline_prefers_key_one(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx_mod = sys.modules["context"]
        memory = type("M", (), {})()
        memory.get_outline = lambda: {"10": "later step", "1": "first target", "2": "second"}
        self.assertEqual(ctx_mod._first_outline_target(memory), "first target")

    def test_reset_between_questions(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        ctx = sys.modules["context"].get_episode_context(reset=True)
        ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "q1"}})
        ctx.system_memory.add_obtained_information("leak")
        self.assertTrue(ctx._analyzed)
        self.assertGreater(ctx.step_count, 0)
        ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "q2-different"}})
        self.assertEqual(ctx.step_count, 1)
        self.assertNotIn("leak", ctx.system_memory.get_obtained_information())

    def test_threadsafe_timeout_patch_off_main_thread(self) -> None:
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        # Point upload MAS to the real project upload so Executor imports succeed.
        root = ROOT / "user_space" / "projects" / "epc-aw-main" / "upload" / "EPC-AW-main"
        upload = self.user_root / "projects" / "epc-aw-main" / "upload" / "EPC-AW-main"
        if upload.exists():
            import shutil

            shutil.rmtree(upload)
        upload.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(root, upload)
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        old_mock = os.environ.get("SCIENCE_EPC_AW_MOCK")
        os.environ["SCIENCE_EPC_AW_MOCK"] = "0"
        os.environ["SCIENCE_EPC_AW_REQUIRE_REAL"] = ""
        try:
            load_entry_module("epc-aw-main", reload=True)
            ctx_mod = sys.modules["context"]
            ctx_mod._ensure_import_paths(root)
            ctx_mod._install_threadsafe_timeouts()
            from MAS.epc_aw.models.executor import Executor  # type: ignore

            self.assertTrue(getattr(Executor.execute_tool_command, "_science_threadsafe", False))
            executor = Executor(llm_engine_name="gpt-4o", root_cache_dir=str(Path(self._tmp.name) / "cache"), verbose=False)

            def run_off_main() -> str:
                self.assertIsNot(threading.current_thread(), threading.main_thread())
                return str(executor.execute_tool_command("Python_Coder_Tool", "# no tool.execute() block"))

            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(run_off_main).result(timeout=30)
            self.assertNotIn("signal only works in main thread", result.lower())
        finally:
            if old_mock is None:
                os.environ.pop("SCIENCE_EPC_AW_MOCK", None)
            else:
                os.environ["SCIENCE_EPC_AW_MOCK"] = old_mock

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
            # messages-level soft switch survives LiteLLM stripping extra_body
            merged2 = ctx_mod._merge_no_think(
                {"messages": [{"role": "user", "content": "What is 1+1?"}]}
            )
            self.assertTrue(
                merged2["messages"][0]["content"].rstrip().endswith("/no_think")
            )
            # idempotent: no double suffix
            merged3 = ctx_mod._merge_no_think(merged2)
            self.assertEqual(
                merged3["messages"][0]["content"].count("/no_think"), 1
            )
            # multimodal content list also gets the suffix on its text part
            merged4 = ctx_mod._merge_no_think(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "describe"}],
                        }
                    ]
                }
            )
            self.assertTrue(
                merged4["messages"][0]["content"][0]["text"].rstrip().endswith("/no_think")
            )
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


class TestEpcAwSearchFetchGate(EpcAwWrapTestBase):
    """Fix D: after a search tool returns a SERP list, the next plan is forced
    to Web_Fetch_Tool on the first SERP URL — the planner LLM is bypassed."""

    def _make_ctx(self):
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        return sys.modules["context"].get_episode_context(reset=True)

    def test_serp_first_url_extracts_first_entry(self) -> None:
        ctx_mod = sys.modules["context"]
        serp = (
            "bing results for: Citibank founded\n"
            "1. Citibank - Wikipedia\n"
            "   URL: https://en.wikipedia.org/wiki/Citibank\n"
            "2. History\n"
            "   URL: https://www.citigroup.com/history"
        )
        self.assertEqual(
            ctx_mod._serp_first_url(serp),
            "https://en.wikipedia.org/wiki/Citibank",
        )
        self.assertEqual(ctx_mod._serp_first_url("no urls here"), "")
        self.assertEqual(ctx_mod._serp_first_url(""), "")

    def test_search_output_forces_web_fetch_next_step(self) -> None:
        from types import SimpleNamespace

        ctx = self._make_ctx()
        question = "Who was president when Citibank was founded?"
        ctx._active_question = question
        ctx.question = question
        ctx.step_count = 1
        ctx.last_tool_name = "Bing_Search_Tool"
        ctx.last_tool_output = (
            "bing results for: Citibank founded\n"
            "1. Citibank - Wikipedia\n"
            "   URL: https://en.wikipedia.org/wiki/Citibank"
        )
        ctx.planner.available_tools = [
            "Python_Coder_Tool",
            "Base_Generator_Tool",
            "Wikipedia_Search_Tool",
            "Bing_Search_Tool",
            "Web_Fetch_Tool",
        ]
        called = {"n": 0}

        def _must_not_run(*_args: object, **_kwargs: object) -> object:
            called["n"] += 1
            return SimpleNamespace(context="c", sub_goal="s", tool_name="Base_Generator_Tool"), ""

        ctx.planner.generate_next_step = _must_not_run  # type: ignore[method-assign]
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": question}})
        args = plan["payload"]["args"]
        self.assertEqual(args["tool_name"], "Web_Fetch_Tool")
        self.assertEqual(args["url"], "https://en.wikipedia.org/wiki/Citibank")
        self.assertIn("Fetch https://en.wikipedia.org/wiki/Citibank", args["sub_goal"])
        self.assertEqual(called["n"], 0)  # planner LLM bypassed
        trace = plan["payload"]["trace"]
        self.assertEqual(trace.get("tool"), "Web_Fetch_Tool")
        self.assertIn("forced_fetch", trace)
        self.assertEqual(trace["forced_fetch"]["url"], "https://en.wikipedia.org/wiki/Citibank")
        # llm_calls key exists even when no engine fired (Fix E)
        self.assertIn("llm_calls", trace)
        self.assertEqual(trace["llm_calls"]["count"], 0)

    def test_fetched_page_does_not_force_again(self) -> None:
        ctx = self._make_ctx()
        question = "Who was president when Citibank was founded?"
        ctx._active_question = question
        ctx.question = question
        ctx.step_count = 2
        ctx.last_tool_name = "Web_Fetch_Tool"
        ctx.last_tool_output = "Citibank was founded in 1812. James Madison was president."
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": question}})
        self.assertNotIn("forced_fetch", plan["payload"]["trace"])

    def test_serp_output_allows_free_choice_after_fetch_rejected(self) -> None:
        ctx = self._make_ctx()
        question = "Who was president when Citibank was founded?"
        ctx._active_question = question
        ctx.question = question
        ctx.step_count = 2
        ctx.last_tool_name = "Bing_Search_Tool"
        ctx.last_tool_output = (
            "wikipedia results for: Citibank\n"
            "1. Citibank\n"
            "   URL: https://en.wikipedia.org/wiki/Citibank"
        )
        # Web_Fetch already failed twice → state machine stands down
        ctx.failed_tools = ["Web_Fetch_Tool"]
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": question}})
        self.assertNotIn("forced_fetch", plan["payload"]["trace"])


class TestEpcAwLlmCallTrace(EpcAwWrapTestBase):
    """Fix E: every window's trace carries llm_calls with prompt/model/n."""

    def _make_ctx(self):
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        return sys.modules["context"]

    def test_engine_recorder_logs_generate_text(self) -> None:
        ctx_mod = self._make_ctx()

        class FakeEngine:
            model_string = "test-model"

            def _generate_text(self, prompt, system_prompt=None, max_tokens=2048, top_p=0.99, n=1, response_format=None):
                return "ok response" if n == 1 else ["a", "b"]

        eng = FakeEngine()
        ctx_mod._wrap_engine_llm_recorder(eng)
        out = eng._generate_text("What is 1+1?", n=1)
        self.assertEqual(out, "ok response")
        calls = ctx_mod.drain_llm_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["model"], "test-model")
        self.assertEqual(calls[0]["n"], 1)
        self.assertEqual(calls[0]["prompt_head"], "What is 1+1?")
        self.assertEqual(calls[0]["response_head"], "ok response")
        # list-of-candidates responses record only the first head
        eng2 = FakeEngine()
        ctx_mod._wrap_engine_llm_recorder(eng2)
        eng2._generate_text("pick", n=2)
        calls2 = ctx_mod.drain_llm_calls()
        self.assertEqual(calls2[0]["n"], 2)
        self.assertEqual(calls2[0]["response_head"], "a")
        # exceptions are logged and re-raised
        class BoomEngine:
            model_string = "boom"

            def _generate_text(self, prompt, **kwargs):
                raise RuntimeError("endpoint down")

        eng3 = BoomEngine()
        ctx_mod._wrap_engine_llm_recorder(eng3)
        with self.assertRaises(RuntimeError):
            eng3._generate_text("q")
        calls3 = ctx_mod.drain_llm_calls()
        self.assertIn("RuntimeError", calls3[0]["response_head"])

    def test_window_trace_carries_llm_calls(self) -> None:
        ctx_mod = self._make_ctx()
        ctx = ctx_mod.get_episode_context(reset=True)
        ctx_mod.record_llm_call(
            {"model": "m", "n": 1, "prompt_head": "p" * 600, "response_head": "r"}
        )
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "1+1"}})
        llm = plan["payload"]["trace"]["llm_calls"]
        self.assertEqual(llm["count"], 1)
        self.assertEqual(len(llm["calls"][0]["prompt_head"]), 500)  # head clamped
        self.assertEqual(llm["calls"][0]["model"], "m")
        # drained: next window starts from zero
        tool = ctx.exec_window({
            "kind": "tool_invoke",
            "dst": "u_epc-aw-main__executor",
            "payload": plan["payload"]["args"],
        })
        self.assertEqual(tool["payload"]["trace"]["llm_calls"]["count"], 0)

    def test_runtime_defaults_match_source_main(self) -> None:
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.paths import project_dir

        self._drop_upload("epc-aw-main")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        import yaml

        runtime = yaml.safe_load(
            (project_dir("epc-aw-main") / "contracts" / "runtime.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(runtime["temperature"], 0.7)
        self.assertEqual(runtime["n"], 1)
        self.assertEqual(runtime["max_tokens"], 4000)


if __name__ == "__main__":
    unittest.main()
