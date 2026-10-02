"""EPC-AW executor tool smoke + completion budget guards."""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

ALL_TOOLS = [
    "Base_Generator_Tool",
    "Python_Coder_Tool",
    "Wikipedia_Search_Tool",
    "Web_Search_Tool",
    "Google_Search_Tool",
]


class TestEpcAwCompletionBudget(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        self._old = {k: os.environ.get(k) for k in (
            "SCIENCE_USER_SPACE_DIR", "SCIENCE_EPC_AW_MOCK", "SCIENCE_EPC_AW_MAX_MODEL_LEN",
            "SCIENCE_EPC_AW_MAX_COMPLETION", "MAS_MAX_COMPLETION_TOKENS",
        )}
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)
        os.environ["SCIENCE_EPC_AW_MOCK"] = "1"
        os.environ["SCIENCE_EPC_AW_MAX_MODEL_LEN"] = "8192"
        os.environ["SCIENCE_EPC_AW_MAX_COMPLETION"] = "2048"

    def tearDown(self) -> None:
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _load_context(self):
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.paths import ensure_project_layout

        create_project("epc-aw-main", title="EPC-AW-main", mode="native_mas")
        root = ensure_project_layout("epc-aw-main")
        solver = root / "upload" / "EPC-AW-main" / "MAS" / "epc_aw" / "solver.py"
        solver.parent.mkdir(parents=True, exist_ok=True)
        solver.write_text("# marker\n", encoding="utf-8")
        (solver.parent / "models").mkdir(parents=True, exist_ok=True)
        for name in ("planner.py", "executor.py", "diagnoser.py"):
            (solver.parent / "models" / name).write_text("# stub\n", encoding="utf-8")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        return sys.modules["context"]

    def test_scaffold_labels_are_abstract(self) -> None:
        from workflow.user_gateway.epc_aw import epc_aw_workflow

        wf = epc_aw_workflow("epc-aw-main")
        labels = {a["id"]: a.get("label") for a in wf["agents"]}
        self.assertEqual(labels["planner"], "epc_aw_planner")
        self.assertEqual(labels["verifier"], "epc_aw_verifier")
        self.assertIn("epc_aw_executor", labels.values())

    def test_clamp_keeps_prompt_plus_completion_under_model_len(self) -> None:
        ctx = self._load_context()
        prompt = "x" * 12000  # ~4000 tokens by rough estimator
        capped = ctx._clamp_max_tokens(16384, prompt)
        prompt_tokens = ctx._estimate_prompt_tokens(prompt)
        self.assertLessEqual(prompt_tokens + capped, 8192)
        self.assertLessEqual(capped, 2048)

    def test_llm_error_plan_does_not_emit_no_matched_tool(self) -> None:
        ctx_mod = self._load_context()
        ctx = ctx_mod.get_episode_context(reset=True)
        ctx.last_trace = {"llm_error": {"error": "BadRequestError", "message": "context length"}}
        ctx.question = "1+1"
        ctx.system_memory.set_outline({"1": "compute"})
        # Force generate_next_step to return an error dict.
        ctx.planner.generate_next_step = lambda *a, **k: ({"error": "BadRequestError", "message": "overflow"}, "")
        plan = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "1+1"}})
        tool = str(plan["payload"]["args"].get("tool_name") or "")
        self.assertFalse(tool.lower().startswith("no matched"), tool)
        self.assertEqual(tool, "Base_Generator_Tool")


class TestEpcAwToolSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = ROOT / "ref_Rep" / "EPC-AW"
        if not (cls.root / "MAS" / "epc_aw").is_dir():
            raise unittest.SkipTest("ref_Rep/EPC-AW missing")

    def setUp(self) -> None:
        self._old = {k: os.environ.get(k) for k in (
            "SCIENCE_EPC_AW_MOCK", "MODEL_Name", "MODEL_NAME", "OPENAI_API_KEY",
            "OPENAI_API_BASE", "OPENAI_BASE_URL",
        )}
        os.environ["SCIENCE_EPC_AW_MOCK"] = "0"
        os.environ.setdefault("MODEL_Name", "Qwen3-4B")
        os.environ.setdefault("MODEL_NAME", os.environ["MODEL_Name"])
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        os.environ.setdefault("OPENAI_API_BASE", "http://127.0.0.1:8000/v1")
        os.environ.setdefault("OPENAI_BASE_URL", os.environ["OPENAI_API_BASE"])

    def tearDown(self) -> None:
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_execute_tool_command_smoke_for_loaded_tools(self) -> None:
        adapted = ROOT / "user_space" / "projects" / "epc-aw-main" / "adapted" / "context.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("epc_aw_adapted_context", adapted)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        root = mod._import_root()
        self.assertIsNotNone(root)
        mod._ensure_import_paths(root)
        mod._map_llm_env()
        os.environ.setdefault("SCIENCE_EPC_AW_MAX_MODEL_LEN", "8192")
        os.environ.setdefault("SCIENCE_EPC_AW_MAX_COMPLETION", "2048")
        with mod._epc_tools_scope(root):
            mod._install_threadsafe_timeouts()
            mod._patch_google_search_model_string()
            mod._guard_wikipedia_sys_exit()
            from MAS.epc_aw.models.executor import (  # type: ignore
                Executor,
                TOOL_NAME_MAPPING_LONG,
            )
            from MAS.epc_aw.models.initializer import Initializer  # type: ignore
            import importlib

            init = Initializer(
                enabled_tools=ALL_TOOLS,
                tool_engine=["Default"] * len(ALL_TOOLS),
                model_string=os.environ.get("MODEL_Name") or "Qwen3-4B",
                verbose=False,
            )
            available = list(init.available_tools or [])
            skipped = [name for name in ALL_TOOLS if name not in available]
            cache = Path(tempfile.mkdtemp(prefix="epc_tool_smoke_"))
            executor = Executor(
                llm_engine_name=os.environ.get("MODEL_Name") or "Qwen3-4B",
                root_cache_dir=str(cache),
                verbose=False,
                max_time=30,
            )
            executor.set_query_cache_dir(str(cache))
            self.assertTrue(getattr(Executor.execute_tool_command, "_science_threadsafe", False))

            # Patch each loaded tool's execute to a sentinel so we validate the
            # Executor import/dispatch path without requiring live LLM/network.
            patched = []
            for name in available:
                meta = TOOL_NAME_MAPPING_LONG.get(name) or {}
                dir_name = meta.get("dir_name") or name.lower().replace("_tool", "")
                class_name = meta.get("class_name") or name
                module = importlib.import_module(f"tools.{dir_name}.tool")
                tool_cls = getattr(module, class_name)
                orig = tool_cls.execute

                def _smoke(self, *args, _name=name, **kwargs):
                    return f"smoke-ok:{_name}"

                tool_cls.execute = _smoke
                patched.append((tool_cls, orig))

            try:
                for name in ALL_TOOLS:
                    if name not in available:
                        continue
                    cmd = f'execution = tool.execute(query="smoke")'

                    def run(tool_name: str = name, command: str = cmd) -> str:
                        self.assertIsNot(threading.current_thread(), threading.main_thread())
                        return str(executor.execute_tool_command(tool_name, command))

                    with ThreadPoolExecutor(max_workers=1) as pool:
                        text = pool.submit(run).result(timeout=60)
                    low = text.lower()
                    self.assertNotIn("signal only works in main thread", low, msg=f"{name}: {text[:300]}")
                    self.assertNotIn("no module named", low, msg=f"{name}: {text[:300]}")
                    self.assertIn(f"smoke-ok:{name}", text, msg=f"{name}: {text[:300]}")
            finally:
                for tool_cls, orig in patched:
                    tool_cls.execute = orig

            print("EPC-AW tool smoke available=", available)
            print("EPC-AW tool smoke skipped=", skipped)
            # Base + Python should load without browser deps.
            for must in ("Base_Generator_Tool", "Python_Coder_Tool"):
                self.assertIn(must, available, msg=f"{must} missing; skipped={skipped}")

    def test_python_coder_snippet_off_main_thread(self) -> None:
        """Fresh tools.* import inside the EPC-AW scope must not call SIGALRM."""
        adapted = ROOT / "user_space" / "projects" / "epc-aw-main" / "adapted" / "context.py"
        import importlib.util

        spec = importlib.util.spec_from_file_location("epc_aw_python_timeout_ctx", adapted)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        root = mod._import_root()
        mod._ensure_import_paths(root)
        self.assertNotIn("Execution timed out", mod._INFRA_TOOL_ERRORS)
        self.assertNotIn("Execution timed out after", mod._INFRA_TOOL_ERRORS)
        # Patch once on whatever tools package is visible now. The eval scope
        # then drops tools.* and imports a fresh copy; that copy must be patched
        # again inside execute_tool_command.
        mod._install_threadsafe_timeouts()
        from MAS.epc_aw.models.executor import Executor  # type: ignore

        with mod._epc_tools_scope(root):
            import importlib

            pct = importlib.import_module("tools.python_coder.tool")
            self.assertFalse(getattr(pct.timeout, "_science_threadsafe", False))

            def execute(self, query):  # type: ignore[no-redef]
                return self.execute_code_snippet("```python\nprint(4 * 12)\n```")

            pct.Python_Coder_Tool.execute = execute  # type: ignore[method-assign]
            executor = Executor(
                llm_engine_name=os.environ.get("MODEL_Name") or "Qwen3-4B",
                root_cache_dir=str(Path(tempfile.mkdtemp(prefix="epc_py_"))),
                verbose=False,
                max_time=30,
            )
            executor.set_query_cache_dir(executor.root_cache_dir)

            def run() -> str:
                self.assertIsNot(threading.current_thread(), threading.main_thread())
                return str(executor.execute_tool_command(
                    "Python_Coder_Tool",
                    'execution = tool.execute(query="4*12")',
                ))

            with ThreadPoolExecutor(max_workers=1) as pool:
                text = pool.submit(run).result(timeout=60)
            self.assertNotIn("signal only works", text.lower())
            self.assertIn("48", text)
            self.assertTrue(getattr(pct.timeout, "_science_threadsafe", False))


class TestEpcAwChromeDemote(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        self._old = {k: os.environ.get(k) for k in (
            "SCIENCE_USER_SPACE_DIR", "SCIENCE_EPC_AW_MOCK", "CHROME_BINARY", "GOOGLE_CHROME_BIN",
        )}
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)
        os.environ["SCIENCE_EPC_AW_MOCK"] = "1"
        os.environ.pop("CHROME_BINARY", None)
        os.environ.pop("GOOGLE_CHROME_BIN", None)

    def tearDown(self) -> None:
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        self._tmp.cleanup()

    def _load_context(self):
        from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev
        from workflow.user_gateway.loader import load_entry_module
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.paths import ensure_project_layout

        create_project("epc-aw-main", title="EPC-AW-main", mode="native_mas")
        root = ensure_project_layout("epc-aw-main")
        solver = root / "upload" / "EPC-AW-main" / "MAS" / "epc_aw" / "solver.py"
        solver.parent.mkdir(parents=True, exist_ok=True)
        solver.write_text("# marker\n", encoding="utf-8")
        (solver.parent / "models").mkdir(parents=True, exist_ok=True)
        for name in ("planner.py", "executor.py", "diagnoser.py"):
            (solver.parent / "models" / name).write_text("# stub\n", encoding="utf-8")
        scaffold_epc_aw_pev("epc-aw-main", title="EPC-AW-main")
        sys.modules.pop("context", None)
        load_entry_module("epc-aw-main", reload=True)
        return sys.modules["context"]

    def test_demote_google_when_no_chrome(self) -> None:
        ctx = self._load_context()
        # Force finder to miss even if host has chrome.
        ctx._find_chrome_binary = lambda: None  # type: ignore
        tools = ctx._patch_google_chrome_or_demote(list(ALL_TOOLS))
        self.assertNotIn("Google_Search_Tool", tools)
        self.assertIn("Wikipedia_Search_Tool", tools)
        self.assertIn("Base_Generator_Tool", tools)

    def test_keep_google_when_chrome_present(self) -> None:
        ctx = self._load_context()
        fake = Path(self._tmp.name) / "fake-chrome"
        fake.write_text("#!/bin/sh\n", encoding="utf-8")
        fake.chmod(0o755)
        ctx._find_chrome_binary = lambda: str(fake)  # type: ignore
        tools = ctx._patch_google_chrome_or_demote(list(ALL_TOOLS))
        self.assertIn("Google_Search_Tool", tools)

    def test_exec_marks_binary_location_failed(self) -> None:
        ctx_mod = self._load_context()
        ctx = ctx_mod.get_episode_context(reset=True)

        def _boom(tool_name: str, command: str) -> str:
            return "Binary Location Must be a String"

        ctx.executor.execute_tool_command = _boom  # type: ignore
        out = ctx.exec_window({
            "kind": "tool_invoke",
            "dst": "u_epc-aw-main__executor",
            "payload": {
                "question": "q",
                "tool_name": "Python_Coder_Tool",
                "sub_goal": "s",
                "context": "c",
            },
        })
        self.assertFalse(out["payload"]["ok"])
        self.assertIn("Python_Coder_Tool", ctx.failed_tools)

    def test_google_navigate_stops_a_hung_get(self) -> None:
        import time

        ctx = self._load_context()

        class _Driver:
            browser_pid = None
            service = None

        def raw_get(url: str, *_args: object, **_kwargs: object) -> None:
            time.sleep(5)

        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            ctx._google_navigate(_Driver(), raw_get, "https://www.google.com", 0.2, (), {}, "")
        self.assertLess(time.monotonic() - started, 2.0)

    def test_run_isolated_returns_when_eval_cancelled(self) -> None:
        import threading
        import time

        from workflow.user_gateway.sandbox import (
            EvalCancelled,
            bind_eval_cancel,
            reset_eval_cancel,
            run_isolated,
        )

        event = threading.Event()
        token = bind_eval_cancel(event)

        def slow() -> None:
            time.sleep(30)

        threading.Timer(0.15, event.set).start()
        started = time.monotonic()
        try:
            with self.assertRaises(EvalCancelled):
                run_isolated(slow, timeout_s=30)
            self.assertLess(time.monotonic() - started, 3.0)
        finally:
            reset_eval_cancel(token)

    def test_exec_does_not_blacklist_timeout(self) -> None:
        ctx_mod = self._load_context()
        ctx = ctx_mod.get_episode_context(reset=True)

        def _boom(tool_name: str, command: str) -> str:
            return "['Execution timed out after 120 seconds']"

        ctx.executor.execute_tool_command = _boom  # type: ignore
        out = ctx.exec_window({
            "kind": "tool_invoke",
            "dst": "u_epc-aw-main__executor",
            "payload": {
                "question": "q",
                "tool_name": "Python_Coder_Tool",
                "sub_goal": "s",
                "context": "c",
            },
        })
        self.assertTrue(out["payload"]["ok"])
        self.assertNotIn("Python_Coder_Tool", ctx.failed_tools)

    def test_find_playwright_chrome(self) -> None:
        ctx = self._load_context()
        found = ctx._find_chrome_binary()
        self.assertTrue(found and Path(found).is_file(), msg=str(found))
        self.assertIn("chrome", Path(found).name)


class TestFastSearchTools(unittest.TestCase):
    def _load(self):
        import importlib.util

        path = ROOT / "user_space" / "projects" / "epc-aw-main" / "adapted" / "context.py"
        spec = importlib.util.spec_from_file_location("epc_aw_fast_search_ctx", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)
        return mod

    def test_wikipedia_search_returns_urls_without_opening_pages(self) -> None:
        import requests

        mod = self._load()
        timeouts = []
        urls = []

        class _Resp:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self._payload

        def fake_get(url: str, params=None, timeout=None, headers=None, **_kwargs):
            timeouts.append(timeout)
            urls.append(url)
            return _Resp({"query": {"search": [{
                "title": "Citibank",
                "snippet": "American <span>bank</span> founded in 1812",
            }]}})

        orig = requests.get
        requests.get = fake_get
        try:
            text = mod._wikipedia_hits("Citibank")
        finally:
            requests.get = orig
        self.assertEqual(timeouts, [(3, 8)])
        self.assertEqual(urls, ["https://en.wikipedia.org/w/api.php"])
        self.assertIn("title: Citibank", text)
        self.assertIn("url: https://en.wikipedia.org/wiki/Citibank", text)
        self.assertIn("snippet: American bank founded in 1812", text)
        self.assertNotIn("extracts", text)

    def test_bing_parser_keeps_title_url_snippet(self) -> None:
        mod = self._load()
        html = (
            '<h2 class=""><a href="https://www.citi.com/">Citi</a></h2>'
            '<div class="b_caption"><p>Citibank offers banking.</p></div>'
            '<h2 class=""><a href="https://www.bing.com/ck/a">skip</a></h2>'
            '<div class="b_caption"><p>ignored</p></div>'
        )
        hits = mod._parse_bing_html(html)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["url"], "https://www.citi.com/")
        text = mod._format_hits(hits)
        self.assertIn("title: Citi", text)
        self.assertIn("url: https://www.citi.com/", text)
        self.assertIn("snippet: Citibank offers banking.", text)

    def test_web_fetch_missing_url_skips_http(self) -> None:
        import requests

        mod = self._load()
        called = {"n": 0}

        def fake_get(*_args, **_kwargs):
            called["n"] += 1
            raise AssertionError("missing url must not fetch")

        orig = requests.get
        requests.get = fake_get
        try:
            text = mod._web_fetch("when was citibank founded", "")
        finally:
            requests.get = orig
        self.assertEqual(text, "Web_Fetch_Tool requires a url")
        self.assertEqual(called["n"], 0)

    def test_web_fetch_branches_on_content_type(self) -> None:
        import requests

        mod = self._load()
        pages = {
            "https://example.com/page": ("text/html", 200, b"<p>Founded in 1812.</p><img alt='logo' src='/logo.png'>", None),
            "https://example.com/pic.png": ("image/png", 200, b"\x89PNG", "12"),
            "https://example.com/file.pdf": ("application/pdf", 200, b"not a pdf", None),
            "https://example.com/denied": ("text/html", 403, b"no", None),
        }

        class _Resp:
            def __init__(self, ctype: str, status: int, body: bytes, length):
                self.headers = {"Content-Type": ctype}
                if length:
                    self.headers["Content-Length"] = length
                self.status_code = status
                self.content = body
                self._body = body

            def close(self) -> None:
                return None

        def fake_get(url: str, timeout=None, headers=None, stream=None, **_kwargs):
            self.assertEqual(timeout, (3, 8))
            ctype, status, body, length = pages[url]
            return _Resp(ctype, status, body, length)

        orig = requests.get
        requests.get = fake_get
        try:
            html = mod._web_fetch("year", "https://example.com/page")
            image = mod._web_fetch("", "https://example.com/pic.png")
            pdf = mod._web_fetch("", "https://example.com/file.pdf")
            denied = mod._web_fetch("", "https://example.com/denied")
        finally:
            requests.get = orig
        self.assertIn("Founded in 1812.", html)
        self.assertIn("https://example.com/logo.png", html)
        self.assertIn("alt: logo", html)
        self.assertIn("type: image", image)
        self.assertIn("bytes: 12", image)
        self.assertNotIn("PNG", image)
        self.assertIn("looks like a scan", pdf)
        self.assertIn("HTTP 403", denied)


if __name__ == "__main__":
    unittest.main()
