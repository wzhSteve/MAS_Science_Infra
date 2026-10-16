"""user_projects runtime clamp: drop Google, allow AgentFlow Web_Fetch."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from science_infra.control.user_projects import (  # noqa: E402
    DEFAULT_AGENTFLOW_RUNTIME,
    SHARED_TOOLS,
    _clamp_runtime,
    _REJECTED_TOOLS,
)


class TestRuntimeAllowlist(unittest.TestCase):
    def test_shared_and_agentflow_defaults_include_web_fetch_not_google(self) -> None:
        expected = [
            "Base_Generator_Tool",
            "Python_Coder_Tool",
            "Wikipedia_Search_Tool",
            "Bing_Search_Tool",
            "Web_Fetch_Tool",
        ]
        self.assertEqual(SHARED_TOOLS, expected)
        self.assertEqual(DEFAULT_AGENTFLOW_RUNTIME["enabled_tools"], expected)
        for name in _REJECTED_TOOLS:
            self.assertNotIn(name, SHARED_TOOLS)
            self.assertNotIn(name, DEFAULT_AGENTFLOW_RUNTIME["enabled_tools"])

    def test_clamp_drops_google_keeps_web_fetch_for_agentflow(self) -> None:
        cfg = _clamp_runtime(
            {
                "enabled_tools": [
                    "Base_Generator_Tool",
                    "Google_Search_Tool",
                    "Ground_Google_Search_Tool",
                    "Web_Fetch_Tool",
                ]
            },
            family="agentflow",
        )
        self.assertNotIn("Google_Search_Tool", cfg["enabled_tools"])
        self.assertNotIn("Ground_Google_Search_Tool", cfg["enabled_tools"])
        self.assertIn("Web_Fetch_Tool", cfg["enabled_tools"])
        self.assertNotIn("allowed_tools", cfg)

    def test_clamp_drops_google_for_epc(self) -> None:
        cfg = _clamp_runtime(
            {
                "enabled_tools": [
                    "Bing_Search_Tool",
                    "Google_Search_Tool",
                    "Web_Fetch_Tool",
                ]
            },
            family="epc_aw",
        )
        self.assertNotIn("Google_Search_Tool", cfg["enabled_tools"])
        self.assertIn("Web_Fetch_Tool", cfg["enabled_tools"])
        self.assertNotIn("allowed_tools", cfg)

    def test_clamp_agentflow_rejects_only_google(self) -> None:
        with self.assertRaises(ValueError):
            _clamp_runtime({"enabled_tools": ["Google_Search_Tool"]}, family="agentflow")


class TestRuntimeAllowedToolsOverlay(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        import os

        self._old = os.environ.get("SCIENCE_USER_SPACE_DIR")
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)
        from workflow.user_gateway.registry import create_project

        create_project("agentflow", title="AgentFlow wrap", mode="io_module")

    def tearDown(self) -> None:
        import os

        if self._old is None:
            os.environ.pop("SCIENCE_USER_SPACE_DIR", None)
        else:
            os.environ["SCIENCE_USER_SPACE_DIR"] = self._old
        self._tmp.cleanup()

    def test_get_returns_allowed_tools_without_writing_yaml(self) -> None:
        from science_infra.control.user_projects import get_project_runtime, _runtime_path

        path = _runtime_path("agentflow")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(
                {
                    "n": 1,
                    "max_steps": 10,
                    "max_time": 300,
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "memory_result_chars": 0,
                    "enabled_tools": ["Base_Generator_Tool", "Web_Fetch_Tool"],
                },
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        before = path.read_text(encoding="utf-8")
        data = get_project_runtime("agentflow")
        self.assertIn("Web_Fetch_Tool", data["allowed_tools"])
        self.assertNotIn("Google_Search_Tool", data["allowed_tools"])
        self.assertIn("Web_Fetch_Tool", data["enabled_tools"])
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        self.assertNotIn("allowed_tools", raw)
        self.assertEqual(path.read_text(encoding="utf-8"), before)

    def test_put_persists_web_fetch_without_writing_allowed_tools(self) -> None:
        from science_infra.control.user_projects import put_project_runtime, _runtime_path

        saved = put_project_runtime(
            "agentflow",
            {
                "enabled_tools": [
                    "Base_Generator_Tool",
                    "Google_Search_Tool",
                    "Web_Fetch_Tool",
                ]
            },
        )
        self.assertNotIn("Google_Search_Tool", saved["enabled_tools"])
        self.assertIn("Web_Fetch_Tool", saved["enabled_tools"])
        self.assertIn("Web_Fetch_Tool", saved["allowed_tools"])
        raw = yaml.safe_load(_runtime_path("agentflow").read_text(encoding="utf-8")) or {}
        self.assertNotIn("allowed_tools", raw)
        self.assertIn("Web_Fetch_Tool", raw["enabled_tools"])
        self.assertNotIn("Google_Search_Tool", raw["enabled_tools"])


if __name__ == "__main__":
    unittest.main()
