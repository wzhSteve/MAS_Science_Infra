"""Local vLLM launch flags. Does not start a server."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TestLocalLlmArgv(unittest.TestCase):
    def test_start_enables_auto_tool_choice(self) -> None:
        from science_infra.control.experiments import create_experiment, save_section
        from science_infra.control import services

        with tempfile.TemporaryDirectory() as tmp:
            env = patch.dict(os.environ, {"SCIENCE_EXPERIMENTS_DIR": tmp})
            env.start()
            self.addCleanup(env.stop)
            create_experiment("llm-flags", name="flags")
            save_section("llm-flags", "llm", {
                "kind": "local",
                "model": "Qwen3-4B",
                "model_path": "/tmp/Qwen3-4B",
                "port": 8000,
                "base_url": "http://127.0.0.1:8000/v1",
            })
            captured = {}

            def _start(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(run_id="llm-test")

            with patch.object(services, "resolve_binding", return_value=None), patch.object(
                services.PROCS, "active", return_value=None
            ), patch.object(services, "_openai_models", return_value=[]), patch.object(
                services, "_free_local_llm_port"
            ), patch.object(services, "_port_open", return_value=False), patch.object(
                services.PROCS, "start", side_effect=_start
            ), patch.object(services.BUS, "publish"):
                out = services.start_local_llm("llm-flags")
            argv = out["argv"]
            self.assertIn("--enable-auto-tool-choice", argv)
            self.assertIn("--tool-call-parser", argv)
            self.assertEqual(argv[argv.index("--tool-call-parser") + 1], "hermes")
            self.assertFalse(out["reused"])
            self.assertIn("--enable-auto-tool-choice", captured["argv"])


if __name__ == "__main__":
    unittest.main()
