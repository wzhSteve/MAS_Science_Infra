"""enable_thinking is one .env key and applies to every LLM role."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from science_infra.control.thinking import read_enable_thinking, write_enable_thinking


class TestEnableThinkingFile(unittest.TestCase):
    def test_write_replaces_only_that_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("OPENAI_API_KEY=keep-me\n# note\nenable_thinking=true\n", encoding="utf-8")
            self.assertFalse(write_enable_thinking(False, path))
            text = path.read_text(encoding="utf-8")
            self.assertIn("OPENAI_API_KEY=keep-me", text)
            self.assertIn("# note", text)
            self.assertIn("enable_thinking=false", text)
            self.assertNotIn("enable_thinking=true", text)
            self.assertFalse(read_enable_thinking(path))
            self.assertEqual(os.environ.get("enable_thinking"), "false")

    def test_missing_key_appends_and_defaults_off(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("OTHER=1\n", encoding="utf-8")
            self.assertTrue(write_enable_thinking(True, path))
            self.assertTrue(read_enable_thinking(path))
            self.assertTrue(path.read_text(encoding="utf-8").strip().endswith("enable_thinking=true"))


if __name__ == "__main__":
    unittest.main()
