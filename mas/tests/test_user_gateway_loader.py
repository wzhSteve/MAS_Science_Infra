"""window_detail_lines rendering (Fix E loader lines)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _msg(kind: str, payload: dict) -> SimpleNamespace:
    return SimpleNamespace(kind=kind, payload=payload)


class TestWindowDetailLines(unittest.TestCase):
    def setUp(self) -> None:
        from workflow.user_gateway.loader import window_detail_lines

        self.render = window_detail_lines

    def test_llm_calls_lines(self) -> None:
        lines = self.render(_msg("plan_step", {
            "trace": {
                "llm_calls": {
                    "count": 3,
                    "calls": [
                        {"prompt_head": "p" * 300, "model": "m", "n": 1, "response_head": "r"},
                        {"prompt_head": "second", "model": "m", "n": 2, "response_head": "x"},
                    ],
                }
            }
        }))
        self.assertIn("  模型  3 次调用", lines)
        prompt_line = next(l for l in lines if l.startswith("  提示词"))
        # head clamped by _short(96)
        self.assertEqual(len(prompt_line), len("  提示词  ") + 96)
        self.assertTrue(prompt_line.endswith("…"))

    def test_llm_calls_zero_or_missing_degrades_silently(self) -> None:
        for payload in ({}, {"trace": {}}, {"trace": {"llm_calls": {"count": 0, "calls": []}}}, {"trace": None}):
            lines = self.render(_msg("plan_step", payload))
            self.assertFalse(any("模型" in l or "提示词" in l for l in lines), (payload, lines))

    def test_llm_calls_malformed_does_not_crash(self) -> None:
        lines = self.render(_msg("tool_result", {"trace": {"llm_calls": {"count": "x", "calls": "nope"}}}))
        self.assertIsInstance(lines, list)

    def test_existing_lines_unchanged(self) -> None:
        lines = self.render(_msg("tool_result", {
            "ok": True, "output": "out",
            "trace": {"tool": "Google_Search_Tool", "sub_goal": "find x", "command": "q=x"},
        }))
        self.assertIn("  工具  Google_Search_Tool", lines)
        self.assertIn("  目标  find x", lines)
        self.assertIn("  命令  q=x", lines)
        self.assertTrue(any(l.startswith("  结果  成功") for l in lines))


if __name__ == "__main__":
    unittest.main()
