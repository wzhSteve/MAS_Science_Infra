"""Dedup-refund acceptance tests: cached no-op repeats must not burn steps.

Covers:
  1. refund applied when a repeat returned cached evidence (step_count back,
     wasted action slot removed, counter incremented).
  2. no refund when stopping / no hit / cap reached / step already zero.
  3. config exposes max_dedup_refunds_per_episode.
  4. _execute_authorized flags _cghe_dedup_hit on a denied-but-cached repeat.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

import context  # noqa: E402
from context import _apply_dedup_refund, _cached_tool_result, _execute_authorized  # noqa: E402


class _FakeMemory:
    def __init__(self, actions: dict) -> None:
        self.actions = actions

    def get_actions(self) -> dict:
        return self.actions


def _ep(step_count: int, actions: dict, *, hit: bool = True, refunds: int = 0):
    return SimpleNamespace(
        step_count=step_count,
        memory=_FakeMemory(actions),
        _cghe_dedup_hit=hit,
        _cghe_dedup_refunds=refunds,
        _cghe_dedup_fingerprint="fp-test",
    )

def _settings(cap: int) -> dict:
    return {"max_dedup_refunds_per_episode": cap}


class RefundMechanics(unittest.TestCase):
    def test_cached_tool_result_found_by_tool_and_command(self) -> None:
        actions = {
            "Action Step 1": {
                "tool_name": "Bing_Search_Tool",
                "command": "execution = tool.execute(query='a')",
                "result": "snippet-a",
            }
        }
        ep = _ep(2, actions)
        found, value = _cached_tool_result(ep, "Bing_Search_Tool", "execution = tool.execute(query='a')")
        self.assertTrue(found)
        self.assertEqual(value, "snippet-a")
        found2, _ = _cached_tool_result(ep, "Bing_Search_Tool", "execution = tool.execute(query='b')")
        self.assertFalse(found2)

    def test_refund_applied_when_hit_and_not_stop(self) -> None:
        actions = {
            "Action Step 1": {"tool_name": "Bing_Search_Tool", "command": "q1", "result": "r1"},
            "Action Step 2": {"tool_name": "Bing_Search_Tool", "command": "q1", "result": "r1"},
        }
        ep = _ep(2, actions, hit=True, refunds=0)
        with patch.object(context, "cghe_settings", return_value=_settings(3)):
            applied = _apply_dedup_refund(ep, stop=False)
        self.assertTrue(applied)
        self.assertEqual(ep.step_count, 1)
        self.assertEqual(ep._cghe_dedup_refunds, 1)
        self.assertNotIn("Action Step 2", ep.memory.actions)
        self.assertIn("Action Step 1", ep.memory.actions)

    def test_no_refund_when_stopping(self) -> None:
        ep = _ep(5, {"Action Step 5": {"tool_name": "Bing_Search_Tool", "command": "q", "result": "r"}})
        with patch.object(context, "cghe_settings", return_value=_settings(3)):
            self.assertFalse(_apply_dedup_refund(ep, stop=True))
        self.assertEqual(ep.step_count, 5)
        self.assertEqual(ep._cghe_dedup_refunds, 0)

    def test_no_refund_without_hit(self) -> None:
        ep = _ep(5, {}, hit=False)
        with patch.object(context, "cghe_settings", return_value=_settings(3)):
            self.assertFalse(_apply_dedup_refund(ep, stop=False))
        self.assertEqual(ep.step_count, 5)
        self.assertEqual(ep._cghe_dedup_refunds, 0)

    def test_no_refund_at_cap(self) -> None:
        ep = _ep(5, {"Action Step 5": {"tool_name": "Bing_Search_Tool", "command": "q", "result": "r"}}, refunds=3)
        with patch.object(context, "cghe_settings", return_value=_settings(3)):
            self.assertFalse(_apply_dedup_refund(ep, stop=False))
        self.assertEqual(ep.step_count, 5)
        self.assertIn("Action Step 5", ep.memory.actions)
        self.assertEqual(ep._cghe_dedup_refunds, 3)

    def test_no_refund_at_step_zero(self) -> None:
        ep = _ep(0, {})
        with patch.object(context, "cghe_settings", return_value=_settings(3)):
            self.assertFalse(_apply_dedup_refund(ep, stop=False))
        self.assertEqual(ep.step_count, 0)

    def test_zero_cap_disables_refund(self) -> None:
        ep = _ep(4, {"Action Step 4": {"tool_name": "Bing_Search_Tool", "command": "q", "result": "r"}})
        with patch.object(context, "cghe_settings", return_value=_settings(0)):
            self.assertFalse(_apply_dedup_refund(ep, stop=False))
        self.assertEqual(ep.step_count, 4)


class ConfigExposure(unittest.TestCase):
    def test_settings_exposes_dedup_cap(self) -> None:
        settings = context.cghe_settings()
        self.assertIn("max_dedup_refunds_per_episode", settings)
        self.assertGreaterEqual(int(settings["max_dedup_refunds_per_episode"]), 0)


class AuthorizedFlagging(unittest.TestCase):
    def test_denied_cached_repeat_sets_dedup_hit(self) -> None:
        ep = SimpleNamespace(
            step_count=3,
            memory=_FakeMemory({
                "Action Step 2": {
                    "tool_name": "Bing_Search_Tool",
                    "command": "execution = tool.execute(query='dup')",
                    "raw_tool_result": "cached-snippets",
                }
            }),
            _cghe_dedup_hit=False,
            _cghe_dedup_fingerprint="",
            _cghe_last_auth=None,
            executor=SimpleNamespace(execute_tool_command=lambda *_a, **_k: "should-not-run"),
            _cghe_directive={},
        )

        def fake_controller() -> object:
            return SimpleNamespace(graph=None)

        ep.cghe_controller = fake_controller  # type: ignore[attr-defined]
        denied = SimpleNamespace(allowed=False, replay=False, reason="same fingerprint must not be repeated again", fingerprint="fp-dup")
        with patch.object(context, "cghe_enabled", return_value=True), patch.object(
            context, "_authorize_cghe_call", return_value=denied
        ), patch.object(context, "_mark_directive_node_failed"):
            result = _execute_authorized(ep, "Bing_Search_Tool", "execution = tool.execute(query='dup')")
        self.assertEqual(result, "cached-snippets")
        self.assertTrue(ep._cghe_dedup_hit)
        self.assertEqual(ep._cghe_dedup_fingerprint, "fp-dup")

    def test_denied_without_cache_leaves_hit_false(self) -> None:
        ep = SimpleNamespace(
            step_count=3,
            memory=_FakeMemory({}),
            _cghe_dedup_hit=False,
            _cghe_dedup_fingerprint="",
            _cghe_last_auth=None,
            executor=SimpleNamespace(execute_tool_command=lambda *_a, **_k: "should-not-run"),
            _cghe_directive={},
        )
        ep.cghe_controller = lambda: SimpleNamespace(graph=None)  # type: ignore[assignment]
        denied = SimpleNamespace(allowed=False, replay=False, reason="budget", fingerprint="fp-x")
        with patch.object(context, "cghe_enabled", return_value=True), patch.object(
            context, "_authorize_cghe_call", return_value=denied
        ), patch.object(context, "_mark_directive_node_failed"):
            result = _execute_authorized(ep, "Bing_Search_Tool", "execution = tool.execute(query='novel')")
        self.assertFalse(ep._cghe_dedup_hit)
        self.assertNotIn("should-not-run", str(result))


if __name__ == "__main__":
    unittest.main()
