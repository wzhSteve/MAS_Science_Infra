"""EPC-AW opt-epc work items: prompts-lite, tolerant JSON, memory dedup.

Work item A - prompts-lite templates:
  * every lite template carries the EXACT same str.format placeholder set as
    its upload/ original (string.Formatter().parse equality);
  * character budgets: planner <=2000, diagnoser <=1800, executor <=1600,
    direct_output <=1000;
  * fake-data format() dry-run raises no KeyError;
  * the open-shim injection is idempotent (installing twice never stacks);
  * redirects land on adapted/prompts/**/*.lite.txt (not upload/ twin paths);
  * builtins.open is never hijacked (isolation safety).

Work item B - tolerant JSON layer:
  * fence-wrapped JSON parses;
  * first balanced JSON substring survives prose around it;
  * total garbage degrades to the {"error":..., ...} payload.

Work item C - memory get_actions dedup:
  * duplicate (tool_name, tool_input) entries collapse;
  * re-install on the same memory does not stack wrappers.
"""

from __future__ import annotations

import builtins
import importlib.util
import sys
import unittest
from pathlib import Path
from string import Formatter
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

EPC_CTX = ROOT / "user_space" / "projects" / "epc-aw-main" / "adapted" / "context.py"
LITE_DIR = ROOT / "user_space" / "projects" / "epc-aw-main" / "adapted" / "prompts"
REF_PROMPTS = ROOT / "user_space" / "projects" / "epc-aw-main" / "upload" / "EPC-AW-main" / "MAS" / "epc_aw" / "prompts"
REF_ROOT = ROOT / "user_space" / "projects" / "epc-aw-main" / "upload" / "EPC-AW-main"


def _load_context_module():
    spec = importlib.util.spec_from_file_location("epc_aw_adapted_context_lite", EPC_CTX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _placeholders(text: str) -> set:
    return {fname for _, fname, _, _ in Formatter().parse(text) if fname is not None}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


LITE_PAIRS = {
    "planner": ("planner/generate_next_step.lite.txt", "planner/generate_next_step.txt", 2000),
    "diagnoser": ("diagnoser/verificate_context.lite.txt", "diagnoser/verificate_context.txt", 1800),
    "executor": ("executor/generate_tool_command.lite.txt", "executor/generate_tool_command.txt", 1600),
    "direct": ("executor/direct_output.lite.txt", "executor/direct_output.txt", 1000),
}


class TestLiteTemplates(unittest.TestCase):
    def _check(self, lite_rel: str, ref_rel: str, cap: int) -> None:
        lite = _read(LITE_DIR / lite_rel)
        orig = _read(REF_PROMPTS / ref_rel)
        self.assertLessEqual(len(lite), cap, f"{lite_rel} exceeds {cap} chars")
        self.assertLess(len(lite), len(orig), f"{lite_rel} must be shorter than the original")
        self.assertEqual(
            _placeholders(lite),
            _placeholders(orig),
            f"{lite_rel} placeholder set must match the original exactly",
        )
        lite.format(**{name: "X" for name in _placeholders(lite)})

    def test_planner_lite(self) -> None:
        self._check(*LITE_PAIRS["planner"])

    def test_diagnoser_lite(self) -> None:
        self._check(*LITE_PAIRS["diagnoser"])

    def test_executor_lite(self) -> None:
        self._check(*LITE_PAIRS["executor"])

    def test_direct_output_lite(self) -> None:
        self._check(*LITE_PAIRS["direct"])


class TestLiteCandidateMapping(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_context_module()

    def test_maps_ref_path_to_adapted_lite(self) -> None:
        ref = str(REF_PROMPTS / "planner" / "generate_next_step.txt")
        cand = self.mod._is_lite_candidate(ref)
        self.assertIsNotNone(cand)
        self.assertEqual(cand, LITE_DIR / "planner" / "generate_next_step.lite.txt")
        self.assertTrue(cand.is_file())

    def test_ignores_unrelated_prompt(self) -> None:
        ref = str(REF_PROMPTS / "planner" / "belief_prediction.txt")
        self.assertIsNone(self.mod._is_lite_candidate(ref))

    def test_ignores_non_prompt_path(self) -> None:
        self.assertIsNone(self.mod._is_lite_candidate("/tmp/foo.txt"))


class TestLiteInjectionIdempotent(unittest.TestCase):
    def test_shim_marker_no_stacking_no_builtins(self) -> None:
        mod = _load_context_module()
        # Ensure EPC-AW import root is on path before installing shims.
        if str(REF_ROOT) not in sys.path:
            sys.path.insert(0, str(REF_ROOT))
        before_builtins = builtins.open
        try:
            import MAS.epc_aw.models.planner as pmod  # type: ignore
            import MAS.epc_aw.models.diagnoser as dmod  # type: ignore
            import MAS.epc_aw.models.executor as emod  # type: ignore
        except Exception:
            self.skipTest("EPC-AW ref modules not importable here")

        with patch.object(mod, "_prompts_lite_enabled", return_value=True):
            mod._install_lite_prompts()
            first = [getattr(getattr(m, "open", None), "_science_prompt_lite", False) for m in (pmod, dmod, emod)]
            open_objs = [getattr(m, "open", None) for m in (pmod, dmod, emod)]
            mod._install_lite_prompts()
            second = [getattr(getattr(m, "open", None), "_science_prompt_lite", False) for m in (pmod, dmod, emod)]
            open_objs2 = [getattr(m, "open", None) for m in (pmod, dmod, emod)]

        self.assertTrue(all(first), "shims must be installed after the first run")
        self.assertEqual(first, second)
        self.assertEqual(open_objs, open_objs2, "second install must not replace/stack shims")
        self.assertIs(builtins.open, before_builtins, "builtins.open must stay untouched")
        self.assertFalse(getattr(builtins.open, "_science_prompt_lite", False))

    def test_module_open_serves_lite_body(self) -> None:
        mod = _load_context_module()
        if str(REF_ROOT) not in sys.path:
            sys.path.insert(0, str(REF_ROOT))
        try:
            import MAS.epc_aw.models.planner as pmod  # type: ignore
        except Exception:
            self.skipTest("EPC-AW planner not importable here")
        with patch.object(mod, "_prompts_lite_enabled", return_value=True):
            mod._install_lite_prompts()
        ref = str(REF_PROMPTS / "planner" / "generate_next_step.txt")
        with pmod.open(ref, "r", encoding="utf-8") as f:
            body = f.read()
        lite = _read(LITE_DIR / "planner" / "generate_next_step.lite.txt")
        self.assertEqual(body, lite)
        self.assertLessEqual(len(body), 2000)


class TestTolerantJson(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_context_module()

    def test_fence_strip(self) -> None:
        fenced = '```json\n{"Analysis": "ok", "Conclusion": "STOP"}\n```'
        cleaned = self.mod._json_tolerant_clean(fenced)
        import json

        self.assertEqual(json.loads(cleaned)["Conclusion"], "STOP")

    def test_balanced_extraction(self) -> None:
        noisy = 'prefix {"a": 1, "b": {"c": 2}} trailing'
        cleaned = self.mod._json_tolerant_clean(noisy)
        import json

        self.assertEqual(json.loads(cleaned), {"a": 1, "b": {"c": 2}})

    def test_garbage_error_payload_no_raise(self) -> None:
        wrapper = self.mod._json_tolerant_wrapper(
            lambda self, prompt, *a, **k: "total garbage"
        )

        class Eng:
            pass

        class MemoryVerification:
            @classmethod
            def model_json_schema(cls):
                return {}

        out = wrapper(Eng(), "prompt", response_format=MemoryVerification)
        self.assertIsInstance(out, dict)
        self.assertEqual(out.get("error"), "json_parse_failed")


class TestMemoryDedup(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_context_module()

    def test_dedup_token_and_install_idempotent(self) -> None:
        self.assertEqual(
            self.mod._action_dedup_token(
                {"tool_name": "Google_Search_Tool", "tool_input": "q"}
            ),
            ("Google_Search_Tool", "q"),
        )
        if str(REF_ROOT) not in sys.path:
            sys.path.insert(0, str(REF_ROOT))
        try:
            from MAS.epc_aw.models.memory import Memory  # type: ignore
        except Exception:
            self.skipTest("EPC-AW Memory not importable here")
        self.mod._install_memory_dedup(limit=12)
        first = Memory.get_actions
        self.mod._install_memory_dedup(limit=12)
        self.assertIs(Memory.get_actions, first)
        self.assertTrue(getattr(Memory.get_actions, "_science_dedup", False))

        mem = Memory()
        # Populate via whatever public API exists; fall back to actions attr.
        actions = {
            "Planner Step 1": {
                "tool_name": "Google_Search_Tool",
                "tool_input": "q",
                "result": "r1",
            },
            "Planner Step 2": {
                "tool_name": "Google_Search_Tool",
                "tool_input": "q",
                "result": "r2",
            },
            "Planner Step 3": {
                "tool_name": "Web_Fetch_Tool",
                "tool_input": "u",
                "result": "x",
            },
        }
        if hasattr(mem, "actions"):
            mem.actions = actions
        else:
            self.skipTest("Memory has no actions attribute to seed")
        out = mem.get_actions()
        self.assertEqual(len(out), 2)
        self.assertIn("Planner Step 2", out)  # newest duplicate kept
        self.assertNotIn("Planner Step 1", out)
        self.assertIn("Planner Step 3", out)


if __name__ == "__main__":
    unittest.main()
