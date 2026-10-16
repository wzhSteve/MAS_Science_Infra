"""EPC-AW cheap/paper profiles: n=1 skips BTS; n>=2 reaches _select_plan consensus."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


_SNAPSHOT = {}


def setUpModule() -> None:
    _SNAPSHOT["path"] = list(sys.path)
    _SNAPSHOT["modules"] = {k: v for k, v in sys.modules.items() if k == "tools" or k.startswith("tools.")}


def tearDownModule() -> None:
    sys.path[:] = _SNAPSHOT.get("path", sys.path)
    for k in [k for k in sys.modules if k == "tools" or k.startswith("tools.")]:
        del sys.modules[k]
    sys.modules.update(_SNAPSHOT.get("modules", {}))

CONTEXT = ROOT / "user_space" / "projects" / "epc-aw-main" / "adapted" / "context.py"
_ENV = ("SCIENCE_EPC_AW_PROFILE", "SCIENCE_EPC_AW_N", "SCIENCE_EPC_AW_MOCK", "SCIENCE_EPC_AW_PROMPTS_LITE")


def _load(profile: str):
    os.environ["SCIENCE_EPC_AW_PROFILE"] = profile
    os.environ["SCIENCE_EPC_AW_MOCK"] = "1"
    os.environ.pop("SCIENCE_EPC_AW_N", None)
    os.environ.pop("SCIENCE_EPC_AW_PROMPTS_LITE", None)
    spec = importlib.util.spec_from_file_location(f"epc_ctx_{profile}", CONTEXT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(CONTEXT.is_file(), "epc-aw-main project missing")
class TestEpcAwProfiles(unittest.TestCase):
    def setUp(self) -> None:
        self._old = {k: os.environ.get(k) for k in _ENV}

    def tearDown(self) -> None:
        for key, value in self._old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _plan(self, mod):
        ctx = mod.get_episode_context(reset=True)
        ctx.question = "What is 2+3?"
        ctx.system_memory.set_outline({"1": "compute"})
        out = ctx.plan_window({"kind": "plan_step", "dst": "planner", "payload": {"question": "What is 2+3?"}})
        return ctx, out

    def test_cheap_profile_is_single_plan_lite(self) -> None:
        mod = _load("cheap")
        self.assertEqual(mod._load_runtime()["n"], 1)
        self.assertEqual(mod._load_runtime()["profile"], "cheap")
        self.assertTrue(mod._prompts_lite_enabled())
        ctx, _ = self._plan(mod)
        self.assertEqual(int(ctx.last_trace.get("bts", {}).get("n", 1)), 1)

    def test_paper_profile_runs_bts(self) -> None:
        mod = _load("paper")
        self.assertGreaterEqual(mod._load_runtime()["n"], 2)
        self.assertFalse(mod._prompts_lite_enabled())
        ctx, out = self._plan(mod)
        self.assertIn("bts", ctx.last_trace)
        self.assertGreaterEqual(int(ctx.last_trace["bts"]["n"]), 2)
        self.assertTrue(out["payload"]["args"].get("tool_name"))

    def test_explicit_n_env_beats_profile(self) -> None:
        mod = _load("paper")
        os.environ["SCIENCE_EPC_AW_N"] = "4"
        self.assertEqual(mod._load_runtime()["n"], 4)

    def test_tolerant_json_hook_does_not_recurse(self) -> None:
        import json
        import time

        mod = _load("cheap")
        orig = json.loads
        try:
            mod._install_json_tolerant_parse_hook()
            bad = '{"a": {"b": [1, 2, {"c": "x"' * 30 + " trailing"
            t0 = time.time()
            with self.assertRaises(ValueError):
                json.loads(bad)
            self.assertLess(time.time() - t0, 1.0)
            self.assertEqual(json.loads('```json\n{"x": 1}\n```'), {"x": 1})
        finally:
            json.loads = orig

    def test_unknown_profile_falls_back_to_yaml(self) -> None:
        mod = _load("nonsense")
        self.assertEqual(mod._load_runtime()["profile"] in {"custom", "cheap", "paper"}, True)


if __name__ == "__main__":
    unittest.main()
