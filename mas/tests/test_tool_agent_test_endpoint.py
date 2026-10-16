"""Control-plane smoke for POST /api/mas/tool-agents/test.

No real network calls: only route registration, request-model validation and
the unknown-agent error path are exercised (mirrors test_stage10_control_ui.py).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(TIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


class TestToolAgentTestEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app
        from science_infra.control.experiments import ensure_experiment

        ensure_experiment("demo")
        cls.client = TestClient(create_app())

    def test_route_registered_and_openapi_schema(self):
        paths = self.client.app.openapi()["paths"]
        self.assertIn("/api/mas/tool-agents/test", paths)
        self.assertIn("post", paths["/api/mas/tool-agents/test"])

    def test_request_model_rejects_missing_agent_id(self):
        r = self.client.post("/api/mas/tool-agents/test", json={"tier": "lite"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("agent_id", r.text)

    def test_unknown_agent_returns_404_without_invoking(self):
        r = self.client.post(
            "/api/mas/tool-agents/test",
            json={"agent_id": "no_such_agent", "tier": "lite", "args": {"query": "x"}},
        )
        self.assertEqual(r.status_code, 404)
        self.assertIn("unknown tool-agent", r.text)

    def test_error_path_returns_ok_false_and_200(self):
        """An invoke crash must surface as ok=False, never an unhandled 500."""
        from unittest.mock import patch

        from tools.tool_agents import TOOL_AGENTS

        with patch.object(
            TOOL_AGENTS["think"],
            "invoke",
            side_effect=RuntimeError("boom: no kernel"),
        ):
            r = self.client.post(
                "/api/mas/tool-agents/test",
                json={"agent_id": "think", "tier": "lite", "args": {"text": "x"}},
            )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertFalse(body["ok"])
        self.assertIn("boom: no kernel", body["output"])
        self.assertIn("error:", body["output"])
        self.assertIsInstance(body["elapsed_ms"], int)


class TestToolAgentTierSmoke(unittest.TestCase):
    """think has no kernel and no LLM: invoke with tier never touches network."""

    def test_think_lite_and_pro_markers(self):
        from tools.tool_agents import TOOL_AGENTS

        for tier in ("lite", "pro"):
            out = TOOL_AGENTS["think"].invoke({"text": "unit"}, tier=tier)
            self.assertTrue(str(out).startswith("tier="), tier)


class TestPaletteTierFields(unittest.TestCase):
    def test_tool_agents_have_tier_fields(self):
        from science_infra.control.services import mas_palette

        pal = mas_palette()
        entries = {t["id"]: t for t in pal["tool_agents"]}
        for tid in ("base_generator", "bing_search", "wikipedia_search", "web_fetch", "python_coder"):
            self.assertFalse(entries[tid]["has_tiers"], tid)
            self.assertEqual(entries[tid]["tiers"], [], tid)
            self.assertEqual(entries[tid]["default_fetch_n"], 0, tid)

    def test_default_fetch_n_stays_zero_without_tiers(self):
        from science_infra.control.services import mas_palette

        with _env_override("MAS_PRO_FETCH_N", "7"):
            pal = mas_palette()
        got = {t["id"]: t["default_fetch_n"] for t in pal["tool_agents"]}
        self.assertEqual(got["bing_search"], 0)
        self.assertEqual(got["wikipedia_search"], 0)


class _env_override:
    """Temporarily set an env var, restoring the previous value on exit."""

    def __init__(self, key: str, value: str):
        self.key = key
        self.value = value
        self.saved = None

    def __enter__(self):
        self.saved = os.environ.pop(self.key, None)
        os.environ[self.key] = self.value
        return self

    def __exit__(self, *_exc):
        if self.saved is not None:
            os.environ[self.key] = self.saved
        else:
            os.environ.pop(self.key, None)
        return False


if __name__ == "__main__":
    unittest.main()
