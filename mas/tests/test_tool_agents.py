"""Unit tests for the five centralized tool-agents (phase 6)."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.tool_agents import (  # noqa: E402
    TOOL_AGENT_ARGS_SCHEMAS,
    TOOL_AGENTS,
    BlankAgentAdapter,
    get_tool_agent,
    validate_tool_args,
)
from workflow.spec import AgentNodeSpec  # noqa: E402


class TestFiveToolAgents(unittest.TestCase):
    def test_five_ids_registered(self):
        for aid in ("base_generator", "wikipedia_search", "bing_search", "web_search", "python_coder", "think"):
            self.assertIn(aid, TOOL_AGENTS)
            self.assertEqual(TOOL_AGENTS[aid].kind, "tool")
            self.assertEqual(TOOL_AGENTS[aid].id, aid)
            self.assertIn(aid, TOOL_AGENT_ARGS_SCHEMAS)

    def test_execute_python_maps_to_python_coder(self):
        ta = get_tool_agent("execute_python")
        self.assertIs(ta, TOOL_AGENTS["python_coder"])

    def test_think_has_no_echo_kernel(self):
        ta = TOOL_AGENTS["think"]
        self.assertIsNone(ta._kernel)
        self.assertTrue(ta.llm_required)

    def test_google_and_web_have_distinct_schemas(self):
        self.assertNotIn("url", TOOL_AGENT_ARGS_SCHEMAS["bing_search"].get("required", []))
        self.assertIn("url", TOOL_AGENT_ARGS_SCHEMAS["web_search"]["required"])
        ok, _ = validate_tool_args("web_search", {"query": "q"})
        self.assertFalse(ok)

    def test_python_coder_kernel_runs(self):
        ta = TOOL_AGENTS["python_coder"]
        saved = ta._llm_invoke
        ta._llm_invoke = None
        try:
            out = ta.invoke({"code": "result = 6 * 7"})
        finally:
            ta._llm_invoke = saved
        self.assertIn("42", str(out))

    def test_args_schema_think_requires_text(self):
        ok, reason = validate_tool_args("think", {})
        self.assertFalse(ok)
        self.assertIn("text", reason)
        ok, _ = validate_tool_args("think", {"text": "x"})
        self.assertTrue(ok)

    def test_python_coder_requires_query_or_code(self):
        ok, reason = validate_tool_args("python_coder", {})
        self.assertFalse(ok)
        self.assertIn("query or code", reason)
        ok, _ = validate_tool_args("python_coder", {"code": "result=1"})
        self.assertTrue(ok)

    def test_blank_adapter_uses_bare_id(self):
        spec = AgentNodeSpec(id="summarizer", kind="blank", system_prompt="sum")
        adapter = BlankAgentAdapter(spec)
        self.assertEqual(adapter.id, "summarizer")
        self.assertFalse(adapter.id.startswith("blank:"))
        self.assertEqual(adapter.tool_name, "summarizer")
        self.assertEqual(adapter.kind, "blank")


if __name__ == "__main__":
    unittest.main()
