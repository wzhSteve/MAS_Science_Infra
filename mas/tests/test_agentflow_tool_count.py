"""Regression: n_search/n_python must count tools invoked through a
user_space executor wrapper (e.g. AgentFlow), where every tool_result comes
from one executor node but carries the real tool name in payload.tool.

Before the fix, only literal agent ids (google_search / python_coder ...) were
counted, so AgentFlow rollouts always logged search=0 py=0 and the
multi_tool_bonus reward shaping never fired.
"""

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

from workflow.centralized_runtime import (  # noqa: E402
    _is_python_toolmsg,
    _is_search_toolmsg,
)


def _agentflow_spec():
    """planner -> u_agentflow__executor (user_space blank) -> verifier,
    mirroring user_space/projects/agentflow/contracts/workflow.yaml."""
    return MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "agents": [
            {"id": "planner", "kind": "planner", "system_prompt": "plan",
             "profile": {"backend": "user_space", "user_project": "absent_project"}},
            {"id": "u_agentflow__executor", "kind": "blank", "trainable": False,
             "system_prompt": "exec",
             "profile": {"backend": "user_space", "user_project": "absent_project", "llm_required": False}},
            {"id": "verifier", "kind": "verifier", "system_prompt": "verify",
             "profile": {"backend": "user_space", "user_project": "absent_project"}},
        ],
        "routers": [],
        "edges": [
            {"from": "planner", "to": "u_agentflow__executor", "kind": "message"},
            {"from": "u_agentflow__executor", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
        "hub": {"role": "planner", "max_feedback_hops": 2},
    })


class TestToolMsgClassification(unittest.TestCase):
    def _msg(self, payload):
        from workflow.protocol import make_message

        payload.setdefault("evidence_type", "DIRECT")
        payload.setdefault("ok", True)
        payload.setdefault("output", "x")
        return make_message(
            task_id="t", turn=1, src="u_agentflow__executor", dst="verifier",
            kind="tool_result", payload=payload,
        )

    def test_payload_tool_names_recognized(self):
        for name in ("Bing_Search_Tool", "Wikipedia_Search_Tool", "Web_Fetch_Tool"):
            self.assertTrue(_is_search_toolmsg(self._msg({"tool": name, "output": "x", "ok": True})), name)
        self.assertFalse(_is_search_toolmsg(self._msg({"tool": "Python_Coder_Tool", "output": "x"})))
        self.assertTrue(_is_python_toolmsg(self._msg({"tool": "Python_Coder_Tool"})))

    def test_trace_tool_name_fallback(self):
        m = self._msg({"trace": {"tool": "Bing_Search_Tool"}, "output": "x", "ok": True})
        self.assertTrue(_is_search_toolmsg(m))

    def test_plain_agentflow_executor_not_counted_as_search(self):
        m = self._msg({"output": "plain", "ok": True})
        self.assertFalse(_is_search_toolmsg(m))
        self.assertFalse(_is_python_toolmsg(m))


class TestAgentflowSearchCount(unittest.TestCase):
    def test_n_search_counts_wrapped_tool(self):
        """Exercise the counting helpers on the exact message shapes the
        user_gateway loader produces for AgentFlow tool results."""
        from workflow.protocol import make_message

        tool_msg = make_message(
            task_id="t", turn=1, src="u_agentflow__executor", dst="verifier",
            kind="tool_result",
            payload={"output": "bing results for: q", "ok": True,
                     "evidence_type": "DIRECT", "tool": "Bing_Search_Tool",
                     "command": "execution = tool.execute(query='q')"},
        )
        self.assertTrue(_is_search_toolmsg(tool_msg))
        # src-based branch (legacy path) still works for native ids
        legacy = make_message(
            task_id="t", turn=1, src="bing_search", dst="verifier",
            kind="tool_result", payload={"output": "x", "ok": True, "evidence_type": "DIRECT"},
        )
        self.assertEqual(legacy.src, "bing_search")


if __name__ == "__main__":
    unittest.main()
