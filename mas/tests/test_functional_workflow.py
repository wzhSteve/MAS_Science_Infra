"""Functional contracts: workflow events, verifier hop, resume prefix, train/collect split."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _multi_spec():
    from workflow.spec import MASSpec

    return MASSpec.model_validate(
        {
            "schema_version": "0.3",
            "topology": "graph",
            "entry_agent": "planner",
            "tools": ["execute_python", "wikipedia_search"],
            "hub": {"system_prompt": "planner prompt"},
            "agents": [
                {"id": "planner", "kind": "planner", "tools": ["execute_python"]},
                {"id": "execute_python", "kind": "tool"},
                {"id": "wikipedia_search", "kind": "tool"},
                {"id": "expert_1", "kind": "blank", "system_prompt": "expert"},
            ],
            "routers": [
                {"id": "route_x", "candidates": ["execute_python", "wikipedia_search", "expert_1"]},
            ],
            "edges": [{"from": "planner", "to": "route_x", "kind": "message"}],
        }
    )


class TestWorkflowEvents(unittest.TestCase):
    def test_mock_hub_event_order(self):
        from workflow.contracts import EventKind
        from workflow.runtime import ExecutionService

        traj = ExecutionService(mock=True).run(
            {"id": "q", "question": "1+1", "answer": "2", "source": "gsm8k", "_mock_answer": "2"}
        )
        kinds = [event.kind for event in traj.events]
        self.assertIn(EventKind.TASK_START, kinds)
        self.assertIn(EventKind.TOOL_CALL, kinds)
        self.assertIn(EventKind.TOOL_RESULT, kinds)
        self.assertIn(EventKind.FINAL_ANSWER, kinds)
        self.assertLess(kinds.index(EventKind.TASK_START), kinds.index(EventKind.TOOL_CALL))
        self.assertLess(kinds.index(EventKind.TOOL_CALL), kinds.index(EventKind.TOOL_RESULT))
        self.assertLess(kinds.index(EventKind.TOOL_RESULT), kinds.index(EventKind.FINAL_ANSWER))

    def test_verifier_failure_feedback_hop(self):
        from workflow.contracts import EventKind
        from workflow.runtime import ExecutionService
        from workflow.spec import load_spec

        spec = load_spec(str(FIXTURES / "hub_verify.yaml"))
        traj = ExecutionService(mock=True, spec=spec).run(
            {"id": "e", "question": "1+1", "answer": "2", "_mock_answer": "2", "_mock_error": "boom"}
        )
        self.assertEqual(traj.meta.get("feedback_hops"), 1)
        self.assertEqual(traj.final_answer, "2")
        feedback = [event for event in traj.events if event.kind == EventKind.FEEDBACK]
        self.assertTrue(feedback[0].payload.get("rerun"))
        self.assertFalse(feedback[-1].payload.get("rerun"))

    def test_resume_messages_keep_snapshot_prefix(self):
        from workflow.archive import branch_point_to_resume_task_fields
        from workflow.runtime import ExecutionService

        traj = ExecutionService(mock=True).run(
            {"id": "resume", "question": "1+1", "answer": "2", "_mock_answer": "2"}
        )
        self.assertTrue(traj.branch_points)
        fields = branch_point_to_resume_task_fields(traj.branch_points[0])
        prefix = fields["resume_messages"]
        self.assertTrue(prefix)
        self.assertEqual(traj.messages[: len(prefix)], prefix)
        self.assertEqual(fields["resume_parent_id"], traj.branch_points[0].parent_rollout_id or "")


class TestTrainCollectSplit(unittest.TestCase):
    def test_collect_uses_compiled_graph_and_train_stays_single_hub(self):
        from unittest.mock import patch

        from workflow.compiler import compile_spec
        from workflow.runtime import ExecutionService, agent_window_rollout, run_compiled_episode

        spec = _multi_spec()
        compiled = compile_spec(spec)
        self.assertTrue(compiled.multi_agent)
        source = (ROOT / "lit_tir_agent.py").read_text(encoding="utf-8")
        rollout = source.split("def rollout(", 1)[1].split("\n    def ", 1)[0]
        self.assertIn("run_episode(", rollout)
        self.assertIn("run_compiled_episode(", rollout)
        self.assertIn("agent_window_rollout", rollout)
        self.assertFalse(agent_window_rollout(spec))

        with patch("workflow.runtime.run_compiled_episode", wraps=run_compiled_episode) as spy:
            traj = ExecutionService(mock=True, spec=spec).run(
                {"id": "multi", "question": "q?", "answer": "42", "_mock_answer": "42"}
            )
        self.assertTrue(spy.called)
        self.assertEqual(traj.meta.get("entry_agent"), "planner")
