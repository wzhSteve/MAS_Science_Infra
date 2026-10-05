"""Centralized MAS integration tests (phase 7).

Three topologies (linear PEV, multi-router fan-out, blank-agent inserted) run
through the new AgentMessage flow. Structural tests use ``MockWindowLLM`` so
they run without the local LLM. A live-LLM test uses the ``local_llm_endpoint``
fixture (skipped when the local vLLM endpoint is down).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from workflow.archive import Archive, register_archive  # noqa: E402
from workflow.centralized_runtime import MockWindowLLM, run_centralized_episode  # noqa: E402
from workflow.compiler import compile_spec  # noqa: E402
from workflow.memory import MemoryStore  # noqa: E402
from workflow.spec import MASSpec  # noqa: E402


def _linear_spec() -> MASSpec:
    """t1: planner -> router -> python_coder -> verifier (linear PEV)."""
    return MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "tools": ["python_coder"],
        "agents": [
            {"id": "planner", "kind": "planner", "tools": ["python_coder"], "system_prompt": "plan"},
            {"id": "python_coder", "kind": "tool", "trainable": False},
            {"id": "verifier", "kind": "verifier", "system_prompt": "verify"},
        ],
        "routers": [{"id": "route_exec", "candidates": ["python_coder"], "strategy": "llm_choice"}],
        "edges": [
            {"from": "planner", "to": "route_exec", "kind": "route"},
            {"from": "route_exec", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
    })


def _multi_router_spec() -> MASSpec:
    """t2: planner fans out to two routers -> two tool-agents -> verifier."""
    return MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "tools": ["python_coder", "think"],
        "agents": [
            {"id": "planner", "kind": "planner", "tools": ["python_coder", "think"], "system_prompt": "plan"},
            {"id": "python_coder", "kind": "tool", "trainable": False},
            {"id": "think", "kind": "tool", "trainable": False},
            {"id": "verifier", "kind": "verifier", "system_prompt": "verify"},
        ],
        "routers": [
            {"id": "route_a", "candidates": ["python_coder"], "strategy": "llm_choice"},
            {"id": "route_b", "candidates": ["think"], "strategy": "llm_choice"},
        ],
        "edges": [
            {"from": "planner", "to": "route_a", "kind": "route"},
            {"from": "planner", "to": "route_b", "kind": "route"},
            {"from": "route_a", "to": "verifier", "kind": "message"},
            {"from": "route_b", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
    })


def _blank_inserted_spec() -> MASSpec:
    """t3: planner -> router -> python_coder -> summarizer (blank) -> verifier."""
    return MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "tools": ["python_coder"],
        "agents": [
            {"id": "planner", "kind": "planner", "tools": ["python_coder"], "system_prompt": "plan"},
            {"id": "python_coder", "kind": "tool", "trainable": False},
            {
                "id": "summarizer",
                "kind": "blank",
                "system_prompt": "Summarize the tool output.",
                "profile": {
                    "input_schema": {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]},
                    "output_schema": {"type": "object", "properties": {"output": {"type": "string"}}, "required": ["output"]},
                },
            },
            {"id": "verifier", "kind": "verifier", "system_prompt": "verify"},
        ],
        "routers": [{"id": "route_exec", "candidates": ["python_coder"], "strategy": "llm_choice"}],
        "edges": [
            {"from": "planner", "to": "route_exec", "kind": "route"},
            {"from": "python_coder", "to": "summarizer", "kind": "message"},
            {"from": "summarizer", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
    })


class TestLinearPEV(unittest.TestCase):
    def test_four_message_cycle(self):
        spec = _linear_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        win = MockWindowLLM({
            "planner": [json.dumps({"next": "python_coder", "args": {"code": "result=42"}, "sub_goal": "compute", "done": True})],
            "verifier": [json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})],
        })
        raw = run_centralized_episode(
            {"id": "t1", "question": "What is 6*7?"}, None, arch, spec, MemoryStore(), compiled,
            window_llm=win,
        )
        # one planner, one router, one tool, one verifier window
        ids = [e.get("agent_id") for e in raw.window_events]
        self.assertIn("planner", ids)
        self.assertIn("route_exec", ids)
        self.assertIn("python_coder", ids)
        self.assertIn("verifier", ids)
        # message kinds: plan_step, tool_invoke, tool_result, verify
        kinds = [m.get("kind") for m in raw.messages]
        self.assertEqual(kinds, ["plan_step", "tool_invoke", "tool_result", "verify"])
        plan_id = raw.messages[0]["msg_id"]
        self.assertEqual(raw.messages[1]["kind"], "tool_invoke")
        self.assertEqual(raw.messages[1]["trace_ref"], plan_id)
        self.assertEqual(raw.messages[2]["trace_ref"], raw.messages[1]["msg_id"])
        self.assertEqual(raw.messages[3]["trace_ref"], raw.messages[2]["msg_id"])
        self.assertEqual(raw.n_python, 1)
        self.assertTrue(raw.format_ok)

    def test_verifier_fail_feedback(self):
        spec = _linear_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        win = MockWindowLLM({
            "planner": [
                json.dumps({"next": "python_coder", "args": {"code": "result=1"}, "sub_goal": "compute", "done": False}),
                json.dumps({"next": "python_coder", "args": {"code": "result=42"}, "sub_goal": "compute", "done": True}),
            ],
            "verifier": [
                json.dumps({"ok": False, "reason": "wrong answer", "step_conclusion": "INCOMPLETE", "slot_updates": []}),
                json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []}),
            ],
        })
        raw = run_centralized_episode(
            {"id": "t1b", "question": "q?"}, None, arch, spec, MemoryStore(), compiled, window_llm=win,
        )
        # two turns: first verify fails, second succeeds
        verify_msgs = [m for m in raw.messages if m.get("kind") == "verify"]
        self.assertEqual(len(verify_msgs), 2)
        self.assertFalse(verify_msgs[0]["payload"]["ok"])
        self.assertTrue(verify_msgs[1]["payload"]["ok"])
        fb = [m for m in raw.messages if m.get("kind") == "feedback"]
        self.assertGreaterEqual(len(fb), 1)
        self.assertTrue(raw.format_ok)


class TestMultiRouterParallel(unittest.TestCase):
    def test_fan_out_two_tools_parallel(self):
        spec = _multi_router_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        win = MockWindowLLM({
            "planner": [json.dumps({"next": ["python_coder", "think"], "args": [{"code": "result=1"}, {"text": "reason"}], "sub_goal": "parallel", "done": False})],
            "verifier": [json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})],
        })
        raw = run_centralized_episode(
            {"id": "t2", "question": "q?"}, None, arch, spec, MemoryStore(), compiled, window_llm=win,
        )
        route_msgs = [m for m in raw.messages if m.get("kind") == "tool_invoke"]
        self.assertEqual(len(route_msgs), 2)
        plan_msg = next(m for m in raw.messages if m.get("kind") == "plan_step")
        for rm in route_msgs:
            self.assertEqual(rm["trace_ref"], plan_msg["msg_id"])
        tool_msgs = [m for m in raw.messages if m.get("kind") == "tool_result"]
        self.assertEqual(len(tool_msgs), 2)
        srcs = sorted(m["src"] for m in tool_msgs)
        self.assertEqual(srcs, ["python_coder", "think"])
        invoke_ids = {m["msg_id"] for m in route_msgs}
        for tm in tool_msgs:
            self.assertIn(tm["trace_ref"], invoke_ids)
        ids = [e.get("agent_id") for e in raw.window_events]
        self.assertIn("python_coder", ids)
        self.assertIn("think", ids)
        self.assertIn("route_a", ids)
        self.assertIn("route_b", ids)


class TestBlankAgentInserted(unittest.TestCase):
    def test_blank_agent_window_uses_bare_id(self):
        spec = _blank_inserted_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        win = MockWindowLLM({
            "planner": [json.dumps({"next": "python_coder", "args": {"code": "result=42"}, "sub_goal": "compute", "done": False})],
            "summarizer": [json.dumps({"output": "the result is 42"})],
            "verifier": [json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})],
        })
        raw = run_centralized_episode(
            {"id": "t3", "question": "q?"}, None, arch, spec, MemoryStore(), compiled, window_llm=win,
        )
        ids = [e.get("agent_id") for e in raw.window_events]
        # summarizer window uses the bare id (no blank: prefix) and sits after python_coder
        self.assertIn("python_coder", ids)
        self.assertIn("summarizer", ids)
        self.assertFalse(any(str(i).startswith("blank:") for i in ids))
        tool_msgs = [m for m in raw.messages if m.get("kind") == "tool_result"]
        srcs = [m["src"] for m in tool_msgs]
        self.assertIn("python_coder", srcs)
        self.assertIn("summarizer", srcs)
        by_src = {m["src"]: m for m in tool_msgs}
        self.assertEqual(by_src["summarizer"]["trace_ref"], by_src["python_coder"]["msg_id"])


class TestIllegalNext(unittest.TestCase):
    def test_next_outside_candidates_does_not_invoke_tool(self):
        spec = _linear_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        win = MockWindowLLM({
            "planner": [
                json.dumps({"next": "ghost", "args": {}, "sub_goal": "x", "done": False}),
                json.dumps({"next": "", "args": {}, "sub_goal": "done", "done": True, "answer": "stop"}),
            ],
            "verifier": [json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})],
        })
        raw = run_centralized_episode(
            {"id": "t-illegal", "question": "q?"}, None, arch, spec, MemoryStore(), compiled, window_llm=win,
        )
        first_cycle = []
        for m in raw.messages:
            first_cycle.append(m.get("kind"))
            if m.get("kind") == "feedback":
                break
        self.assertEqual(first_cycle, ["plan_step", "feedback"])
        self.assertNotIn("tool_invoke", first_cycle)
        self.assertNotIn("tool_result", first_cycle)
        self.assertNotIn("python_coder", [
            e.get("agent_id") for e in raw.window_events if e.get("kind") == "after_tool"
        ])


class TestContractValidationFail(unittest.TestCase):
    def test_tool_args_invalid_evidence_error(self):
        spec = _linear_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        # planner picks python_coder but with wrong args (missing code/query ok? schema allows empty)
        # use a tool with required args: think requires text
        spec2 = _multi_router_spec()
        compiled2 = compile_spec(spec2)
        win = MockWindowLLM({
            "planner": [json.dumps({"next": "think", "args": {}, "sub_goal": "think", "done": False})],
            "verifier": [json.dumps({"ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": []})],
        })
        raw = run_centralized_episode(
            {"id": "t4", "question": "q?"}, None, arch, spec2, MemoryStore(), compiled2, window_llm=win,
        )
        tool_msg = next(m for m in raw.messages if m.get("kind") == "tool_result")
        # think requires text; empty args -> invalid -> evidence_type ERROR
        self.assertFalse(tool_msg["payload"]["ok"])
        self.assertEqual(tool_msg["payload"]["evidence_type"], "ERROR")


def _executor_window_spec() -> MASSpec:
    """EPC-AW shape: planner → blank executor → verifier, no router."""
    return MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "tools": [],
        "agents": [
            {"id": "planner", "kind": "planner", "system_prompt": "plan"},
            {"id": "executor", "kind": "blank", "system_prompt": "execute"},
            {"id": "verifier", "kind": "verifier", "system_prompt": "verify"},
        ],
        "edges": [
            {"from": "planner", "to": "executor", "kind": "message"},
            {"from": "executor", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
    })


class TestExecutionNodesAreWorkflowAgents(unittest.TestCase):
    def test_pev_nodes_are_planner_executor_verifier(self):
        from workflow.execution_recording import ExecutionRecorder

        spec = _executor_window_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        win = MockWindowLLM({
            "planner": [json.dumps({
                "next": "executor",
                "args": {"query": "Citibank founded"},
                "sub_goal": "Find the founding year",
                "done": False,
            })],
            "executor": [json.dumps({
                "output": "Citibank was founded in 1812 via Wikipedia_Search_Tool",
                "trace": {"tool": "Wikipedia_Search_Tool", "command": "Citibank founded"},
            })],
            "verifier": [json.dumps({
                "ok": True,
                "reason": "ok",
                "step_conclusion": "COMPLETE",
                "slot_updates": [],
                "ready_to_stop": True,
                "answer": "1812",
            })],
        })
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ExecutionRecorder(Path(tmp), "rollout", "attempt")
            raw = run_centralized_episode(
                {"id": "pev", "question": "When was Citibank founded?"},
                None, arch, spec, MemoryStore(), compiled,
                window_llm=win, execution_recorder=recorder,
            )
            self.assertFalse(raw.error, raw.error)
            nodes = [node for node in recorder.trace.nodes if node.kind == "execution"]
            self.assertEqual([node.agent_id for node in nodes], ["planner", "executor", "verifier"])
            self.assertTrue(all(node.agent_kind == "agent" for node in nodes))
            executor = nodes[1]
            detail = json.dumps(recorder._details[executor.node_id], ensure_ascii=False, default=str)
            self.assertIn("Wikipedia_Search_Tool", detail)
            self.assertNotIn("Wikipedia_Search_Tool", [node.agent_id for node in nodes])
            before = len(recorder.trace.nodes)
            self.assertEqual(recorder.begin("Wikipedia_Search_Tool", "tool", 2, {"query": "x"}), "")
            self.assertEqual(len(recorder.trace.nodes), before)
            self.assertTrue(any("Wikipedia_Search_Tool" in issue for issue in recorder.trace.errors))

    def test_declared_tool_agent_stays_a_node(self):
        from workflow.execution_recording import ExecutionRecorder

        spec = _linear_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        win = MockWindowLLM({
            "planner": [json.dumps({
                "next": "python_coder", "args": {"code": "result=42"}, "sub_goal": "compute", "done": True,
            })],
            "verifier": [json.dumps({
                "ok": True, "reason": "ok", "step_conclusion": "COMPLETE", "slot_updates": [],
            })],
        })
        with tempfile.TemporaryDirectory() as tmp:
            recorder = ExecutionRecorder(Path(tmp), "rollout", "attempt")
            run_centralized_episode(
                {"id": "tools", "question": "1+1"},
                None, arch, spec, MemoryStore(), compiled,
                window_llm=win, execution_recorder=recorder,
            )
            ids = [node.agent_id for node in recorder.trace.nodes]
            self.assertEqual(ids, ["planner", "python_coder", "verifier"])
            self.assertNotIn("route_exec", ids)


# Live local-LLM test (skipped when the local vLLM endpoint is down) ---------

class TestLiveLocalLLM(unittest.TestCase):
    def setUp(self):  # noqa: D401
        from tests.conftest import _endpoint_up  # type: ignore

        if not _endpoint_up():
            self.skipTest("local LLM not running")
        super().setUp()

    def test_linear_pev_live(self):
        from tests.conftest import LOCAL_LLM_BASE, LOCAL_LLM_NAME  # type: ignore
        from workflow.centralized_runtime import LocalWindowLLM
        from workflow.runtime import LLMConfig

        spec = _linear_spec()
        compiled = compile_spec(spec)
        arch = register_archive(Archive())
        llm = LLMConfig(endpoint=LOCAL_LLM_BASE, model=LOCAL_LLM_NAME, api_key="EMPTY")
        win = LocalWindowLLM(llm)
        raw = run_centralized_episode(
            {"id": "live1", "question": "What is 6 times 7? Use python_coder to compute."},
            llm, arch, spec, MemoryStore(), compiled, window_llm=win,
        )
        ids = [e.get("agent_id") for e in raw.window_events]
        self.assertIn("planner", ids, f"error={raw.error} kinds={[m.get('kind') for m in raw.messages]}")
        self.assertIn("python_coder", ids)
        self.assertIn("route_exec", ids)
        self.assertTrue(
            raw.final_answer or any(m.get("kind") == "tool_result" for m in raw.messages),
            f"no answer and no tool_result; error={raw.error}",
        )


if __name__ == "__main__":
    unittest.main()
