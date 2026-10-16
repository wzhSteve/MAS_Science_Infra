"""A drag-built centralized graph with empty prompts still walks one PEV turn.

The canvas writes the same defaults, and an older graph whose prompts were
saved as empty strings falls back to them at runtime.
"""

from __future__ import annotations

import json
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

from workflow.archive import Archive, register_archive  # noqa: E402
from workflow.centralized_runtime import MockWindowLLM, run_centralized_episode  # noqa: E402
from workflow.compiler import compile_spec  # noqa: E402
from workflow.memory import MemoryStore  # noqa: E402
from workflow.prompts import DEFAULT_PLANNER_PROMPT, DEFAULT_VERIFIER_PROMPT  # noqa: E402
from workflow.spec import MASSpec  # noqa: E402


def _drag_spec(*, planner_prompt: str = "", verifier_prompt: str = "") -> MASSpec:
    return MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "hub": {"role": "planner", "skills": [], "system_prompt": planner_prompt},
        "tools": ["base_generator"],
        "agents": [
            {
                "id": "planner",
                "kind": "planner",
                "role": "planner",
                "system_prompt": planner_prompt,
                "trainable": True,
            },
            {"id": "base_generator", "kind": "tool", "role": "tool", "trainable": False},
            {
                "id": "verifier",
                "kind": "verifier",
                "role": "verifier",
                "system_prompt": verifier_prompt,
                "trainable": False,
            },
        ],
        "routers": [{"id": "router", "candidates": ["base_generator"], "strategy": "from_plan"}],
        "edges": [
            {"from": "planner", "to": "router", "kind": "route"},
            {"from": "router", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
    })


class _RecordingLLM(MockWindowLLM):
    def __init__(self, responses):
        super().__init__(responses)
        self.seen: list[tuple[str, str]] = []
        self.users: list[tuple[str, str]] = []

    def invoke(self, system_prompt: str, user_text: str, *, agent_id: str, kind: str) -> str:
        self.seen.append((agent_id, system_prompt))
        self.users.append((agent_id, user_text))
        return super().invoke(system_prompt, user_text, agent_id=agent_id, kind=kind)


class TestDragDefaultMas(unittest.TestCase):
    def test_empty_prompts_walk_one_pev_turn(self) -> None:
        spec = _drag_spec()
        compiled = compile_spec(spec)
        self.assertTrue(compiled.ok, compiled.reason)
        plan = json.dumps({
            "next": "base_generator",
            "args": {"query": "drag"},
            "sub_goal": "answer",
            "done": False,
        })
        done = json.dumps({"next": "", "args": {}, "sub_goal": "", "done": True})
        verify = json.dumps({
            "ok": True,
            "reason": "done",
            "step_conclusion": "COMPLETE",
            "slot_updates": [],
        })
        llm = _RecordingLLM({"planner": [plan, done], "verifier": [verify]})
        archive = register_archive(Archive())
        raw = run_centralized_episode(
            {"id": "drag", "question": "what is 1+1?"},
            None,
            archive,
            spec,
            MemoryStore(),
            compiled,
            window_llm=llm,
        )
        kinds = [message.get("kind") or message.get("type") for message in raw.messages]
        self.assertIn("tool_result", kinds)
        self.assertTrue(any(item[0] == "verifier" for item in llm.seen))
        prompts = {agent: text for agent, text in llm.seen}
        self.assertEqual(prompts["planner"], DEFAULT_PLANNER_PROMPT)
        self.assertEqual(prompts["verifier"], DEFAULT_VERIFIER_PROMPT)
        self.assertTrue(raw.final_answer or raw.messages)

    def test_written_prompts_are_not_replaced(self) -> None:
        spec = _drag_spec(planner_prompt="keep planner", verifier_prompt="keep verifier")
        compiled = compile_spec(spec)
        llm = _RecordingLLM({"planner": [], "verifier": []})
        run_centralized_episode(
            {"id": "keep", "question": "q"},
            None,
            register_archive(Archive()),
            spec,
            MemoryStore(),
            compiled,
            window_llm=llm,
        )
        prompts = {agent: text for agent, text in llm.seen}
        self.assertEqual(prompts["planner"], "keep planner")
        self.assertEqual(prompts["verifier"], "keep verifier")

    def test_three_invalid_tool_names_stop_the_episode(self) -> None:
        spec = _drag_spec()
        compiled = compile_spec(spec)
        bad = json.dumps({"next": "agent1", "args": {}, "sub_goal": "search", "done": False})
        llm = _RecordingLLM({"planner": [bad, bad, bad, bad]})
        raw = run_centralized_episode(
            {"id": "bad-next", "question": "q"},
            None,
            register_archive(Archive()),
            spec,
            MemoryStore(),
            compiled,
            window_llm=llm,
        )
        self.assertEqual(sum(1 for agent, _text in llm.seen if agent == "planner"), 3)
        self.assertIn("agent1", raw.error or "")

    def test_complete_without_answer_field_stops_after_one_tool(self) -> None:
        spec = _drag_spec()
        compiled = compile_spec(spec)
        bad = json.dumps({"next": "agent1", "args": {}, "sub_goal": "search", "done": False})
        good = json.dumps({
            "next": "base_generator",
            "args": {"query": "drag"},
            "sub_goal": "answer",
            "done": False,
        })
        again = json.dumps({
            "next": "base_generator",
            "args": {"query": "again"},
            "sub_goal": "again",
            "done": False,
        })
        verify = json.dumps({
            "ok": False,
            "reason": "John de Mol Jr.",
            "step_conclusion": "COMPLETE",
            "slot_updates": [],
        })
        llm = _RecordingLLM({"planner": [bad, good, again], "verifier": [verify]})
        raw = run_centralized_episode(
            {"id": "complete", "question": "who?"},
            None,
            register_archive(Archive()),
            spec,
            MemoryStore(),
            compiled,
            window_llm=llm,
        )
        planner_users = [text for agent, text in llm.users if agent == "planner"]
        self.assertEqual(len(planner_users), 2)
        self.assertIn("base_generator", planner_users[1])
        self.assertIn("set done to true and next to empty", planner_users[1])
        tools = [message for message in raw.messages if message.get("kind") == "tool_result"]
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["src"], "base_generator")
        self.assertEqual(raw.final_answer, "John de Mol Jr.")
        self.assertFalse(raw.error)
