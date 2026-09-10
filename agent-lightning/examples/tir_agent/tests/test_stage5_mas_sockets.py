"""v1.1: Skill.run on the hot path, two-layer Memory, YAML tools bind mock."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestSkillHotPath(unittest.TestCase):
    def test_react_loop_is_invoked(self):
        from workflow import Collector

        traj = Collector(mock=True).collect_one(
            {"id": "s", "question": "1+1", "answer": "2", "source": "gsm8k", "_mock_answer": "2"}
        )
        self.assertIn("react_loop", traj.meta.get("skills") or [])

    def test_custom_skill_and_route(self):
        from workflow.plugins import REGISTRY
        from workflow.runtime import ExecutionService
        from workflow.spec import MASSpec, HubSpec

        class Probe:
            name = "probe"
            called = False

            def run(self, ctx):
                Probe.called = True
                if "task" not in ctx:
                    raise AssertionError("skill ctx missing task")
                return {"ok": True, "output": "ok", "route": "hub"}

        saved = dict(REGISTRY._skills)
        try:
            REGISTRY.register_skill(Probe())
            spec = MASSpec(hub=HubSpec(skills=["react_loop", "probe"]))
            traj = ExecutionService(mock=True, spec=spec).run(
                {"id": "p", "question": "1+1", "answer": "2", "_mock_answer": "2"}
            )
            self.assertTrue(Probe.called)
            self.assertEqual(traj.meta.get("skill_route"), "hub")
            self.assertEqual(traj.meta.get("skills"), ["react_loop", "probe"])
        finally:
            REGISTRY._skills.clear()
            REGISTRY._skills.update(saved)

    def test_unknown_skill_in_spec_raises(self):
        from workflow.runtime import ExecutionService
        from workflow.spec import MASSpec, HubSpec

        spec = MASSpec(hub=HubSpec(skills=["causal_analysis"]))
        with self.assertRaises(KeyError):
            ExecutionService(mock=True, spec=spec).run({"id": "x", "question": "q", "answer": "a"})


class TestYamlToolsBindMock(unittest.TestCase):
    def test_drop_wikipedia_from_yaml(self):
        from workflow.runtime import ExecutionService
        from workflow.spec import load_spec

        text = """
schema_version: "0.1.0"
topology: hub_react
hub:
  role: orchestrator
  skills:
    - react_loop
tools:
  - web_search
  - execute_python
llm:
  kind: api
  model: ""
  base_url: ""
memory:
  agent: messages
  system: none
archive:
  window: post_first_tool
"""
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(text)
            path = f.name
        spec = load_spec(path)
        self.assertNotIn("wikipedia_search", spec.tools)
        traj = ExecutionService(mock=True, spec=spec).run(
            {"id": "y", "question": "1+1", "answer": "2", "_mock_answer": "2"}
        )
        self.assertEqual(traj.meta.get("tools"), ["web_search", "execute_python"])
        self.assertNotIn("wikipedia_search", traj.meta.get("tools") or [])
        names = [e.payload.get("name") for e in traj.events if e.payload.get("name")]
        self.assertNotIn("wikipedia_search", names)

    def test_search_only_tools_mock(self):
        from workflow.runtime import ExecutionService
        from workflow.spec import MASSpec

        spec = MASSpec(tools=["web_search"])
        traj = ExecutionService(mock=True, spec=spec).run(
            {"id": "z", "question": "capital", "answer": "Paris", "_mock_answer": "Paris"}
        )
        self.assertEqual(traj.meta.get("n_search"), 1)
        self.assertEqual(traj.meta.get("n_python"), 0)


class TestMemoryTwoLayer(unittest.TestCase):
    def test_agent_write_and_no_system_when_none(self):
        from workflow.contracts import EventKind
        from workflow.runtime import ExecutionService

        svc = ExecutionService(mock=True)
        traj = svc.run({"id": "m", "question": "1+1", "answer": "2", "_mock_answer": "2"})
        kinds = [e.kind for e in traj.events]
        self.assertIn(EventKind.MEMORY_READ, kinds)
        self.assertIn(EventKind.MEMORY_WRITE, kinds)
        scopes = [e.payload.get("scope") for e in traj.events if e.kind == EventKind.MEMORY_WRITE]
        self.assertEqual(scopes, ["agent"])
        mem = traj.meta.get("memory") or []
        self.assertEqual(mem[0]["scope"], "agent")
        self.assertEqual(mem[0]["owner"], "hub")
        self.assertIsNotNone(mem[0].get("content"))
        self.assertEqual(len(svc.memory.read("system")), 0)

    def test_system_kv_when_enabled(self):
        from workflow.contracts import EventKind
        from workflow.runtime import ExecutionService
        from workflow.spec import MASSpec, MemorySpec

        spec = MASSpec(memory=MemorySpec(agent="messages", system="kv"))
        svc = ExecutionService(mock=True, spec=spec)
        traj = svc.run({"id": "sys", "question": "1+1", "answer": "2", "_mock_answer": "2"})
        scopes = [e.payload.get("scope") for e in traj.events if e.kind == EventKind.MEMORY_WRITE]
        self.assertIn("system", scopes)
        self.assertEqual(len(svc.memory.read("system", "mas")), 1)
        self.assertEqual(len([m for m in traj.meta.get("memory") or [] if m.get("scope") == "system"]), 1)

    def test_second_run_reads_prior_agent_memory(self):
        from workflow.runtime import ExecutionService

        svc = ExecutionService(mock=True)
        svc.run({"id": "a", "question": "1+1", "answer": "2", "_mock_answer": "2"})
        traj2 = svc.run({"id": "b", "question": "2+2", "answer": "4", "_mock_answer": "4"})
        reads = [e for e in traj2.events if e.kind.value == "memory_read"]
        self.assertTrue(reads)
        self.assertGreaterEqual(int(reads[0].payload.get("n") or 0), 1)


class TestRunnerInjection(unittest.TestCase):
    def test_custom_runner(self):
        from workflow.runtime import ExecutionService, MockRunner

        class Marker(MockRunner):
            called = False

            def run(self, task, llm, archive, spec, memory):
                Marker.called = True
                return super().run(task, llm, archive, spec, memory)

        svc = ExecutionService(mock=True, runner=Marker())
        svc.run({"id": "r", "question": "1+1", "answer": "2", "_mock_answer": "2"})
        self.assertTrue(Marker.called)
        self.assertNotIn("agentlightning", sys.modules)


if __name__ == "__main__":
    unittest.main()
