"""Schema 0.3 tests: sugar expansion, tool-agent compiler, ToolAgentInvoker (P1)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestSchema03Sugar(unittest.TestCase):
    def test_old_yaml_expands_implicit_tool_agents(self):
        from workflow.spec import load_spec

        for yaml_path in ("experiments/arpo_e2e/workflow.yaml", "experiments/demo/workflow.yaml"):
            spec = load_spec(yaml_path)
            kinds = {a.id: a.kind for a in spec.agents}
            self.assertEqual(kinds.get("hub"), "hub", yaml_path)
            for t in spec.tools:
                self.assertEqual(kinds.get(t), "tool", f"{yaml_path}:{t}")
            # router field exists, defaults empty
            self.assertEqual(spec.routers, [])

    def test_compile_old_yaml_ok(self):
        from workflow.compiler import compile_spec, tool_agent_ids, trainable_agents
        from workflow.spec import load_spec

        spec = load_spec("experiments/arpo_e2e/workflow.yaml")
        c = compile_spec(spec)
        self.assertTrue(c.ok, c.reason)
        self.assertEqual(sorted(tool_agent_ids(spec)), sorted(spec.tools))
        self.assertEqual(trainable_agents(spec), ["hub"])
        self.assertFalse(c.multi_agent)

    def test_router_candidates_validation(self):
        from workflow.compiler import compile_spec
        from workflow.spec import MASSpec, _normalize_schema03

        raw = {
            "schema_version": "0.3",
            "topology": "graph",
            "entry_agent": "planner",
            "agents": [
                {"id": "planner", "role": "planner"},
                {"id": "executor", "role": "executor"},
            ],
            "edges": [{"from": "planner", "to": "executor", "kind": "route"}],
            "routers": [
                {"id": "pick", "candidates": ["planner", "executor"], "strategy": "llm_choice"}
            ],
        }
        _normalize_schema03(raw)
        c = compile_spec(MASSpec.model_validate(raw))
        self.assertTrue(c.ok, c.issues)

        raw["routers"][0]["candidates"] = ["planner", "ghost"]
        _normalize_schema03(raw)
        c2 = compile_spec(MASSpec.model_validate(raw))
        self.assertFalse(c2.ok)
        self.assertTrue(any("ghost" in i for i in c2.issues))


class TestToolAgentInvoker(unittest.TestCase):
    def test_registry_wraps_all_tools(self):
        from tools.tool_agents import TOOL_AGENTS

        self.assertTrue(len(TOOL_AGENTS) >= 3)
        for aid, agent in TOOL_AGENTS.items():
            self.assertEqual(agent.kind, "tool")
            self.assertEqual(agent.id, aid)

    def test_invoker_round_trip(self):
        from tools.tool_agents import TOOL_AGENTS
        from tir_agent import ToolAgentInvoker

        invoker = ToolAgentInvoker({})
        self.assertIn("execute_python", invoker.known_ids)
        out = invoker.invoke("execute_python", {"code": "print(21*2)"})
        self.assertIn("42", str(out))

        # fallback to legacy map for unregistered tool id
        legacy_invoker = ToolAgentInvoker({"execute_python": TOOL_AGENTS["execute_python"]})
        self.assertIn("no output", str(legacy_invoker.invoke("execute_python", {"code": "1"})))

    def test_unknown_tool_returns_none(self):
        from tir_agent import ToolAgentInvoker

        invoker = ToolAgentInvoker({})
        self.assertIsNone(invoker.invoke("ghost_tool", {}))


if __name__ == "__main__":
    unittest.main()
