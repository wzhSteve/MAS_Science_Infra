"""Agent-framework unit tests (Stage A: registry / router / memory / PEV)."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "mas"))

from workflow.agents import AgentRegistry, BlankAgent  # noqa: E402
from workflow.compiler import compile_spec  # noqa: E402
from workflow.contracts import MemoryItem  # noqa: E402
from workflow.memory import MemoryStore  # noqa: E402
from workflow.spec import AgentNodeSpec, MASSpec, RouterSpec  # noqa: E402


def _spec_with_tools() -> MASSpec:
    return MASSpec.model_validate(
        {
            "schema_version": "0.3",
            "topology": "hub_react",
            "tools": ["execute_python"],
            "hub": {"system_prompt": "test"},
        }
    )


def _spec_with_agents_and_router() -> MASSpec:
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
                {
                    "id": "expert_1",
                    "kind": "blank",
                    "system_prompt": "You are expert 1",
                    "profile": {"skills": ["physics"], "memory": {"policy": "append_latest", "max_items": 2}},
                },
            ],
            "routers": [
                {"id": "route_x", "candidates": ["execute_python", "wikipedia_search", "expert_1"]},
            ],
            "edges": [
                {"from": "planner", "to": "route_x", "kind": "message"},
            ],
        }
    )


class TestAgentRegistry(unittest.TestCase):
    def test_dual_backend_binding_pure_default(self):
        reg = AgentRegistry.from_spec(_spec_with_tools())
        self.assertIn("execute_python", reg.tool_agents)
        self.assertEqual(reg.tool_agents["execute_python"].backend, "pure")

    def test_llm_required_not_bound_in_training_mode(self):
        spec = _spec_with_tools()
        spec.agents = [AgentNodeSpec(id="execute_python", kind="tool", profile={"llm_required": True})]
        reg = AgentRegistry.from_spec(spec, allow_llm_backends=False)
        ta = reg.tool_agents["execute_python"]
        self.assertEqual(ta.backend, "pure")  # training keeps pure backend
        out = ta.invoke({"code": "result = 6 * 7"})
        self.assertIn("42", out)

    def test_llm_backend_bound_in_test_mode(self):
        spec = _spec_with_tools()
        spec.agents = [AgentNodeSpec(id="execute_python", kind="tool", profile={"llm_required": True})]
        reg = AgentRegistry.from_spec(spec, allow_llm_backends=True)
        self.assertEqual(reg.tool_agents["execute_python"].backend, "llm")

    def test_pure_backend_round_trip(self):
        reg = AgentRegistry.from_spec(_spec_with_tools())
        out = reg.tool_agents["execute_python"].invoke({"code": "result = 41 + 1"})
        self.assertIn("42", str(out))

    def test_epc_aw_import_not_polluted(self):
        AgentRegistry.from_spec(_spec_with_tools())
        agl_mods = [m for m in sys.modules if "agentlightning" in m or "epc_aw" in m]
        self.assertEqual(agl_mods, [])


class TestRouterRuntime(unittest.TestCase):
    def test_router_compiles_into_tools_for(self):
        spec = _spec_with_agents_and_router()
        c = compile_spec(spec)
        self.assertTrue(c.ok, c.issues)
        self.assertIn("route_x", c.routers)
        self.assertEqual(c.routers["route_x"].candidates[0], "execute_python")

    def test_router_candidates_are_subset_of_agents(self):
        spec = _spec_with_agents_and_router()
        c = compile_spec(spec)
        for cand in c.routers["route_x"].candidates:
            self.assertIn(cand, c.agents)

    def test_bind_tools_via_router_upstream(self):
        spec = _spec_with_agents_and_router()
        c = compile_spec(spec)
        # upstream of route_x is planner → tool-agent candidates routed there
        self.assertIn("execute_python", c.tools_for.get("planner", []))
        self.assertIn("wikipedia_search", c.tools_for.get("planner", []))

    def test_window_event_carries_router_id(self):
        from tir_agent import TirAgent

        spec = _spec_with_agents_and_router()
        routers = {r.id: {"candidates": r.candidates, "strategy": r.strategy} for r in spec.routers}
        agent = TirAgent(
            endpoint="http://localhost:1",
            model_name="test",
            enabled_tools=["execute_python"],
            routers=routers,
            agent_id="planner",
        )
        self.assertEqual(agent.agent_id, "planner")
        self.assertEqual(agent._router_by_tool["execute_python"], "route_x")

    def test_invalid_router_candidate_fails_compile(self):
        spec = _spec_with_agents_and_router()
        spec.routers = [RouterSpec(id="bad", candidates=["nonexistent"])]
        c = compile_spec(spec)
        self.assertFalse(c.ok)
        self.assertTrue(any("nonexistent" in i for i in c.issues))


class TestBlankAgentRouting(unittest.TestCase):
    """W1: kind=blank router candidates routed via tool-call shells."""

    def _spec(self) -> MASSpec:
        return MASSpec.model_validate(
            {
                "schema_version": "0.3",
                "topology": "graph",
                "entry_agent": "planner",
                "tools": ["execute_python"],
                "hub": {"system_prompt": "planner prompt"},
                "agents": [
                    {"id": "planner", "kind": "planner", "tools": ["execute_python"]},
                    {"id": "execute_python", "kind": "tool"},
                    {
                        "id": "expert_phys",
                        "kind": "blank",
                        "system_prompt": "You are a physicist.",
                        "profile": {"skills": ["physics"]},
                    },
                ],
                "routers": [
                    {"id": "route_x", "candidates": ["execute_python", "expert_phys"]},
                ],
                "edges": [
                    {"from": "planner", "to": "route_x", "kind": "message"},
                ],
            }
        )

    def test_blank_candidate_compiles_into_tools_for(self):
        c = compile_spec(self._spec())
        self.assertTrue(c.ok, c.issues)
        self.assertIn("blank:expert_phys", c.tools_for.get("planner", []))
        self.assertIn("execute_python", c.tools_for.get("planner", []))

    def test_blank_prefixed_candidate_also_compiles(self):
        """UI writes blank:<id> candidates — compiler accepts both forms."""
        spec = self._spec()
        spec.routers = [
            RouterSpec(id="route_x", candidates=["execute_python", "blank:expert_phys"])
        ]
        c = compile_spec(spec)
        self.assertTrue(c.ok, c.issues)
        self.assertIn("blank:expert_phys", c.tools_for.get("planner", []))

    def test_blank_candidate_also_valid_agent_node(self):
        c = compile_spec(self._spec())
        self.assertIn("expert_phys", c.agents)
        self.assertEqual(c.agents["expert_phys"].kind, "blank")

    def test_blank_agent_adapter_invoke_mock_llm(self):
        from tools.tool_agents import BlankAgentAdapter

        spec = self._spec()
        blank = next(a for a in spec.agents if a.kind == "blank")
        adapter = BlankAgentAdapter.from_agent(blank)

        class _MockLLM:
            def invoke(self, prompt):
                return f"MOCK::{prompt}"

        adapter.bind_llm(_MockLLM())
        self.assertEqual(adapter.id, "blank:expert_phys")
        self.assertEqual(adapter.kind, "blank")
        self.assertIn("expert_phys", adapter.description)
        out = adapter.invoke({"input": "What is F=ma?"})
        self.assertIn("You are a physicist.", out)
        self.assertIn("What is F=ma?", out)
        self.assertIn("MOCK::", out)

    def test_blank_agent_adapter_without_llm_returns_error(self):
        from tools.tool_agents import BlankAgentAdapter

        spec = self._spec()
        blank = next(a for a in spec.agents if a.kind == "blank")
        adapter = BlankAgentAdapter.from_agent(blank)
        out = adapter.invoke({"input": "q"})
        self.assertIn("no bound LLM", out)

    def test_tir_agent_binds_blank_candidate_end_to_end(self):
        from tir_agent import TirAgent

        spec = self._spec()
        agent = TirAgent.from_spec(
            spec,
            endpoint="http://localhost:1",
            model_name="test",
            enabled_tools=["execute_python", "blank:expert_phys"],
            system_prompt="planner prompt",
            agent_id="planner",
        )
        # adapter bound on the invoker
        self.assertIn("blank:expert_phys", agent.tool_agent_invoker.known_ids)
        self.assertIn("blank:expert_phys", agent._blank_adapters)
        # router context registered for the prefixed form
        self.assertEqual(agent._router_by_tool.get("blank:expert_phys"), "route_x")
        # replace the bound LLM with a mock and route one hop through the shell
        adapter = agent._blank_adapters["blank:expert_phys"]

        class _MockLLM:
            def invoke(self, prompt):
                return f"EXPERT_SAID::{len(prompt)}"

        adapter.bind_llm(_MockLLM())
        out = agent.tool_agent_invoker.invoke("blank:expert_phys", {"input": "solve"})
        self.assertIn("EXPERT_SAID::", out)

    def test_call_tools_window_event_marks_blank(self):
        """call_tools on a blank:<id> hop records tool_id=blank:<id> + agent_kind=blank."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from tir_agent import TirAgent

        spec = self._spec()
        agent = TirAgent.from_spec(
            spec,
            endpoint="http://localhost:1",
            model_name="test",
            enabled_tools=["blank:expert_phys"],
            system_prompt="planner prompt",
            agent_id="planner",
        )
        adapter = agent._blank_adapters["blank:expert_phys"]

        class _MockLLM:
            def invoke(self, prompt):
                return "expert answer"

        adapter.bind_llm(_MockLLM())
        state = {
            "question": "q?",
            "num_turns": 1,
            "asked_finalize": False,
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="q?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "blank:expert_phys", "args": {"input": "solve"}, "id": "call_1"}
                    ],
                ),
            ],
            "turn_records": [],
            "n_search": 0,
            "n_python": 0,
            "branch_messages": [],
            "window_snapshots": [],
            "window_events": [],
        }
        out = agent.call_tools(state)
        self.assertEqual(len(out["messages"]), 4)  # +1 ToolMessage
        tm = out["messages"][-1]
        self.assertEqual(getattr(tm, "tool_call_id", ""), "call_1")
        self.assertIn("expert answer", str(tm.content))
        ev = out["window_events"][-1]
        self.assertEqual(ev["tool_id"], "blank:expert_phys")
        self.assertEqual(ev["metrics"].get("agent_kind"), "blank")
        self.assertEqual(ev["metrics"].get("router_id"), "route_x")


class TestTwoLayerMemory(unittest.TestCase):
    def test_agent_scope_isolation(self):
        mem = MemoryStore()
        mem.write_agent("a1", {"v": 1}, memory_scope="agent")
        mem.write_agent("a2", {"v": 2}, memory_scope="agent")
        self.assertEqual(len(mem.read_agent("a1", memory_scope="agent")), 1)
        self.assertEqual(len(mem.read_agent("a2", memory_scope="agent")), 1)
        self.assertNotEqual(
            mem.read_agent("a1", memory_scope="agent")[0].content["v"],
            mem.read_agent("a2", memory_scope="agent")[0].content["v"],
        )

    def test_shared_scope_interop(self):
        mem = MemoryStore()
        mem.write_agent("a1", {"v": 1}, memory_scope="shared")
        mem.write_agent("a2", {"v": 2}, memory_scope="shared")
        # both write to hub buffer → 2 items visible to both agents
        self.assertEqual(len(mem.read_agent("a1", memory_scope="shared")), 2)
        self.assertEqual(len(mem.read_agent("a2", memory_scope="shared")), 2)

    def test_append_latest_policy_truncation(self):
        mem = MemoryStore()
        for i in range(5):
            mem.write_agent("a1", {"i": i}, memory_scope="agent")
        items = mem.read_agent(
            "a1", memory_scope="agent", policy={"policy": "append_latest", "max_items": 2}
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(items[-1].content["i"], 4)

    def test_profile_memory_policy_from_spec(self):
        from workflow.runtime import _node_memory_settings

        spec = _spec_with_agents_and_router()
        scope, policy = _node_memory_settings(spec, "expert_1")
        self.assertEqual(scope, "agent")
        self.assertEqual(policy, {"policy": "append_latest", "max_items": 2})
        scope2, _ = _node_memory_settings(spec, "planner")
        self.assertEqual(scope2, "agent")  # explicit agent default in schema 0.3


class TestPevExecutor(unittest.TestCase):
    def test_planner_tools_are_tool_agents(self):
        spec = _spec_with_agents_and_router()
        c = compile_spec(spec)
        tool_set = set(c.tools_for.get("planner", []))
        self.assertTrue({"execute_python", "wikipedia_search"} & tool_set)

    def test_mock_graph_episode_per_agent_window_events(self):
        from workflow.archive import Archive, register_archive
        from workflow.runtime import MockRunner, run_compiled_episode

        spec = _spec_with_agents_and_router()
        arch = register_archive(Archive())
        raw = run_compiled_episode(
            {"id": "t1", "question": "q?", "answer": "42"},
            None,
            arch,
            spec,
            MemoryStore(),
            MockRunner(),
        )
        self.assertIsNotNone(raw)
        agent_ids = {e.get("agent_id") for e in raw.window_events}
        self.assertIn("planner", agent_ids)
        self.assertTrue(all(e.get("agent_id") != "hub" for e in raw.window_events))
        snaps = {s.get("agent_id") for s in raw.window_snapshots}
        self.assertIn("planner", snaps)

    def test_tir_runner_passes_agent_id(self):
        from workflow.archive import Archive, register_archive
        from workflow.runtime import TirRunner, run_compiled_episode
        seen = {}

        class SpyRunner(TirRunner):
            def run(self, task, llm, archive, spec, memory, agent_id="hub"):
                seen["agent_id"] = agent_id
                from workflow.runtime import run_mock_episode

                return run_mock_episode(task, archive, spec=spec, memory=memory, agent_id=agent_id)

        spec = _spec_with_agents_and_router()
        arch = register_archive(Archive())
        run_compiled_episode(
            {"id": "t2", "question": "q?", "answer": "1"},
            None,
            arch,
            spec,
            MemoryStore(),
            SpyRunner(),
        )
        self.assertEqual(seen.get("agent_id"), "planner")


if __name__ == "__main__":
    unittest.main()
