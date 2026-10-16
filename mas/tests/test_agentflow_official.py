"""AgentFlow user project wraps the vendored AgentFlow-main (upload/) faithfully.

Real official Planner / Executor / Verifier / Memory / tool objects; only the
engine's network call is scripted per role.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

PROJECT = ROOT / "user_space" / "projects" / "agentflow"
ADAPTED = PROJECT / "adapted"
OFFICIAL = ROOT / "user_space" / "projects" / "agentflow" / "upload" / "AgentFlow-main" / "agentflow" / "agentflow"

PLAN = (
    "Justification: a generalist can add numbers.\n"
    "Context: none\nSub-Goal: compute 2+2\nTool Name: Generalist_Solution_Generator_Tool"
)
COMMAND = 'Generated Command:\n```python\nexecution = tool.execute(query="2+2")\n```'
ANALYSIS = "Concise Summary: add two numbers."
DIRECT = "1. **Process Summary:** used the generalist.\n2. **Answer:** 4"


def _load(name: str):
    if str(ADAPTED) not in sys.path:
        sys.path.insert(0, str(ADAPTED))
    spec = importlib.util.spec_from_file_location(name, ADAPTED / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@unittest.skipUnless(OFFICIAL.is_dir(), "upload/AgentFlow-main missing")
class AgentFlowOfficialCase(unittest.TestCase):
    verdicts: List[str]

    def setUp(self) -> None:
        self.engine = _load("engine")
        self.tools = _load("tools_local")
        self.ctx_mod = _load("context")
        self.calls: List[Dict[str, object]] = []
        self.verdicts = ["Explanation: enough.\nConclusion: STOP"]
        self.plans = [PLAN]
        test = self

        def fake_complete(engine_self, prompt, system_prompt, max_tokens):
            test.calls.append({"role": engine_self.role, "trainable": engine_self.trainable, "prompt": prompt})
            role = engine_self.role
            if role == "planner":
                return test.plans.pop(0) if len(test.plans) > 1 else test.plans[0]
            if role == "planner_fixed":
                return DIRECT if "Process Summary" in prompt else ANALYSIS
            if role == "executor":
                return COMMAND
            if role == "verifier":
                return test.verdicts.pop(0) if len(test.verdicts) > 1 else test.verdicts[0]
            return "4"

        p = patch.object(self.engine.LocalChatEngine, "_complete", fake_complete)
        p.start()
        self.addCleanup(p.stop)
        runtime = dict(self.ctx_mod.RUNTIME_DEFAULTS, max_steps=3)
        r = patch.object(self.ctx_mod, "load_runtime", return_value=runtime)
        r.start()
        self.addCleanup(r.stop)
        self.ctx = self.ctx_mod.EpisodeContext()

    def _round(self, question: str = "What is 2+2?"):
        plan = self.ctx.plan_window({"payload": {"question": question}})
        tool = self.ctx.exec_window({"payload": {}})
        verify = self.ctx.verify_window({"payload": {}})
        return plan, tool, verify


class TestComponents(AgentFlowOfficialCase):
    def test_official_toolbox_and_engine_roles(self) -> None:
        comps = self.ctx.components()
        self.assertEqual(
            comps.available_tools,
            sorted([
                "Bing_Search_Tool", "Generalist_Solution_Generator_Tool",
                "Python_Code_Generator_Tool", "Web_Fetch_Tool",
                "Wikipedia_RAG_Search_Tool",
            ]),
        )
        self.assertIn("Bing", comps.toolbox_metadata["Bing_Search_Tool"]["tool_description"])
        self.assertTrue(comps.planner.llm_engine.trainable)
        for eng in (
            comps.planner.llm_engine_fixed, comps.verifier.llm_engine_fixed,
            comps.executor.llm_generate_tool_command,
        ):
            self.assertFalse(eng.trainable)
        for inst in comps.executor.tool_instances_cache.values():
            if hasattr(inst, "llm_engine"):
                self.assertFalse(inst.llm_engine.trainable)

    def test_official_classes_are_used(self) -> None:
        comps = self.ctx.components()
        for obj in (comps.planner, comps.verifier, comps.executor):
            self.assertTrue(str(OFFICIAL) in sys.modules[type(obj).__module__].__file__)

    def test_bing_backend_is_local(self) -> None:
        comps = self.ctx.components()
        tool = comps.executor.tool_instances_cache["Bing_Search_Tool"]
        with patch.object(self.tools, "local_bing_search", return_value="serp") as fake:
            self.assertEqual(tool.execute(query="q"), "serp")
        fake.assert_called_once_with("q")

    def test_official_executor_runs_local_bing(self) -> None:
        """Regression: Executor always calls set_custom_output_dir before execute."""
        import tempfile

        comps = self.ctx.components()
        tool = comps.executor.tool_instances_cache["Bing_Search_Tool"]
        self.assertTrue(callable(getattr(tool, "set_custom_output_dir", None)))
        comps.executor.set_query_cache_dir(tempfile.mkdtemp(prefix="af_bing_"))
        with patch.object(self.tools, "local_bing_search", return_value="bing results for: q"):
            out = comps.executor.execute_tool_command(
                "Bing_Search_Tool",
                'execution = tool.execute(query="Oliver Reed Royal Flash")',
            )
        self.assertEqual(out, ["bing results for: q"])
        self.assertFalse(str(out).startswith("Error"))

    def test_rejects_non_official_tool(self) -> None:
        with self.assertRaises(ValueError):
            self.tools.build_toolbox(["Google_Search_Tool"], lambda: None)

    def test_web_fetch_backend_is_local(self) -> None:
        comps = self.ctx.components()
        tool = comps.executor.tool_instances_cache["Web_Fetch_Tool"]
        self.assertEqual(list(tool.input_types), ["url"])
        with patch("mas.tools.search.fetch_page", return_value="page") as fake:
            self.assertEqual(tool.execute(url="https://example.com"), "page")
        fake.assert_called_once()
        self.assertEqual(fake.call_args[0][0], "https://example.com")

    def test_official_executor_runs_local_web_fetch(self) -> None:
        """Regression: Executor always calls set_custom_output_dir before execute."""
        import tempfile

        comps = self.ctx.components()
        tool = comps.executor.tool_instances_cache["Web_Fetch_Tool"]
        self.assertTrue(callable(getattr(tool, "set_custom_output_dir", None)))
        comps.executor.set_query_cache_dir(tempfile.mkdtemp(prefix="af_fetch_"))
        with patch("mas.tools.search.fetch_page", return_value="fetched page") as fake:
            out = comps.executor.execute_tool_command(
                "Web_Fetch_Tool",
                'execution = tool.execute(url="https://example.com")',
            )
        self.assertEqual(out, ["fetched page"])
        self.assertFalse(str(out).startswith("Error"))
        fake.assert_called_once()
        self.assertEqual(fake.call_args[0][0], "https://example.com")


class TestEpisode(AgentFlowOfficialCase):
    def test_one_step_stop_runs_official_loop(self) -> None:
        plan, tool, verify = self._round()
        self.assertEqual(plan["payload"]["next"], self.ctx_mod.EXECUTOR_ID)
        self.assertEqual(plan["payload"]["args"]["tool_name"], "Generalist_Solution_Generator_Tool")
        self.assertEqual(tool["payload"]["tool"], "Base_Generator_Tool")
        self.assertEqual(tool["payload"]["output"], '["4"]')
        self.assertTrue(verify["payload"]["ready_to_stop"])
        self.assertEqual(verify["payload"]["answer"], "<answer>4</answer>")
        roles = [c["role"] for c in self.calls]
        self.assertEqual(
            roles,
            ["planner_fixed", "planner", "executor", "tool", "verifier", "planner_fixed", "planner_fixed"],
        )
        self.assertEqual(verify["payload"]["trace"].get("final_output"), ANALYSIS)
        self.assertNotIn("<answer>", str(verify["payload"]["trace"].get("final_output") or ""))

    def test_only_generate_next_step_is_trainable(self) -> None:
        self.verdicts = ["Conclusion: CONTINUE", "Conclusion: CONTINUE", "Conclusion: STOP"]
        for _ in range(3):
            self._round()
        trainable = [c for c in self.calls if c["trainable"]]
        self.assertEqual(len(trainable), 3)
        self.assertTrue(all("Determine the optimal next step" in str(c["prompt"]) for c in trainable))

    def test_shared_memory_feeds_next_plan(self) -> None:
        self.verdicts = ["Conclusion: CONTINUE", "Conclusion: STOP"]
        self._round()
        self._round()
        planner_prompts = [c["prompt"] for c in self.calls if c["role"] == "planner"]
        self.assertIn("Action Step 1", str(planner_prompts[1]))

    def test_restore_continues_from_generate_next_step(self) -> None:
        self.verdicts = ["Conclusion: CONTINUE", "Conclusion: STOP", "Conclusion: STOP"]
        plan, tool, _verify = self._round()
        replay = plan["payload"]["agentflow_state"]
        kept = tool["payload"]["agentflow_state"]
        self.assertEqual(replay["step_count"], 0)
        self.assertNotIn("Action Step 1", replay["actions"])
        self.assertIn("Action Step 1", kept["actions"])
        self.assertTrue(kept["query_analysis"])
        self.ctx_mod.reset_for_tests()
        fresh = self.ctx_mod.EpisodeContext()
        before = len(self.calls)
        restored = fresh.plan_window({"question": "2+2", "agentflow_state": kept})
        new_calls = self.calls[before:]
        self.assertTrue(new_calls)
        self.assertEqual(new_calls[0]["role"], "planner")
        self.assertNotIn("planner_fixed", [c["role"] for c in new_calls])
        self.assertIn("Action Step 1", str(new_calls[0]["prompt"]))
        self.assertFalse(restored["payload"]["done"])

    def test_max_steps_forces_direct_output(self) -> None:
        self.verdicts = ["Conclusion: CONTINUE"]
        for i in range(3):
            _plan, _tool, verify = self._round()
            self.assertEqual(verify["payload"]["ready_to_stop"], i == 2)
        self.assertEqual(verify["payload"]["answer"], "<answer>4</answer>")
        self.assertTrue(verify["payload"]["trace"]["budget_stop"])

    def test_unknown_tool_follows_official_not_found_path(self) -> None:
        self.plans = [PLAN.replace("Generalist_Solution_Generator_Tool", "Nonexistent_Tool")]
        _plan, tool, _verify = self._round()
        self.assertFalse(tool["payload"]["ok"])
        self.assertIn("tool was not found", tool["payload"]["output"])
        self.assertNotIn("executor", [c["role"] for c in self.calls])

    def test_episodes_are_isolated_by_episode_id(self) -> None:
        from workflow.user_gateway.sandbox import bind_episode_id, reset_episode_id

        @contextmanager
        def episode(eid: str):
            token = bind_episode_id(eid)
            try:
                yield
            finally:
                reset_episode_id(token)

        self.verdicts = ["Conclusion: CONTINUE"]
        with episode("a"):
            self.ctx.plan_window({"payload": {"question": "qa"}})
        with episode("b"):
            self.ctx.plan_window({"payload": {"question": "qb"}})
            self.ctx.exec_window({"payload": {}})
        with episode("a"):
            self.ctx.exec_window({"payload": {}})
        self.assertEqual(self.ctx._episodes["a"].question, "qa")
        self.assertEqual(len(self.ctx._episodes["a"].memory.get_actions()), 1)
        self.assertEqual(len(self.ctx._episodes["b"].memory.get_actions()), 1)


class TestFrozenScope(AgentFlowOfficialCase):
    def test_frozen_roles_enter_scope_trainable_does_not(self) -> None:
        entered: List[str] = []

        @contextmanager
        def scope(role: str):
            entered.append(role)
            yield

        with patch.object(self.engine, "frozen_llm_scope", scope):
            self._round()
        self.assertNotIn("planner", entered)
        self.assertEqual(
            entered,
            ["planner_fixed", "executor", "tool", "verifier", "planner_fixed", "planner_fixed"],
        )

    def test_public_scope_names_frozen_span(self) -> None:
        from workflow.agent_tracing import agent_node_name, agent_span_pattern
        import re

        pattern = agent_span_pattern(["planner"])
        self.assertIsNotNone(re.search(pattern, agent_node_name("planner")))
        self.assertIsNone(re.search(pattern, agent_node_name("frozen__planner_fixed")))


class TestWrapAnswer(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load("context") if OFFICIAL.is_dir() else None

    @unittest.skipUnless(OFFICIAL.is_dir(), "upload/AgentFlow-main missing")
    def test_variants(self) -> None:
        wrap = self.mod.wrap_answer
        self.assertEqual(wrap(DIRECT), "<answer>4</answer>")
        self.assertEqual(wrap("blah <answer>Paris</answer>"), "<answer>Paris</answer>")
        self.assertEqual(wrap("Answer: The answer is 18.", source="gsm8k"), "<answer>18</answer>")
        self.assertEqual(wrap(""), "")


class TestContracts(unittest.TestCase):
    def test_workflow_is_trainable_window_graph(self) -> None:
        from workflow.compiler import trainable_agents
        from workflow.runtime import agent_window_rollout
        from workflow.spec import load_spec

        spec = load_spec(str(PROJECT / "contracts" / "workflow.yaml"))
        self.assertTrue(agent_window_rollout(spec))
        self.assertEqual(trainable_agents(spec), ["planner"])
        self.assertIsNone(getattr(spec, "sampling", None) and spec.sampling.sites or None)
        verifier = next(a for a in spec.agents if a.id == "verifier")
        self.assertEqual((verifier.meta or {}).get("wraps"), "AgentFlow.Verifier")

    def test_detect_family(self) -> None:
        from workflow.user_gateway.detect import detect_mas_family

        self.assertEqual(detect_mas_family("agentflow", "AgentFlow"), "agentflow")
        self.assertEqual(detect_mas_family("other", "zip", wraps="AgentFlow.PEV"), "agentflow")


class TestTokenIdHook(unittest.TestCase):
    def test_parsed_completion_carries_injected_ids(self) -> None:
        import httpx
        from openai import OpenAI

        engine = _load("engine")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [{
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "4"},
                    }],
                },
            )

        def fake_encode(messages, completion):
            self.assertEqual(completion, "4")
            self.assertEqual(messages[-1]["content"], "2+2")
            return [11, 12], [13]

        with patch.object(engine, "encode_chat_ids", fake_encode):
            http_client = httpx.Client(transport=httpx.MockTransport(handler))
            engine._install_token_hook(http_client)
            client = OpenAI(base_url="http://example.test/v1", api_key="EMPTY", http_client=http_client)
            resp = client.chat.completions.with_raw_response.create(
                model="m", messages=[{"role": "user", "content": "2+2"}]
            ).parse()
        self.assertEqual(list(resp.prompt_token_ids), [11, 12])
        self.assertEqual(list(resp.response_token_ids[0]), [13])
        self.assertEqual(list(resp.choices[0].token_ids), [13])

    def test_server_ids_are_left_alone(self) -> None:
        engine = _load("engine")
        payload = {"object": "chat.completion", "prompt_token_ids": [7], "choices": []}
        self.assertIs(engine.attach_token_ids(b"{}", payload), payload)


if __name__ == "__main__":
    unittest.main()
