"""Stage 1: MAS construct + data loop (no GPU / no agentlightning)."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORBIDDEN = {"agentlightning", "verl", "ray", "lit_tir_agent"}


class TestWorkflowDeps(unittest.TestCase):
    def test_workflow_ast_forbids_rl_imports(self):
        wf = ROOT / "workflow"
        errs = []
        for path in wf.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            found: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        found.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    found.add(node.module.split(".")[0])
            for name in FORBIDDEN & found:
                errs.append(f"{path.relative_to(ROOT)} imports {name}")
        self.assertEqual(errs, [])


class TestSpecAndPlugins(unittest.TestCase):
    def test_load_hub_react_yaml(self):
        from workflow.plugins import REGISTRY
        from workflow.spec import load_spec

        spec = load_spec()
        self.assertEqual(spec.topology, "hub_react")
        self.assertEqual(spec.memory.agent, "messages")
        self.assertEqual(spec.memory.system, "none")
        self.assertIn("execute_python", spec.tools)
        self.assertIn("web_search", spec.tools)
        for name in spec.hub.skills:
            REGISTRY.get_skill(name)
        REGISTRY.get_role("hub")
        REGISTRY.get_role("planner")
        REGISTRY.get_role("executor")
        with self.assertRaises(KeyError):
            REGISTRY.get_skill("causal_analysis")

    def test_missing_spec_falls_back_to_defaults(self):
        from workflow.spec import load_spec

        spec = load_spec("/tmp/no-such-hub-react.yaml")
        self.assertEqual(spec.topology, "hub_react")
        self.assertIn("execute_python", spec.tools)


class TestCanonicalReward(unittest.TestCase):
    def test_algos_reexports_workflow(self):
        import algos.rewards as ar
        import workflow.rewards as wr

        self.assertIs(ar.compute_outcome_reward, wr.compute_outcome_reward)

    def test_format_and_accuracy(self):
        from workflow.rewards import compute_outcome_reward

        self.assertEqual(
            compute_outcome_reward(None, "2", source="gsm8k", format_ok=False, n_search=0, n_python=0),
            -1.0,
        )
        self.assertEqual(
            compute_outcome_reward("9", "2", source="gsm8k", format_ok=True, n_search=0, n_python=0),
            0.0,
        )
        self.assertEqual(
            compute_outcome_reward("2", "2", source="gsm8k", format_ok=True, n_search=0, n_python=1),
            1.0,
        )
        self.assertAlmostEqual(
            compute_outcome_reward("2", "2", source="gsm8k", format_ok=True, n_search=1, n_python=1),
            1.1,
        )


class TestMockCollect(unittest.TestCase):
    def test_mock_events_reward_and_memory(self):
        from workflow import Collector, batch_to_train_signal
        from workflow.contracts import EventKind

        tasks = [
            {"id": "demo-1", "question": "What is 1+1?", "answer": "2", "source": "gsm8k", "_mock_answer": "2"},
            {"id": "demo-2", "question": "What is 2+2?", "answer": "4", "source": "gsm8k", "_mock_answer": "4"},
        ]
        batch = Collector(mock=True, n=1).collect(tasks)
        self.assertEqual(batch.meta.get("n_trajectories"), 2)
        t0 = batch.trajectories[0]
        kinds = {e.kind for e in t0.events}
        self.assertIn(EventKind.TOOL_CALL, kinds)
        self.assertIn(EventKind.TOOL_RESULT, kinds)
        self.assertIn(EventKind.FINAL_ANSWER, kinds)
        self.assertGreaterEqual(float(t0.final_reward or 0), 0.0)
        self.assertEqual(t0.meta.get("n_python"), 1)
        self.assertEqual(t0.meta.get("n_search"), 0)
        self.assertTrue(t0.meta.get("format_ok"))
        mem = t0.meta.get("memory") or []
        self.assertEqual(mem[0]["scope"], "agent")
        self.assertEqual(mem[0]["owner"], "hub")
        self.assertNotIn("agentlightning", sys.modules)
        sig = batch_to_train_signal(batch, algo="grpo")
        self.assertEqual(sig.advantage.name, "grpo")

    def test_group_n(self):
        from workflow import Collector

        batch = Collector(mock=True, n=2).collect(
            [{"id": "g", "question": "1+1", "answer": "2", "source": "gsm8k", "_mock_answer": "2"}]
        )
        self.assertEqual(batch.meta.get("n_trajectories"), 2)
        self.assertEqual(batch.meta.get("group_n"), 2)

    def test_run_without_llm_raises(self):
        from workflow.runtime import ExecutionService

        with self.assertRaises(RuntimeError):
            ExecutionService(mock=False).run({"id": "x", "question": "1+1", "answer": "2"})


class TestContextBudget(unittest.TestCase):
    def test_fit_messages_and_completion_cap(self):
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        from tir_agent import clip_text, fit_messages_for_context, remaining_completion_tokens

        self.assertIn("truncated", clip_text("x" * 400, 80))
        msgs = [
            SystemMessage(content="sys"),
            HumanMessage(content="q"),
            ToolMessage(content="wiki " * 2000, tool_call_id="c1"),
        ]
        fitted = fit_messages_for_context(msgs, max_prompt_tokens=900)
        self.assertLess(len(str(fitted[-1].content)), 900)
        n = remaining_completion_tokens(fitted, max_model_len=2560, max_tokens=1024)
        self.assertLessEqual(n + 8, 2560)
        self.assertGreaterEqual(n, 64)
        # 2160 prompt + 1024 completion would overflow 2560
        overflow = remaining_completion_tokens(
            [SystemMessage(content="m" * 6000)],
            max_model_len=2560,
            max_tokens=1024,
        )
        self.assertEqual(overflow, 64)


if __name__ == "__main__":
    unittest.main()
