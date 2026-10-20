"""Memory prompt, episode isolation, and the GRPO / APPO mask contract. CPU only."""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MAS = ROOT / "mas"
ADAPTED = ROOT / "user_space" / "projects" / "simple_pev" / "adapted"
for path in (str(ROOT), str(MAS), str(ADAPTED)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _load_context():
    spec = importlib.util.spec_from_file_location("simple_pev_context_fidelity", ADAPTED / "context.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Mem:
    def __init__(self, actions):
        self._actions = actions

    def get_actions(self):
        return self._actions


class TestMemoryPrompt(unittest.TestCase):
    def test_prompt_shows_every_step_command_and_truncated_result(self) -> None:
        import sft_prompts

        long = "R" * 5000
        memory = _Mem({
            "Action Step 1": {
                "tool_name": "Bing_Search_Tool",
                "sub_goal": "find the birth year",
                "command": "execution = tool.execute(query='a')",
                "result": "snippet-one",
            },
            "Action Step 2": {
                "tool_name": "Wikipedia_RAG_Search_Tool",
                "sub_goal": "open the page",
                "command": "execution = tool.execute(query='b')",
                "result": long,
            },
            "Action Step 3": {
                "tool_name": "Web_Fetch_Tool",
                "sub_goal": "read the body",
                "command": "execution = tool.execute(url='https://example.test')",
                "result": "fetched body",
            },
            "Action Step 4": {
                "tool_name": "Bing_Search_Tool",
                "sub_goal": "check the other name",
                "command": "execution = tool.execute(query='d')",
                "result": "snippet-four",
            },
        })
        text = sft_prompts.recent_steps(memory, limit=4000)
        for step in ("Action Step 1", "Action Step 2", "Action Step 3", "Action Step 4"):
            self.assertIn(step, text)
        self.assertIn("command: execution = tool.execute(query='a')", text)
        self.assertIn("command: execution = tool.execute(url='https://example.test')", text)
        self.assertIn("...", text)
        self.assertNotIn("R" * 4000, text)
        result_line = text.split("result: ", 2)[2].split("\n", 1)[0]
        self.assertLessEqual(len(result_line), 4000)
        self.assertTrue(result_line.endswith("..."))

        class Planner:
            def llm_engine(self, prompt, **_kwargs):
                self.prompt = prompt
                return "Context: c\nSub-Goal: s\nTool Name: Bing_Search_Tool"

        planner = Planner()
        planner.generate_next_step = MethodType(sft_prompts.generate_next_step, planner)
        planner.generate_next_step("Who is older?", None, "notes", memory, 4, 6)
        self.assertIn("Action Step 1", planner.prompt)
        self.assertIn("command:", planner.prompt)
        self.assertIn("Recent steps:", planner.prompt)
        self.assertNotIn("R" * 4000, planner.prompt)


class TestEpisodeIsolation(unittest.TestCase):
    def test_empty_id_does_not_reuse_the_previous_question(self) -> None:
        ctx_mod = _load_context()
        ctx = ctx_mod.EpisodeContext()
        ctx.components = lambda: SimpleNamespace(log_key=threading.local())

        class FakeEpisode:
            def __init__(self, comps, question, source):
                del comps, source
                self.question = question
                self.done = False

        with patch.object(ctx_mod, "Episode", FakeEpisode):
            first = ctx.episode({"question": "question-a"}, create=True)
            empty = ctx.episode({}, create=True)
            again = ctx.episode({}, create=False)
        self.assertEqual(first.question, "question-a")
        self.assertIsNot(first, empty)
        self.assertIs(empty, again)
        self.assertNotEqual(empty.question, "question-a")


class TestTrainStepCap(unittest.TestCase):
    def test_training_uses_three_steps_eval_keeps_the_default(self) -> None:
        ctx_mod = _load_context()
        env = os.environ.copy()
        env.pop("AGENTFLOW_MAX_STEPS", None)
        env.pop("TIR_ALGO", None)
        env.pop("AGENTFLOW_TRAIN", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(ctx_mod.load_runtime()["max_steps"], 10)
        with patch.dict(os.environ, {**env, "TIR_ALGO": "flow_grpo"}, clear=True):
            self.assertEqual(ctx_mod.load_runtime()["max_steps"], 3)
        with patch.dict(os.environ, {**env, "TIR_ALGO": "grpo", "AGENTFLOW_MAX_STEPS": "6"}, clear=True):
            self.assertEqual(ctx_mod.load_runtime()["max_steps"], 6)


class TestFrozenSpanAndResponseMask(unittest.TestCase):
    def test_agent_match_drops_frozen_roles(self) -> None:
        import re

        from workflow.agent_tracing import agent_node_name, agent_span_pattern

        pattern = agent_span_pattern(["planner"])
        self.assertIsNotNone(re.search(pattern, agent_node_name("planner")))
        for role in ("frozen__planner_fixed", "frozen__executor", "frozen__verifier"):
            self.assertIsNone(re.search(pattern, agent_node_name(role)))

    def test_loss_mask_covers_the_response_slice_only(self) -> None:
        """dp_actor takes response_mask from the right-hand response length.

        Memory tokens sit in the prompt half, so they change logits and receive
        no policy gradient.
        """
        prompt_len, resp_len = 6, 4
        attention = torch.ones(1, prompt_len + resp_len)
        attention[0, :2] = 0
        attention[0, -1] = 0
        response_mask = attention[:, -resp_len:]
        self.assertEqual(tuple(response_mask.shape), (1, resp_len))
        self.assertEqual(int(response_mask.sum()), resp_len - 1)
        self.assertTrue(torch.equal(response_mask[0], torch.tensor([1, 1, 1, 0])))


class TestConstantGroupAdvantage(unittest.TestCase):
    def test_all_zero_and_all_one_groups_have_zero_advantage(self) -> None:
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        mask = torch.ones(4, 3)
        index = np.array(["q", "q", "q", "q"])
        for value in (0.0, 1.0):
            rewards = torch.zeros(4, 3)
            rewards[:, -1] = value
            adv, _returns = compute_grpo_outcome_advantage(rewards, mask, index)
            self.assertEqual(tuple(adv.shape), (4, 3))
            self.assertTrue(torch.count_nonzero(adv) == 0)

    def test_zero_variance_groups_are_counted(self) -> None:
        from verl import DataProto

        from rl.hooks.advantage import count_zero_variance_groups

        rewards = torch.zeros(6, 2)
        rewards[:4, -1] = 0
        rewards[4, -1] = 0
        rewards[5, -1] = 1
        batch = DataProto.from_dict(tensors={"token_level_rewards": rewards})
        batch.non_tensor_batch["uid"] = np.array(["flat", "flat", "flat", "flat", "mixed", "mixed"])
        self.assertEqual(count_zero_variance_groups(batch), 1)

    def test_response_pad_does_not_carry_advantage(self) -> None:
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        rewards = torch.zeros(2, 3)
        rewards[:, -1] = torch.tensor([1.0, 0.0])
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        adv, _returns = compute_grpo_outcome_advantage(
            rewards, mask, np.array(["q", "q"]),
        )
        self.assertEqual(float(adv[0, -1]), 0.0)
        self.assertEqual(float(adv[1, -1]), 0.0)
        self.assertNotEqual(float(adv[0, 0]), 0.0)


class TestTemperatureRule(unittest.TestCase):
    def test_higher_temperature_is_kept_only_when_contrast_increases(self) -> None:
        from probe_temperature import choose_temperature, summarize_group

        same = summarize_group([
            {"reward": 0, "answer": "A", "tools": ["Bing"]},
            {"reward": 0, "answer": "B", "tools": ["Wiki"]},
        ])
        self.assertFalse(same["contrast"])
        self.assertEqual(same["unique_answers"], 2)
        mixed = summarize_group([
            {"reward": 0, "answer": "A", "tools": ["Bing"]},
            {"reward": 1, "answer": "B", "tools": ["Wiki"]},
        ])
        self.assertTrue(mixed["contrast"])
        self.assertEqual(choose_temperature({
            "0.5": {"contrast_groups": 0},
            "0.7": {"contrast_groups": 1},
            "1.0": {"contrast_groups": 1},
        }), 0.7)
        self.assertEqual(choose_temperature({
            "0.5": {"contrast_groups": 2},
            "0.7": {"contrast_groups": 1},
            "1.0": {"contrast_groups": 4},
        }), 1.0)


class TestAppoBranchMask(unittest.TestCase):
    def test_branch_row_loss_mask_is_zero(self) -> None:
        from verl import DataProto

        from rl.hooks.advantage import apply_appo_credit

        scores = torch.zeros(2, 4)
        scores[0, -1] = 1.0
        scores[1, -1] = 0.0
        batch = DataProto.from_dict(tensors={
            "advantages": torch.ones(2, 4),
            "token_level_scores": scores,
            "token_level_rewards": scores.clone(),
            "response_mask": torch.ones(2, 4),
            "returns": torch.ones(2, 4),
            "loss_mask": torch.ones(2, 4),
        })
        batch.non_tensor_batch["rollout_role"] = np.array(["parent", "branch"], dtype=object)
        batch.non_tensor_batch["data_id_list"] = np.array(["t", "t"], dtype=object)
        batch.non_tensor_batch["rollout_id_list"] = np.array(["p", "c"], dtype=object)
        batch.non_tensor_batch["resume_parent_id"] = np.array(["", "p"], dtype=object)
        batch.non_tensor_batch["appo_branch_token"] = np.array(["", 1], dtype=object)
        out = apply_appo_credit(batch, discount=0.9)
        self.assertTrue(torch.equal(out.batch["loss_mask"][1], torch.zeros(4)))
        self.assertTrue(torch.equal(out.batch["advantages"][1], torch.zeros(4)))
        self.assertTrue(torch.equal(out.batch["loss_mask"][0], torch.ones(4)))


if __name__ == "__main__":
    unittest.main()
