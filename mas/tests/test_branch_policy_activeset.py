"""Unit tests for official-ARPO-aligned branch_policy + ActiveSetSession."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class _FakeRaw:
    messages: List[Dict[str, Any]] = field(default_factory=list)
    branch_messages: List[Dict[str, Any]] = field(default_factory=list)
    h_root: float = 0.2
    h_tool: float = 0.9
    consecutive_high: int = 0
    error: Optional[str] = None
    final_answer: str = "1"
    format_ok: bool = True
    n_search: int = 0
    n_python: int = 1


class TestBranchPolicy(unittest.TestCase):
    def test_allocate_forks_beam_and_remaining(self):
        from rl.hooks.branch_policy import allocate_forks

        self.assertEqual(allocate_forks(remaining=5, beam_size=3, n_sources=1), [2])
        self.assertEqual(allocate_forks(remaining=1, beam_size=3, n_sources=2), [1, 0])
        self.assertEqual(allocate_forks(remaining=0, beam_size=3, n_sources=1), [0])
        self.assertEqual(allocate_forks(remaining=4, beam_size=1, n_sources=1), [0])

    def test_arpo_higher_delta_forks_more_often(self):
        from rl.hooks.branch_policy import arpo_should_fork

        class _Draw:
            def __init__(self, v: float):
                self.v = v

            def random(self):
                return self.v

        # draw=0.4, ΔH=0 → prob=0.4 <= 0.5 → fork
        self.assertTrue(
            arpo_should_fork(0.2, 0.2, branch_probability=0.5, entropy_weight=0.5, rng=_Draw(0.4))
        )
        # draw=0.99, ΔH=0 → prob=0.99 > 0.5 → reject
        self.assertFalse(
            arpo_should_fork(0.2, 0.2, branch_probability=0.5, entropy_weight=0.5, rng=_Draw(0.99))
        )
        # draw=0.99, ΔH=1.5 → prob=clamp(0.99-0.75)=0.24 <= 0.5 → fork
        self.assertTrue(
            arpo_should_fork(1.5, 0.0, branch_probability=0.5, entropy_weight=0.5, rng=_Draw(0.99))
        )

    def test_distribute_root_budgets(self):
        from rl.hooks.branch_policy import distribute_root_budgets

        self.assertEqual(distribute_root_budgets(4, 2), [2, 2])
        self.assertEqual(distribute_root_budgets(5, 2), [3, 2])
        self.assertEqual(sum(distribute_root_budgets(7, 3)), 7)


class TestActiveSetSession(unittest.TestCase):
    def test_plan_and_local_execute_fills_budget(self):
        from workflow.active_set import ActiveSetConfig, ActiveSetSession

        calls = {"n": 0}

        def run_ep(task: Dict[str, Any]) -> _FakeRaw:
            calls["n"] += 1
            prefix = [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "execute_python"}]},
                {"role": "tool", "content": "42"},
            ]
            if task.get("resume_messages"):
                return _FakeRaw(messages=list(task["resume_messages"]) + [{"role": "assistant", "content": "<answer>42</answer>"}], branch_messages=list(task["resume_messages"]), h_root=0.2, h_tool=0.9)
            return _FakeRaw(messages=prefix + [{"role": "assistant", "content": "<answer>42</answer>"}], branch_messages=prefix, h_root=0.2, h_tool=0.9)

        class _AlwaysFork:
            def random(self):
                return 0.0

        sess = ActiveSetSession(
            ActiveSetConfig(
                group_budget=4,
                beam_size=3,
                max_branch_depth=2,
                execute_local=True,
                use_official_arpo_gate=True,
                branch_probability=0.5,
                entropy_weight=0.5,
            ),
            rng=_AlwaysFork(),
        )
        result = sess.run({"question": "q", "answer": "42"}, run_ep, reward_fn=lambda r, t: 1.0)
        self.assertEqual(len(result.completed), 4)
        self.assertGreaterEqual(result.branch_local_count, 1)
        self.assertEqual(result.metrics["n_completed"], 4)

    def test_train_path_plans_without_local_execute(self):
        from workflow.active_set import ActiveSetConfig, ActiveSetSession

        def run_ep(task: Dict[str, Any]) -> _FakeRaw:
            prefix = [
                {"role": "user", "content": "q"},
                {"role": "tool", "content": "1"},
            ]
            return _FakeRaw(branch_messages=prefix, h_root=0.1, h_tool=1.0)

        class _AlwaysFork:
            def random(self):
                return 0.0

        sess = ActiveSetSession(
            ActiveSetConfig(group_budget=4, beam_size=3, execute_local=False),
            rng=_AlwaysFork(),
        )
        result = sess.run({"question": "q"}, run_ep)
        self.assertEqual(len(result.completed), 1)
        self.assertEqual(len(result.plans), 2)  # beam_size-1=2
        self.assertEqual(result.branch_local_count, 2)


if __name__ == "__main__":
    unittest.main()
