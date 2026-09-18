"""Daemon expand_in_runner materialization (no GPU / no live AGL store)."""

from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


@contextmanager
def _agl_modules_quarantined():
    """Import rl.hooks.daemon (pulls agentlightning) without leaking it.

    Later stage tests assert ``"agentlightning" not in sys.modules`` (layer
    independence red line); scrub AGL modules on exit so this suite can run
    in the same discover process.
    """
    import gc

    try:
        yield
    finally:
        agl = [k for k in list(sys.modules) if k == "agentlightning" or k.startswith("agentlightning.")]
        for k in agl:
            del sys.modules[k]
        gc.collect()


class TestDaemonExpandEnqueue(unittest.TestCase):
    def test_enqueue_from_runner_expansions_counts_and_fills(self):
        from workflow.active_set import dump_local_expansion

        with _agl_modules_quarantined(), tempfile.TemporaryDirectory() as td:
            # Import after path setup; daemon pulls agentlightning.
            from rl.hooks.daemon import TirAgentModeDaemon

            import workflow.active_set as active_set

            active_set.EXPANSION_DIR = td

            daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
            daemon.mode = "v1"
            daemon.tir_algo = "arpo"
            daemon.tir_config = {"expand_in_runner": True, "beam_size": 2}
            daemon._full_group_n = 4
            daemon.train_rollout_n = 4
            daemon.llm_timeout_seconds = 30
            daemon._store_enqueue_branch_count = 0
            daemon._branch_local_count_total = 0
            daemon._pending_original_by_data_id = {
                "q1": {"data_id": "q1", "question": "1+1", "answer": "2"},
            }
            daemon._task_id_to_original_sample = {
                "r0": {"data_id": "q1", "question": "1+1", "answer": "2"},
            }
            daemon._rollout_meta = {"r0": {"branch_local_count": 2}}
            daemon._completed_rollouts_v0 = {}
            daemon._enqueued_expansion_parents = set()
            daemon._incremental_branch_total = 0
            daemon._rollout_trees = {}

            dump_local_expansion(
                "r0",
                {
                    "branch_local_count": 2,
                    "plans": [
                        {
                            "resume_messages": [{"role": "user", "content": "x"}],
                            "parent_id": "r0",
                            "depth": 1,
                            "reason": "arpo_branch",
                            "meta": {},
                        },
                        {
                            "resume_messages": [{"role": "user", "content": "y"}],
                            "parent_id": "r0",
                            "depth": 1,
                            "reason": "arpo_branch",
                            "meta": {},
                        },
                    ],
                },
            )

            created: List[Any] = []

            async def _enqueue_many(requests):
                out = []
                for i, req in enumerate(requests):
                    rid = f"new{i}"
                    r = MagicMock()
                    r.rollout_id = rid
                    r.metadata = {"data_id": "q1"}
                    out.append(r)
                    created.append(req)
                return out

            daemon.store = MagicMock()
            daemon.store.enqueue_many_rollouts = AsyncMock(side_effect=_enqueue_many)
            daemon._total_tasks_queued = 1

            n = asyncio.get_event_loop().run_until_complete(daemon._enqueue_from_runner_expansions())
            # remaining = 4 - 1 = 3 → 2 branch + 1 global
            self.assertEqual(n, 3)
            self.assertEqual(daemon._store_enqueue_branch_count, 2)
            branch_reqs = [c for c in created if (c.metadata or {}).get("tir_branch")]
            self.assertEqual(len(branch_reqs), 2)
            self.assertTrue(created[0].input.get("resume_messages"))
            self.assertIsNone(created[2].input.get("resume_messages"))

    def test_stamp_expand_budgets_distributes(self):
        with _agl_modules_quarantined():
            from rl.hooks.daemon import TirAgentModeDaemon

        daemon = TirAgentModeDaemon.__new__(TirAgentModeDaemon)
        daemon.tir_config = {
            "max_branch_depth": 2,
            "beam_size": 2,
            "branch_probability": 0.5,
            "entropy_weight": 0.5,
            "use_official_arpo_gate": True,
        }
        daemon._full_group_n = 4
        daemon._task_id_to_original_sample = {
            "a": {"data_id": "d", "question": "q"},
            "b": {"data_id": "d", "question": "q"},
        }
        daemon._stamp_expand_budgets()
        budgets = [
            daemon._task_id_to_original_sample["a"]["sampling_budget"],
            daemon._task_id_to_original_sample["b"]["sampling_budget"],
        ]
        self.assertEqual(sorted(budgets), [2, 2])
        self.assertTrue(daemon._task_id_to_original_sample["a"]["expand_in_runner"])


if __name__ == "__main__":
    unittest.main()
