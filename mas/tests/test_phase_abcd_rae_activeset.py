"""Unit tests for RAE chain, ActiveSetScheduler ready-batch, and full A^RAE."""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestResumeBoundaryAndMeta(unittest.TestCase):
    def test_resume_boundary_from_messages(self):
        from workflow.active_set import resume_boundary_from_messages

        self.assertEqual(resume_boundary_from_messages(None), 0)
        self.assertEqual(resume_boundary_from_messages([{"role": "user"}, {"role": "assistant"}]), 2)

    def test_plan_meta_includes_boundary_and_action_key(self):
        from types import SimpleNamespace

        from workflow.active_set import ActiveSetConfig, ActiveSetSession
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite, BranchSiteReward

        prefix = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "execute_python", "id": "1"}]},
            {"role": "tool", "content": "42"},
        ]
        raw = SimpleNamespace(
            branch_messages=prefix,
            window_events=[{
                "event_id": "w1",
                "snapshot_ref": "archive:s1",
                "agent_id": "hub",
                "kind": "after_tool",
                "tool_id": "execute_python",
                "turn": 1,
                "metrics": {"h_root": 0.1, "h_tool": 0.9},
            }],
            window_snapshots=[{
                "event_id": "w1",
                "messages": prefix,
            }],
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "execute_python", "id": "1"}]},
                {"role": "tool", "content": "42"},
            ],
            h_root=0.1,
            h_tool=0.9,
            consecutive_high=0,
            error=None,
        )
        site = BranchSite(
            id="s1",
            enabled=True,
            anchor=BranchAnchor(kind="after_tool"),
            gate=BranchGate(type="always"),
            reward=BranchSiteReward(scheme="rae_adjudicate"),
        )
        sess = ActiveSetSession(ActiveSetConfig(beam_size=2, sites=[site], run_probes=False))
        plans = sess.plan_forks_from_raw(raw, parent_id="p0", remaining=2)
        self.assertTrue(plans)
        self.assertGreater(plans[0].meta.get("resume_boundary", 0), 0)
        self.assertIn("action_key", plans[0].meta)
        self.assertEqual(plans[0].meta.get("reward_scheme"), "rae_adjudicate")


class TestProbesHBranch(unittest.TestCase):
    def test_collect_h_branch_from_probes(self):
        from types import SimpleNamespace

        from workflow.active_set import ActiveSetConfig, ActiveSetSession
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite, BranchSiteReward

        calls = {"n": 0}

        def run_ep(task: Dict[str, Any]):
            calls["n"] += 1
            ok = calls["n"] % 2 == 1
            return SimpleNamespace(
                format_ok=ok,
                final_answer="1" if ok else "",
                error=None,
                branch_messages=task.get("resume_messages") or [],
            )

        raw = SimpleNamespace(
            branch_messages=[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
            h_root=0.2,
            h_tool=0.8,
            consecutive_high=0,
            error=None,
        )
        site = BranchSite(
            id="dual",
            enabled=True,
            anchor=BranchAnchor(kind="after_tool"),
            gate=BranchGate(type="dual_entropy", params={"probe_k": 2, "u_threshold": 0.0}),
            reward=BranchSiteReward(scheme="rae_adjudicate"),
        )
        sess = ActiveSetSession(ActiveSetConfig(sites=[site], probe_k=2, run_probes=True))
        hb = sess.collect_h_branch(raw, {"_rollout_id": "root"}, run_ep, site=site)
        self.assertIsNotNone(hb)
        self.assertGreaterEqual(calls["n"], 2)


class TestRaeFullAndDeadEnd(unittest.TestCase):
    def test_renorm_and_adjudicate(self):
        from rl.hooks.rae_advantage import (
            adjudicate_action_group,
            apply_dead_end_backprop_verdicts,
            renorm_target_probs,
        )

        self.assertEqual(adjudicate_action_group([False, False], k_min=2), "invalidate")
        tgt = renorm_target_probs([0.5, 0.5], ["validate", "invalidate"])
        self.assertAlmostEqual(sum(tgt), 1.0, places=5)
        self.assertGreater(tgt[0], tgt[1])

        verdicts = apply_dead_end_backprop_verdicts(
            ["none", "invalidate", "invalidate"],
            roles=["parent", "child", "child"],
            parent_ids=["", "p", "p"],
            data_ids=["d", "d", "d"],
            depth=1,
        )
        self.assertEqual(verdicts[0], "invalidate")

    def test_zero_prefix_with_boundary(self):
        import torch
        from verl import DataProto

        from rl.hooks.rae_advantage import zero_prefix_advantages_for_children

        adv = torch.ones(2, 5)
        batch = DataProto.from_dict(tensors={"advantages": adv})
        batch.non_tensor_batch["rollout_role"] = __import__("numpy").array(["parent", "child"], dtype=object)
        batch.non_tensor_batch["resume_boundary"] = __import__("numpy").array([0, 3], dtype=object)
        out = zero_prefix_advantages_for_children(batch)
        self.assertTrue(torch.allclose(out.batch["advantages"][1, :3], torch.zeros(3)))
        self.assertTrue(torch.allclose(out.batch["advantages"][1, 3:], torch.ones(2)))


class TestActiveSetSchedulerReadyBatch(unittest.TestCase):
    def test_slow_tool_does_not_block_other_llm(self):
        from types import SimpleNamespace

        from workflow.active_set import ActiveSetConfig
        from workflow.active_set_scheduler import ActiveSetScheduler, LeafState

        llm_order: List[str] = []
        tool_sleep = {"A": 0.01, "B": 0.15, "C": 0.01}

        def llm_step(payload: Dict[str, Any]) -> Dict[str, Any]:
            lid = str(payload.get("leaf_id") or "")
            llm_order.append(lid)
            # first call: request tool; second: done
            msgs = list(payload.get("messages") or [])
            n_assistant = sum(1 for m in msgs if m.get("role") == "assistant")
            if n_assistant == 0:
                return {
                    "messages": msgs + [{"role": "assistant", "content": "call", "tool_calls": [{"name": "t", "id": lid}]}],
                    "tool_calls": [{"name": "t", "id": lid, "args": {}}],
                    "done": False,
                }
            return {
                "messages": msgs + [{"role": "assistant", "content": "<answer>1</answer>"}],
                "tool_calls": [],
                "done": True,
                "raw_partial": SimpleNamespace(
                    branch_messages=msgs,
                    h_root=0.1,
                    h_tool=0.1,
                    consecutive_high=0,
                    error=None,
                    format_ok=True,
                    final_answer="1",
                ),
            }

        def tool_step(payload: Dict[str, Any], tool_calls: List[Any]) -> Dict[str, Any]:
            tid = ""
            if tool_calls:
                tid = str(tool_calls[0].get("id") or "")
            # map leaf by tool id suffix
            delay = 0.05
            for k, v in tool_sleep.items():
                if k in tid:
                    delay = v
                    break
            time.sleep(delay)
            msgs = list(payload.get("messages") or []) + [{"role": "tool", "content": "ok"}]
            return {"messages": msgs, "done": False}

        # Seed three leaves by using group_budget=3 and forcing always-fork after root —
        # simpler: manually drive scheduler with three ready roots via custom run.
        # Use execute path: one root that finishes without fork; instead call internal leaves.
        cfg = ActiveSetConfig(group_budget=1, beam_size=1, execute_local=True, run_probes=False, sites=[])
        sched = ActiveSetScheduler(cfg, llm_step=llm_step, tool_step=tool_step, max_workers=4, max_rounds=8)

        # Directly test tool drain skew metric with three waiting leaves
        from workflow.active_set_scheduler import Leaf

        leaves = [
            Leaf(leaf_id="A", task={}, messages=[], state=LeafState.WAITING_TOOL, pending_tools=[{"id": "A"}]),
            Leaf(leaf_id="B", task={}, messages=[], state=LeafState.WAITING_TOOL, pending_tools=[{"id": "B"}]),
            Leaf(leaf_id="C", task={}, messages=[], state=LeafState.WAITING_TOOL, pending_tools=[{"id": "C"}]),
        ]
        t0 = time.time()
        for L in leaves:
            L.tool_started_at = t0
        sched._drain_tools(leaves)
        ready_ids = [L.leaf_id for L in leaves if L.state == LeafState.READY]
        self.assertEqual(set(ready_ids), {"A", "B", "C"})
        self.assertGreater(sched.metrics.as_dict()["tool_wait_skew_max"], 0.05)
        # A and C should finish before B wall time would serialize (0.15*3)
        self.assertLess(time.time() - t0, 0.35)


class TestOnTokenContract(unittest.TestCase):
    def test_event_kind_and_snapshot_token_ids(self):
        from workflow.contracts import EventKind, Snapshot

        self.assertEqual(EventKind.ON_TOKEN.value, "on_token")
        snap = Snapshot(archive_id="a", messages=[{"role": "user", "content": "q"}], token_ids=[1, 2, 3])
        self.assertEqual(snap.token_ids, [1, 2, 3])

    def test_site_matches_on_token(self):
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite
        from workflow.gates import site_matches_event

        site = BranchSite(
            id="tok",
            enabled=True,
            anchor=BranchAnchor(kind="on_token", agent_id="hub"),
            gate=BranchGate(type="always"),
        )
        self.assertTrue(site_matches_event(site, event_kind="on_token", agent_id="hub"))
        self.assertFalse(site_matches_event(site, event_kind="after_tool", agent_id="hub"))


class TestPlanForksTrajectorySites(unittest.TestCase):
    """Runtime regression for UI trajectory sites (site.anchor.kind self-match).

    Note: current plan_forks matches each site against its own anchor.kind, not a
    live runtime event stream. Assertions follow that semantics.
    """

    def test_after_agent_turn_always_emits_plan(self):
        from types import SimpleNamespace

        from workflow.active_set import ActiveSetConfig, ActiveSetSession
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite, BranchSiteReward

        raw = SimpleNamespace(
            branch_messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "plan step"},
            ],
            h_root=0.1,
            h_tool=0.2,
            consecutive_high=0,
            error=None,
        )
        site = BranchSite(
            id="turn_hub",
            enabled=True,
            anchor=BranchAnchor(kind="after_agent_turn", agent_id="hub"),
            gate=BranchGate(type="always"),
            reward=BranchSiteReward(scheme="shared_outcome"),
        )
        sess = ActiveSetSession(ActiveSetConfig(beam_size=2, sites=[site], run_probes=False))
        plans = sess.plan_forks_from_raw(raw, parent_id="p0", remaining=2)
        self.assertEqual(plans, [])

    def test_after_verifier_fail_gate(self):
        from types import SimpleNamespace

        from workflow.active_set import ActiveSetConfig, ActiveSetSession
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite, BranchSiteReward

        raw = SimpleNamespace(
            branch_messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "ans"},
            ],
            h_root=0.0,
            h_tool=0.0,
            consecutive_high=0,
            error=None,
        )
        site = BranchSite(
            id="ver_fail",
            enabled=True,
            anchor=BranchAnchor(kind="after_verifier", agent_id="verifier"),
            gate=BranchGate(type="verifier_fail"),
            reward=BranchSiteReward(scheme="shared_outcome"),
        )
        sess = ActiveSetSession(
            ActiveSetConfig(beam_size=2, sites=[site], run_probes=False, verifier_ok=False)
        )
        plans = sess.plan_forks_from_raw(raw, parent_id="root", remaining=2)
        self.assertEqual(plans, [])

        sess_ok = ActiveSetSession(
            ActiveSetConfig(beam_size=2, sites=[site], run_probes=False, verifier_ok=True)
        )
        self.assertEqual(sess_ok.plan_forks_from_raw(raw, parent_id="root", remaining=2), [])

    def test_agent_turn_site_uses_messages_fallback(self):
        from types import SimpleNamespace

        from workflow.active_set import ActiveSetConfig, ActiveSetSession
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite, BranchSiteReward

        # Only after_tool site would need branch_messages; agent_turn uses messages fallback.
        raw = SimpleNamespace(
            branch_messages=None,
            messages=[
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "thinking"},
            ],
            h_root=0.05,
            h_tool=0.05,
            consecutive_high=0,
            error=None,
        )
        tool_site = BranchSite(
            id="tool_only",
            enabled=True,
            anchor=BranchAnchor(kind="after_tool", tool_id="execute_python"),
            gate=BranchGate(type="always"),
            reward=BranchSiteReward(scheme="shared_outcome"),
        )
        turn_site = BranchSite(
            id="turn_only",
            enabled=True,
            anchor=BranchAnchor(kind="after_agent_turn", agent_id="hub"),
            gate=BranchGate(type="always"),
            reward=BranchSiteReward(scheme="shared_outcome"),
        )
        # Explicit sites require real Window events and snapshots.
        sess = ActiveSetSession(
            ActiveSetConfig(beam_size=2, sites=[tool_site, turn_site], run_probes=False)
        )
        plans = sess.plan_forks_from_raw(raw, parent_id="p0", remaining=2)
        self.assertEqual(plans, [])

        # Explicit Tool sites do not use legacy messages fallback either.
        sess_tool = ActiveSetSession(
            ActiveSetConfig(beam_size=2, sites=[tool_site], run_probes=False)
        )
        tool_plans = sess_tool.plan_forks_from_raw(raw, parent_id="p0", remaining=2)
        self.assertEqual(tool_plans, [])

    def test_window_events_match_and_fallback(self):
        """P2: real WindowEndEvent stream matching + legacy fallback when empty."""
        from types import SimpleNamespace

        from workflow.active_set import ActiveSetConfig, ActiveSetSession
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite, BranchSiteReward
        from workflow.gates import site_matches_event

        site = BranchSite(
            id="tool_site",
            enabled=True,
            anchor=BranchAnchor(kind="after_tool", tool_id="execute_python"),
            gate=BranchGate(type="always"),
            reward=BranchSiteReward(scheme="shared_outcome"),
        )

        # 1) events present and match → plan emitted, event_kind from site anchor
        raw = SimpleNamespace(
            branch_messages=[{"role": "user", "content": "q"}],
            window_events=[
                {"event_id": "w1", "snapshot_ref": "archive:s1", "agent_id": "hub", "kind": "after_tool", "tool_id": "execute_python", "turn": 1},
            ],
            window_snapshots=[
                {"event_id": "w1", "messages": [{"role": "user", "content": "q"}]},
            ],
            h_root=0.1,
            h_tool=0.2,
            consecutive_high=0,
            error=None,
        )
        sess = ActiveSetSession(ActiveSetConfig(beam_size=2, sites=[site], run_probes=False))
        plans = sess.plan_forks_from_raw(raw, parent_id="p0", remaining=2)
        self.assertTrue(plans)
        self.assertEqual(plans[0].meta.get("event_kind"), "after_tool")
        self.assertEqual(plans[0].site_id, "tool_site")

        # 2) events present but none match → no plan (stream is source of truth)
        raw_mismatch = SimpleNamespace(
            branch_messages=[{"role": "user", "content": "q"}],
            window_events=[
                {"agent_id": "hub", "kind": "after_verifier", "tool_id": None, "turn": 2},
            ],
            h_root=0.1,
            h_tool=0.2,
            consecutive_high=0,
            error=None,
        )
        sess2 = ActiveSetSession(ActiveSetConfig(beam_size=2, sites=[site], run_probes=False))
        self.assertEqual(sess2.plan_forks_from_raw(raw_mismatch, parent_id="p0", remaining=2), [])

        # 3) explicit sites do not self-match when no Window event exists
        raw_no_events = SimpleNamespace(
            branch_messages=[{"role": "user", "content": "q"}],
            window_events=[],
            h_root=0.1,
            h_tool=0.2,
            consecutive_high=0,
            error=None,
        )
        sess3 = ActiveSetSession(ActiveSetConfig(beam_size=2, sites=[site], run_probes=False))
        plans3 = sess3.plan_forks_from_raw(raw_no_events, parent_id="p0", remaining=2)
        self.assertEqual(plans3, [])

        # 4) Agent Window no longer aliases a Tool Result event
        turn_site = BranchSite(
            id="turn_site",
            enabled=True,
            anchor=BranchAnchor(kind="after_agent_turn", agent_id="hub"),
            gate=BranchGate(type="always"),
            reward=BranchSiteReward(scheme="shared_outcome"),
        )
        self.assertFalse(
            site_matches_event(
                turn_site,
                event_kind="after_agent_turn",
                window_events=[{"agent_id": "hub", "kind": "after_tool", "turn": 1}],
            )
        )


class TestKHopAndVerdictTree(unittest.TestCase):
    """P2: k-hop credit + RAE verdict writeback onto RolloutTree nodes."""

    def _tree(self):
        return {
            "tree_id": "q1",
            "nodes": [
                {"node_id": "root", "parent_id": None, "role": "root", "reward": 0.2},
                {"node_id": "c0", "parent_id": "root", "role": "child", "reward": 0.8},
                {"node_id": "c0a", "parent_id": "c0", "role": "child", "reward": 0.4},
            ],
        }

    def test_k_hop_equal_weight_mean(self):
        from rl.hooks.rae_advantage import k_hop_cumulative_reward

        tree = self._tree()
        # k=0: leaf only
        self.assertAlmostEqual(k_hop_cumulative_reward(tree, "c0a", k=0), 0.4)
        # k=1: leaf + parent
        self.assertAlmostEqual(k_hop_cumulative_reward(tree, "c0a", k=1), (0.4 + 0.8) / 2)
        # k=2: whole path
        self.assertAlmostEqual(k_hop_cumulative_reward(tree, "c0a", k=2), (0.4 + 0.8 + 0.2) / 3)
        # no reward info → None
        bare = {"nodes": [{"node_id": "x", "parent_id": None, "reward": None}]}
        self.assertIsNone(k_hop_cumulative_reward(bare, "x", k=2))

    def test_apply_verdicts_to_tree(self):
        from rl.hooks.rae_advantage import apply_verdicts_to_tree

        tree = self._tree()
        n = apply_verdicts_to_tree(tree, {"c0": "invalidate", "c0a": "validate"})
        self.assertGreaterEqual(n, 2)
        by_id = {n["node_id"]: n for n in tree["nodes"]}
        self.assertEqual(by_id["c0"]["verdict"], "invalidate")
        self.assertEqual(by_id["c0a"]["verdict"], "validate")

        # node_ids remap path: children in order get store rollout ids
        tree2 = self._tree()
        apply_verdicts_to_tree(tree2, {"rid_0": "invalidate"}, node_ids=["rid_0", "rid_1"])
        children = [n for n in tree2["nodes"] if n["role"] != "root"]
        self.assertEqual(children[0]["verdict"], "invalidate")


if __name__ == "__main__":
    unittest.main()
