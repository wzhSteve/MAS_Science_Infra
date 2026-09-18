"""RolloutTree contract tests (new_framework P0)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestRolloutTreeContract(unittest.TestCase):
    def test_leaves_and_path_to_root(self):
        from workflow.contracts import RolloutTree, RolloutTreeNode

        tree = RolloutTree(
            tree_id="q1",
            nodes=[
                RolloutTreeNode(node_id="root", role="root"),
                RolloutTreeNode(node_id="a", parent_id="root", depth=1, role="child"),
                RolloutTreeNode(node_id="b", parent_id="root", depth=1, role="child"),
                RolloutTreeNode(node_id="a1", parent_id="a", depth=2, role="child"),
            ],
        )
        self.assertEqual(sorted(tree.leaves()), ["a1", "b"])
        self.assertEqual(tree.path_to_root("a1"), ["a1", "a", "root"])
        self.assertEqual(tree.path_to_root("root"), ["root"])
        # cycle-safe
        tree2 = RolloutTree(
            tree_id="q2",
            nodes=[
                RolloutTreeNode(node_id="x", parent_id="y"),
                RolloutTreeNode(node_id="y", parent_id="x"),
            ],
        )
        self.assertEqual(len(tree2.path_to_root("x")), 2)

    def test_json_round_trip(self):
        from workflow.contracts import RolloutTree, RolloutTreeNode

        tree = RolloutTree(
            tree_id="q1",
            query="2+3=?",
            nodes=[
                RolloutTreeNode(node_id="root", role="root"),
                RolloutTreeNode(node_id="c0", parent_id="root", depth=1, metrics={"h_tool": 0.9}),
            ],
            outcomes={"c0": {"answer": "5", "reward": 1.0}},
        )
        parsed = RolloutTree.model_validate(json.loads(tree.model_dump_json()))
        self.assertEqual(parsed.leaves(), ["c0"])
        self.assertEqual(parsed.outcomes["c0"]["reward"], 1.0)

    def test_expansion_payload_dual_format(self):
        from workflow.active_set import ActiveSetResult, expansion_payload_from_result, tree_from_plans
        from workflow.active_set import ForkPlan

        plan = ForkPlan(
            resume_messages=[{"role": "user", "content": "q"}],
            parent_id="p0",
            depth=1,
            reason="always_branch",
            site_id="turn_hub",
            role="child",
            meta={"h_root": 0.1, "h_tool": 0.2, "event_kind": "after_agent_turn"},
        )
        result = ActiveSetResult(plans=[plan], branch_local_count=1, global_fill_count=0)
        payload = expansion_payload_from_result(result)

        # dual format: tree present, legacy flat plans preserved
        self.assertIn("tree", payload)
        self.assertIn("plans", payload)
        self.assertEqual(payload["branch_local_count"], 1)
        self.assertEqual(len(payload["plans"]), 1)
        self.assertEqual(payload["plans"][0]["site_id"], "turn_hub")

        tree = payload["tree"]
        self.assertEqual(tree["tree_id"], "p0")
        root = tree["nodes"][0]
        self.assertEqual(root["role"], "root")
        child = tree["nodes"][1]
        self.assertEqual(child["parent_id"], "p0")
        self.assertEqual(child["metrics"].get("event_kind"), "after_agent_turn")

    def test_tree_from_plans_empty(self):
        from workflow.active_set import tree_from_plans

        tree = tree_from_plans("p0", [])
        self.assertEqual(len(tree.nodes), 1)
        self.assertEqual(tree.nodes[0].role, "root")
        self.assertEqual(tree.leaves(), ["p0"])

    def test_tree_event_contract(self):
        from workflow.contracts import RolloutTreeEvent

        ev = RolloutTreeEvent(event="node_added", tree_id="q1", node_id="c0", payload={"h": 0.5})
        self.assertEqual(ev.event, "node_added")
        with self.assertRaises(Exception):
            RolloutTreeEvent.model_validate({"event": "unknown_kind_x", "tree_id": "q1", "bogus": 1})


if __name__ == "__main__":
    unittest.main()
