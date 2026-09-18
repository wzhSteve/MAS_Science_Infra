"""Unit tests for BranchSite gates and RAE adjudication helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for p in (str(ROOT), str(REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


class TestGates(unittest.TestCase):
    def test_always_and_tool_gates(self):
        from workflow.contracts import BranchGate
        from workflow.gates import GateContext, evaluate_gate

        always = evaluate_gate(BranchGate(type="always"), GateContext())
        self.assertTrue(always.passed)

        ok = evaluate_gate(BranchGate(type="tool_ok"), GateContext(tool_ok=True))
        self.assertTrue(ok.passed)
        bad = evaluate_gate(BranchGate(type="tool_ok"), GateContext(tool_ok=False))
        self.assertFalse(bad.passed)

        err = evaluate_gate(BranchGate(type="tool_error"), GateContext(tool_ok=False))
        self.assertTrue(err.passed)

    def test_verifier_and_failure(self):
        from workflow.contracts import BranchGate
        from workflow.gates import GateContext, evaluate_gate

        self.assertTrue(
            evaluate_gate(BranchGate(type="verifier_fail"), GateContext(verifier_ok=False)).passed
        )
        self.assertTrue(
            evaluate_gate(BranchGate(type="failure_trigger"), GateContext(final_failed=True)).passed
        )

    def test_dual_entropy(self):
        from workflow.contracts import BranchGate
        from workflow.gates import GateContext, evaluate_gate

        gate = BranchGate(type="dual_entropy", params={"u_threshold": 0.05})
        low = evaluate_gate(gate, GateContext(h_root=0.1, h_tool=0.2, h_branch=0.1))
        self.assertFalse(low.passed)  # U=0.01
        high = evaluate_gate(gate, GateContext(h_root=0.1, h_tool=0.6, h_branch=0.2))
        self.assertTrue(high.passed)  # U=0.1

    def test_site_matches_anchor(self):
        from workflow.contracts import BranchAnchor, BranchGate, BranchSite
        from workflow.gates import site_matches_event

        site = BranchSite(
            id="s1",
            enabled=True,
            anchor=BranchAnchor(kind="after_tool", agent_id="hub", tool_id="execute_python"),
            gate=BranchGate(type="always"),
        )
        self.assertTrue(
            site_matches_event(
                site, event_kind="after_tool", agent_id="hub", tool_id="execute_python"
            )
        )
        self.assertFalse(
            site_matches_event(site, event_kind="after_tool", agent_id="hub", tool_id="web_search")
        )


class TestRaeAdjudicate(unittest.TestCase):
    def test_adjudicate_action_group(self):
        from rl.hooks.rae_advantage import adjudicate_action_group

        self.assertEqual(adjudicate_action_group([False, False], p_plus=0.8, k_min=2), "invalidate")
        self.assertEqual(adjudicate_action_group([True, True, True], p_plus=0.8, k_min=2), "validate")
        self.assertEqual(adjudicate_action_group([True, False], p_plus=0.8, k_min=2), "abstain")
        self.assertEqual(adjudicate_action_group([True], p_plus=0.8, k_min=2), "abstain")

    def test_sample_policy_resolved_sites(self):
        from workflow.contracts import SamplePolicy

        sp = SamplePolicy(mode="rae", barriers=["after_tool"], sites=[])
        sites = sp.resolved_sites()
        self.assertTrue(len(sites) >= 1)
        self.assertEqual(sites[0].anchor.kind, "after_tool")


class TestBranchSitesTsParity(unittest.TestCase):
    """Python-side candidate shape mirrors UI helpers enough for contracts roundtrip."""

    def test_branch_site_roundtrip(self):
        from workflow.contracts import BranchSite

        raw = {
            "id": "site_after_tool_hub_execute_python",
            "enabled": True,
            "anchor": {"kind": "after_tool", "agent_id": "hub", "tool_id": "execute_python"},
            "gate": {"type": "entropy_delta", "params": {}},
            "fork": {"beam_size": 2, "share_observation": True},
            "reward": {"scheme": "rae_adjudicate", "p_plus": 0.8},
        }
        site = BranchSite.model_validate(raw)
        dumped = site.model_dump(mode="json")
        self.assertEqual(dumped["reward"]["scheme"], "rae_adjudicate")
        self.assertTrue(dumped["enabled"])


if __name__ == "__main__":
    unittest.main()
