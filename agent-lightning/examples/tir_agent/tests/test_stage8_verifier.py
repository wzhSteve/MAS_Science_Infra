"""v1.3: verifier skill routes fail back to hub for one hop (no PEV / no MAS_structagent)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestVerifierRegistry(unittest.TestCase):
    def test_default_registry_has_verifier_not_in_hub_skills(self):
        from workflow.plugins import REGISTRY
        from workflow.spec import load_spec

        REGISTRY.get_skill("verifier")
        REGISTRY.get_role("verifier")
        spec = load_spec()
        self.assertIsNone(spec.hub.verify)
        self.assertNotIn("verifier", spec.hub.skills)


class TestVerifierHop(unittest.TestCase):
    def test_error_retries_once_and_succeeds(self):
        from workflow.contracts import EventKind
        from workflow.runtime import ExecutionService
        from workflow.spec import load_spec

        spec = load_spec(str(FIXTURES / "hub_verify.yaml"))
        self.assertEqual(spec.hub.verify, "verifier")
        traj = ExecutionService(mock=True, spec=spec).run(
            {
                "id": "e",
                "question": "1+1",
                "answer": "2",
                "_mock_answer": "2",
                "_mock_error": "boom",
            }
        )
        self.assertIsNone(traj.meta.get("error"))
        self.assertEqual(traj.final_answer, "2")
        self.assertEqual(traj.meta.get("feedback_hops"), 1)
        kinds = [e.kind for e in traj.events]
        self.assertIn(EventKind.FEEDBACK, kinds)
        self.assertIn(EventKind.ERROR, kinds)
        self.assertIn("verifier", traj.meta.get("skills") or [])
        fb = [e for e in traj.events if e.kind == EventKind.FEEDBACK]
        self.assertGreaterEqual(len(fb), 2)
        self.assertTrue(fb[0].payload.get("rerun"))
        self.assertFalse(fb[-1].payload.get("rerun"))

    def test_success_no_rerun(self):
        from workflow.contracts import EventKind
        from workflow.runtime import ExecutionService
        from workflow.spec import load_spec

        spec = load_spec(str(FIXTURES / "hub_verify.yaml"))
        traj = ExecutionService(mock=True, spec=spec).run(
            {"id": "ok", "question": "1+1", "answer": "2", "_mock_answer": "2"}
        )
        self.assertEqual(traj.final_answer, "2")
        self.assertEqual(traj.meta.get("feedback_hops"), 0)
        fb = [e for e in traj.events if e.kind == EventKind.FEEDBACK]
        self.assertEqual(len(fb), 1)
        self.assertTrue(fb[0].payload.get("ok"))
        self.assertFalse(fb[0].payload.get("rerun"))

    def test_default_spec_error_has_no_feedback(self):
        from workflow.contracts import EventKind
        from workflow.runtime import ExecutionService

        traj = ExecutionService(mock=True).run(
            {"id": "e", "question": "x", "answer": "0", "_mock_error": "boom"}
        )
        self.assertTrue(traj.meta.get("error"))
        self.assertNotIn(EventKind.FEEDBACK, {e.kind for e in traj.events})
        self.assertEqual(traj.meta.get("feedback_hops") or 0, 0)

    def test_zero_hops_does_not_rerun(self):
        from workflow.runtime import ExecutionService
        from workflow.spec import HubSpec, MASSpec

        spec = MASSpec(hub=HubSpec(skills=["react_loop"], verify="verifier", max_feedback_hops=0))
        traj = ExecutionService(mock=True, spec=spec).run(
            {"id": "e", "question": "x", "answer": "0", "_mock_error": "boom"}
        )
        self.assertTrue(traj.meta.get("error"))
        self.assertEqual(traj.meta.get("feedback_hops"), 0)

    def test_persist_error_stops_after_max_hops(self):
        from workflow.runtime import ExecutionService
        from workflow.spec import load_spec

        spec = load_spec(str(FIXTURES / "hub_verify.yaml"))
        traj = ExecutionService(mock=True, spec=spec).run(
            {
                "id": "e",
                "question": "x",
                "answer": "0",
                "_mock_error": "boom",
                "_mock_error_persist": True,
            }
        )
        self.assertTrue(traj.meta.get("error"))
        self.assertEqual(traj.meta.get("feedback_hops"), 1)

    def test_unknown_verify_skill_raises(self):
        from workflow.runtime import ExecutionService
        from workflow.spec import HubSpec, MASSpec

        spec = MASSpec(hub=HubSpec(verify="causal_analysis"))
        with self.assertRaises(KeyError):
            ExecutionService(mock=True, spec=spec).run(
                {"id": "x", "question": "1+1", "answer": "2", "_mock_answer": "2"}
            )


if __name__ == "__main__":
    unittest.main()
