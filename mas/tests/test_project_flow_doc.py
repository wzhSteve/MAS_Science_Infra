"""Fix F-1: flow-doc API structure (agents / steps / prompts)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


class FlowDocTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.user_root = Path(self._tmp.name) / "user_space"
        self.user_root.mkdir()
        self._old = os.environ.get("SCIENCE_USER_SPACE_DIR")
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(self.user_root)
        # fake upload/ prompts tree
        self.fake_repo = Path(self._tmp.name)
        for rel in (
            "user_space/projects/agentflow/upload/AgentFlow-main/agentflow/agentflow/models/planner.py",
            "user_space/projects/agentflow/upload/AgentFlow-main/agentflow/agentflow/models/executor.py",
            "user_space/projects/agentflow/upload/AgentFlow-main/agentflow/agentflow/models/verifier.py",
            "user_space/projects/agentflow/upload/AgentFlow-main/agentflow/agentflow/models/memory.py",
            # epc-aw prompts tree (13 files, structure mirrors agentflow; 5th diagnoser differs)
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/planner/analyze_query.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/planner/generate_next_step.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/planner/report_generation.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/planner/belief_prediction.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/executor/report_generation.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/executor/belief_prediction.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/executor/generate_tool_command.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/executor/direct_output.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/diagnoser/report_generation.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/diagnoser/belief_prediction.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/diagnoser/verificate_context.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/diagnoser/update_outline.txt",
            "user_space/projects/epc-aw-main/upload/EPC-AW-main/MAS/epc_aw/prompts/diagnoser/epc_aw_diagnosis.txt",
        ):
            path = self.fake_repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"prompt body of {path.name}", encoding="utf-8")
        # register the two projects
        from workflow.user_gateway.registry import create_project

        create_project("agentflow", title="AgentFlow wrap", mode="io_module")
        create_project("epc-aw-main", title="EPC-AW-main", mode="io_module")

    def tearDown(self) -> None:
        if self._old is None:
            os.environ.pop("SCIENCE_USER_SPACE_DIR", None)
        else:
            os.environ["SCIENCE_USER_SPACE_DIR"] = self._old
        self._tmp.cleanup()

    def _flow_doc(self, project_id: str):
        import science_infra.control.user_projects as up

        with patch.object(up.Path, "cwd", lambda: self.fake_repo):
            # mas_root().parent is repo_root(); point it at the fake tree
            with patch.object(up, "mas_root", lambda: self.fake_repo / "mas"):
                return up.get_project_flow_doc(project_id)


class TestProjectFlowDoc(FlowDocTestBase):
    def test_agentflow_doc_structure(self) -> None:
        doc = self._flow_doc("agentflow")
        self.assertEqual(doc["project_id"], "agentflow")
        self.assertEqual(doc["source_root"], "user_space/projects/agentflow/upload/AgentFlow-main/agentflow/agentflow")
        self.assertEqual([a["id"] for a in doc["agents"]], ["planner", "executor", "verifier"])
        self.assertEqual(sorted(p["name"] for p in doc["prompts"]),
                         ["executor/executor.py", "planner/planner.py", "verifier/verifier.py"])
        for item in doc["prompts"]:
            self.assertIn("prompt body of", item["content"])
        methods = [s["method"] for s in doc["steps"]]
        self.assertEqual(methods, [
            "analyze_query", "generate_next_step", "extract_context_subgoal_and_tool",
            "generate_tool_command", "execute_tool_command", "memory.add_action",
            "verificate_context", "extract_conclusion", "generate_direct_output",
        ])
        for banned in ("compute_bts_for_plans", "update_outline", "aifm_diagnosis"):
            self.assertNotIn(banned, methods)
        final = doc["steps"][-1]
        self.assertEqual(final["agent"], "planner")
        trainable = [s for s in doc["steps"] if "可训" in s["desc"]]
        self.assertEqual([s["method"] for s in trainable], ["generate_next_step"])

    def test_epc_aw_uses_epc_aw_diagnosis(self) -> None:
        doc = self._flow_doc("epc-aw-main")
        self.assertEqual(doc["source_root"], "user_space/projects/epc-aw-main/upload/EPC-AW-main")
        methods = [s["method"] for s in doc["steps"]]
        self.assertIn("epc_aw_diagnosis", methods)
        self.assertNotIn("aifm_diagnosis", methods)
        self.assertEqual(len(doc["prompts"]), 13)

    def test_runtime_active_flags(self) -> None:
        af = self._flow_doc("agentflow")
        self.assertTrue(all(s.get("runtime_active") is True for s in af["steps"]))
        epc = self._flow_doc("epc-aw-main")
        self.assertEqual(len(epc["steps"]), 12)
        for step in epc["steps"]:
            self.assertTrue(step.get("runtime_active") is True, f"epc-aw 步骤 {step['seq']} 应为 True")
        bts = next(s for s in epc["steps"] if "report_generation" in s["method"])
        self.assertNotIn("运行时", bts["desc"])

    def test_prompt_counts_constant(self) -> None:
        epc = self._flow_doc("epc-aw-main")
        self.assertEqual(len(epc["prompts"]), 13)
        epc_diag = sorted(p["name"] for p in epc["prompts"] if p["agent"] == "diagnoser")
        self.assertEqual(epc_diag.count("diagnoser/epc_aw_diagnosis.txt"), 1)

    def test_unknown_project_404(self) -> None:
        import science_infra.control.user_projects as up

        with self.assertRaises(FileNotFoundError):
            self._flow_doc("nope")

    def test_prompt_read_failure_sets_error(self) -> None:
        bad = self.fake_repo / "user_space/projects/agentflow/upload/AgentFlow-main/agentflow/agentflow/models/planner.py"
        # a directory named *.txt makes read_text raise (root ignores chmod)
        bad.unlink()
        bad.mkdir()
        try:
            doc = self._flow_doc("agentflow")
        finally:
            bad.rmdir()
        item = next(p for p in doc["prompts"] if p["name"].endswith("planner.py"))
        self.assertEqual(item["content"], "")
        self.assertIn("error", item)

    def test_route_registered_in_app(self) -> None:
        from science_infra.control.app import create_app

        app = create_app()
        paths = {getattr(r, "path", "") for r in app.routes}
        self.assertIn("/api/mas/user-projects/{project_id}/flow-doc", paths)


if __name__ == "__main__":
    unittest.main()
