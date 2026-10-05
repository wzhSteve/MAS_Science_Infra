"""Functional contracts for the documented Science Studio surface (no GPU)."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(TIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from science_infra.control.experiments import VALID_ALGOS, ensure_experiment, load_bundle


class TestStudioMeta(unittest.TestCase):
    def test_meta_algos_include_arpo_and_appo(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        with TestClient(create_app()) as client:
            r = client.get("/api/meta")
        self.assertEqual(r.status_code, 200, r.text)
        algos = r.json()["algos"]
        self.assertEqual(algos, list(VALID_ALGOS))
        self.assertIn("appo", algos)
        self.assertIn("arpo", algos)
        self.assertEqual(
            set(algos),
            {"grpo", "arpo", "appo", "aepo", "igpo", "gigpo", "rae"},
        )


class TestStudioPaletteAndSampling(unittest.TestCase):
    def test_palette_exposes_templates_and_user_projects(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        with TestClient(create_app()) as client:
            r = client.get("/api/mas/palette")
        self.assertEqual(r.status_code, 200, r.text)
        pal = r.json()
        self.assertIn("templates", pal)
        self.assertTrue(pal["templates"])
        self.assertIn("user_projects", pal)
        self.assertIsInstance(pal["user_projects"], list)

    def test_sampling_preview_is_not_tool_anchored(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        ensure_experiment("demo")
        workflow = load_bundle("demo")["workflow"]
        sampling = dict(workflow.get("sampling") or {})
        sampling["mode"] = "arpo"
        workflow["sampling"] = sampling
        with TestClient(create_app()) as client:
            r = client.post("/api/mas/sampling/preview", json={"data": workflow})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        opps = body.get("opportunities") or []
        self.assertTrue(opps, "arpo preview should list agent/router/verifier windows")
        kinds = {str((item.get("anchor") or {}).get("kind") or "") for item in opps}
        self.assertNotIn("after_tool", kinds)
        self.assertTrue(kinds & {"after_agent_turn", "after_verifier", "on_edge"})
        for item in opps:
            agent_id = str((item.get("anchor") or {}).get("agent_id") or item.get("node_id") or "")
            self.assertFalse(
                agent_id in {"wikipedia_search", "google_search", "web_search", "python_coder", "think"},
                f"tool node should not be a sampling candidate: {agent_id}",
            )


class TestStudioRolloutAndEval(unittest.TestCase):
    def test_mock_rollout_run_shape(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        ensure_experiment("demo")
        workflow = load_bundle("demo")["workflow"]
        with TestClient(create_app()) as client:
            r = client.post(
                "/api/mas/rollout-runs?experiment_id=demo",
                json={
                    "workflow": workflow,
                    "task": {"question": "1+1=?"},
                    "execution": "mock",
                },
            )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("run_id", body)
        self.assertIn(body.get("status"), {"succeeded", "failed"})

    def test_eval_sources_shape(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        ensure_experiment("demo")
        with TestClient(create_app()) as client:
            r = client.get("/api/mas/eval-sources?experiment_id=demo")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        for key in ("val_files", "val_exists", "test_files", "test_exists"):
            self.assertIn(key, body)

    def test_runs_default_to_train_kind(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        ensure_experiment("demo")
        with TestClient(create_app()) as client:
            r = client.get("/api/rl/runs?experiment_id=demo")
            self.assertEqual(r.status_code, 200, r.text)
            for row in r.json().get("runs") or []:
                self.assertEqual(row.get("kind"), "train")
            r2 = client.get("/api/rl/runs?experiment_id=demo&kinds=eval")
        self.assertEqual(r2.status_code, 200, r2.text)
        for row in r2.json().get("runs") or []:
            self.assertEqual(row.get("kind"), "eval")

    def test_run_scoped_rollout_trees_route(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app

        ensure_experiment("demo")
        with TestClient(create_app()) as client:
            r = client.get("/api/rl/runs/aaaaaaaaaaaa/rollout-trees?experiment_id=demo")
        self.assertEqual(r.status_code, 404, r.text)
        self.assertIn("训练运行", r.text)


class TestStudioAssistantJail(unittest.TestCase):
    def test_assistant_write_stays_in_user_space(self):
        from science_infra.control.assistant import assistant_tools
        from workflow.user_gateway.registry import create_project
        from workflow.user_gateway.tools import clear_user_tools

        tmp = tempfile.TemporaryDirectory()
        user_root = Path(tmp.name) / "user_space"
        user_root.mkdir()
        old = os.environ.get("SCIENCE_USER_SPACE_DIR")
        os.environ["SCIENCE_USER_SPACE_DIR"] = str(user_root)
        clear_user_tools()
        try:
            create_project("studio-jail", title="Studio jail", mode="native_mas")
            tools = {t["name"]: t["fn"] for t in assistant_tools("studio-jail")}
            result = tools["write_file"](path="mas/tir_agent.py", content="nope")
            self.assertIn("adapted/tir_agent.py", result)
            self.assertFalse((TIR / "tir_agent.py").read_text(encoding="utf-8").startswith("nope"))
            adapted = user_root / "projects" / "studio-jail" / "adapted" / "tir_agent.py"
            self.assertTrue(adapted.is_file())
            self.assertEqual(adapted.read_text(encoding="utf-8"), "nope")
        finally:
            clear_user_tools()
            if old is None:
                os.environ.pop("SCIENCE_USER_SPACE_DIR", None)
            else:
                os.environ["SCIENCE_USER_SPACE_DIR"] = old
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
