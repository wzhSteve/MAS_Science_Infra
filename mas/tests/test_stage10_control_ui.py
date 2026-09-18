"""Control plane + MASSpec 0.2 smoke (no GPU)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TIR = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TIR) not in sys.path:
    sys.path.insert(0, str(TIR))


class TestMasSpecExecutable(unittest.TestCase):
    def test_masspec_graph_executable_flag(self):
        from workflow.spec import AgentNodeSpec, EdgeSpec, MASSpec

        hub = MASSpec(topology="hub_react")
        self.assertTrue(hub.is_executable()[0])

        graph_ok = MASSpec(
            schema_version="0.2.0",
            topology="graph",
            agents=[AgentNodeSpec(id="hub", role="orchestrator")],
        )
        self.assertTrue(graph_ok.is_executable()[0])

        graph_bad = MASSpec(
            schema_version="0.2.0",
            topology="graph",
            agents=[AgentNodeSpec(id="planner", role="planner")],
            edges=[EdgeSpec(**{"from": "planner", "to": "execute_python", "kind": "message"})],
        )
        ok, reason = graph_bad.is_executable()
        self.assertFalse(ok)
        self.assertTrue("tool" in reason or "execute_python" in reason)

        pev = MASSpec(
            schema_version="0.2.0",
            topology="graph",
            entry_agent="planner",
            agents=[
                AgentNodeSpec(id="planner", role="planner", trainable=True),
                AgentNodeSpec(id="executor", role="executor", tools=["execute_python"], trainable=True),
                AgentNodeSpec(id="verifier", role="verifier", trainable=False),
            ],
            edges=[
                EdgeSpec(**{"from": "planner", "to": "executor", "kind": "route"}),
                EdgeSpec(**{"from": "executor", "to": "verifier", "kind": "message"}),
                EdgeSpec(**{"from": "verifier", "to": "planner", "kind": "feedback"}),
                EdgeSpec(**{"from": "executor", "to": "execute_python", "kind": "tool_call"}),
            ],
        )
        self.assertTrue(pev.is_executable()[0])


class TestControlApi(unittest.TestCase):
    def test_control_api_collect_diagnose(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app
        from science_infra.control.experiments import ensure_experiment

        ensure_experiment("demo")
        client = TestClient(create_app())
        r = client.post("/api/mas/collect?experiment_id=demo", json={"mock": True, "n": 1})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["n"], 1)
        r = client.post("/api/harness/diagnose?experiment_id=demo", json={})
        self.assertEqual(r.status_code, 200)
        r = client.get("/api/monitor/demo")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("n", 0), 1)
        r = client.get("/api/runs?experiment_id=demo")
        self.assertEqual(r.status_code, 200)
        runs = r.json().get("runs") or []
        if runs:
            rid = runs[0]["run_id"]
            g = client.get(f"/api/runs/{rid}")
            self.assertEqual(g.status_code, 200)
            self.assertIn("log_tail", g.json())


class TestRlYamlCli(unittest.TestCase):
    def test_rl_yaml_cli_flag_present(self):
        import train_tir_agent as t

        self.assertTrue(hasattr(t, "load_rl_yaml"))
        self.assertTrue(hasattr(t, "_deep_merge"))
