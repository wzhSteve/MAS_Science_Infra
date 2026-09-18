"""Stage 11: GPU API, compiler, graph collect, active-agent mapping."""

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


class TestCompiler(unittest.TestCase):
    def test_compile_rejects_agent_tool_call(self):
        from workflow.compiler import compile_spec
        from workflow.spec import AgentNodeSpec, EdgeSpec, MASSpec

        spec = MASSpec(
            topology="graph",
            entry_agent="planner",
            agents=[
                AgentNodeSpec(id="planner", role="planner"),
                AgentNodeSpec(id="executor", role="executor"),
            ],
            edges=[EdgeSpec(**{"from": "planner", "to": "executor", "kind": "tool_call"})],
        )
        compiled = compile_spec(spec)
        self.assertFalse(compiled.ok)

    def test_compile_pev_and_trainable(self):
        from workflow.compiler import compile_spec, trainable_agents
        from workflow.spec import AgentNodeSpec, EdgeSpec, MASSpec

        spec = MASSpec(
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
        compiled = compile_spec(spec)
        self.assertTrue(compiled.ok, compiled.reason)
        self.assertTrue(compiled.multi_agent)
        self.assertEqual(compiled.entry_agent, "planner")
        names = trainable_agents(spec)
        self.assertIn("planner", names)
        self.assertNotIn("verifier", names)


class TestCompiledCollect(unittest.TestCase):
    def test_graph_mock_collect_walks_agents(self):
        from workflow.runtime import ExecutionService
        from workflow.spec import AgentNodeSpec, EdgeSpec, HubSpec, MASSpec

        spec = MASSpec(
            topology="graph",
            entry_agent="planner",
            hub=HubSpec(skills=["react_loop"]),
            agents=[
                AgentNodeSpec(id="planner", role="planner", skills=["react_loop"], trainable=True),
                AgentNodeSpec(id="executor", role="executor", tools=["execute_python"], trainable=True),
            ],
            edges=[
                EdgeSpec(**{"from": "planner", "to": "executor", "kind": "route"}),
                EdgeSpec(**{"from": "executor", "to": "execute_python", "kind": "tool_call"}),
            ],
        )
        traj = ExecutionService(mock=True, spec=spec).run(
            {"id": "g", "question": "1+1", "answer": "2", "_mock_answer": "2"}
        )
        self.assertEqual(traj.meta.get("entry_agent"), "planner")
        self.assertEqual(traj.meta.get("compiled"), "graph_compiled")
        enters = [e for e in traj.events if (e.payload or {}).get("phase") == "enter"]
        self.assertGreaterEqual(len(enters), 2)


class TestGpuApi(unittest.TestCase):
    def test_gpus_api_and_recommend(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app
        from science_infra.control.experiments import apply_gpu_selection, recommend_rl

        rec = recommend_rl(1)
        self.assertEqual(rec["profile"], "fast")
        self.assertEqual(rec["n_runners"], 1)
        client = TestClient(create_app())
        r = client.get("/api/gpus")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("gpus", body)
        self.assertIn("count", body)
        self.assertIn("recommend", body)

        rl = apply_gpu_selection({"profile": "a800_2gpu", "rollout_per_gpu": 2}, [0])
        self.assertEqual(rl["profile"], "fast")
        self.assertEqual(rl["trainer"]["n_gpus_per_node"], 1)
        self.assertEqual(rl["devices"]["ids"], [0])

    def test_mas_palette_templates_executable(self):
        from science_infra.control.services import mas_palette
        from workflow.spec import MASSpec

        pal = mas_palette()
        ids = {t["id"] for t in pal["templates"]}
        self.assertIn("hub_react", ids)
        self.assertIn("pev_draft", ids)
        for t in pal["templates"]:
            spec = MASSpec.model_validate(t["workflow"])
            ok, reason = spec.is_executable()
            self.assertTrue(ok, f"{t['id']}: {reason}")

    def test_rl_yaml_skips_device_keys(self):
        import train_tir_agent as t

        merged = t._deep_merge(
            {"trainer": {"n_gpus_per_node": 1}, "actor_rollout_ref": {"rollout": {"n": 2}}},
            {
                "devices": {"ids": [0]},
                "rollout_per_gpu": 8,
                "trainer": {"n_gpus_per_node": 1},
                "actor_rollout_ref": {"rollout": {"n": 2}},
            },
        )
        self.assertNotIn("devices", merged)
        self.assertNotIn("rollout_per_gpu", merged)
        self.assertEqual(merged["actor_rollout_ref"]["rollout"]["n"], 2)

    def test_agl_health_and_rewrite(self):
        from fastapi.testclient import TestClient

        from science_infra.control.app import create_app
        from science_infra.control.services import rewrite_agl_payload

        client = TestClient(create_app())
        r = client.get("/api/agl/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("ok", body)
        self.assertEqual(body.get("ui_path"), "/agl/metrics")

        html = b'<script src="/assets/index.js"></script><fetch("/v1/agl/health")>'
        out = rewrite_agl_payload(html, "text/html").decode()
        self.assertIn("/agl/assets/", out)
        self.assertIn("/agl/v1/", out)
        js = b'{path:"/",element:X},{path:"metrics"};to:"/metrics";url:"v1/agl/resources"'
        out_js = rewrite_agl_payload(js, "application/javascript").decode()
        self.assertIn('{path:"/agl",element:', out_js)
        self.assertIn('to:"/agl/metrics"', out_js)
        self.assertIn("/agl/v1/agl/resources", out_js)
        self.assertNotIn('{path:"/",element:', out_js)

    def test_run_sh_help_train_flags(self):
        import subprocess

        text = subprocess.check_output(["bash", str(ROOT / "run.sh"), "help"], text=True)
        self.assertIn("--gpu", text)
        self.assertIn("--n-runners", text)
        self.assertIn("--rl-yaml", text)
