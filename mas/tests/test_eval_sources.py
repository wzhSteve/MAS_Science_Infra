"""Path selection for parquet eval. Does not start training or inference."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
for _p in (str(ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from science_infra.control.process_manager import PROCS  # noqa: E402
from science_infra.control.rollout_runs import eval_parquet_paths, format_eval_trace  # noqa: E402
from science_infra.control.training_logs import read_log  # noqa: E402


class TestEvalParquetPaths(unittest.TestCase):
    def test_test_parquet_sits_beside_val(self):
        with tempfile.TemporaryDirectory() as tmp:
            val = Path(tmp) / "val.parquet"
            val.write_bytes(b"")
            paths = eval_parquet_paths({"rl": {"data": {"val_files": str(val)}}})
        self.assertEqual(paths["val_files"], str(val))
        self.assertEqual(paths["test_files"], str(val.with_name("test.parquet")))
        self.assertTrue(paths["val_exists"])
        self.assertFalse(paths["test_exists"])

    def test_missing_files_are_absent(self):
        missing = Path("/tmp/does-not-exist-mas-eval") / "val.parquet"
        paths = eval_parquet_paths({"rl": {"data": {"val_files": str(missing)}}})
        self.assertEqual(paths["test_files"], str(missing.with_name("test.parquet")))
        self.assertFalse(paths["val_exists"])
        self.assertFalse(paths["test_exists"])

    def test_explicit_test_files_override_sibling(self):
        chosen = REPO / "ref_Rep" / "HIVE" / "test" / "gaia" / "data" / "data.json"
        paths = eval_parquet_paths({"rl": {"data": {
            "val_files": "/tmp/does-not-exist-mas-eval/val.parquet",
            "test_files": str(chosen),
        }}})
        self.assertEqual(paths["test_files"], str(chosen))
        self.assertTrue(paths["test_exists"])
        self.assertFalse(paths["val_exists"])

    def test_trace_lines_show_agent_hops(self):
        lines = format_eval_trace("[1/1]", [
            {
                "turn": 0, "src": "planner", "dst": "route_exec", "kind": "plan_step",
                "payload": {"next": "python_coder", "args": {"code": "result=42"}, "sub_goal": "compute", "done": True},
            },
            {
                "turn": 0, "src": "route_exec", "dst": "python_coder", "kind": "tool_invoke",
                "payload": {"code": "result=42"},
            },
            {
                "turn": 0, "src": "python_coder", "dst": "verifier", "kind": "tool_result",
                "payload": {"output": "42", "ok": True, "evidence_type": "DIRECT"},
            },
            {
                "turn": 0, "src": "verifier", "dst": "planner", "kind": "verify",
                "payload": {"ok": True, "step_conclusion": "COMPLETE", "reason": "ok"},
            },
        ])
        text = "\n".join(lines)
        self.assertIn("planner → route_exec  plan_step", text)
        self.assertIn("route_exec → python_coder  tool_invoke", text)
        self.assertIn("python_coder → verifier  tool_result", text)
        self.assertIn("verifier → planner  verify", text)
        self.assertIn("next=python_coder", text)

    def test_trace_expands_command_analysis_outline(self):
        lines = format_eval_trace("[2/5]", [
            {
                "turn": 1, "src": "planner", "dst": "route_exec", "kind": "plan_step",
                "payload": {
                    "next": "u_epc-aw-main__executor",
                    "args": {"tool_name": "Python_Coder_Tool"},
                    "sub_goal": "compute jewels of Aaron then Siobhan",
                    "done": False,
                    "trace": {
                        "outline": {"1": "compute jewels of Aaron then Siobhan"},
                        "tool": "Python_Coder_Tool",
                        "bts": {"n": 1, "selected": 0},
                        "analysis": "need the first quantity",
                    },
                },
            },
            {
                "turn": 1, "src": "u_epc-aw-main__executor", "dst": "verifier", "kind": "tool_result",
                "payload": {
                    "output": "23",
                    "ok": True,
                    "evidence_type": "DIRECT",
                    "trace": {"command": "print((40/2+5)-2)", "output": "23", "analysis": "arithmetic"},
                },
            },
            {
                "turn": 1, "src": "verifier", "dst": "planner", "kind": "verify",
                "payload": {
                    "ok": False,
                    "step_conclusion": "INCOMPLETE",
                    "reason": "CONTINUE",
                    "ready_to_stop": False,
                    "trace": {
                        "analysis": "still missing Siobhan",
                        "new_info": "Aaron has 23",
                        "outline_updated": {"1": "compute Siobhan"},
                    },
                },
            },
        ], detail=True)
        text = "\n".join(lines)
        self.assertIn("command = print((40/2+5)-2)", text)
        self.assertIn("analysis =", text)
        self.assertIn("outline.1 =", text)
        self.assertIn("bts n=1 selected=0", text)
        self.assertIn("outline' =", text)

    def test_eval_log_is_readable_like_training_stdout(self):
        row = PROCS.start_inline(kind="eval", experiment_id="demo", meta={"split": "val"})
        run_id = row["run_id"]
        try:
            PROCS.append_log("demo", run_id, "MAS 测试 · val · 1 条")
            block = read_log("demo", run_id, None, 64 * 1024, None)
            self.assertEqual(block["state"]["kind"], "eval")
            self.assertIn("MAS 测试", block["text"])
            self.assertFalse(block["terminal"])
            PROCS.finish_inline("demo", run_id, state="succeeded", message="完成", returncode=0)
            done = read_log("demo", run_id, block["next_offset"], 64 * 1024, block["generation"])
            self.assertTrue(done["terminal"])
            self.assertEqual(done["state"]["state"], "succeeded")
        finally:
            PROCS._pending.pop(run_id, None)
            run_dir = PROCS.run_dir("demo", run_id)
            if run_dir.exists():
                for child in run_dir.iterdir():
                    child.unlink()
                run_dir.rmdir()

    def test_stop_eval_sets_cancel_and_stopping(self):
        import threading

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import science_infra.control.rollout_runs as rr

        row = PROCS.start_inline(kind="eval", experiment_id="demo", meta={"split": "val"})
        run_id = row["run_id"]
        try:
            with rr._eval_guard:
                rr._eval_running["demo"] = run_id
                rr._eval_cancel[run_id] = threading.Event()
            app = FastAPI()
            app.include_router(rr.router)
            client = TestClient(app)
            from unittest.mock import patch

            with patch("science_infra.control.services.stop_local_llm", return_value={"running": False, "killed": []}) as stop_llm:
                resp = client.post(f"/api/mas/eval-runs/{run_id}/stop?experiment_id=demo")
            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["state"], "stopping")
            self.assertTrue(rr._eval_cancel[run_id].is_set())
            stop_llm.assert_called_once_with("demo")
            self.assertIn(run_id, rr._eval_release_llm)
            status = PROCS.status(run_id, "demo") or {}
            self.assertEqual(status.get("state"), "stopping")
            self.assertTrue(status.get("running"))
            raw = json.loads((PROCS.run_dir("demo", run_id) / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(raw.get("state"), "stopping")
            log = (PROCS.run_dir("demo", run_id) / "stdout.log").read_text(encoding="utf-8")
            self.assertIn("停止测试", log)
            self.assertIn("本地 vLLM", log)
        finally:
            with rr._eval_guard:
                if rr._eval_running.get("demo") == run_id:
                    rr._eval_running.pop("demo", None)
                rr._eval_cancel.pop(run_id, None)
                rr._eval_release_llm.discard(run_id)
            PROCS._pending.pop(run_id, None)
            run_dir = PROCS.run_dir("demo", run_id)
            if run_dir.exists():
                for child in run_dir.iterdir():
                    child.unlink()
                run_dir.rmdir()

    def test_eval_stops_vllm_only_when_started(self):
        from unittest.mock import patch

        import science_infra.control.rollout_runs as rr

        calls = {"stop": 0}

        def _stop(_exp: str):
            calls["stop"] += 1
            return {"running": False}

        def _run(started: bool, reused: bool) -> None:
            calls["stop"] = 0
            row = PROCS.start_inline(kind="eval", experiment_id="demo", meta={"split": "val"})
            run_id = row["run_id"]
            try:
                with patch("science_infra.control.services.ensure_local_llm_ready", return_value={
                    "started": started,
                    "reused": reused,
                    "base_url": "http://127.0.0.1:8000/v1",
                    "models": ["m"],
                }), patch("science_infra.control.services.stop_local_llm", side_effect=_stop), patch.object(
                    rr, "_execute_question", return_value=({"final_answer": "1", "status": "ok"}, [])
                ):
                    rr._run_eval_job(
                        "demo",
                        run_id,
                        workflow={"agents": []},
                        tasks=[{"id": "t1", "question": "1+1", "answer": "2", "source": "gsm8k"}],
                        llm=None,
                        model_summary={"name": "m"},
                        model_public={"model": "m", "base_url": "http://127.0.0.1:8000/v1"},
                        split="val",
                        parquet="/tmp/val.parquet",
                        local_llm=True,
                    )
            finally:
                PROCS._pending.pop(run_id, None)
                run_dir = PROCS.run_dir("demo", run_id)
                if run_dir.exists():
                    for child in run_dir.iterdir():
                        child.unlink()
                    run_dir.rmdir()

        _run(started=True, reused=False)
        self.assertEqual(calls["stop"], 1)
        _run(started=False, reused=True)
        self.assertEqual(calls["stop"], 0)


if __name__ == "__main__":
    unittest.main()
