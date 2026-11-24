"""Frozen vLLM sleep/wake client. No GPU."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from rl.hooks.frozen_vllm import (
    after_planner_sleep,
    before_planner_wake,
    enabled,
    planner_sleep_level,
    prometheus_gauge_sum,
    sleep,
)

REPO = Path(__file__).resolve().parents[2]


class _Server(ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.events: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def do_POST(self):
        server: _Server = self.server  # type: ignore[assignment]
        server.events.append(f"POST {self.path}")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if self.path.startswith("/v1/chat/completions"):
            self.wfile.write(b'{"choices":[{"text":"ok"}]}')
        else:
            self.wfile.write(b"{}")

    def do_GET(self):
        server: _Server = self.server  # type: ignore[assignment]
        server.events.append(f"GET {self.path}")
        if self.path.startswith("/metrics"):
            body = (
                'vllm:num_requests_running{model_name="Qwen3-4B",engine="0"} 0\n'
                'vllm:num_requests_waiting{model_name="Qwen3-4B",engine="0"} 0\n'
            ).encode()
        else:
            body = b'{"is_sleeping": true}'
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_rank0_wakes_before_planner_and_sleeps_at_level_2(monkeypatch):
    server = _Server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("FCR_FROZEN_VLLM_URL", f"http://{host}:{port}")
    monkeypatch.setenv("FCR_FROZEN_SLEEP_SETTLE_S", "0")
    monkeypatch.setenv("FCR_FROZEN_IDLE_POLL_S", "0")
    order: list[str] = []

    def barrier():
        order.append("barrier")

    before_planner_wake(0, barrier)
    order.append("planner_wake")
    assert planner_sleep_level() == 2
    after_planner_sleep(0, barrier)
    server.shutdown()

    assert server.events[0] == "POST /wake_up"
    assert any(item.startswith("POST /v1/chat/completions") for item in server.events)
    assert "GET /metrics" in server.events
    assert server.events.index("GET /metrics") < server.events.index("POST /sleep?level=2")
    assert "POST /sleep?level=2" in server.events
    assert "GET /is_sleeping" in server.events
    assert order[0] == "barrier"
    assert order[1] == "planner_wake"


def test_non_rank0_does_not_call_http(monkeypatch):
    server = _Server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("FCR_FROZEN_VLLM_URL", f"http://{host}:{port}")
    before_planner_wake(1, lambda: None)
    after_planner_sleep(1, lambda: None)
    server.shutdown()
    assert server.events == []


def test_unset_url_keeps_planner_sleep_level_1(monkeypatch):
    monkeypatch.delenv("FCR_FROZEN_VLLM_URL", raising=False)
    assert enabled() is False
    assert planner_sleep_level() == 1
    before_planner_wake(0, lambda: (_ for _ in ()).throw(AssertionError("barrier")))
    after_planner_sleep(0, lambda: (_ for _ in ()).throw(AssertionError("barrier")))


def test_off_switch_disables_client(monkeypatch):
    monkeypatch.setenv("FCR_FROZEN_VLLM_URL", "off")
    assert enabled() is False
    assert planner_sleep_level() == 1


def test_sharding_manager_calls_the_hooks():
    text = (REPO / "rl/verl_backend/verl/workers/sharding_manager/fsdp_vllm.py").read_text(encoding="utf-8")
    assert "before_planner_wake(self.tp_rank, self._world_barrier)" in text
    assert "self.inference_engine.sleep(level=planner_sleep_level())" in text
    assert "after_planner_sleep(self.tp_rank, self._world_barrier)" in text


def test_memory_budget_and_resume_step11():
    cfg = yaml.safe_load((REPO / "experiments/fcr-step4-retrain/rl.yaml").read_text(encoding="utf-8"))
    util = float(cfg["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"])
    assert util == 0.45
    assert util + 0.18 <= 0.70
    assert cfg["trainer"]["resume_from_path"].endswith("global_step_13")
    assert cfg["trainer"]["total_training_steps"] == 17
    assert cfg["env"]["FCR_FROZEN_VLLM_URL"] == "http://127.0.0.1:8001"
    script = (REPO / "scripts/serve_fcr_verifier.sh").read_text(encoding="utf-8")
    assert "--tensor-parallel-size" in script
    assert "--enable-sleep-mode" in script
    assert 'FCR_VERIFIER_GPU_UTIL:-0.18' in script
    assert 'FCR_VERIFIER_TP:-2' in script
    assert "VLLM_SERVER_DEV_MODE=1" in script
    assert "reaping stale vllm" in script
    launch = (REPO / "experiments/fcr-step4-retrain/launch_r7.sh").read_text(encoding="utf-8")
    assert "global_step_13" in launch
    assert "a GPU vLLM process is resident" not in launch


def test_http_failure_still_reaches_barrier(monkeypatch):
    monkeypatch.setenv("FCR_FROZEN_VLLM_URL", "http://127.0.0.1:9")
    seen = []
    after_planner_sleep(0, lambda: seen.append("sleep"))
    before_planner_wake(0, lambda: seen.append("wake"))
    assert seen == ["sleep", "wake"]


def test_is_sleeping_false_does_not_raise(monkeypatch):
    class Deny(_Handler):
        def do_GET(self):
            if self.path.startswith("/metrics"):
                body = (
                    'vllm:num_requests_running{engine="0"} 0\n'
                    'vllm:num_requests_waiting{engine="0"} 0\n'
                ).encode()
            else:
                body = json.dumps({"is_sleeping": False}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class DenyServer(_Server):
        def __init__(self):
            ThreadingHTTPServer.__init__(self, ("127.0.0.1", 0), Deny)
            self.events = []

    server = DenyServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("FCR_FROZEN_VLLM_URL", f"http://{host}:{port}")
    monkeypatch.setenv("FCR_FROZEN_SLEEP_SETTLE_S", "0")
    monkeypatch.setenv("FCR_FROZEN_IDLE_POLL_S", "0")
    try:
        sleep(timeout=5)
    finally:
        server.shutdown()
    server_text = (REPO / "rl/verl_backend/verl/workers/rollout/async_server.py").read_text(encoding="utf-8")
    assert "refusing to cancel a collective and save" in server_text
    assert "ray.cancel" not in server_text


def test_gauge_sum_adds_both_engines():
    text = "\n".join(
        [
            "# HELP vllm:num_requests_running x",
            'vllm:num_requests_running{engine="0"} 1',
            'vllm:num_requests_running{engine="1"} 2',
            'vllm:num_requests_waiting{engine="0"} 0',
        ]
    )
    assert prometheus_gauge_sum(text, "vllm:num_requests_running") == 3
    assert prometheus_gauge_sum(text, "vllm:num_requests_waiting") == 0
    assert prometheus_gauge_sum(text, "vllm:missing") is None


def test_sleep_waits_until_requests_drain(monkeypatch):
    class BusyThenIdle(_Handler):
        def do_GET(self):
            server: _Server = self.server  # type: ignore[assignment]
            server.events.append(f"GET {self.path}")
            if self.path.startswith("/metrics"):
                n = sum(1 for item in server.events if item == "GET /metrics")
                running = 2 if n < 3 else 0
                body = (
                    f'vllm:num_requests_running{{engine="0"}} {running}\n'
                    'vllm:num_requests_waiting{engine="0"} 0\n'
                ).encode()
            else:
                body = b'{"is_sleeping": true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class BusyServer(_Server):
        def __init__(self):
            ThreadingHTTPServer.__init__(self, ("127.0.0.1", 0), BusyThenIdle)
            self.events = []

    server = BusyServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("FCR_FROZEN_VLLM_URL", f"http://{host}:{port}")
    monkeypatch.setenv("FCR_FROZEN_SLEEP_SETTLE_S", "0")
    monkeypatch.setenv("FCR_FROZEN_IDLE_POLL_S", "0")
    monkeypatch.setenv("FCR_FROZEN_IDLE_TIMEOUT", "5")
    try:
        sleep(timeout=5)
    finally:
        server.shutdown()
    sleep_at = server.events.index("POST /sleep?level=2")
    metrics_before = [item for item in server.events[:sleep_at] if item == "GET /metrics"]
    assert len(metrics_before) == 3


def test_busy_engine_refuses_to_sleep(monkeypatch):
    class AlwaysBusy(_Handler):
        def do_GET(self):
            body = (
                'vllm:num_requests_running{engine="0"} 1\n'
                'vllm:num_requests_waiting{engine="0"} 4\n'
            ).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    class BusyServer(_Server):
        def __init__(self):
            ThreadingHTTPServer.__init__(self, ("127.0.0.1", 0), AlwaysBusy)
            self.events = []

    server = BusyServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    monkeypatch.setenv("FCR_FROZEN_VLLM_URL", f"http://{host}:{port}")
    monkeypatch.setenv("FCR_FROZEN_SLEEP_SETTLE_S", "0")
    monkeypatch.setenv("FCR_FROZEN_IDLE_POLL_S", "0")
    monkeypatch.setenv("FCR_FROZEN_IDLE_TIMEOUT", "0")
    try:
        try:
            sleep(timeout=5)
        except RuntimeError as exc:
            assert "refusing to sleep" in str(exc)
        else:
            raise AssertionError("sleep posted while requests were running")
    finally:
        server.shutdown()
    assert "POST /sleep?level=2" not in server.events
