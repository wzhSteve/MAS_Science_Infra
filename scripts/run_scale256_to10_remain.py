"""Finish to-10 after ARPO hang: ARPO from global_step_9, then APPO."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8787"
EXPERIMENTS = ("scale256-arpo", "scale256-appo")
TERMINAL = {"succeeded", "failed", "cancelled", "error"}
POLL_SECONDS = 60


def _request(method: str, path: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.loads(response.read().decode())


def _activity(experiment_id: str) -> dict:
    return _request("GET", f"/api/rl/activity?experiment_id={experiment_id}")


def _run(experiment_id: str, run_id: str) -> dict:
    return _request(
        "GET",
        f"/api/rl/runs/{run_id}?experiment_id={experiment_id}&tail=0",
    )


def main() -> int:
    for experiment_id in EXPERIMENTS:
        active = _activity(experiment_id).get("run")
        if active and active.get("running"):
            print(f"refuse: {experiment_id} already running {active.get('run_id')}", flush=True)
            return 1
    for experiment_id in EXPERIMENTS:
        started = _request(
            "POST",
            f"/api/rl/train?experiment_id={experiment_id}",
            {"stop_llm": True, "confirm_gpu": True},
        )
        run_id = str(started.get("run_id") or "")
        print(f"started {experiment_id} run_id={run_id}", flush=True)
        if not run_id:
            print(f"no run_id for {experiment_id}: {started}", flush=True)
            return 1
        while True:
            time.sleep(POLL_SECONDS)
            try:
                row = _run(experiment_id, run_id)
            except urllib.error.URLError as exc:
                print(f"poll error {experiment_id} {exc}", flush=True)
                continue
            state = str(row.get("state") or "")
            print(f"{time.strftime('%F %T')} {experiment_id} {run_id} {state}", flush=True)
            if state in TERMINAL:
                if state != "succeeded":
                    print(f"stop chain: {experiment_id} {state}", flush=True)
                    return 1
                break
    print("arpo+appo reached 10 steps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
