"""Start the four resume runs through the control plane, one at a time."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8787"
EXPERIMENTS = ("scale256-grpo", "scale256-flow-grpo", "scale256-arpo", "scale256-appo")
ACTIVE = {"preparing", "starting", "running", "stopping"}


def _get(path: str) -> dict:
    last_error: Exception | None = None
    for _ in range(30):
        try:
            with urllib.request.urlopen(BASE + path, timeout=60) as response:
                payload = json.load(response)
            return payload if isinstance(payload, dict) else {}
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"control plane unavailable: {last_error}")


def _post(path: str) -> dict:
    request = urllib.request.Request(
        BASE + path,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.load(response)
    return payload if isinstance(payload, dict) else {}


def _runs(experiment_id: str) -> list:
    payload = _get(f"/api/rl/runs?experiment_id={experiment_id}&kinds=train&limit=100")
    rows = payload.get("runs") or []
    return [row for row in rows if isinstance(row, dict)]


def _ours(rows: list, started: float) -> list:
    return [row for row in rows if float(row.get("started_at") or 0) >= started - 30]


def main() -> int:
    started = float(sys.argv[1]) if len(sys.argv) > 1 else time.time()
    for experiment_id in EXPERIMENTS:
        while True:
            rows = _ours(_runs(experiment_id), started)
            active = next((row for row in rows if row.get("running") or row.get("state") in ACTIVE), None)
            if active:
                print(f"waiting {experiment_id} {active.get('run_id')} {active.get('state')}", flush=True)
                time.sleep(60)
                continue
            if any(row.get("state") == "succeeded" for row in rows):
                print(f"finished {experiment_id}", flush=True)
                break
            failed = next((row for row in rows if row.get("state") in {"failed", "cancelled"}), None)
            if failed:
                print(f"stopped {experiment_id} {failed.get('state')} {failed.get('message')}", flush=True)
                return 1
            busy = False
            for other in EXPERIMENTS:
                other_rows = _ours(_runs(other), started)
                if any(row.get("running") or row.get("state") in ACTIVE for row in other_rows):
                    busy = True
                    break
            if busy:
                time.sleep(60)
                continue
            print(f"starting {experiment_id}", flush=True)
            try:
                launched = _post(f"/api/rl/train?experiment_id={experiment_id}")
            except urllib.error.HTTPError as error:
                print(error.read().decode("utf-8", errors="replace"), flush=True)
                return 1
            print(f"launched {experiment_id} {launched.get('run_id')} {launched.get('state')}", flush=True)
            time.sleep(20)
    return 0


if __name__ == "__main__":
    sys.exit(main())
