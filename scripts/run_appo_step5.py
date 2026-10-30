"""After the in-flight 4-step APPO run finishes, start step 5 from global_step_4."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8787"
EXP = "scale256-appo"
CURRENT = "e45553a1e548"
CKPT = Path(
    "/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning/"
    "simple_pev_scale256_appo_valid/global_step_4"
)
POLL = 30


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


def main() -> int:
    print(f"wait for {CURRENT} to succeed, then resume step 5", flush=True)
    while True:
        try:
            row = _request("GET", f"/api/rl/runs/{CURRENT}?experiment_id={EXP}&tail=0")
        except urllib.error.URLError as exc:
            print(f"poll error {exc}", flush=True)
            time.sleep(POLL)
            continue
        state = str(row.get("state") or "")
        print(f"{time.strftime('%F %T')} {CURRENT} {state}", flush=True)
        if state == "succeeded":
            break
        if state in {"failed", "cancelled", "error"}:
            print(f"current run {state}; not starting step 5", flush=True)
            return 1
        time.sleep(POLL)
    deadline = time.time() + 1800
    while time.time() < deadline:
        if (CKPT / "actor" / "model_world_size_2_rank_0.pt").is_file():
            break
        print("waiting for global_step_4 weights", flush=True)
        time.sleep(10)
    else:
        print("global_step_4 missing", flush=True)
        return 1
    started = _request(
        "POST",
        f"/api/rl/train?experiment_id={EXP}",
        {"stop_llm": True, "confirm_gpu": True},
    )
    run_id = started.get("run_id")
    print(f"started step-5 run_id={run_id}", flush=True)
    return 0 if run_id else 1


if __name__ == "__main__":
    sys.exit(main())
