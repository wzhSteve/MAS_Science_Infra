"""Continue GRPO, Flow-GRPO, ARPO, then APPO until each has 20 training steps.

If a run stalls (quiet stdout + occupied idle GPUs), stop it and resume from
the latest saved checkpoint. Does not change RL algorithms.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8787"
ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
EXPERIMENTS = (
    "scale256-grpo",
    "scale256-flow-grpo",
    "scale256-arpo",
    "scale256-appo",
)
CKPT_DIRS = {
    "scale256-grpo": ROOT
    / "mas/checkpoints/AgentLightning/simple_pev_scale256_grpo_valid",
    "scale256-flow-grpo": ROOT
    / "mas/checkpoints/AgentLightning/simple_pev_scale256_flow_grpo_valid",
    "scale256-arpo": ROOT
    / "mas/checkpoints/AgentLightning/simple_pev_scale256_arpo_valid",
    "scale256-appo": ROOT
    / "mas/checkpoints/AgentLightning/simple_pev_scale256_appo_valid",
}
TERMINAL = {"succeeded", "failed", "cancelled", "error"}
POLL_SECONDS = 60
STALL_SECONDS = 3600
GPU_IDLE_UTIL = 2
GPU_HELD_MIB = 20000
GPU_FREE_WAIT = 1800


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


def _gpu_stats() -> tuple[list[int], list[int]]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    utils: list[int] = []
    mems: list[int] = []
    for line in out.strip().splitlines():
        util_s, mem_s = [part.strip() for part in line.split(",")]
        utils.append(int(float(util_s)))
        mems.append(int(float(mem_s)))
    return utils, mems


def _log_age_seconds(row: dict) -> float | None:
    path = Path(str(row.get("log_path") or ""))
    if not path.is_file():
        return None
    return time.time() - path.stat().st_mtime


def _is_stalled(row: dict) -> bool:
    if not row.get("running"):
        return False
    age = _log_age_seconds(row)
    if age is None or age < STALL_SECONDS:
        return False
    utils, mems = _gpu_stats()
    if not utils:
        return False
    held = max(mems) >= GPU_HELD_MIB
    idle = max(utils) <= GPU_IDLE_UTIL
    return held and idle


def _latest_ckpt(experiment_id: str) -> Path:
    ckpt_root = CKPT_DIRS[experiment_id]
    steps: list[tuple[int, Path]] = []
    for path in ckpt_root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if not match:
            continue
        weight = path / "actor" / "model_world_size_2_rank_0.pt"
        if weight.is_file():
            steps.append((int(match.group(1)), path))
    if not steps:
        raise FileNotFoundError(f"no resumable ckpt under {ckpt_root}")
    return max(steps, key=lambda item: item[0])[1]


def _point_resume_yaml(experiment_id: str, ckpt: Path) -> None:
    yaml_path = ROOT / "experiments" / experiment_id / "rl_ui_resume.yaml"
    text = yaml_path.read_text()
    updated, n = re.subn(
        r"(?m)^  resume_from_path: .+$",
        f"  resume_from_path: {ckpt}",
        text,
        count=1,
    )
    if n != 1:
        raise RuntimeError(f"could not patch resume_from_path in {yaml_path}")
    yaml_path.write_text(updated)
    print(f"resume_from_path -> {ckpt}", flush=True)


def _wait_gpu_free() -> None:
    deadline = time.time() + GPU_FREE_WAIT
    while time.time() < deadline:
        _, mems = _gpu_stats()
        if mems and max(mems) == 0:
            print("gpu free", flush=True)
            return
        print(f"waiting gpu free mem={mems}", flush=True)
        time.sleep(10)
    raise TimeoutError("gpu did not free after stop")


def _start(experiment_id: str) -> str:
    started = _request(
        "POST",
        f"/api/rl/train?experiment_id={experiment_id}",
        {"stop_llm": True, "confirm_gpu": True},
    )
    run_id = str(started.get("run_id") or "")
    print(f"started {experiment_id} run_id={run_id}", flush=True)
    if not run_id:
        raise RuntimeError(f"no run_id for {experiment_id}: {started}")
    return run_id


def _run_one_experiment(experiment_id: str) -> int:
    while True:
        run_id = _start(experiment_id)
        while True:
            time.sleep(POLL_SECONDS)
            try:
                row = _run(experiment_id, run_id)
            except urllib.error.URLError as exc:
                print(f"poll error {experiment_id} {exc}", flush=True)
                continue
            state = str(row.get("state") or "")
            age = _log_age_seconds(row)
            age_s = f"{age:.0f}s" if age is not None else "na"
            print(
                f"{time.strftime('%F %T')} {experiment_id} {run_id} {state} log_age={age_s}",
                flush=True,
            )
            if state == "succeeded":
                return 0
            if state in TERMINAL:
                print(f"stop chain: {experiment_id} {state}", flush=True)
                return 1
            if _is_stalled(row):
                print(f"stall detected {experiment_id} {run_id}; stopping", flush=True)
                try:
                    _request("POST", f"/api/rl/stop?experiment_id={experiment_id}")
                except urllib.error.URLError as exc:
                    print(f"stop error {exc}", flush=True)
                _wait_gpu_free()
                ckpt = _latest_ckpt(experiment_id)
                _point_resume_yaml(experiment_id, ckpt)
                break


def main() -> int:
    for experiment_id in EXPERIMENTS:
        active = _activity(experiment_id).get("run")
        if active and active.get("running"):
            print(
                f"refuse: {experiment_id} already running {active.get('run_id')}",
                flush=True,
            )
            return 1
    for experiment_id in EXPERIMENTS:
        rc = _run_one_experiment(experiment_id)
        if rc != 0:
            return rc
    print("all four reached 20 steps", flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        EXPERIMENTS = tuple(sys.argv[1:])
    sys.exit(main())
