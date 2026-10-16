#!/usr/bin/env python3
"""Sequential 4-algorithm AgentFlow RL via MAS Control (UI-visible runs).

For each algo (grpo -> arpo -> appo -> flow_grpo):
  1. Sync experiments/agentflow_4algo/rl_{algo}.yaml into the UI-used rl.yaml
  2. build_training_plan + launch_training (writes artifacts/runs/<run_id>/)
  3. Wait until the run reaches a terminal state

Usage:
  .venv/bin/python scripts/agentflow_4algo_ui_run.py --steps 1     # smoke
  .venv/bin/python scripts/agentflow_4algo_ui_run.py --steps 10    # 10-step
  .venv/bin/python scripts/agentflow_4algo_ui_run.py --algos grpo,arpo
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from science_infra.control.process_manager import PROCS, TERMINAL_STATES  # noqa: E402
from science_infra.control.training import build_training_plan, launch_training  # noqa: E402

EXP_DIR = ROOT / "experiments" / "agentflow_4algo"
EXP_ID = "agentflow_4algo"
RL_MAIN = EXP_DIR / "rl.yaml"
BACKUP = EXP_DIR / "rl.yaml.bak"


def _set_total_steps(steps: int) -> None:
    for path in EXP_DIR.glob("rl_*.yaml"):
        text = path.read_text(encoding="utf-8")
        for old in ("total_training_steps: 10", "total_training_steps: 1"):
            text = text.replace(old, f"total_training_steps: {steps}")
        # 20-row hotpot train / batch 8 -> 3 steps per epoch; 5 epochs covers 10
        # steps. For a smoke of N steps keep epochs >= ceil(N / 3) so the
        # dataloader actually yields N batches before early-stop.
        text = text.replace("total_epochs: 5", f"total_epochs: {max(1, (steps + 2) // 3)}")
        text = text.replace("total_epochs: 1", f"total_epochs: {max(1, (steps + 2) // 3)}")
        path.write_text(text, encoding="utf-8")


def sync_algo(algo: str) -> None:
    src = EXP_DIR / f"rl_{algo}.yaml"
    shutil.copyfile(src, RL_MAIN)


def wait_run(run_id: str, timeout_s: int = 6 * 3600) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        row = PROCS.status(run_id)
        if row and row.get("state") in TERMINAL_STATES:
            return row
        time.sleep(5)
    raise TimeoutError(f"run {run_id} did not finish in {timeout_s}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algos", default="grpo,arpo,appo,flow_grpo")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--keep-rl", action="store_true", help="do not restore rl.yaml at the end")
    ap.add_argument("--wait", action="store_true", default=True)
    args = ap.parse_args()
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]

    _set_total_steps(int(args.steps))
    # Back up the UI-facing rl.yaml so we can restore the user's last choice.
    shutil.copyfile(RL_MAIN, BACKUP)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    os.environ.setdefault("TIR_MAX_MODEL_LEN", "2560")

    summary = []
    try:
        for algo in algos:
            print(f"\n=== [{time.strftime('%F %T')}] >>> sync+launch {algo} ===", flush=True)
            sync_algo(algo)
            plan = build_training_plan(EXP_ID)
            if not plan.ready:
                issues = [(i["code"], i["message"]) for i in plan.blocking_issues]
                raise RuntimeError(f"preflight not ready: {issues}")
            out = launch_training(
                EXP_ID,
                request_id=f"{algo}-{args.steps}step-{int(time.time())}",
                preflight_revision=plan.revision,
                stop_local_llm=True,
            )
            run_id = out["run_id"]
            print(f"launched run_id={run_id} algo={out.get('algo')} gpus={out.get('cuda_visible_devices')}", flush=True)
            row = wait_run(run_id)
            rc = row.get("returncode")
            state = row.get("state")
            print(f"<<< {algo} run_id={run_id} state={state} returncode={rc}", flush=True)
            summary.append((algo, run_id, state, rc))
            if state != "succeeded":
                raise RuntimeError(f"{algo} ended with state={state}")
    finally:
        if not args.keep_rl and BACKUP.exists():
            shutil.copyfile(BACKUP, RL_MAIN)

    print("\n=== ALL RUNS DONE ===")
    for algo, run_id, state, rc in summary:
        print(f"{algo}: {state} (run_id={run_id}, rc={rc})")
    return 0 if all(s == "succeeded" for _, _, s, _ in summary) else 1


if __name__ == "__main__":
    raise SystemExit(main())
