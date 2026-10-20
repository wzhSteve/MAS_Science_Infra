#!/usr/bin/env bash
# Launch the 256-trajectory runs through the UI control plane.
#   probe     one step per algorithm, no validation
#   continue  steps 2-5 from global_step_1, validate at step 5
# Each algorithm is its own experiment (scale256-grpo, scale256-flow-grpo,
# scale256-arpo, scale256-appo). The run record, stdout log, reward curve,
# and rollout tree — training rows and the step-5 validation — show up there.
# Baseline yaml files under this directory stay at 5 steps / resume_mode=disable.
set -euo pipefail
MODE="${1:-}"
if [[ "$MODE" != "probe" && "$MODE" != "continue" ]]; then
  echo "usage: $0 probe|continue" >&2
  exit 2
fi
ROOT="/root/autodl-tmp/MAS_Science_Infra"
SCALE="$ROOT/experiments/simple-pev/scale256"
export AGENTFLOW_TEMPERATURE="${AGENTFLOW_TEMPERATURE:-0.7}"
export AGENTFLOW_SFT_PROMPTS=1
unset AGENTFLOW_MAX_STEPS || true
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env" || true
  set +a
fi
export AGENTFLOW_TEMPERATURE="${AGENTFLOW_TEMPERATURE:-0.7}"
export AGENTFLOW_SFT_PROMPTS=1
unset AGENTFLOW_MAX_STEPS || true
PYTHON="${PYTHON:-/root/autodl-tmp/MAS_Science_Infra/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="/root/autodl-tmp/AgentFlow/.venv/bin/python"
fi
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/mas:$ROOT/rl/verl_backend${PYTHONPATH:+:$PYTHONPATH}"
export MODE SCALE ROOT
"$PYTHON" - <<'PY'
import os
import time
from pathlib import Path

import yaml

from science_infra.control.experiments import ensure_experiment, exp_dir
from science_infra.control.process_manager import PROCS, TERMINAL_STATES
from science_infra.control.training import build_training_plan, launch_training

mode = os.environ["MODE"]
scale = Path(os.environ["SCALE"])
root = Path(os.environ["ROOT"])
workflow = yaml.safe_load((scale / "workflow.yaml").read_text(encoding="utf-8"))
names = {
    "grpo": "scale256-grpo",
    "flow_grpo": "scale256-flow-grpo",
    "arpo": "scale256-arpo",
    "appo": "scale256-appo",
}

def dump(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

for algo, exp_id in names.items():
    ckpt = root / "mas" / "checkpoints" / "AgentLightning" / f"simple_pev_scale256_{algo}" / "global_step_1" / "actor"
    if mode == "continue" and not ckpt.is_dir():
        raise SystemExit(f"missing checkpoint {ckpt}")
    ensure_experiment(exp_id, name=f"scale256 {algo}")
    exp = exp_dir(exp_id)
    dump(exp / "workflow.yaml", workflow)
    rl = yaml.safe_load((scale / f"rl_{algo}.yaml").read_text(encoding="utf-8"))
    trainer = rl.setdefault("trainer", {})
    if mode == "probe":
        trainer["total_training_steps"] = 1
        trainer["test_freq"] = -1
        trainer["resume_mode"] = "disable"
    else:
        trainer["total_training_steps"] = 5
        trainer["test_freq"] = 5
        trainer["resume_mode"] = "auto"
    dump(exp / "rl.yaml", rl)
    plan = build_training_plan(exp_id)
    if not plan.ready:
        raise SystemExit(f"{exp_id} preflight failed: {plan.blocking_issues}")
    started = launch_training(
        exp_id,
        request_id=f"scale256-{mode}-{algo}-{time.time_ns()}",
        preflight_revision=plan.revision,
        stop_local_llm=True,
    )
    run_id = started["run_id"]
    print(f"scale256 {mode} {algo} experiment={exp_id} run={run_id} log={started.get('log_path')}", flush=True)
    # The process already snapshotted effective-rl.yaml. Put the 5-step
    # baseline back so the UI form is not left on the one-step probe.
    baseline = yaml.safe_load((scale / f"rl_{algo}.yaml").read_text(encoding="utf-8"))
    dump(exp / "rl.yaml", baseline)
    while True:
        row = PROCS.status(run_id, exp_id) or {}
        state = str(row.get("state") or "")
        if state in TERMINAL_STATES:
            break
        time.sleep(20)
    print(f"scale256 {mode} {algo} finished state={state} returncode={row.get('returncode')}", flush=True)
    if state != "succeeded":
        raise SystemExit(f"{exp_id} {mode} ended in {state}")
PY
