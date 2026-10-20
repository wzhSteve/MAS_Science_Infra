#!/usr/bin/env bash
# Run the four 5-step fidelity trains only when the temperature probe
# found at least one group with both reward 0 and reward 1.
set -euo pipefail
ROOT="/root/autodl-tmp/MAS_Science_Infra"
PROBE="$ROOT/user_space/projects/simple_pev/artifacts/temperature_probe.json"
MAS20="$ROOT/user_space/projects/simple_pev/artifacts/mas20_accuracy.json"
DECISION="$ROOT/user_space/projects/simple_pev/artifacts/fidelity_short_train.json"
WF="$ROOT/experiments/simple-pev/workflow.yaml"
# The 20-question check is the gate when the temperature probe never finished.
# Planner temperature then stays at the runtime default, 0.7.
if [[ ! -f "$PROBE" ]]; then
  if [[ ! -f "$MAS20" ]]; then
    echo "missing $PROBE and $MAS20" >&2
    exit 2
  fi
  export MAS20 DECISION
  set +e
  python - <<'PY'
import json, os
from pathlib import Path
report = json.loads(Path(os.environ["MAS20"]).read_text(encoding="utf-8"))
correct = int(report.get("correct") or 0)
decision = {
    "source": "mas20_accuracy",
    "correct": correct,
    "accuracy": report.get("accuracy"),
    "recommended_temperature": 0.7,
    "run_short_train": bool(report.get("run_short_train")) and correct > 1,
}
if not decision["run_short_train"]:
    decision["reason"] = "20-question accuracy is still near zero; 5-step runs stay off"
Path(os.environ["DECISION"]).write_text(json.dumps(decision, indent=2), encoding="utf-8")
raise SystemExit(0 if decision["run_short_train"] else 3)
PY
  status=$?
  set -e
  if [[ $status -eq 3 ]]; then
    echo "near-zero 20-question accuracy; short train skipped"
    exit 0
  fi
  if [[ $status -ne 0 ]]; then
    echo "mas20 decision failed with status $status" >&2
    exit "$status"
  fi
  temp=0.7
  export AGENTFLOW_TEMPERATURE="$temp"
else
export PROBE DECISION
set +e
python - <<'PY'
import json, os
from pathlib import Path
report = json.loads(Path(os.environ["PROBE"]).read_text(encoding="utf-8"))
total = 0
for block in (report.get("by_temperature") or {}).values():
    total += int(block.get("contrast_groups") or 0)
temp = float(report.get("recommended_temperature") or 0.7)
decision = {
    "contrast_groups": total,
    "recommended_temperature": temp,
    "run_short_train": total > 0,
}
if total <= 0:
    decision["reason"] = "no group contained both reward 0 and 1; 5-step runs stay off"
Path(os.environ["DECISION"]).write_text(json.dumps(decision, indent=2), encoding="utf-8")
raise SystemExit(0 if total > 0 else 3)
PY
status=$?
set -e
if [[ $status -eq 3 ]]; then
  echo "no 0/1 contrast; short train skipped"
  exit 0
fi
if [[ $status -ne 0 ]]; then
  echo "probe decision failed with status $status" >&2
  exit "$status"
fi
temp=$(python -c 'import json,os; print(json.load(open(os.environ["PROBE"]))["recommended_temperature"])')
export AGENTFLOW_TEMPERATURE="$temp"
fi
export AGENTFLOW_SFT_PROMPTS=1
# Train cap is train_max_steps. A leftover eval cap would keep six turns.
unset AGENTFLOW_MAX_STEPS || true
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env" || true
  set +a
fi
export AGENTFLOW_TEMPERATURE="$temp"
export AGENTFLOW_SFT_PROMPTS=1
unset AGENTFLOW_MAX_STEPS || true
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
fi
cd "$ROOT/mas"
export PYTHONPATH="$ROOT:$ROOT/mas:$ROOT/rl/verl_backend${PYTHONPATH:+:$PYTHONPATH}"
for algo in grpo flow_grpo arpo appo; do
  "$PYTHON" train_tir_agent.py a800_2gpu \
    --rl-yaml "$ROOT/experiments/simple-pev/fidelity/rl_${algo}.yaml" \
    --workflow-yaml "$WF"
done
