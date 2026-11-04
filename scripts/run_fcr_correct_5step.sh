#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/MAS_Science_Infra
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rl/verl_backend:$ROOT/mas${PYTHONPATH:+:$PYTHONPATH}"
unset PYTORCH_CUDA_ALLOC_CONF
PY="$ROOT/.venv/bin/python"
SPEC="$ROOT/experiments/fcr_short"
LOG="$SPEC/correct_5step.log"
mkdir -p "$SPEC/events"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_online() {
  local name="$1"
  rm -f "$SPEC/events/${name}.jsonl" "$SPEC/events/${name}_val.jsonl"
  log "START $name"
  "$PY" "$ROOT/scripts/run_fcr_short_job.py" \
    --rl-yaml "$SPEC/${name}.yaml" --name "$name" --wait-gpu-idle \
    >> "$LOG" 2>&1 || { log "FAIL $name exit=$?"; return 1; }
  rm -rf "$ROOT/mas/checkpoints/AgentLightning/${name}/global_step_"* 2>/dev/null || true
  log "END $name"
}

{
  echo ""
  echo "===== correct 5step $(date '+%F %T') ====="
} >> "$LOG"

run_online fcr_grpo_corr || true
run_online fcr_branch_corr || true
"$PY" "$ROOT/scripts/summarize_correct_5step.py" | tee -a "$LOG"
log "correct 5step complete"
