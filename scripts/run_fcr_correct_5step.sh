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

log_ckpts() {
  local name="$1"
  local dir="$ROOT/mas/checkpoints/AgentLightning/${name}"
  log "CKPTS $name dir=$dir"
  if [[ -f "$dir/best_val.json" ]]; then
    log "CKPTS $name best_val.json=$(cat "$dir/best_val.json")"
  else
    log "CKPTS $name best_val.json=missing"
  fi
  if [[ -f "$dir/latest_checkpointed_iteration.txt" ]]; then
    log "CKPTS $name latest=$(cat "$dir/latest_checkpointed_iteration.txt")"
  fi
  local steps
  steps=$(ls -d "$dir"/global_step_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ' || true)
  log "CKPTS $name kept_steps=${steps:-none}"
}

run_online() {
  local name="$1"
  rm -f "$SPEC/events/${name}.jsonl" "$SPEC/events/${name}_val.jsonl"
  log "START $name"
  "$PY" "$ROOT/scripts/run_fcr_short_job.py" \
    --rl-yaml "$SPEC/${name}.yaml" --name "$name" --wait-gpu-idle \
    >> "$LOG" 2>&1 || { log "FAIL $name exit=$?"; return 1; }
  # Keep best-val + last global_step_* (do not delete checkpoints).
  log_ckpts "$name"
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
