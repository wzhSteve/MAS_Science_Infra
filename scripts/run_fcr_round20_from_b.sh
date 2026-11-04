#!/usr/bin/env bash
# Resume round20 from H2 (A already done). Fresh ≤20h wall clock.
# Memory harden: FCR_GRAD_DIAGNOSTICS=0, vLLM util 0.28, MAX_ROWS=1 on pred arms.
set -euo pipefail
ROOT=/root/autodl-tmp/MAS_Science_Infra
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rl/verl_backend:$ROOT/mas${PYTHONPATH:+:$PYTHONPATH}"
unset PYTORCH_CUDA_ALLOC_CONF
PY="$ROOT/.venv/bin/python"
SPEC="$ROOT/experiments/fcr_short"
LOG="$SPEC/round20_resume.log"
START=$(date +%s)
BUDGET_H=${BUDGET_H:-20}
DEADLINE=$((START + BUDGET_H * 3600))
SKIP_B1=${SKIP_B1:-0}

elapsed_h() { awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.2f", (n-s)/3600}'; }
remaining_h() { awk -v d="$DEADLINE" -v n="$(date +%s)" 'BEGIN{printf "%.2f", (d-n)/3600}'; }
lt() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 < b+0)}'; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0 >= b+0)}'; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_online() {
  local name="$1" yaml="$2"
  local rem
  rem=$(remaining_h)
  if lt "$rem" 3.2; then
    log "SKIP $name remaining=${rem}h < 3.2h"
    return 1
  fi
  log "START $name rem=${rem}h"
  "$PY" "$ROOT/scripts/run_fcr_short_job.py" --rl-yaml "$yaml" --name "$name" --wait-gpu-idle \
    >> "$LOG" 2>&1 || { log "FAIL $name exit=$?"; return 1; }
  rm -rf "$ROOT/mas/checkpoints/AgentLightning/${name}/global_step_"* 2>/dev/null || true
  log "END $name elapsed=$(elapsed_h)h"
  return 0
}

{
  echo ""
  echo "===== resume restart $(date '+%F %T') budget=${BUDGET_H}h SKIP_B1=${SKIP_B1} ====="
} >> "$LOG"
log "round20 resume-from-B begin budget=${BUDGET_H}h harden=util0.28,MAX_ROWS1,grad_diag0"

if [[ ! -f "$SPEC/offline_pred/fcr_r20_A5_metrics.jsonl" ]]; then
  log "missing A5 metrics; refuse resume"
  exit 1
fi

if [[ "$SKIP_B1" != "1" ]]; then
  run_online fcr_r20_B1 "$SPEC/fcr_r20_B1.yaml" || true
else
  log "SKIP_B1=1; keep prior B1 metrics"
fi
run_online fcr_r20_B3 "$SPEC/fcr_r20_B3.yaml" || true

run_online fcr_r20_C1 "$SPEC/fcr_r20_C1.yaml" || true
rem=$(remaining_h)
if ge "$rem" 3.5; then
  run_online fcr_r20_C2 "$SPEC/fcr_r20_C2.yaml" || true
else
  log "SKIP C2 rem=${rem}h"
fi

log "round20 resume complete elapsed=$(elapsed_h)h"
"$PY" "$ROOT/scripts/summarize_fcr_round20.py" || true
