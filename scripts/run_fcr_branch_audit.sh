#!/usr/bin/env bash
# Frozen-policy event audit, then matched-budget branch comparison. Wall clock <= 20h.
set -euo pipefail
ROOT=/root/autodl-tmp/MAS_Science_Infra
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rl/verl_backend:$ROOT/mas${PYTHONPATH:+:$PYTHONPATH}"
unset PYTORCH_CUDA_ALLOC_CONF
PY="$ROOT/.venv/bin/python"
SPEC="$ROOT/experiments/fcr_short"
LOG="$SPEC/branch_audit.log"
START=$(date +%s)
DEADLINE=$((START + 20 * 3600))

elapsed_h() { awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.2f", (n-s)/3600}'; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_online() {
  local name="$1"
  rm -f "$SPEC/events/${name}.jsonl"
  mkdir -p "$SPEC/events"
  log "START $name elapsed=$(elapsed_h)h"
  "$PY" "$ROOT/scripts/run_fcr_short_job.py" \
    --rl-yaml "$SPEC/${name}.yaml" --name "$name" --wait-gpu-idle \
    >> "$LOG" 2>&1 || { log "FAIL $name exit=$?"; return 1; }
  rm -rf "$ROOT/mas/checkpoints/AgentLightning/${name}/global_step_"* 2>/dev/null || true
  log "END $name elapsed=$(elapsed_h)h"
  return 0
}

{
  echo ""
  echo "===== branch audit $(date '+%F %T') ====="
} >> "$LOG"

run_online fcr_audit || { log "audit job failed"; exit 1; }
"$PY" "$ROOT/scripts/audit_fcr_events.py" \
  --events "$SPEC/events/fcr_audit.jsonl" \
  --out "$SPEC/EVENT_AUDIT.md" \
  --name fcr_audit | tee -a "$LOG"

run_online fcr_eval_nobranch || true
run_online fcr_eval_fail || true
run_online fcr_eval_confirmed || true

elapsed=$(elapsed_h)
awk -v e="$elapsed" 'BEGIN{exit !(e+0 >= 18)}' && {
  log "SKIP fcr_eval_random elapsed=${elapsed}h >= 18h"
} || run_online fcr_eval_random || true

"$PY" "$ROOT/scripts/summarize_branch_paired.py" | tee -a "$LOG"
log "branch audit complete elapsed=$(elapsed_h)h"
