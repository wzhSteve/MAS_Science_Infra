#!/usr/bin/env bash
# 20h hard budget runner: A1/A5 offline → B1/B3 → C1 → (optional) C2
set -euo pipefail
ROOT=/root/autodl-tmp/MAS_Science_Infra
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rl/verl_backend:$ROOT/mas${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv/bin/python"
SPEC="$ROOT/experiments/fcr_short"
LOG="$SPEC/round20.log"
START=$(date +%s)
BUDGET_H=${BUDGET_H:-20}
DEADLINE=$((START + BUDGET_H * 3600))

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

mkdir -p "$SPEC/offline_pred"
: > "$LOG"
log "round20 begin budget=${BUDGET_H}h"

# --- Phase0 smoke ---
log "pytest smoke"
"$PY" -m pytest -q "$ROOT/mas/tests/test_fcr_grpo.py" \
  -k "bce_recovers or persistent_replay or budget_violation_counts or branch_repair or failure_or_uncertainty or bss_in_binary" \
  | tee -a "$LOG"

# --- H1 offline A1 / A5 ---
log "START offline A1"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/run_fcr_pred_offline.py" --name fcr_r20_A1 --mode a1 --steps 30 --seed 42 \
  | tee -a "$SPEC/fcr_r20_A1.log"
log "START offline A5"
CUDA_VISIBLE_DEVICES=0 "$PY" "$ROOT/scripts/run_fcr_pred_offline.py" --name fcr_r20_A5 --mode a5 --steps 30 --seed 42 \
  | tee -a "$SPEC/fcr_r20_A5.log"
log "offline A done elapsed=$(elapsed_h)h"

# Gate: if both collapsed (final mean_p≈0 and bss<=0), stop
"$PY" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
ok = True
for name in ("fcr_r20_A1", "fcr_r20_A5"):
    rows = [json.loads(l) for l in (Path("experiments/fcr_short/offline_pred")/f"{name}_metrics.jsonl").read_text().splitlines() if l.strip()]
    last = rows[-1]
    print(name, {k: last[k] for k in ("mean_p","bss","positive_recall","accuracy")})
# H1 pass if A5 mean_p_given_pos or bss better than A1, and A5 mean_p not ~0 forever
a1 = [json.loads(l) for l in Path("experiments/fcr_short/offline_pred/fcr_r20_A1_metrics.jsonl").read_text().splitlines() if l.strip()][-1]
a5 = [json.loads(l) for l in Path("experiments/fcr_short/offline_pred/fcr_r20_A5_metrics.jsonl").read_text().splitlines() if l.strip()][-1]
both_dead = a1["mean_p"] < 0.02 and a5["mean_p"] < 0.02 and a1["bss"] <= 0 and a5["bss"] <= 0
print("STOP_BEFORE_B" if both_dead else "CONTINUE_B")
PY

if grep -q STOP_BEFORE_B "$LOG"; then
  log "A gate failed; stop before online B/C"
  exit 0
fi

# --- H2 online ---
run_online fcr_r20_B1 "$SPEC/fcr_r20_B1.yaml" || true
run_online fcr_r20_B3 "$SPEC/fcr_r20_B3.yaml" || true

# --- H3 ---
run_online fcr_r20_C1 "$SPEC/fcr_r20_C1.yaml" || true
rem=$(remaining_h)
if ge "$rem" 3.5; then
  run_online fcr_r20_C2 "$SPEC/fcr_r20_C2.yaml" || true
else
  log "SKIP C2 rem=${rem}h"
fi

log "round20 complete elapsed=$(elapsed_h)h"
"$PY" "$ROOT/scripts/summarize_fcr_round20.py" || true
