#!/usr/bin/env bash
# Correction-first pilot: smoke, then BR-Strict / GRPO / B / Failure-only,
# then BR-Weak and Random only if Strict produced events and no budget violation.
set -euo pipefail
ROOT=/root/autodl-tmp/MAS_Science_Infra
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rl/verl_backend:$ROOT/mas${PYTHONPATH:+:$PYTHONPATH}"
unset PYTORCH_CUDA_ALLOC_CONF
PY="$ROOT/.venv/bin/python"
SPEC="$ROOT/experiments/fcr_short"
LOG="$SPEC/correction_pilot.log"
START=$(date +%s)

elapsed_h() { awk -v s="$START" -v n="$(date +%s)" 'BEGIN{printf "%.2f", (n-s)/3600}'; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_online() {
  local name="$1" yaml="$2"
  log "START $name"
  "$PY" "$ROOT/scripts/run_fcr_short_job.py" --rl-yaml "$yaml" --name "$name" --wait-gpu-idle \
    >> "$LOG" 2>&1 || { log "FAIL $name exit=$?"; return 1; }
  rm -rf "$ROOT/mas/checkpoints/AgentLightning/${name}/global_step_"* 2>/dev/null || true
  log "END $name elapsed=$(elapsed_h)h"
  return 0
}

{
  echo ""
  echo "===== correction pilot $(date '+%F %T') ====="
} >> "$LOG"

log "pytest"
"$PY" -m pytest -q "$ROOT/mas/tests/test_fcr_grpo.py" \
  -k "correct_pattern or correct_reward or strict_quad or failure_confirmed_does_not or random_gate or synthetic_batch or reward_density or strict_pattern" \
  | tee -a "$LOG"

run_online fcr_br_strict_smoke "$SPEC/fcr_br_strict_smoke.yaml" || { log "smoke failed"; exit 1; }

"$PY" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
p = Path("/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning/fcr_br_strict_smoke/metrics.jsonl")
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
assert rows, "smoke metrics empty"
row = rows[-1]
need = [
    "fcr/R_answer_mean", "fcr/R_correct_mean", "fcr/strict_event_count",
    "fcr/weak_event_count", "fcr/budget_violation_count", "fcr/mean_p",
    "fcr/branch_uplift", "training/reward",
]
missing = [k for k in need if k not in row]
print("smoke_steps", len(rows), "missing", missing)
print({k: row.get(k) for k in need})
assert not missing, missing
assert float(row.get("fcr/budget_violation_count") or 0) == 0
print("SMOKE_OK")
PY

run_online fcr_br_strict "$SPEC/fcr_br_strict.yaml" || true
run_online fcr_grpo_base "$SPEC/fcr_grpo_base.yaml" || true
run_online fcr_b_only "$SPEC/fcr_b_only.yaml" || true
run_online fcr_fail_only "$SPEC/fcr_fail_only.yaml" || true

gate=$("$PY" - <<'PY'
import json
from pathlib import Path
p = Path("/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning/fcr_br_strict/metrics.jsonl")
if not p.exists():
    print("STOP_EXTRA")
    raise SystemExit(0)
rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
if not rows:
    print("STOP_EXTRA")
    raise SystemExit(0)
strict = sum(float(r.get("fcr/strict_event_count") or 0) for r in rows)
budget = max(float(r.get("fcr/budget_violation_count") or 0) for r in rows)
print(f"strict_sum={strict} budget_max={budget}")
print("RUN_EXTRA" if strict > 0 and budget == 0 else "STOP_EXTRA")
PY
)
echo "$gate" | tee -a "$LOG"
if echo "$gate" | grep -q RUN_EXTRA; then
  run_online fcr_br_weak "$SPEC/fcr_br_weak.yaml" || true
  run_online fcr_random "$SPEC/fcr_random.yaml" || true
else
  log "skip weak/random: $gate"
fi

log "correction pilot complete elapsed=$(elapsed_h)h"
"$PY" "$ROOT/scripts/summarize_fcr_correction.py" || true
