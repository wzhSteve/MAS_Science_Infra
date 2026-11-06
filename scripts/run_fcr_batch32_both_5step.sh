#!/usr/bin/env bash
# Run fcr_grpo_corr (resume to 5) then fcr_branch_corr (fresh 5) at batch=32, n_runners=16.
set -euo pipefail
ROOT=/root/autodl-tmp/MAS_Science_Infra
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/rl/verl_backend:$ROOT/mas${PYTHONPATH:+:$PYTHONPATH}"
unset PYTORCH_CUDA_ALLOC_CONF
PY="$ROOT/.venv/bin/python"
SPEC="$ROOT/experiments/fcr_short"
LOG="$SPEC/batch32_both_5step.log"
mkdir -p "$SPEC/events"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log_ckpts() {
  local name="$1"
  local dir="$ROOT/mas/checkpoints/AgentLightning/${name}"
  log "CKPTS $name latest=$(cat "$dir/latest_checkpointed_iteration.txt" 2>/dev/null || echo missing)"
  if [[ -f "$dir/best_val_checkpoint.json" ]]; then
    log "CKPTS $name best=$(cat "$dir/best_val_checkpoint.json")"
  else
    log "CKPTS $name best_val_checkpoint.json=missing"
  fi
  local steps
  steps=$(ls -d "$dir"/global_step_* 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ' || true)
  log "CKPTS $name kept_steps=${steps:-none}"
}

backup_metrics() {
  local name="$1"
  local dir="$ROOT/mas/checkpoints/AgentLightning/${name}"
  if [[ -s "$dir/metrics.jsonl" ]]; then
    cp -a "$dir/metrics.jsonl" "$dir/metrics.jsonl.pre_launch_bak"
  fi
}

merge_metrics_bak() {
  local name="$1"
  local dir="$ROOT/mas/checkpoints/AgentLightning/${name}"
  local bak="$dir/metrics.jsonl.pre_launch_bak"
  local cur="$dir/metrics.jsonl"
  [[ -f "$bak" && -f "$cur" ]] || return 0
  "$PY" - <<PY
import json
from pathlib import Path
bak = Path("$bak")
cur = Path("$cur")
def rows(p):
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
br, cr = rows(bak), rows(cur)
def step(r):
    return int(r.get("training/global_step") or r.get("step") or 0)
have = {step(r) for r in cr}
merged = list(cr)
for r in br:
    if step(r) not in have:
        merged.append(r)
        have.add(step(r))
merged.sort(key=step)
cur.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
print("merged metrics steps", sorted(have))
PY
}

run_one() {
  local name="$1"
  backup_metrics "$name"
  # do not wipe events on resume arm; wipe on fresh branch
  if [[ "$name" == "fcr_branch_corr" ]]; then
    rm -f "$SPEC/events/${name}.jsonl" "$SPEC/events/${name}_val.jsonl"
  fi
  log "START $name"
  "$PY" "$ROOT/scripts/run_fcr_short_job.py" \
    --rl-yaml "$SPEC/${name}.yaml" --name "$name" --wait-gpu-idle \
    >>"$LOG" 2>&1 || { log "FAIL $name exit=$?"; merge_metrics_bak "$name"; return 1; }
  merge_metrics_bak "$name"
  log_ckpts "$name"
  log "END $name"
}

{
  echo ""
  echo "===== batch32 both 5step $(date '+%F %T') ====="
  echo "ETA_HINT: grpo ~4 remaining steps * 3.2h + branch 5 * 3.2h ≈ 29h"
} >>"$LOG"

log "pipeline begin"
run_one fcr_grpo_corr || true
run_one fcr_branch_corr || true
log "pipeline complete"
"$PY" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
root = Path("/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning")
for name in ("fcr_grpo_corr", "fcr_branch_corr"):
    p = root / name / "metrics.jsonl"
    print(f"\n## {name}")
    if not p.exists() or not p.stat().st_size:
        print("no metrics")
        continue
    for line in p.read_text().splitlines():
        m = json.loads(line)
        s = m.get("training/global_step") or m.get("step")
        print(
            f"step{s}: train={m.get('training/reward')} val={m.get('val/reward')} "
            f"infra={m.get('fcr/infra_drop_rate')} n={m.get('training/n_rollouts')} "
            f"Ra={m.get('val/fcr/R_answer_mean')} Rc={m.get('val/fcr/R_correct_mean')}"
        )
PY
log "batch32 both 5step complete"
