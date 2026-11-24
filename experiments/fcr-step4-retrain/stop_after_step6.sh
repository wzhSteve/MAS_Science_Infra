#!/bin/bash
# Watch r7 training log; when step 6 metrics (or global_step_6 ckpt) land,
# kill train_tir_agent so T2 live-val can run on the fresh checkpoint.
set -e
LOG=${1:-experiments/fcr-step4-retrain/r7_train.log}
CKPT=/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning/fcr_step4_retrain/global_step_6
cd /root/autodl-tmp/MAS_Science_Infra
echo "watching for step6 completion in $LOG"
while true; do
  if [ -f "$CKPT/actor/model_world_size_2_rank_0.pt" ]; then
    # Prefer a ckpt newer than this watcher start (r7 rewrite of step6).
    # Fallback: also accept "step:6 - training/reward" in the r7 log.
    if grep -qE "step:6 - training/reward" "$LOG" 2>/dev/null; then
      echo "step6 complete — stopping train_tir_agent"
      pkill -f train_tir_agent || true
      echo STOPPED_AFTER_STEP6
      exit 0
    fi
  fi
  if ! pgrep -f train_tir_agent >/dev/null 2>&1; then
    echo TRAIN_DIED_EARLY
    exit 1
  fi
  sleep 60
done
