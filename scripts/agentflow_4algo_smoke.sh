#!/usr/bin/env bash
# Gate B: 1-step smoke for GRPO / ARPO / APPO / Flow-GRPO on 2x A800.
set -uo pipefail
cd /root/autodl-tmp/MAS_Science_Infra
source .venv/bin/activate
export PYTHONPATH=/root/autodl-tmp/MAS_Science_Infra:/root/autodl-tmp/MAS_Science_Infra/mas
export TIR_MAX_MODEL_LEN=8192
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ "${CUDA_VISIBLE_DEVICES}" != *","* ]]; then
  export CUDA_VISIBLE_DEVICES=0,1
fi
export MAS_BASE_URL=${MAS_BASE_URL:-http://127.0.0.1:8000/v1}
export MAS_MODEL_NAME=${MAS_MODEL_NAME:-Qwen3-4B}

LOG_DIR=/root/autodl-tmp/MAS_Science_Infra/logs/agentflow_4algo_smoke
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/_smoke_master.log"
: > "$MASTER"

cleanup_between() {
  ray stop --force >/dev/null 2>&1 || true
  sleep 8
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$MASTER"
}

fail=0
for ALGO in grpo arpo appo flow_grpo; do
  CFG=experiments/agentflow_4algo/smoke/rl_${ALGO}.yaml
  LOG=$LOG_DIR/${ALGO}.log
  echo "=================================================================" | tee -a "$MASTER"
  echo "[$(date '+%F %T')] >>> smoke $ALGO start" | tee -a "$MASTER"
  echo "=================================================================" | tee -a "$MASTER"
  python -m mas.train_tir_agent a800_2gpu --rl-yaml "$CFG" > "$LOG" 2>&1
  RC=$?
  echo "[$(date '+%F %T')] <<< smoke $ALGO exit=$RC" | tee -a "$MASTER"

  if grep -q "cannot reshape tensor of 0 elements" "$LOG"; then
    echo "[$(date '+%F %T')] FAIL $ALGO: reshape crash" | tee -a "$MASTER"
    fail=1
  fi
  if grep -q "must be divisible by minimal possible batch size" "$LOG"; then
    echo "[$(date '+%F %T')] FAIL $ALGO: batch not divisible" | tee -a "$MASTER"
    fail=1
  fi
  if grep -qiE "CUDA out of memory|OutOfMemoryError" "$LOG"; then
    echo "[$(date '+%F %T')] FAIL $ALGO: OOM" | tee -a "$MASTER"
    fail=1
  fi
  if ! grep -q "AgentFlow llm_model=" "$LOG"; then
    echo "[$(date '+%F %T')] FAIL $ALGO: no AgentFlow rollout" | tee -a "$MASTER"
    fail=1
  else
    echo "[$(date '+%F %T')] OK $ALGO: AgentFlow rollout seen" | tee -a "$MASTER"
  fi
  if grep -q "No non-empty triplets" "$LOG"; then
    echo "[$(date '+%F %T')] WARN $ALGO: empty triplets skipped" | tee -a "$MASTER"
  fi
  if [[ $RC -ne 0 ]]; then
    fail=1
  fi

  cleanup_between
done

echo "=================================================================" | tee -a "$MASTER"
if [[ $fail -eq 0 ]]; then
  echo "[$(date '+%F %T')] SMOKE ALL PASS" | tee -a "$MASTER"
else
  echo "[$(date '+%F %T')] SMOKE HAD FAILURES" | tee -a "$MASTER"
fi
exit $fail
