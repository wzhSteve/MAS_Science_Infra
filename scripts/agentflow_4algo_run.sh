#!/usr/bin/env bash
# Sequential 4-algorithm 10-step AgentFlow RL training on 2x A800.
# Each algo runs in foreground of this script (true sequential).
# Logs -> /root/autodl-tmp/MAS_Science_Infra/logs/agentflow_4algo_10step/
set -uo pipefail
cd /root/autodl-tmp/MAS_Science_Infra
source .venv/bin/activate
# ``mas/train_tir_agent.py`` imports ``from workflow.*`` and ``from rl.*`` (no
# ``mas.`` prefix), so both the repo root (for rl/, tir_agent) and mas/ (for
# workflow/, tool_agents, mas.*) must be on PYTHONPATH.
# NOTE: do NOT prepend ref_Rep/agentflow to PYTHONPATH — that repo ships
# top-level ``types.py`` / ``logging.py`` which shadow stdlib and break
# the venv editable install (.pth bootstrap). The AgentFlow native tools
# are loaded lazily inside context.py via a temporary sys.path append.
export PYTHONPATH=/root/autodl-tmp/MAS_Science_Infra:/root/autodl-tmp/MAS_Science_Infra/mas
export TIR_MAX_MODEL_LEN=8192
# Force both A800s. Shell defaults like CUDA_VISIBLE_DEVICES=0 would otherwise
# shrink trainer.n_gpus_per_node to 1 inside train_tir_agent.py.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# If the ambient env only exposed one GPU, still prefer both for this run.
if [[ "${CUDA_VISIBLE_DEVICES}" != *","* ]]; then
  export CUDA_VISIBLE_DEVICES=0,1
fi
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here —
# VERL's vllm rollout worker asserts it is unset, since vLLM's CuMemAllocator
# manages its own memory pool and is incompatible with expandable_segments.
export MAS_BASE_URL=${MAS_BASE_URL:-http://127.0.0.1:8000/v1}
export MAS_MODEL_NAME=${MAS_MODEL_NAME:-Qwen3-4B}

LOG_DIR=/root/autodl-tmp/MAS_Science_Infra/logs/agentflow_4algo_10step
mkdir -p "$LOG_DIR"

# vLLM is spawned by VERL rollout workers; no external mock needed.
echo "[$(date '+%F %T')] starting 4-algo run (vLLM will be spawned by VERL)"

fail=0
run_algo() {
  local ALGO=$1
  local CFG=experiments/agentflow_4algo/rl_${ALGO}.yaml
  local LOG=$LOG_DIR/${ALGO}.log
  echo "================================================================="
  echo "[$(date '+%F %T')] >>> $ALGO start"
  echo "================================================================="
  python -m mas.train_tir_agent a800_2gpu \
    --rl-yaml "$CFG" > "$LOG" 2>&1
  local RC=$?
  echo "[$(date '+%F %T')] <<< $ALGO exit=$RC"
  if grep -q "cannot reshape tensor of 0 elements" "$LOG"; then
    echo "[$(date '+%F %T')] FAIL $ALGO: reshape crash"
    RC=1
  fi
  if grep -qiE "CUDA out of memory|OutOfMemoryError" "$LOG"; then
    echo "[$(date '+%F %T')] FAIL $ALGO: OOM"
    RC=1
  fi
  if [[ $RC -ne 0 ]]; then
    fail=1
  fi
  ray stop --force >/dev/null 2>&1 || true
  sleep 8
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  return $RC
}

run_algo grpo || true
run_algo arpo || true
run_algo appo || true
run_algo flow_grpo || true

echo "================================================================="
if [[ $fail -eq 0 ]]; then
  echo "[$(date '+%F %T')] ALL 4 ALGORITHMS DONE (PASS)"
else
  echo "[$(date '+%F %T')] ALL 4 ALGORITHMS DONE (HAD FAILURES)"
fi
echo "================================================================="
ls -la /root/autodl-tmp/MAS_Science_Infra/checkpoints/AgentLightning/ 2>/dev/null | tail -20
exit $fail
