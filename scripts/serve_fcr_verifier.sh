#!/usr/bin/env bash
# Serve a DEDICATED OpenAI-compatible vLLM for the frozen FCR roles
# (verifier / executor / tool text calls), decoupled from the rollout engine.
#
# Why: with frozen roles pinned at llm.yaml's http://127.0.0.1:8000/v1, verifier
# prefill queues behind actor generations. In run 9a45bbd456ad that produced 15+
# "Request timed out" errors in steps 5-7, crashed fcr/label_coverage
# 0.999 -> 0.370, and starved the Correct reward of the (t, t+1) labels it needs.
#
# Usage (before launching training):
#   ./scripts/serve_fcr_verifier.sh
#   FCR_VERIFIER_DEVICE=cpu ./scripts/serve_fcr_verifier.sh
# GPU default is TP=2 on both visible cards, util 0.18, sleep mode on.
# Training sleeps this server at level 2 before the FSDP update.
# Then export for the training process (rl.yaml `env:` also carries these):
#   export AGENTFLOW_FIXED_BASE_URL=http://127.0.0.1:8001/v1
#   export AGENTFLOW_FIXED_MODEL=Qwen3-4B
#   export FCR_FROZEN_VLLM_URL=http://127.0.0.1:8001
set -euo pipefail

MODEL="${FCR_VERIFIER_MODEL:-/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B}"
PORT="${FCR_VERIFIER_PORT:-8001}"
NAME="${FCR_VERIFIER_NAME:-Qwen3-4B}"
# Keep >= training budget (rl.yaml max_prompt+max_response = 10240 today): a
# smaller window 400s the clamped verifier request and drops its label.
MAX_LEN="${FCR_VERIFIER_MAX_LEN:-10240}"
# TP=2 puts about 4GB of Qwen3-4B weights on each A800. 0.18 is ~14GB/card,
# leaving the planner rollout (util 0.45) room on the same cards.
# max_num_seqs matches val --n-runners 32. Train uses 16 runners, so 32 slots cover both.
GPU_UTIL="${FCR_VERIFIER_GPU_UTIL:-0.18}"
TP="${FCR_VERIFIER_TP:-2}"
MAX_NUM_SEQS="${FCR_VERIFIER_MAX_NUM_SEQS:-32}"
EAGER_FLAG=()
if [ "${FCR_VERIFIER_ENFORCE_EAGER:-1}" != "0" ]; then
  EAGER_FLAG=(--enforce-eager)
fi

if [ ! -f "$MODEL/config.json" ]; then
  echo "error: model weights not found at $MODEL" >&2
  exit 1
fi

# Resolve the vLLM CLI: explicit override, repo venv, then PATH.
VLLM_BIN="${FCR_VERIFIER_VLLM:-}"
if [ -z "$VLLM_BIN" ]; then
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  if [ -x "$REPO_DIR/.venv/bin/vllm" ]; then
    VLLM_BIN="$REPO_DIR/.venv/bin/vllm"
  else
    VLLM_BIN="vllm"
  fi
fi

# A crashed engine stops listening but keeps its CUDA context. Starting a
# second one on the same cards faults the GPU and leaves utilization stuck
# at 100% with 0 MiB reported. Reap that leftover, and do not stack a second
# engine when this port is already healthy.
if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "verifier already serving on :${PORT}; not starting another engine"
  exit 0
fi
while read -r pid cmd; do
  case "$cmd" in
    *"vllm serve"*"--port ${PORT} "*|*"vllm serve"*"--port ${PORT}"|*"vllm serve"*"--port=${PORT}"*)
      echo "reaping stale vllm pid ${pid} (port ${PORT} is not serving)"
      kill -TERM "$pid" 2>/dev/null || true
      ;;
  esac
done < <(ps -eo pid=,args=)
sleep 1
while read -r pid cmd; do
  case "$cmd" in
    *"vllm serve"*"--port ${PORT} "*|*"vllm serve"*"--port ${PORT}"|*"vllm serve"*"--port=${PORT}"*)
      kill -KILL "$pid" 2>/dev/null || true
      ;;
  esac
done < <(ps -eo pid=,args=)

DEVICE="${FCR_VERIFIER_DEVICE:-gpu}"
if [ "$DEVICE" = "cpu" ]; then
  # Planner FSDP and vLLM TP use both GPUs. CPU keeps the frozen verifier off them.
  echo "serving dedicated verifier on :$PORT (model=$NAME, device=cpu, max_len=$MAX_LEN)"
  unset CUDA_VISIBLE_DEVICES
  exec "$VLLM_BIN" serve "$MODEL" \
    --served-model-name "$NAME" \
    --port "$PORT" \
    --host 127.0.0.1 \
    --trust-remote-code \
    --device cpu \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs 1
fi

if [ "$TP" -lt 2 ]; then
  echo "error: FCR_VERIFIER_TP must be >= 2 so both GPUs serve the frozen engine" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
# vLLM 0.9.2 mounts /sleep, /wake_up, and /is_sleeping only in dev mode.
export VLLM_SERVER_DEV_MODE=1
echo "serving dedicated verifier on :$PORT (model=$NAME, tp=$TP, cuda=$CUDA_VISIBLE_DEVICES, gpu_util=$GPU_UTIL, max_num_seqs=$MAX_NUM_SEQS, max_len=$MAX_LEN, sleep_mode=on)"
exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$NAME" \
  --port "$PORT" \
  --host 127.0.0.1 \
  --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --enable-sleep-mode \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_UTIL" \
  "${EAGER_FLAG[@]}"
