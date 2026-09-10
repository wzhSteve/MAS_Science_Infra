#!/usr/bin/env bash
# 1-step smoke training for math_gsm.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
cd "${ROOT}"

if [[ ! -f data/train.parquet || ! -f data/val.parquet ]]; then
  echo "Parquet data missing; running prepare_data.sh (small subset)..."
  TRAIN_LIMIT=64 VAL_LIMIT=16 bash scripts/prepare_data.sh
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Required for verl 0.5 + vllm 0.9 async server
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True — incompatible with vLLM CuMemAllocator.
# Optional: restart ray if leftover sessions cause issues.
if [[ -x "${ROOT}/../../scripts/restart_ray.sh" ]]; then
  bash "${ROOT}/../../scripts/restart_ray.sh" || true
fi

echo "Running fast (1-step) smoke training..."
"${PYTHON}" train_math_agent.py fast --n-runners 2
