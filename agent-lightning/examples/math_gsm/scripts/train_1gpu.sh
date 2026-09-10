#!/usr/bin/env bash
# 1x A800 fallback only. Prefer scripts/train_2gpu.sh with a second A800
# instead of further reducing hyperparameters.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
cd "${ROOT}"

if [[ ! -f data/train.parquet || ! -f data/val.parquet ]]; then
  echo "Parquet data missing; preparing GSM8K..."
  bash scripts/prepare_data.sh
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
if [[ -x "${ROOT}/../../scripts/restart_ray.sh" ]]; then
  bash "${ROOT}/../../scripts/restart_ray.sh" || true
fi

echo "WARNING: 1-GPU fallback. For full hyperparams, add a second A800 and use scripts/train_2gpu.sh."
echo "Starting 1-GPU A800 training..."
"${PYTHON}" train_math_agent.py a800 --n-runners "${N_RUNNERS:-4}"
