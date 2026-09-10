#!/usr/bin/env bash
# 1-step smoke training for tir_agent (offline search).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
cd "${ROOT}"

if [[ ! -f data/train.parquet || ! -f data/val.parquet ]]; then
  echo "Parquet data missing; preparing a small mixed subset..."
  TRAIN_LIMIT=32 VAL_LIMIT=8 bash scripts/prepare_data.sh
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export TIR_OFFLINE_SEARCH="${TIR_OFFLINE_SEARCH:-1}"
ALGO="${ALGO:-grpo}"

if [[ -x "${ROOT}/../../scripts/restart_ray.sh" ]]; then
  bash "${ROOT}/../../scripts/restart_ray.sh" || true
fi

echo "Running fast (1-step) smoke training algo=${ALGO} ..."
"${PYTHON}" train_tir_agent.py fast --algo "${ALGO}" --n-runners "${N_RUNNERS:-2}"
