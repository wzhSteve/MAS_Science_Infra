#!/usr/bin/env bash
# Recommended full training: 2x A800 + Qwen3-4B with full hyperparams.
# Prefer adding a second A800 over shrinking batch/sequence settings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
VENV_BIN="$(dirname "${PYTHON}")"
export PATH="${VENV_BIN}:${PATH}"
cd "${ROOT}"

if [[ ! -f data/train.parquet || ! -f data/val.parquet ]]; then
  echo "Parquet data missing; preparing GSM8K..."
  bash scripts/prepare_data.sh
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

GPU_COUNT="$("${PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "ERROR: a800_2gpu needs >=2 visible GPUs, found ${GPU_COUNT}."
  echo "Add a second A800 (or set CUDA_VISIBLE_DEVICES=0,1), then re-run."
  echo "Do not fall back to shrinking hyperparams unless you intentionally want scripts/train_1gpu.sh."
  exit 1
fi

echo "Cleaning leftover GPU / ray processes from previous runs..."
# Stop ray if available, then force-kill stray WorkerDict / train / vLLM holders.
if command -v ray >/dev/null 2>&1; then
  ray stop --force 2>/dev/null || true
fi
pkill -f 'ray::WorkerDict' 2>/dev/null || true
pkill -f 'ray::PatchedvLLMServer' 2>/dev/null || true
pkill -f 'train_math_agent.py' 2>/dev/null || true
sleep 2

if [[ -x "${ROOT}/../../scripts/restart_ray.sh" ]]; then
  bash "${ROOT}/../../scripts/restart_ray.sh" || true
fi

echo "Starting recommended 2-GPU A800 training (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})..."
echo "Note: gpu_memory_utilization=0.35 is for colocated FSDP+vLLM; train_batch_size stays 32."
"${PYTHON}" train_math_agent.py a800_2gpu --n-runners "${N_RUNNERS:-8}"
