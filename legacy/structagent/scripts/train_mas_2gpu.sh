#!/usr/bin/env bash
# Recommended full MAS_structagent training: 2x A800 + Qwen3-4B, GRPO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
VENV_BIN="$(dirname "${PYTHON}")"
export PATH="${VENV_BIN}:${PATH}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export MAS_OPENAI_COMPATIBLE="${MAS_OPENAI_COMPATIBLE:-1}"
export WEB_SEARCH_PROXY="${WEB_SEARCH_PROXY:-http://127.0.0.1:7890}"
export GOOGLE_CHROME_PROXY="${GOOGLE_CHROME_PROXY:-${WEB_SEARCH_PROXY}}"
unset TIR_OFFLINE_SEARCH || true

need_data=0
if [[ ! -f data/train.parquet || ! -f data/val.parquet ]]; then
  need_data=1
else
  N_TRAIN="$("${PYTHON}" -c "import pandas as pd; print(len(pd.read_parquet('data/train.parquet')))")"
  if [[ "${N_TRAIN}" -lt 200 ]]; then
    echo "train.parquet has ${N_TRAIN} rows (<200); rebuilding mixed GSM8K+HotpotQA..."
    need_data=1
  fi
fi
if [[ "${need_data}" -eq 1 ]]; then
  TRAIN_LIMIT="${TRAIN_LIMIT:-1024}" VAL_LIMIT="${VAL_LIMIT:-200}" bash scripts/prepare_data.sh
fi

GPU_COUNT="$("${PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${GPU_COUNT}" -lt 2 ]]; then
  echo "ERROR: a800_2gpu needs >=2 visible GPUs, found ${GPU_COUNT}."
  exit 1
fi

echo "Cleaning leftover GPU / ray processes from previous runs..."
if command -v ray >/dev/null 2>&1; then
  ray stop --force 2>/dev/null || true
fi
pkill -f 'ray::WorkerDict' 2>/dev/null || true
pkill -f 'ray::PatchedvLLMServer' 2>/dev/null || true
pkill -f 'train_mas_agent.py' 2>/dev/null || true
pkill -f 'train_tir_agent.py' 2>/dev/null || true
sleep 2

if [[ -x "${ROOT}/../../scripts/restart_ray.sh" ]]; then
  bash "${ROOT}/../../scripts/restart_ray.sh" || true
fi

echo "Starting 2-GPU MAS training (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})..."
"${PYTHON}" train_mas_agent.py a800_2gpu --n-runners "${N_RUNNERS:-8}" --max-steps "${MAX_STEPS:-8}"
