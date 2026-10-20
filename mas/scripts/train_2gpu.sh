#!/usr/bin/env bash
# Recommended full training: 2x A800 + Qwen3-4B, vanilla GRPO.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
VENV_BIN="$(dirname "${PYTHON}")"
export PATH="${VENV_BIN}:${PATH}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
# Prefer repo-root .env (SCIENCE_INFRA_PROXY); default matches SSH RemoteForward 17890.
REPO_ROOT="$(cd "${ROOT}/.." && pwd)"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env" || true
  set +a
fi
_PROXY_DEFAULT="${SCIENCE_INFRA_PROXY:-${WEB_SEARCH_PROXY:-http://127.0.0.1:17890}}"
export SCIENCE_INFRA_PROXY="${SCIENCE_INFRA_PROXY:-${_PROXY_DEFAULT}}"
export WEB_SEARCH_PROXY="${WEB_SEARCH_PROXY:-${SCIENCE_INFRA_PROXY}}"
export GOOGLE_CHROME_PROXY="${GOOGLE_CHROME_PROXY:-${SCIENCE_INFRA_PROXY}}"
unset TIR_OFFLINE_SEARCH || true
ALGO="${ALGO:-grpo}"

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
pkill -f 'train_tir_agent.py' 2>/dev/null || true
sleep 2

if [[ -x "${ROOT}/../../scripts/restart_ray.sh" ]]; then
  bash "${ROOT}/../../scripts/restart_ray.sh" || true
fi

echo "Starting 2-GPU A800 training (algo=${ALGO}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, WEB_SEARCH_PROXY=${WEB_SEARCH_PROXY})..."
"${PYTHON}" train_tir_agent.py a800_2gpu --algo "${ALGO}" --n-runners "${N_RUNNERS:-8}"
