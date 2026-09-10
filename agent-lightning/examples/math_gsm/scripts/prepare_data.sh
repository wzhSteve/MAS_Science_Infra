#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
cd "${ROOT}"

# AutoDL / CN networks: use HF mirror unless caller overrides.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

VAL_LIMIT="${VAL_LIMIT:-200}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"

ARGS=(--output-dir data --val-limit "${VAL_LIMIT}")
if [[ -n "${TRAIN_LIMIT}" ]]; then
  ARGS+=(--train-limit "${TRAIN_LIMIT}")
fi

echo "Preparing GSM8K parquet under ${ROOT}/data ..."
echo "HF_ENDPOINT=${HF_ENDPOINT}"
"${PYTHON}" prepare_data.py "${ARGS[@]}"
