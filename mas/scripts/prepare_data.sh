#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
cd "${ROOT}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

VAL_LIMIT="${VAL_LIMIT:-200}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
GSM_RATIO="${GSM_RATIO:-0.5}"

ARGS=(--output-dir data --val-limit "${VAL_LIMIT}" --gsm-ratio "${GSM_RATIO}")
if [[ -n "${TRAIN_LIMIT}" ]]; then
  ARGS+=(--train-limit "${TRAIN_LIMIT}")
fi
if [[ "${OFFLINE:-}" == "1" ]]; then
  ARGS+=(--offline)
fi
if [[ "${INCLUDE_NQ:-}" == "1" ]]; then
  ARGS+=(--include-nq)
fi

echo "Preparing mixed GSM8K + HotpotQA parquet under ${ROOT}/data ..."
echo "HF_ENDPOINT=${HF_ENDPOINT}"
"${PYTHON}" prepare_data.py "${ARGS[@]}"
