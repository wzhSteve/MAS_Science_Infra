#!/bin/bash

set -ex

# Prefer project venv ray if available
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ -f "${ROOT}/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv/bin/activate"
fi

# Empty CUDA_VISIBLE_DEVICES hides all GPUs from Ray (resources show 0 GPU).
# Default to GPU 0 on single-card AutoDL boxes; override when calling this script.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ray stop -v --force --grace-period 60
ps aux
env RAY_DEBUG=legacy HYDRA_FULL_ERROR=1 VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" ray start --head --dashboard-host=0.0.0.0
