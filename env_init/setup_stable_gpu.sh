#!/bin/bash
set -ex

# Expect: cwd = project root, .venv already activated (called from env_init/setup.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# python -m uv pip install --upgrade uv pip

uv pip install --no-cache-dir packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
uv pip install --no-cache-dir torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --no-cache-dir transformers==4.53.3
uv pip install --no-cache-dir flash-attn==2.8.1 --no-build-isolation
uv pip install --no-cache-dir vllm==0.9.2
uv pip install --no-cache-dir verl==0.5.0
# VERL trainer.logger often includes tensorboard
uv pip install --no-cache-dir tensorboard==2.21.0

# science-infra extras: live,dev (not AgentFlow's [dev,agent])
uv pip install --no-cache-dir -e ".[live,dev]"
uv pip install --no-cache-dir -e "./agent-lightning"
