#!/bin/bash

# Ensure script exits on error
set -e

# Project root = parent of env_init/ (works from any cwd: bash setup.sh / bash env_init/setup.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"
echo "Project root: ${ROOT}"

# Use domestic PyPI mirror (AutoDL / China)
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://mirrors.aliyun.com/pypi/simple}"
export UV_INDEX_URL="${UV_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple}"
export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
echo "Using PyPI mirror: $UV_DEFAULT_INDEX"

# Install UV (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "UV not installed, installing..."
    pip install uv -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST"
fi

# Create and activate UV virtual environment at project root
if [ ! -d ".venv" ]
then
    echo "Creating UV virtual environment..."
    uv venv -p 3.11
fi

echo "Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

# Install env_init requirements + editable agent-lightning (replaces old agentflow/ layout)
echo "Installing env_init requirements..."
uv pip install -r "${SCRIPT_DIR}/requirements.txt"
echo "Installing agent-lightning (editable, no-deps)..."
uv pip install --no-deps -e ./agent-lightning

# Install project dependencies (development mode)
echo "Installing project dependencies (development mode)..."
uv pip install -e .

# uv pip install omegaconf
# uv pip install codetiming
# uv pip install pyvers multiprocess
uv pip install dashscope
uv pip install fire

# Install additional dependency packages

echo "Installing AutoGen..."
uv pip install "autogen-agentchat" "autogen-ext[openai]"

echo "Installing LiteLLM..."
uv pip install "litellm[proxy]"

echo "Installing MCP..."
uv pip install mcp

echo "Installing OpenAI Agents..."
uv pip install openai-agents

echo "Installing LangChain related packages..."
uv pip install langgraph "langchain[openai]" langchain-community langchain-text-splitters

echo "Installing SQL related dependencies..."
uv pip install sqlparse nltk

bash "${SCRIPT_DIR}/setup_stable_gpu.sh"

rm -rf verl

# System deps: jq (YAML tooling) + Node.js 20 (webui Vite build)
echo "Installing system packages (jq)..."
if command -v apt-get >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && [[ "$(id -u)" -ne 0 ]]; then
    sudo apt-get update
    sudo apt-get install -y jq
  else
    apt-get update
    apt-get install -y jq
  fi
fi
uv pip install yq

echo "Installing Node.js / building webui..."
bash "${SCRIPT_DIR}/install_nodejs.sh"
bash "${SCRIPT_DIR}/build_webui.sh"

# Restart Ray service (defaults CUDA_VISIBLE_DEVICES=0 if unset)
echo "Restarting Ray service..."
bash "${SCRIPT_DIR}/restart_ray.sh"

echo "Ray server is reflushed."
echo "Env init done. UI: ./run.sh ui   | train needs Ray + GPU via env_init/restart_ray.sh"
