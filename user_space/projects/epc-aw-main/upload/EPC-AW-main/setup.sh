#!/bin/bash

# Ensure script exits on error
set -e

# Switch to project root directory
# cd YOUR_ROOT_PATH

# Install UV (if not already installed)
if ! command -v uv &> /dev/null
then
    echo "UV not installed, installing..."
    pip install uv
fi

# Create and activate UV virtual environment
if [ ! -d ".venv" ]
then
    echo "Creating UV virtual environment..."
    uv venv -p 3.11
fi

 echo "Activating virtual environment..."
source .venv/bin/activate

cd MAS
uv pip install -r requirements.txt
uv pip install --no-deps -e .
cd ..

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

# bash scripts/setup_stable_gpu.sh

rm -rf verl

# Restart Ray service
echo "Restarting Ray service..."
# bash scripts/restart_ray.sh

# echo "Ray server is reflushed. "

# sudo apt-get update
# sudo apt-get install -y jq
uv pip install yq
