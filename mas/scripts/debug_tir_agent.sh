#!/usr/bin/env bash
# Debug LitTirAgent against an OpenAI-compatible endpoint.
# If OPENAI_API_BASE is unset, start a local vLLM server for Qwen3-4B on 1 GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/root/autodl-tmp/AgentFlow/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/LLM/Qwen3-4B}"
PORT="${PORT:-8000}"
cd "${ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WEB_SEARCH_PROXY="${WEB_SEARCH_PROXY:-http://127.0.0.1:7890}"
export GOOGLE_CHROME_PROXY="${GOOGLE_CHROME_PROXY:-${WEB_SEARCH_PROXY}}"
unset TIR_OFFLINE_SEARCH || true

STARTED_VLLM=0
VLLM_PID=""

cleanup() {
  if [[ "${STARTED_VLLM}" -eq 1 && -n "${VLLM_PID}" ]]; then
    echo "Stopping temporary vLLM (pid=${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if [[ -z "${OPENAI_API_BASE:-}" && -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "OPENAI_API_BASE not set; starting local vLLM on port ${PORT} (GPU ${CUDA_VISIBLE_DEVICES}) ..."
  "${PYTHON}" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --dtype auto \
    --gpu-memory-utilization 0.7 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > /tmp/tir_agent_vllm_debug.log 2>&1 &
  VLLM_PID=$!
  STARTED_VLLM=1

  echo "Waiting for vLLM to become ready..."
  for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "vLLM is ready."
      break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
      echo "vLLM exited early. Log:"
      tail -n 80 /tmp/tir_agent_vllm_debug.log || true
      exit 1
    fi
    sleep 2
    if [[ "${i}" -eq 120 ]]; then
      echo "Timed out waiting for vLLM. Log:"
      tail -n 80 /tmp/tir_agent_vllm_debug.log || true
      exit 1
    fi
  done

  export OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1"
  export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
  export OPENAI_MODEL="${OPENAI_MODEL:-${MODEL_PATH}}"
fi

export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_MODEL="${OPENAI_MODEL:-${MODEL_PATH}}"

echo "Debugging TIR agent with OPENAI_API_BASE=${OPENAI_API_BASE:-${OPENAI_BASE_URL}}"
"${PYTHON}" tir_agent.py
