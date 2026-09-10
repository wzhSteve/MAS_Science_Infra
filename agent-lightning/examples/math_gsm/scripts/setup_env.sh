#!/usr/bin/env bash
# Install / repair AgentFlow .venv for agent-lightning math_gsm training.
set -euo pipefail

VENV_BIN="${VENV_BIN:-/root/autodl-tmp/AgentFlow/.venv/bin}"
AGL_ROOT="${AGL_ROOT:-/root/autodl-tmp/agent-lightning}"
PYTHON="${VENV_BIN}/python"
UV_BIN="$(command -v uv || true)"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python not found at ${PYTHON}"
  exit 1
fi

if [[ -z "${UV_BIN}" ]]; then
  echo "ERROR: uv not found on PATH. Install uv or set PATH."
  exit 1
fi

uv_pip() {
  "${UV_BIN}" pip install --python "${PYTHON}" "$@"
}

echo "==> Using Python: ${PYTHON}"
"${PYTHON}" -c 'import sys; print(sys.version)'
echo "==> Using uv: ${UV_BIN}"

echo "==> Upgrading vllm / installing verl - keep torch 2.7 for AgentFlow compatibility"
uv_pip --upgrade "vllm==0.9.2" "verl==0.5.0"

echo "==> Installing agentlightning editable from ${AGL_ROOT}"
uv_pip -e "${AGL_ROOT}"

echo "==> Installing LangChain and data deps for math_gsm"
uv_pip \
  "langgraph<1.0" \
  "langchain[openai]<1.0" \
  "langchain-community" \
  "langchain-text-splitters<1.0" \
  "datasets" \
  "pandas" \
  "numpy" \
  "pyarrow" \
  "termcolor"

# Compatibility pins after transitive upgrades from vllm/agentlightning
echo "==> Pinning fastapi and transformers for litellm/vllm compatibility"
uv_pip "fastapi>=0.115,<0.116" "transformers==4.53.3"

echo "==> Smoke imports"
"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
import vllm
print("vllm", vllm.__version__)
import verl
print("verl OK", getattr(verl, "__version__", "?"))
import agentlightning as agl
print("agentlightning", getattr(agl, "__version__", "?"))
print("SETUP_OK")
PY

echo "==> setup_env.sh finished successfully"
echo ""
echo "Optional official stack if the above still conflicts:"
echo "  uv pip install --python ${PYTHON} torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128"
echo "  uv pip install --python ${PYTHON} vllm==0.10.2 verl==0.5.0"
