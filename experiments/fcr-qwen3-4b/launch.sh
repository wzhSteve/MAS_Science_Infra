#!/bin/bash
# Planner uses both GPUs: rollout TP=2, FSDP world size 2.
# Frozen executor/verifier is a second TP=2 vLLM on the same cards (util 0.18).
# Fresh run from LLM/Qwen3-4B. resume_mode=disable, no checkpoint load.
# cgroup is 240GB. Step 13 logged perf/cpu_memory_used_gb=227 with 8 runners.
# Each runner is about 1GB, so 16 runners is the memory bound. 24 fills the
# cgroup and the kernel SIGKILLs the job.
set -e
cd /root/autodl-tmp/MAS_Science_Infra

# API keys live in .env (git-ignored); never hard-code them in this script.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

if ! curl -s --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  echo "ERROR: verifier :8001 not reachable. Start it first:"
  echo "  FCR_VERIFIER_TP=2 nohup ./scripts/serve_fcr_verifier.sh > experiments/fcr-qwen3-4b/verifier_8001.log 2>&1 &"
  exit 1
fi
python3 - <<'PY'
import subprocess
gpus = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
)
uuid_to_idx = {}
for line in gpus.strip().splitlines():
    idx, uuid = [part.strip() for part in line.split(",", 1)]
    uuid_to_idx[uuid] = idx
apps = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
    text=True,
)
def cmdline(pid: str) -> str:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ")
    except OSError:
        return ""
    return raw.decode("utf-8", "replace")

def parent(pid: str) -> str:
    try:
        status = open(f"/proc/{pid}/status", encoding="utf-8").read()
    except OSError:
        return ""
    for row in status.splitlines():
        if row.startswith("PPid:"):
            return row.split()[1]
    return ""

seen = set()
for line in apps.strip().splitlines():
    if not line.strip():
        continue
    uuid, pid = [part.strip() for part in line.split(",", 1)]
    probe = pid
    matched = False
    for _ in range(8):
        if "vllm" in cmdline(probe).lower():
            matched = True
            break
        probe = parent(probe)
        if not probe or probe in {"0", "1"}:
            break
    if matched:
        seen.add(uuid_to_idx.get(uuid))
missing = [idx for idx in ("0", "1") if idx not in seen]
if missing:
    raise SystemExit(
        "ERROR: frozen vLLM TP=2 is not resident on GPU "
        + ",".join(missing)
        + ". Start it with: FCR_VERIFIER_TP=2 ./scripts/serve_fcr_verifier.sh"
    )
PY
echo "preflight OK (verifier :8001 on GPU 0 and 1, train from Qwen3-4B)"

export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY in .env or shell}"
export OPENAI_API_BASE="https://www.dmxapi.cn/v1/"
export OPENAI_API_BASE_URL="https://www.dmxapi.cn/v1/"
export MODEL_Name="qwen3.5-27b"
export SERVER_MODEL="qwen3.5-27b"
export OPENAI_MODEL="qwen3.5-27b"
export AI_ASSISTANT_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY in .env or shell}"
export AI_ASSISTANT_API_BASE="https://www.dmxapi.cn/v1/"
export AI_ASSISTANT_MODEL="qwen3.5-27b"
export EVALUATION_MODEL_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY in .env or shell}"
export EVALUATION_MODEL_API_BASE="https://www.dmxapi.cn/v1/"
export EVALUATION_MODEL="gpt-4o"
export TIR_MAS_HEALTH="0"
export SCIENCE_INFRA_PROXY="http://127.0.0.1:17890"
export WEB_SEARCH_PROXY="http://127.0.0.1:17890"
export TIR_SEARCH_DESC_CHARS="2000"
export BRIGHT_DATA_API_KEY="${BRIGHT_DATA_API_KEY:?set BRIGHT_DATA_API_KEY}"
export BRIGHT_DATA_SERP_ZONES="serp_api1"
export BRIGHT_DATA_SERP_ZONE="serp_api1"
export BRIGHT_DATA_API_KEY_2="${BRIGHT_DATA_API_KEY_2:?set BRIGHT_DATA_API_KEY_2}"
export BRIGHT_DATA_SERP_ZONES_2="serp_api1"
export enable_thinking="false"
export CUDA_VISIBLE_DEVICES="0,1"
export FCR_FROZEN_VLLM_URL="http://127.0.0.1:8001"
export PYTHONUNBUFFERED="1"
export VLLM_USE_V1="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FCR_GRAD_DIAGNOSTICS="0"
export FCR_BRIER_MAX_ROWS="0"
export AGENTFLOW_LLM_TIMEOUT="300"
export BRIGHT_DATA_TIMEOUT="45"
export BRIGHT_DATA_MAX_RETRIES="2"
export AGENTFLOW_ECHO_CONCURRENCY="8"
export MAS_SEARCH_CACHE=""
export PYTHONPATH="/root/autodl-tmp/MAS_Science_Infra:/root/autodl-tmp/MAS_Science_Infra/rl/verl_backend:/root/autodl-tmp/MAS_Science_Infra/mas"
export TIR_RUN_ID="fcr-qwen3-4b"
export TIR_EXPERIMENT_ID="fcr-qwen3-4b"
export TIR_ROLLOUT_TREE_DIR="/root/autodl-tmp/MAS_Science_Infra/experiments/fcr-qwen3-4b/artifacts/rollout-trees"
export NO_PROXY="localhost,127.0.0.1,::1"
export RAY_memory_usage_threshold="0.96"
unset MODEL_NAME OPENAI_MODEL MODEL TIR_OFFLINE_SEARCH PYTORCH_CUDA_ALLOC_CONF HTTP_PROXY HTTPS_PROXY
exec /root/autodl-tmp/MAS_Science_Infra/.venv/bin/python mas/train_tir_agent.py a800_2gpu --algo fcr_grpo --rl-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr-qwen3-4b/rl.yaml --workflow-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr_short/workflow.yaml --n-runners 16 --active-agents planner
