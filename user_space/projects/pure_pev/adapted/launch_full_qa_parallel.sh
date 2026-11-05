#!/usr/bin/env bash
# Launch HotpotQA + Bamboogle pure_pev full evals in parallel.
# Both processes use .env LLM (OPENAI_* / MODEL_Name) and Bright Data KEY_2.
set -euo pipefail

ROOT="/root/autodl-tmp/MAS_Science_Infra"
PROJECT="$ROOT/user_space/projects/pure_pev"
PY="$ROOT/.venv/bin/python"
SCRIPT="$PROJECT/adapted/eval_full_qa.py"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
OUT="$PROJECT/artifacts/full_qa_${STAMP}"
mkdir -p "$OUT"

# Load .env into this shell so child processes inherit the pinned values.
set -a
# shellcheck disable=SC1091
source <(grep -E '^(OPENAI_API_KEY|OPENAI_API_BASE|OPENAI_API_BASE_URL|MODEL_Name|OPENAI_MODEL|SERVER_MODEL|BRIGHT_DATA_API_KEY_2|BRIGHT_DATA_SERP_ZONES_2)=' "$ROOT/.env" | sed 's/\r$//')
set +a

export OPENAI_API_BASE="${OPENAI_API_BASE:-https://www.dmxapi.cn/v1/}"
export OPENAI_API_BASE_URL="${OPENAI_API_BASE_URL:-$OPENAI_API_BASE}"
export MODEL_Name="${MODEL_Name:-qwen3.5-27b}"
export OPENAI_MODEL="${OPENAI_MODEL:-$MODEL_Name}"
export SERVER_MODEL="${SERVER_MODEL:-$MODEL_Name}"
# Force both workers onto KEY_2 (do not use primary BRIGHT_DATA_API_KEY).
export BRIGHT_DATA_API_KEY="${BRIGHT_DATA_API_KEY_2:?BRIGHT_DATA_API_KEY_2 missing}"
export BRIGHT_DATA_SERP_ZONES="${BRIGHT_DATA_SERP_ZONES_2:-serp_api1}"
export BRIGHT_DATA_SERP_ZONE="${BRIGHT_DATA_SERP_ZONES%%,*}"
export AGENTFLOW_SFT_PROMPTS=1
export AGENTFLOW_TOP_LOGPROBS=5
export AGENTFLOW_MAX_STEPS="${AGENTFLOW_MAX_STEPS:-10}"
export AGENTFLOW_LLM_TIMEOUT="${AGENTFLOW_LLM_TIMEOUT:-180}"
export enable_thinking=false

{
  echo "OUT=$OUT"
  echo "MODEL_Name=$MODEL_Name"
  echo "OPENAI_API_BASE_URL=$OPENAI_API_BASE_URL"
  echo "BRIGHT_DATA_API_KEY=${BRIGHT_DATA_API_KEY:0:8}..."
  echo "BRIGHT_DATA_SERP_ZONES=$BRIGHT_DATA_SERP_ZONES"
} | tee "$OUT/launch_env.txt"

echo "starting hotpotqa (KEY_2) ..."
nohup "$PY" "$SCRIPT" \
  --dataset hotpotqa \
  --serp-account secondary \
  --out "$OUT" \
  >"$OUT/hotpotqa.log" 2>&1 &
echo $! >"$OUT/hotpotqa.pid"

echo "starting bamboogle (KEY_2) ..."
nohup "$PY" "$SCRIPT" \
  --dataset bamboogle \
  --serp-account secondary \
  --out "$OUT" \
  >"$OUT/bamboogle.log" 2>&1 &
echo $! >"$OUT/bamboogle.pid"

echo "pids: hotpotqa=$(cat "$OUT/hotpotqa.pid") bamboogle=$(cat "$OUT/bamboogle.pid")"
echo "logs: $OUT/hotpotqa.log $OUT/bamboogle.log"
echo "$OUT" >"$PROJECT/artifacts/full_qa_latest.path"
