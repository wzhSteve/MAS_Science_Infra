#!/usr/bin/env bash
# Launch two pure_pev full evals in parallel.
# Both processes use .env LLM (OPENAI_* / MODEL_Name) and Bright Data KEY_2.
#
# Usage:
#   ./launch_full_qa_parallel.sh [stamp] [dataset_a] [dataset_b]
# Defaults: stamp=now, dataset_a=hotpotqa, dataset_b=bamboogle
set -euo pipefail

ROOT="/root/autodl-tmp/MAS_Science_Infra"
PROJECT="$ROOT/user_space/projects/pure_pev"
PY="$ROOT/.venv/bin/python"
SCRIPT="$PROJECT/adapted/eval_full_qa.py"
STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
DS_A="${2:-hotpotqa}"
DS_B="${3:-bamboogle}"
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
  echo "datasets=$DS_A,$DS_B"
  echo "MODEL_Name=$MODEL_Name"
  echo "OPENAI_API_BASE_URL=$OPENAI_API_BASE_URL"
  echo "BRIGHT_DATA_API_KEY=${BRIGHT_DATA_API_KEY:0:8}..."
  echo "BRIGHT_DATA_SERP_ZONES=$BRIGHT_DATA_SERP_ZONES"
} | tee "$OUT/launch_env.txt"

echo "starting $DS_A (KEY_2) ..."
nohup "$PY" "$SCRIPT" \
  --dataset "$DS_A" \
  --serp-account secondary \
  --out "$OUT" \
  >"$OUT/${DS_A}.log" 2>&1 &
echo $! >"$OUT/${DS_A}.pid"

echo "starting $DS_B (KEY_2) ..."
nohup "$PY" "$SCRIPT" \
  --dataset "$DS_B" \
  --serp-account secondary \
  --out "$OUT" \
  >"$OUT/${DS_B}.log" 2>&1 &
echo $! >"$OUT/${DS_B}.pid"

echo "pids: ${DS_A}=$(cat "$OUT/${DS_A}.pid") ${DS_B}=$(cat "$OUT/${DS_B}.pid")"
echo "logs: $OUT/${DS_A}.log $OUT/${DS_B}.log"
echo "$OUT" >"$PROJECT/artifacts/full_qa_latest.path"
printf '%s\n%s\n' "$DS_A" "$DS_B" >"$OUT/datasets.txt"
