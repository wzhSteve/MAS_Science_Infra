#!/bin/bash
# r5 relaunch: steps 7-10 on top of the complete global_step_6 checkpoint
# (val 0.1843, identity gap 6.7e-16 PASS). r4 died at the step-7 wake_up with
# CUDA OOM on GPU1 (verifier 20.8 GiB co-resident with training rank).
#
# r5 changes vs launch_r4.sh:
#   ① n_runners 8 → 24  (rl.yaml; Bing stays serialized by the file-lock gate)
#   ② verifier on GPU0 @ util 0.30, echo concurrency 4 → 8 (start it first!)
#   ③ test_freq 1 → 2   (val only on steps 8 and 10, saves ~45 min/odd step)
#   + search hardening: AGENTFLOW_LLM_TIMEOUT 90 → 60,
#     BRIGHT_DATA_TIMEOUT 45 → 15, BRIGHT_DATA_MAX_RETRIES 2 → 1
#     (step6: 326/382 infra faults were fetch/Bing failures; fail fast and
#      drop the episode instead of stalling the runner for minutes.)
set -e
cd /root/autodl-tmp/MAS_Science_Infra

# --- 0. Preflight: verifier must be up on :8001 before training starts ---
if ! curl -s --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  echo "ERROR: verifier :8001 not reachable. Start it first:"
  echo "  nohup ./scripts/serve_fcr_verifier.sh > experiments/fcr-step4-retrain/verifier_8001.log 2>&1 &"
  exit 1
fi
echo "verifier :8001 OK"

export OPENAI_API_KEY="sk-UcCcBysddZ5poOiALoCRjI1h4peMBr4lnDtVuRUD4OLMRnsE"
export OPENAI_API_BASE="https://www.dmxapi.cn/v1/"
export OPENAI_API_BASE_URL="https://www.dmxapi.cn/v1/"
export MODEL_Name="qwen3.5-27b"
export SERVER_MODEL="qwen3.5-27b"
export OPENAI_MODEL="qwen3.5-27b"
export AI_ASSISTANT_API_KEY="sk-UcCcBysddZ5poOiALoCRjI1h4peMBr4lnDtVuRUD4OLMRnsE"
export AI_ASSISTANT_API_BASE="https://www.dmxapi.cn/v1/"
export AI_ASSISTANT_MODEL="qwen3.5-27b"
export EVALUATION_MODEL_API_KEY="sk-UcCcBysddZ5poOiALoCRjI1h4peMBr4lnDtVuRUD4OLMRnsE"
export EVALUATION_MODEL_API_BASE="https://www.dmxapi.cn/v1/"
export EVALUATION_MODEL="gpt-4o"
export TIR_MAS_HEALTH="0"
export SCIENCE_INFRA_PROXY="http://127.0.0.1:17890"
export WEB_SEARCH_PROXY="http://127.0.0.1:17890"
export TIR_SEARCH_DESC_CHARS="2000"
export BRIGHT_DATA_API_KEY="f4f5ffba-a865-48d1-b0a1-9d08246ce8b6"
export BRIGHT_DATA_SERP_ZONES="serp_api1"
export BRIGHT_DATA_SERP_ZONE="serp_api1"
export BRIGHT_DATA_API_KEY_2="b9731637-55ac-4b34-b06a-992f3ed32361"
export BRIGHT_DATA_SERP_ZONES_2="serp_api1"
export enable_thinking="false"
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONUNBUFFERED="1"
export VLLM_USE_V1="1"
export FCR_GRAD_DIAGNOSTICS="0"
export FCR_BRIER_MAX_ROWS="0"
# r5 search hardening (env-level mirror of rl.yaml env: block; rl.yaml wins on merge)
export AGENTFLOW_LLM_TIMEOUT="60"
export BRIGHT_DATA_TIMEOUT="15"
export BRIGHT_DATA_MAX_RETRIES="1"
# r5 echo concurrency for the :8001 echo prefill path
export AGENTFLOW_ECHO_CONCURRENCY="8"
export PYTHONPATH="/root/autodl-tmp/MAS_Science_Infra:/root/autodl-tmp/MAS_Science_Infra/rl/verl_backend:/root/autodl-tmp/MAS_Science_Infra/mas"
export TIR_RUN_ID="fcr-step4-retrain"
export TIR_EXPERIMENT_ID="fcr-step4-retrain"
export TIR_ROLLOUT_TREE_DIR="/root/autodl-tmp/MAS_Science_Infra/experiments/fcr-step4-retrain/artifacts/rollout-trees"
export NO_PROXY="localhost,127.0.0.1,::1"
unset MODEL_NAME OPENAI_MODEL MODEL TIR_OFFLINE_SEARCH PYTORCH_CUDA_ALLOC_CONF HTTP_PROXY HTTPS_PROXY
# ① n_runners 24 (matches rl.yaml n_runners; CLI wins over rl.yaml)
exec /root/autodl-tmp/MAS_Science_Infra/.venv/bin/python mas/train_tir_agent.py a800_2gpu --algo fcr_grpo --rl-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr-step4-retrain/rl.yaml --workflow-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr_short/workflow.yaml --n-runners 24 --active-agents planner
