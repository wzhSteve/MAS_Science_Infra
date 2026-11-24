#!/bin/bash
# r6 launch: replay steps 6-10 from the best checkpoint global_step_5
# (val 0.3304). The r5 chain (resume step7) crashed in step8's optimizer with
# GPU0 OOM (verifier ballooned to 42.9G at util 0.30); verifier is now capped
# at util 0.36 + max_num_seqs 24 (verified stable <=30G).
#
# r6 changes vs launch_r5.sh:
#   r7: tool-layer search cache REMOVED — all tool calls hit the network live.
#   + resume back to global_step_5 (rl.yaml)
#   + tir.judge_fails observability (see rl/rewards/llm_judge.py) so the
#     step6 val collapse (0.3304 -> 0.1843) can be attributed: judge drift
#     vs real regression.
# Kept from r5c: 24 runners, verifier GPU0@0.36, echo 8, search fast-fail
# (LLM_TIMEOUT=60, BRIGHT_DATA_TIMEOUT=15, RETRIES=1), test_freq=2.
set -e
cd /root/autodl-tmp/MAS_Science_Infra

# --- 0. Preflight: verifier up on :8001 and step5 checkpoint present ---
if ! curl -s --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  echo "ERROR: verifier :8001 not reachable. Start it first:"
  echo "  nohup ./scripts/serve_fcr_verifier.sh > experiments/fcr-step4-retrain/verifier_8001.log 2>&1 &"
  exit 1
fi
CKPT=/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning/fcr_step4_retrain/global_step_5
if [ ! -f "$CKPT/actor/model_world_size_2_rank_0.pt" ]; then
  echo "ERROR: resume checkpoint incomplete: $CKPT"
  exit 1
fi
echo "preflight OK (verifier :8001, ckpt step5)"

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
# r5c search fast-fail (env-level mirror of rl.yaml env:; rl.yaml wins on merge)
export AGENTFLOW_LLM_TIMEOUT="60"
export BRIGHT_DATA_TIMEOUT="15"
export BRIGHT_DATA_MAX_RETRIES="1"
# r5c echo concurrency for the :8001 echo prefill path
export AGENTFLOW_ECHO_CONCURRENCY="8"
# r7: cache removed; live search everywhere.
export MAS_SEARCH_CACHE=""
export PYTHONPATH="/root/autodl-tmp/MAS_Science_Infra:/root/autodl-tmp/MAS_Science_Infra/rl/verl_backend:/root/autodl-tmp/MAS_Science_Infra/mas"
export TIR_RUN_ID="fcr-step4-retrain"
export TIR_EXPERIMENT_ID="fcr-step4-retrain"
export TIR_ROLLOUT_TREE_DIR="/root/autodl-tmp/MAS_Science_Infra/experiments/fcr-step4-retrain/artifacts/rollout-trees"
export NO_PROXY="localhost,127.0.0.1,::1"
unset MODEL_NAME OPENAI_MODEL MODEL TIR_OFFLINE_SEARCH PYTORCH_CUDA_ALLOC_CONF HTTP_PROXY HTTPS_PROXY
# val-only check: same env as r6, different rl.yaml (val_only) + log
exec /root/autodl-tmp/MAS_Science_Infra/.venv/bin/python mas/train_tir_agent.py a800_2gpu --algo fcr_grpo --rl-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr-step4-retrain/rl_valcheck.yaml --workflow-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr_short/workflow.yaml --n-runners 24 --active-agents planner
