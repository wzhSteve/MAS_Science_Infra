#!/bin/bash
# Val-only for one checkpoint. STEP=5|11|12|13
# Does not write into the training checkpoint directory.
set -e
cd /root/autodl-tmp/MAS_Science_Infra
STEP="${1:?step}"
CKPT="/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning/fcr_step4_retrain/global_step_${STEP}"
if [ ! -f "$CKPT/actor/model_world_size_2_rank_0.pt" ]; then
  echo "ERROR: missing $CKPT"
  exit 1
fi
if ! curl -s --max-time 3 http://127.0.0.1:8001/v1/models >/dev/null 2>&1; then
  echo "ERROR: verifier :8001 not reachable"
  exit 1
fi
python3 - <<PY
from pathlib import Path
p = Path("experiments/fcr-step4-retrain/rl_valonly.yaml")
t = p.read_text()
import re
t = re.sub(r"resume_from_path: .*/global_step_\\d+",
           "resume_from_path: ${CKPT}", t)
p.write_text(t)
print("resume", "${CKPT}")
PY
set -a
# shellcheck disable=SC2046
eval "$(grep '^export ' experiments/fcr-step4-retrain/launch_r7.sh)"
set +a
# launch_r7 exports this then unsets it. Expandable segments break vLLM's pool.
unset PYTORCH_CUDA_ALLOC_CONF
export AGENTFLOW_FROZEN_MAX_TOKENS="${AGENTFLOW_FROZEN_MAX_TOKENS:-1024}"
export TIR_RUN_ID="fcr-step4-valonly-${STEP}"
export TIR_EXPERIMENT_ID="fcr-step4-valonly-${STEP}"
export CUDA_VISIBLE_DEVICES="0,1"
export PYTHONUNBUFFERED="1"
export VLLM_USE_V1="1"
export PYTHONPATH="/root/autodl-tmp/MAS_Science_Infra:/root/autodl-tmp/MAS_Science_Infra/rl/verl_backend:/root/autodl-tmp/MAS_Science_Infra/mas"
exec .venv/bin/python mas/train_tir_agent.py a800_2gpu --algo fcr_grpo \
  --rl-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr-step4-retrain/rl_valonly.yaml \
  --workflow-yaml /root/autodl-tmp/MAS_Science_Infra/experiments/fcr_short/workflow.yaml \
  --n-runners 32 --active-agents planner
