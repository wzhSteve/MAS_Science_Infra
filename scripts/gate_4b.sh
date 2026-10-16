#!/usr/bin/env bash
# One-command Qwen3-4B gate on 2×A800.
#   GPU1: frozen/serving vLLM (Mode A + AgentFlow fixed engines)   GPU0: training
#
#   bash scripts/gate_4b.sh            # everything
#   STAGES="mode_a flow" bash scripts/gate_4b.sh
#
# Stages: mode_a (AgentFlow / EPC cheap / EPC paper n=2, no training)
#         flow   (AgentFlow Flow-GRPO 3 steps: planner-only, 0/1 reward)
#         af_grpo (same windows, plain GRPO)   epc (EPC cheap GRPO 3 steps)
#         tir    (Tir hub × 8 presets × 3 steps)
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH=$ROOT:$ROOT/mas MAS_SEARCH_CACHE=${MAS_SEARCH_CACHE:-1}

MODEL=${MODEL:-$ROOT/LLM/Qwen3-4B}
SERVED=${SERVED:-Qwen3-4B}
PORT=${PORT:-8001}
BASE_URL=http://127.0.0.1:$PORT/v1
STAGES=${STAGES:-"mode_a flow af_grpo epc tir"}
TIR_ALGOS=${TIR_ALGOS:-"grpo arpo appo aepo igpo gigpo rae flow_grpo"}
OUT=${OUT:-$ROOT/artifacts/gate_4b}
LOG=$ROOT/logs/gate_4b
mkdir -p "$OUT" "$LOG"
MASTER=$OUT/gate_master.log
fail=0

say() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }
has() { [[ " $STAGES " == *" $1 "* ]]; }

ensure_server() {
  curl -sf "$BASE_URL/models" >/dev/null && return 0
  say "starting $SERVED on GPU1:$PORT"
  CUDA_VISIBLE_DEVICES=1 setsid nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" --host 0.0.0.0 \
    --gpu-memory-utilization 0.45 --max-model-len 16384 --trust-remote-code \
    > "$OUT/vllm_serve.log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do curl -sf "$BASE_URL/models" >/dev/null && return 0; sleep 5; done
  say "FAIL vLLM server did not come up"; exit 1
}

mode_a() {  # name workflow [extra args...]
  local name=$1 wf=$2; shift 2
  if python scripts/gate_mode_a.py --workflow "$wf" --base-url "$BASE_URL" --model "$SERVED" \
      --out "$OUT/$name.json" "$@" > "$LOG/$name.log" 2>&1; then
    say "OK   mode_a $name $(grep '^REPORT' "$LOG/$name.log" | cut -c8-260)"
  else
    say "FAIL mode_a $name (see $LOG/$name.log)"; fail=1
  fi
}

train() {  # name rl_yaml max_steps_cap
  local name=$1 cfg=$2 cap=${3:-0} f=$LOG/$1.log
  ray stop --force >/dev/null 2>&1; sleep 5
  say ">>> train $name ($cfg)"
  CUDA_VISIBLE_DEVICES=0 python -m mas.train_tir_agent a800 --rl-yaml "$cfg" > "$f" 2>&1
  local rc=$? ok=1
  [[ $rc -eq 0 ]] || { say "FAIL $name exit=$rc"; ok=0; }
  grep -qE "cannot reshape tensor of 0 elements" "$f" && { say "FAIL $name: 0-length reshape"; ok=0; }
  grep -qiE "CUDA out of memory|OutOfMemoryError" "$f" && { say "FAIL $name: OOM"; ok=0; }
  grep -qE "step:3 .*training/reward" "$f" || { say "FAIL $name: step 3 metrics missing"; ok=0; }
  grep -qE "actor/pg_loss" "$f" || { say "FAIL $name: no actor/pg_loss"; ok=0; }
  if [[ $cap -gt 0 ]]; then
    python - "$f" "$cap" <<'PY' || ok=0
import re, sys
text = open(sys.argv[1], errors="ignore").read()
cap = int(sys.argv[2])
bad = []
for m in re.finditer(r"step:(\d+) .*?training/n_rollouts:(\d+).*?training/n_triplets:(\d+)", text):
    step, n_roll, n_trip = map(int, m.groups())
    if n_trip > n_roll * cap:
        bad.append(f"step {step}: {n_trip} triplets > {n_roll}x{cap} planner turns")
rewards = {float(r) for r in re.findall(r"\[Rollout [^\]]+\][\s\S]{0,2500}?R=(-?[0-9.]+)", text)}
if not rewards:
    bad.append("no rollout reward lines")
elif not rewards <= {0.0, 1.0}:
    bad.append(f"non-binary rewards {sorted(rewards)}")
for line in bad:
    print("  ", line)
sys.exit(1 if bad else 0)
PY
    [[ $ok -eq 1 ]] || say "FAIL $name: planner-only / binary reward check"
  fi
  if [[ $ok -eq 1 ]]; then
    say "OK   $name $(grep -oE 'step:3 .*training/reward:[-0-9.e]+' "$f" | grep -oE 'training/reward:[-0-9.e]+' | tail -1)"
  else
    fail=1
  fi
}

ensure_server
if has mode_a; then
  mode_a af_mode_a experiments/agentflow_flow_grpo_4b/workflow.yaml --n 3
  SCIENCE_EPC_AW_PROFILE=cheap mode_a epc_cheap_mode_a experiments/epc_aw_4b/cheap/workflow.yaml --n 3
  SCIENCE_EPC_AW_PROFILE=paper SCIENCE_EPC_AW_N=2 \
    mode_a epc_paper_mode_a experiments/epc_aw_4b/paper/workflow.yaml --n 2 --require-bts 2
fi

export AGENTFLOW_MAX_STEPS=3 AGENTFLOW_FIXED_BASE_URL=$BASE_URL AGENTFLOW_FIXED_MODEL=$SERVED
if has flow; then
  train af_flow_grpo experiments/agentflow_flow_grpo_4b/rl.yaml 3
  grep -q "tir_algo=flow_grpo" "$LOG/af_flow_grpo.log" || { say "FAIL flow: tir_algo"; fail=1; }
  grep -q "reward outcome=binary" "$LOG/af_flow_grpo.log" || { say "FAIL flow: reward outcome"; fail=1; }
  grep -q "agent_match='^(?:agent__planner)\$'" "$LOG/af_flow_grpo.log" || { say "FAIL flow: agent_match"; fail=1; }
fi
has af_grpo && train af_grpo experiments/agentflow_flow_grpo_4b/rl_grpo.yaml
if has epc; then
  SCIENCE_EPC_AW_PROFILE=cheap train epc_cheap_grpo experiments/epc_aw_4b/cheap/rl.yaml
fi
if has tir; then
  for algo in $TIR_ALGOS; do
    train "tir_$algo" "experiments/tir_hub_4b/rl_$algo.yaml"
  done
fi
ray stop --force >/dev/null 2>&1

if [[ $fail -eq 0 ]]; then say "GATE-4B ALL PASS"; else say "GATE-4B HAD FAILURES"; fi
exit $fail
