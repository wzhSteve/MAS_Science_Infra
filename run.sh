#!/usr/bin/env bash
# Agent_Science_Infra 功能测试入口
#
# 用法:
#   ./run.sh              # 等同 smoke（无 GPU）
#   ./run.sh help         # 打印每条命令的中文说明
#   ./run.sh smoke        # 环境 + mock 采集 + 诊断 + HTML
#   ./run.sh tests        # unittest 全量（stage1–11）
#   ./run.sh all          # smoke + tests
#   ./run.sh live-vllm    # 本地 vLLM 驱动 TirAgent：LLM → tool → 答案 → reward
#   ./run.sh live-api     # .env API 驱动 TirAgent（同链路）
#   ./run.sh live-api-data  # 从 data/*.parquet 抽 N 条经 API 跑 TirAgent
#   ./run.sh live         # 等同 live-api
#   ./run.sh ui           # 启动 Science Control UI（uv .venv + webui）
#   ./run.sh ui-test      # 探测 GPU/RL 控制 API（可 start/stop 训练子进程）
#   ./run.sh branch-ui-test  # Branch rollout sites/Collect（可选 --train）
#   ./run.sh traj-test       # Rollout Sampling trajectoryGraph vitest
#
# 配置: Agent_Science_Infra/.env（见 .env.example）
# Python: 必须用本仓库 uv .venv（scripts/setup_uv_env.sh），禁止依赖 miniconda base
# 产物目录: artifacts/run_smoke/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIR="${ROOT}/mas"
VENV_PY="${ROOT}/.venv/bin/python"
OUT="${ROOT}/artifacts/run_smoke"
TRAJ="${OUT}/traj.json"
STATUS_HTML="${OUT}/status.html"
DASH_HTML="${OUT}/dashboard.html"
LIVE_TRAJ="${OUT}/traj_live.json"
LIVE_VLLM_TRAJ="${OUT}/traj_live_vllm.json"
LIVE_API_TRAJ="${OUT}/traj_live_api.json"
LIVE_API_DATA_TRAJ="${OUT}/traj_api_data.json"
LIVE_API_DATA_TASKS="${OUT}/tasks_api_data.json"
LIVE_TASKS="${OUT}/tir_demo_tasks.json"
DATA_DIR="${ROOT}/data"
VLLM_LOG="${VLLM_LOG:-/tmp/vllm_science_infra.log}"
VLLM_PID_FILE="${OUT}/vllm.pid"
UI_LOG="${OUT}/ui.log"
UI_PID_FILE="${OUT}/ui.pid"

# Load Agent_Science_Infra/.env (existing non-empty env wins)
load_root_env() {
  local env_file="${SCIENCE_INFRA_ENV:-${ROOT}/.env}"
  [[ -f "${env_file}" ]] || return 0
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ "${line}" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line//[[:space:]]/}" ]] && continue
    if [[ "${line}" =~ ^[[:space:]]*export[[:space:]]+(.*)$ ]]; then
      line="${BASH_REMATCH[1]}"
    fi
    [[ "${line}" != *"="* ]] && continue
    local key="${line%%=*}"
    local val="${line#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    key="${key#"${key%%[![:space:]]*}"}"
    val="${val#"${val%%[![:space:]]*}"}"
    val="${val%"${val##*[![:space:]]}"}"
    if [[ "${val}" =~ ^\"(.*)\"$ ]]; then
      val="${BASH_REMATCH[1]}"
    elif [[ "${val}" =~ ^\'(.*)\'$ ]]; then
      val="${BASH_REMATCH[1]}"
    fi
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${val}"
    fi
  done < "${env_file}"
}

load_root_env

# shellcheck disable=SC2034
RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
CYAN=$'\033[0;36m'
YELLOW=$'\033[0;33m'
BOLD=$'\033[1m'
NC=$'\033[0m'

die() {
  echo "${RED}ERROR:${NC} $*" >&2
  exit 1
}

step() {
  local n="$1" total="$2" title="$3"
  echo
  echo "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"
  echo "${CYAN}${BOLD}[$n/$total]${NC} ${BOLD}${title}${NC}"
  echo "${CYAN}${BOLD}════════════════════════════════════════════════════════════${NC}"
}

run_cmd() {
  echo "${YELLOW}+ $*${NC}"
  "$@"
}

# Always use repo uv .venv — never PATH science-infra from miniconda.
resolve_python() {
  if [[ -x "${VENV_PY}" ]]; then
    echo "${VENV_PY}"
    return 0
  fi
  die "未找到 uv 虚拟环境: ${VENV_PY}
请用 uv 创建（不要用 miniconda）:
  bash scripts/setup_uv_env.sh
或:
  uv venv --python 3.11 --prompt science-infra
  uv pip install -e '.[live,dev]' --python .venv/bin/python"
}

science_infra() {
  local py
  py="$(resolve_python)"
  export PYTHONPATH="${ROOT}:${TIR}${PYTHONPATH:+:${PYTHONPATH}}"
  # Prefer venv console script if present; else -m
  if [[ -x "${ROOT}/.venv/bin/science-infra" ]]; then
    "${ROOT}/.venv/bin/science-infra" "$@"
  else
    "${py}" -m science_infra.ui.cli "$@"
  fi
}

venv_python() {
  resolve_python
}

ensure_out() {
  mkdir -p "${OUT}"
}

ensure_cli() {
  local py
  py="$(resolve_python)"
  export PYTHONPATH="${ROOT}:${TIR}${PYTHONPATH:+:${PYTHONPATH}}"
  if "${py}" -c "import science_infra.ui.cli" 2>/dev/null; then
    return 0
  fi
  echo "${YELLOW}提示: .venv 中未安装 science-infra，用 uv 安装…${NC}"
  command -v uv >/dev/null 2>&1 || die "需要 uv。curl -LsSf https://astral.sh/uv/install.sh | sh"
  (cd "${ROOT}" && uv pip install -e '.[live,dev]' --python "${py}") \
    || die "uv pip install 失败。请运行: bash scripts/setup_uv_env.sh"
}

ensure_live_deps() {
  local py
  py="$(resolve_python)"
  if "${py}" -c "import langchain, langgraph" 2>/dev/null; then
    return 0
  fi
  echo "${YELLOW}提示: live 需要 langchain/langgraph，用 uv 安装 [live]…${NC}"
  command -v uv >/dev/null 2>&1 || die "需要 uv"
  (cd "${ROOT}" && uv pip install -e '.[live,dev]' --python "${py}") \
    || die "安装 live 依赖失败"
  "${py}" -c "import langchain, langgraph" \
    || die "仍无法 import langchain。请: bash scripts/setup_uv_env.sh"
}

# Demo task that encourages execute_python → answer → reward=1
write_tir_demo_tasks() {
  ensure_out
  cat > "${LIVE_TASKS}" <<'EOF'
[
  {
    "id": "tir-demo-1",
    "question": "What is 17+25? Use the execute_python tool if helpful, then put the final number in <answer>...</answer>.",
    "answer": "42",
    "source": "gsm8k"
  }
]
EOF
  echo "wrote ${LIVE_TASKS}"
}

# Verify Trajectory has tool_call + final_answer + positive reward
assert_tir_pipeline() {
  local traj_path="$1"
  local py
  py="$(resolve_python)"
  "${py}" - "${traj_path}" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
raw = json.loads(path.read_text(encoding="utf-8"))
batch = raw["batch"] if isinstance(raw, dict) and "batch" in raw else raw
trajs = batch.get("trajectories") if isinstance(batch, dict) else None
if not trajs:
    print("FAIL: no trajectories in", path, file=sys.stderr)
    sys.exit(1)
t = trajs[0]
events = t.get("events") or []
kinds = [e.get("kind") for e in events]
has_tool = "tool_call" in kinds
has_answer = bool(str(t.get("final_answer") or "").strip()) or "final_answer" in kinds
reward = t.get("final_reward")
ok_reward = reward is not None and float(reward) > 0
print(f"pipeline check: tool_call={has_tool} answer={has_answer!r} reward={reward} events={kinds}")
if not has_tool:
    print("FAIL: missing tool_call (expected LLM → tool)", file=sys.stderr)
    sys.exit(2)
if not has_answer:
    print("FAIL: missing final answer", file=sys.stderr)
    sys.exit(3)
if not ok_reward:
    print("FAIL: reward not positive", file=sys.stderr)
    sys.exit(4)
print("pipeline OK: LLM → tool → answer → reward")
PY
}

# Assert every trajectory in a batch has tool + answer + reward>0
assert_tir_batch() {
  local traj_path="$1"
  local min_ok="${2:-1}"
  local py
  py="$(resolve_python)"
  MIN_OK="${min_ok}" "${py}" - "${traj_path}" <<'PY'
import json, os, sys
from pathlib import Path
path = Path(sys.argv[1])
min_ok = int(os.environ.get("MIN_OK", "1"))
raw = json.loads(path.read_text(encoding="utf-8"))
batch = raw["batch"] if isinstance(raw, dict) and "batch" in raw else raw
trajs = batch.get("trajectories") if isinstance(batch, dict) else None
if not trajs:
    print("FAIL: no trajectories", file=sys.stderr)
    sys.exit(1)
ok = 0
for t in trajs:
    kinds = [e.get("kind") for e in (t.get("events") or [])]
    has_tool = "tool_call" in kinds
    ans = str(t.get("final_answer") or "").strip()
    reward = t.get("final_reward")
    good = has_tool and bool(ans) and reward is not None and float(reward) > 0
    ok += int(good)
    tid = (t.get("task") or {}).get("id")
    print(f"  {tid}: tool={has_tool} answer={ans!r} reward={reward} {'OK' if good else 'FAIL'}")
print(f"batch success {ok}/{len(trajs)} (need >= {min_ok})")
if ok < min_ok:
    sys.exit(2)
print("batch pipeline OK")
PY
}

# Sample N tasks from data/*.parquet into JSON for CLI collect
sample_data_tasks() {
  local parquet="$1"
  local n="$2"
  local source_filter="${3:-gsm8k}"
  local out_json="$4"
  local py
  py="$(resolve_python)"
  [[ -f "${parquet}" ]] || die "数据文件不存在: ${parquet}
请准备 ${DATA_DIR}/val.parquet 或 train.parquet"

  PARQUET="${parquet}" N="${n}" SOURCE="${source_filter}" OUT_JSON="${out_json}" "${py}" <<'PY'
import ast, json, os
from pathlib import Path
import pandas as pd

parquet = Path(os.environ["PARQUET"])
n = int(os.environ["N"])
source = (os.environ.get("SOURCE") or "").strip()
out = Path(os.environ["OUT_JSON"])
df = pd.read_parquet(parquet)
if source and source.lower() not in ("", "all", "*"):
    df = df[df["source"].astype(str) == source]
if len(df) == 0:
    raise SystemExit(f"no rows after filter source={source!r} in {parquet}")
df = df.head(n)
tasks = []
for row in df.to_dict(orient="records"):
    answers = row.get("answers")
    if isinstance(answers, str):
        try:
            answers = ast.literal_eval(answers)
        except Exception:
            answers = [row.get("answer")]
    tasks.append(
        {
            "id": str(row.get("id") or ""),
            "question": str(row.get("question") or ""),
            "answer": str(row.get("answer") or ""),
            "answers": answers if isinstance(answers, list) else [str(row.get("answer") or "")],
            "source": str(row.get("source") or "gsm8k"),
        }
    )
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {out} n={len(tasks)} from {parquet} source={source or 'all'}")
for t in tasks:
    print(f"  - {t['id']} gold={t['answer']}")
PY
}

vllm_endpoint() {
  local host="${VLLM_HOST:-127.0.0.1}"
  local port="${VLLM_PORT:-8000}"
  echo "http://${host}:${port}/v1"
}

vllm_healthy() {
  local ep
  ep="$(vllm_endpoint)"
  curl -sf -m 3 "${ep%/v1}/health" >/dev/null 2>&1 \
    || curl -sf -m 3 "${ep}/models" >/dev/null 2>&1
}

ensure_vllm() {
  local py host port model_path served mem maxlen
  py="$(resolve_python)"
  host="${VLLM_HOST:-127.0.0.1}"
  port="${VLLM_PORT:-8000}"
  model_path="${VLLM_MODEL_PATH:-/root/autodl-tmp/LLM/Qwen3-4B}"
  served="${VLLM_SERVED_NAME:-Qwen3-4B}"
  mem="${VLLM_GPU_MEM:-0.45}"
  maxlen="${VLLM_MAX_MODEL_LEN:-8192}"

  if vllm_healthy; then
    echo "vLLM already up at $(vllm_endpoint)"
    return 0
  fi

  [[ -d "${model_path}" ]] || die "本地模型不存在: ${model_path}
设置 VLLM_MODEL_PATH=... 或下载权重后再跑 ./run.sh live-vllm"

  command -v nvidia-smi >/dev/null 2>&1 || die "live-vllm 需要 GPU（未找到 nvidia-smi）"
  "${py}" -c "import vllm" 2>/dev/null || die "venv 中无 vllm。请: bash scripts/setup_uv_env.sh（从 AgentFlow 同步）"

  ensure_out
  echo "启动 vLLM: model=${model_path} served=${served} port=${port}（Hermes tool-call）"
  echo "日志: ${VLLM_LOG}"
  # Do not put the module path in this shell's argv in a way pkill -f would kill us later.
  (
    cd "${ROOT}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
    nohup "${py}" -m vllm.entrypoints.openai.api_server \
      --model "${model_path}" \
      --served-model-name "${served}" \
      --host "${host}" \
      --port "${port}" \
      --dtype auto \
      --max-model-len "${maxlen}" \
      --gpu-memory-utilization "${mem}" \
      --enable-auto-tool-choice \
      --tool-call-parser hermes \
      >"${VLLM_LOG}" 2>&1 &
    echo $! >"${VLLM_PID_FILE}"
  )
  echo "vLLM pid=$(cat "${VLLM_PID_FILE}" 2>/dev/null || echo '?')"

  local i
  for i in $(seq 1 120); do
    if vllm_healthy; then
      echo "vLLM ready ($(vllm_endpoint)) after ~${i}s"
      return 0
    fi
    if [[ -f "${VLLM_PID_FILE}" ]] && ! kill -0 "$(cat "${VLLM_PID_FILE}")" 2>/dev/null; then
      die "vLLM 进程已退出。查看日志: tail -80 ${VLLM_LOG}"
    fi
    sleep 2
  done
  die "等待 vLLM 超时（~240s）。查看: tail -80 ${VLLM_LOG}"
}

probe_openai_compat() {
  local endpoint="$1" model="$2" key="${3:-dummy}"
  local py
  py="$(resolve_python)"
  ENDPOINT="${endpoint}" MODEL="${model}" KEY="${key}" "${py}" <<'PY'
import os, sys, httpx
base = os.environ["ENDPOINT"].rstrip("/")
model = os.environ["MODEL"]
key = os.environ.get("KEY") or "dummy"
url = base + "/chat/completions"
try:
    r = httpx.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 4},
        timeout=60.0,
    )
except Exception as e:
    print(f"FAIL probe: {e}", file=sys.stderr)
    sys.exit(1)
print(f"probe status={r.status_code}")
if r.status_code >= 400:
    print(r.text[:400], file=sys.stderr)
    sys.exit(2)
print("probe OK")
PY
}

run_tir_live_collect() {
  local endpoint="$1" model="$2" out_traj="$3" key="${4:-dummy}"
  export OPENAI_API_KEY="${key}"
  write_tir_demo_tasks
  run_cmd science_infra collect \
    --n 1 \
    --tasks "${LIVE_TASKS}" \
    --endpoint "${endpoint}" \
    --model "${model}" \
    --out "${out_traj}"
  run_cmd science_infra diagnose "${out_traj}"
  run_cmd science_infra status "${out_traj}" --html "${out_traj%.json}.html"
  assert_tir_pipeline "${out_traj}"
}

cmd_help() {
  cat <<'EOF'
Agent_Science_Infra / run.sh — 功能测试说明
==========================================

用法
  ./run.sh [smoke|tests|all|live-vllm|live-api|live-api-data|live|train|ui|ui-test|branch-ui-test|traj-test|arpo-train-test|feature-test|help]
  ./run.sh train [fast|a800|a800_2gpu] [--gpu 0] [--n-runners 1] [--algo grpo] [--rl-yaml PATH]
  ./run.sh ui [--host 0.0.0.0] [--port 8787] [--rebuild|--no-build] [--daemon|--stop]
  默认: smoke

子命令
  help          打印本说明
  smoke         无 GPU 冒烟：doctor → 层依赖 → mock 采集 → diagnose → HTML
  tests         跑 tir_agent unittest（stage1–11）
  all           smoke + tests
  feature-test  功能分组测试：按功能域单测（--list 看全部域；详见下方章节）
  live-vllm     本地 vLLM 驱动 TirAgent：LLM → tool → 答案 → reward
  live-api      .env / OPENAI_* API 驱动 TirAgent（单条演示题）
  live-api-data 从 data/*.parquet 抽样经 API 跑 TirAgent（默认 5 条 gsm8k）
  live          等同 live-api
  train         RL 训练（默认 fast + 1 卡 + n_runners=1；参数可选）
  ui            启动 Control WebUI（总命令：uv .venv + 构建 webui + serve）
  ui-test       对已启动（或临时拉起）的 UI 测 GPU/RL 控制 API

配置文件
  Agent_Science_Infra/.env   （模板见 .env.example）
  也可: export SCIENCE_INFRA_ENV=/path/to/custom.env
  优先级: 已有环境变量 > .env > 代码默认值

Python 环境（uv，禁止 miniconda base）
  本仓库 .venv（Python 3.11）
  首次: bash scripts/setup_uv_env.sh
  run.sh 始终调用 .venv/bin/python，不会用 /root/miniconda3/bin/science-infra

产物目录
  artifacts/run_smoke/
    traj.json              mock 采集
    status.html / dashboard.html
    tir_demo_tasks.json    live 演示任务（17+25）
    traj_live_vllm.json    本地 vLLM TirAgent
    traj_live_api.json     远程 API TirAgent（单条）
    tasks_api_data.json    从 parquet 抽出的任务
    traj_api_data.json     data 集 API 采集结果
    traj_*.html            对应 status 页

────────────────────────────────────────
live-vllm
────────────────────────────────────────

  1) 若 http://127.0.0.1:8000/health 不可用，则启动:
       python -m vllm.entrypoints.openai.api_server
         --enable-auto-tool-choice --tool-call-parser hermes
  2) collect 演示任务（鼓励 execute_python）
  3) diagnose + status HTML
  4) 断言轨迹含 tool_call、final_answer、reward>0

  环境变量（可选）:
    VLLM_HOST VLLM_PORT VLLM_MODEL_PATH VLLM_SERVED_NAME
    VLLM_GPU_MEM VLLM_MAX_MODEL_LEN VLLM_LOG CUDA_VISIBLE_DEVICES

────────────────────────────────────────
live-api / live
────────────────────────────────────────

  读 .env 的 OPENAI_API_BASE / OPENAI_MODEL / OPENAI_API_KEY，
  对同一演示任务跑 TirAgent，并断言 LLM → tool → 答案 → reward。

────────────────────────────────────────
live-api-data
────────────────────────────────────────

  从 data/val.parquet（或 DATA_PARQUET）按 source 过滤抽样 N 条，经 API 跑 TirAgent。

  用法:
    ./run.sh live-api-data
    ./run.sh live-api-data data/val.parquet 5 gsm8k
    DATA_N=5 DATA_SOURCE=gsm8k ./run.sh live-api-data

  环境变量:
    DATA_PARQUET   默认 data/val.parquet
    DATA_N         默认 5
    DATA_SOURCE    默认 gsm8k（设 all 不过滤）
    DATA_MIN_OK    默认与 DATA_N 相同：至少多少条 reward>0 且含 tool

────────────────────────────────────────
smoke / tests / train
────────────────────────────────────────

  smoke / tests 见既有说明（mock 冒烟、unittest）。

  train 用法:
    ./run.sh train
    ./run.sh train fast --gpu 0 --n-runners 1 --algo grpo
    ./run.sh train --rl-yaml experiments/demo/rl.yaml --gpu 0

  自动: nvidia-smi 计数；1 张卡 → profile=fast、CUDA_VISIBLE_DEVICES=0。
  a800_2gpu 在可见卡 < 2 时拒绝。缺 parquet 时提示 scripts/prepare_data.sh。

────────────────────────────────────────
ui / ui-test / branch-ui-test / traj-test
────────────────────────────────────────

  总命令（用仓库 uv .venv，不要用 miniconda）:

    ./run.sh ui
    ./run.sh ui --port 8787 --rebuild
    ./run.sh ui --daemon          # 后台，日志 artifacts/run_smoke/ui.log
    ./run.sh ui --stop

  浏览器: http://127.0.0.1:8787/    API: /docs
  顶栏勾选 GPU → Collect / Diagnose / Train；RL 页可改每题采样与 n_runners。

  测控制面（不跑完 GRPO 一步；会 start 再立刻 stop 训练子进程）:

    ./run.sh ui-test
    ./run.sh ui-test --no-train
    UI_PORT=8787 ./run.sh ui-test

  Branch rollout 验收（sites + Collect；可选短训扫 expansion）:

    ./run.sh branch-ui-test
    ./run.sh branch-ui-test --train
    ./run.sh branch-ui-test --algo rae --train
    ./run.sh branch-ui-test --pev-fixture
    手册: docs/BRANCH_ROLLOUT_UI_TEST.md / docs/ROLLOUT_SAMPLING_UI_TEST.md

  Rollout Sampling 轨迹推导单测（vitest，无需 UI）:

    ./run.sh traj-test
    手册: docs/ROLLOUT_SAMPLING_UI_TEST.md

  ARPO 训练验收（10 轮采样/reward/loss 断言，训练结束后恢复 YAML）:

    ./run.sh arpo-train-test
    ./run.sh arpo-train-test --steps 10 --timeout 900
    ./run.sh arpo-train-test --negative    # 追加 n_branch=0 负例
    手册: docs/ARPO_TRAIN_TEST.md (Phase 4) / docs/MAS_AGENT_FRAMEWORK_TEST.md

────────────────────────────────────────
feature-test — 按功能域测试（推荐日常用）
────────────────────────────────────────

  把 146 个单测 + 前端 vitest + 冒烟拆成 16 个功能域，可单测一个功能，
  也可 --all 全量。每个域独立输出 PASS/FAIL + 耗时，结尾汇总矩阵。

  查看全部功能域:

    ./run.sh feature-test --list

  测单个功能（示例）:

    ./run.sh feature-test mas-core          # MAS 基础层（spec/compiler/红线/采集）
    ./run.sh feature-test agent-framework   # agent 化（registry/router/memory/PEV）
    ./run.sh feature-test branch            # 分支采样（gates/RAE/active set）
    ./run.sh feature-test rollout-tree      # RolloutTree 契约
    ./run.sh feature-test schema03          # schema 0.3 sugar/tool-agent
    ./run.sh feature-test daemon            # Daemon expansion enqueue
    ./run.sh feature-test realtime         # 实时 Harness（JSONL/SSE）
    ./run.sh feature-test frontend          # 前端 vitest（轨迹 + router 节点）
    ./run.sh feature-test smoke             # 无 GPU 冒烟（等价 ./run.sh smoke）

  测多个功能:

    ./run.sh feature-test mas-core rl harness

  全量（全部单测域 + frontend，不含 smoke）:

    ./run.sh feature-test --all

  全部功能域一览（--list 输出）:
    mas-core rl harness branch rollout-tree agent-framework schema03
    daemon realtime cli control-ui gpu-compiler verifier e2e frontend smoke

  手册: docs/MAS_AGENT_FRAMEWORK_TEST.md (§2 测试矩阵)

EOF
}

cmd_smoke() {
  ensure_out
  ensure_cli
  local total=6

  step 1 "$total" "doctor — 环境 / GPU / AGL / MASSpec 探测"
  echo "说明: 检查本机能否跑 infra；不采集、不训练。"
  run_cmd science_infra doctor

  step 2 "$total" "check_workflow_deps — workflow/ 不得依赖 RL"
  echo "说明: AST 扫描 workflow/*.py，禁止 import agentlightning/verl/ray。"
  run_cmd env PYTHONPATH="${TIR}" "$(resolve_python)" "${TIR}/scripts/check_workflow_deps.py"

  step 3 "$total" "collect --mock — 无 LLM 采集 Trajectory + TrainSignal"
  echo "说明: 验证 MAS 数据层可独立于 VERL；产物 → ${TRAJ}"
  run_cmd science_infra collect --mock --n 2 --out "${TRAJ}"

  step 4 "$total" "diagnose — Harness 插件读轨迹"
  echo "说明: 对 mock JSON 跑注册诊断器，打印 Hypothesis 列表。"
  run_cmd science_infra diagnose "${TRAJ}"

  step 5 "$total" "status — 写出状态 HTML"
  echo "说明: mean_reward / 曲线 / 事件 / Harness → ${STATUS_HTML}"
  run_cmd science_infra status "${TRAJ}" --html "${STATUS_HTML}"

  step 6 "$total" "dashboard — 写出 dashboard HTML"
  echo "说明: dashboard 子命令别名 → ${DASH_HTML}"
  run_cmd science_infra dashboard "${TRAJ}" --html "${DASH_HTML}"

  echo
  echo "${GREEN}${BOLD}smoke OK${NC}"
  echo "  traj:      ${TRAJ}"
  echo "  status:    ${STATUS_HTML}"
  echo "  dashboard: ${DASH_HTML}"
}

cmd_tests() {
  [[ -d "${TIR}" ]] || die "找不到 mas/: ${TIR}"
  ensure_cli
  step 1 1 "unittest — test_infra_v1.py discover stage1–11"
  echo "说明: MAS / RL overlay / Harness / CLI / verifier / e2e / UI 回归。"
  (
    cd "${TIR}"
    export PYTHONPATH="${ROOT}:${TIR}${PYTHONPATH:+:${PYTHONPATH}}"
    run_cmd "$(resolve_python)" tests/test_infra_v1.py
  )
  echo
  echo "${GREEN}${BOLD}tests OK${NC}"
}

cmd_all() {
  cmd_smoke
  cmd_tests
  echo
  echo "${GREEN}${BOLD}all OK（smoke + tests）${NC}"
}

cmd_live_vllm() {
  ensure_out
  ensure_cli
  ensure_live_deps
  local total=4
  local endpoint model

  step 1 "$total" "ensure local vLLM（Hermes tool-call）"
  ensure_vllm
  endpoint="$(vllm_endpoint)"
  model="${VLLM_SERVED_NAME:-Qwen3-4B}"

  step 2 "$total" "probe OpenAI-compat endpoint"
  probe_openai_compat "${endpoint}" "${model}" "dummy"

  step 3 "$total" "TirAgent collect — LLM → tool → 答案 → reward"
  echo "endpoint=${endpoint} model=${model}"
  echo "产物 → ${LIVE_VLLM_TRAJ}"
  run_tir_live_collect "${endpoint}" "${model}" "${LIVE_VLLM_TRAJ}" "dummy"

  step 4 "$total" "汇总"
  echo
  echo "${GREEN}${BOLD}live-vllm OK${NC}"
  echo "  traj:   ${LIVE_VLLM_TRAJ}"
  echo "  status: ${LIVE_VLLM_TRAJ%.json}.html"
  echo "  vllm:   ${endpoint}  (log: ${VLLM_LOG})"
}

cmd_live_api() {
  ensure_out
  ensure_cli
  ensure_live_deps
  local total=3
  local endpoint model key

  endpoint="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
  model="${OPENAI_MODEL:-${MODEL:-}}"
  key="${OPENAI_API_KEY:-dummy}"

  if [[ -z "${endpoint}" ]]; then
    die "live-api 需要 endpoint。请在 ${ROOT}/.env 设置 OPENAI_API_BASE，或:
  export OPENAI_API_BASE=https://.../v1
  export OPENAI_MODEL=...
  export OPENAI_API_KEY=...
  ./run.sh live-api"
  fi
  if [[ -z "${model}" ]]; then
    die "live-api 需要 OPENAI_MODEL（或 MODEL）"
  fi

  step 1 "$total" "probe API endpoint"
  echo "endpoint=${endpoint} model=${model}"
  probe_openai_compat "${endpoint}" "${model}" "${key}" \
    || die "API 探测失败（常为 401 无效令牌）。请更新 .env 中 OPENAI_API_KEY 后重试。"

  step 2 "$total" "TirAgent collect — LLM → tool → 答案 → reward"
  echo "产物 → ${LIVE_API_TRAJ}"
  run_tir_live_collect "${endpoint}" "${model}" "${LIVE_API_TRAJ}" "${key}"

  step 3 "$total" "汇总"
  echo
  echo "${GREEN}${BOLD}live-api OK${NC}"
  echo "  traj:   ${LIVE_API_TRAJ}"
  echo "  status: ${LIVE_API_TRAJ%.json}.html"
}

cmd_live_api_data() {
  # ./run.sh live-api-data [parquet] [n] [source]
  ensure_out
  ensure_cli
  ensure_live_deps
  local total=4
  local endpoint model key
  local parquet n source min_ok
  local py

  parquet="${1:-${DATA_PARQUET:-${DATA_DIR}/val.parquet}}"
  n="${2:-${DATA_N:-5}}"
  source="${3:-${DATA_SOURCE:-gsm8k}}"
  min_ok="${DATA_MIN_OK:-${n}}"
  py="$(resolve_python)"

  endpoint="${OPENAI_API_BASE:-${OPENAI_BASE_URL:-}}"
  model="${OPENAI_MODEL:-${MODEL:-}}"
  key="${OPENAI_API_KEY:-dummy}"

  if [[ -z "${endpoint}" ]]; then
    die "live-api-data 需要 OPENAI_API_BASE（.env）"
  fi
  if [[ -z "${model}" ]]; then
    die "live-api-data 需要 OPENAI_MODEL（.env）"
  fi
  [[ -f "${parquet}" ]] || die "找不到数据: ${parquet}"

  step 1 "$total" "probe API endpoint"
  echo "endpoint=${endpoint} model=${model}"
  probe_openai_compat "${endpoint}" "${model}" "${key}" \
    || die "API 探测失败。请更新 .env 中 OPENAI_API_KEY。"

  step 2 "$total" "从 parquet 抽样任务"
  echo "parquet=${parquet} n=${n} source=${source}"
  sample_data_tasks "${parquet}" "${n}" "${source}" "${LIVE_API_DATA_TASKS}"

  step 3 "$total" "TirAgent API collect（单进程逐条，共 ${n} 条）"
  echo "产物 → ${LIVE_API_DATA_TRAJ}"
  export OPENAI_API_KEY="${key}"
  export PYTHONPATH="${ROOT}:${TIR}${PYTHONPATH:+:${PYTHONPATH}}"
  TASKS_JSON="${LIVE_API_DATA_TASKS}" \
  OUT_TRAJ="${LIVE_API_DATA_TRAJ}" \
  ENDPOINT="${endpoint}" \
  MODEL="${model}" \
  "${py}" <<'PY'
import json, os, sys
from pathlib import Path

from workflow import Collector, batch_to_train_signal
from workflow.contracts import TrajectoryBatch

tasks = json.loads(Path(os.environ["TASKS_JSON"]).read_text(encoding="utf-8"))
endpoint = os.environ["ENDPOINT"]
model = os.environ["MODEL"]
out = Path(os.environ["OUT_TRAJ"])

collector = Collector(mock=False, endpoint=endpoint, model=model, n=1)
trajs = []
for i, task in enumerate(tasks):
    print(f"--- collect {i}/{len(tasks)-1} id={task.get('id')} ---", flush=True)
    traj = collector.collect_one(dict(task))
    trajs.append(traj)
    print(
        f"  reward={traj.final_reward} answer={traj.final_answer!r} "
        f"n_python={traj.n_python} n_search={traj.n_search}",
        flush=True,
    )

vals = [float(t.final_reward) for t in trajs if t.final_reward is not None]
mean = sum(vals) / len(vals) if vals else 0.0
batch = TrajectoryBatch(
    trajectories=trajs,
    meta={"n_trajectories": len(trajs), "mean_reward": mean, "source": "live-api-data"},
)
signal = batch_to_train_signal(batch, algo="grpo")
payload = {
    "batch": batch.model_dump(mode="json"),
    "train_signal": {
        "advantage": signal.advantage.model_dump(mode="json"),
        "loss": signal.loss.model_dump(mode="json"),
        "meta": signal.meta,
    },
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {out} n={len(trajs)} mean_reward={mean}", flush=True)
PY
  run_cmd science_infra diagnose "${LIVE_API_DATA_TRAJ}"
  run_cmd science_infra status "${LIVE_API_DATA_TRAJ}" --html "${LIVE_API_DATA_TRAJ%.json}.html"

  step 4 "$total" "断言 batch：tool + answer + reward"
  assert_tir_batch "${LIVE_API_DATA_TRAJ}" "${min_ok}"

  echo
  echo "${GREEN}${BOLD}live-api-data OK${NC}"
  echo "  tasks:  ${LIVE_API_DATA_TASKS}"
  echo "  traj:   ${LIVE_API_DATA_TRAJ}"
  echo "  status: ${LIVE_API_DATA_TRAJ%.json}.html"
}

# Backward-compatible alias
cmd_live() {
  cmd_live_api
}

cmd_train() {
  [[ -d "${TIR}" ]] || die "找不到 mas/: ${TIR}"
  ensure_cli
  local py
  py="$(resolve_python)"

  local profile="fast"
  local algo="grpo"
  local n_runners="1"
  local gpu_opt=""
  local rl_yaml=""

  if [[ "${1:-}" =~ ^(fast|a800|a800_2gpu)$ ]]; then
    profile="$1"
    shift
  fi
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpu)
        gpu_opt="${2:-}"
        shift 2
        ;;
      --gpu=*)
        gpu_opt="${1#*=}"
        shift
        ;;
      --n-runners)
        n_runners="${2:-}"
        shift 2
        ;;
      --n-runners=*)
        n_runners="${1#*=}"
        shift
        ;;
      --algo)
        algo="${2:-}"
        shift 2
        ;;
      --algo=*)
        algo="${1#*=}"
        shift
        ;;
      --rl-yaml)
        rl_yaml="${2:-}"
        shift 2
        ;;
      --rl-yaml=*)
        rl_yaml="${1#*=}"
        shift
        ;;
      fast|a800|a800_2gpu)
        profile="$1"
        shift
        ;;
      *)
        die "未知 train 参数: $1
用法: ./run.sh train [fast|a800|a800_2gpu] [--gpu 0] [--n-runners 1] [--algo grpo] [--rl-yaml PATH]"
        ;;
    esac
  done

  ensure_agl_dashboard 0

  if ! "${py}" -c "import agentlightning" 2>/dev/null; then
    die "train 需要 agentlightning（在 uv .venv 中）。示例:
  uv pip install -e '.[rl]' --python .venv/bin/python
或先跑: ./run.sh smoke（无 RL）"
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    die "train 需要 GPU（未找到 nvidia-smi）。无 GPU 请用: ./run.sh smoke 或 ./run.sh all"
  fi
  if ! nvidia-smi -L >/dev/null 2>&1; then
    die "nvidia-smi -L 失败，无法枚举 GPU。无 GPU 请用: ./run.sh smoke"
  fi

  local gpu_count
  gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)"
  gpu_count="${gpu_count:-0}"
  if [[ "${gpu_count}" -lt 1 ]]; then
    die "未枚举到 GPU。无 GPU 请用: ./run.sh smoke"
  fi

  if [[ -z "${gpu_opt}" ]]; then
    if [[ "${profile}" == "a800_2gpu" && "${gpu_count}" -ge 2 ]]; then
      gpu_opt="0,1"
    else
      gpu_opt="0"
    fi
  fi
  gpu_opt="$(echo "${gpu_opt}" | tr -d ' ')"
  local n_sel
  n_sel="$(echo "${gpu_opt}" | awk -F',' '{print NF}')"

  if [[ "${profile}" == "a800_2gpu" && "${n_sel}" -lt 2 ]]; then
    die "profile=a800_2gpu 需要至少 2 张 GPU（当前 --gpu=${gpu_opt}，机器 ${gpu_count} 张）。请用: ./run.sh train fast --gpu 0"
  fi
  if [[ "${n_sel}" -lt 2 && "${profile}" == "a800_2gpu" ]]; then
    die "可见卡不足 2 张，不能用 a800_2gpu"
  fi
  if [[ "${gpu_count}" -lt 2 && "${profile}" == "a800_2gpu" ]]; then
    die "机器只有 ${gpu_count} 张 GPU，禁止 a800_2gpu。请用 fast 或 a800。"
  fi

  local train_pq="${TIR}/data/train.parquet"
  local val_pq="${TIR}/data/val.parquet"
  if [[ ! -f "${train_pq}" || ! -f "${val_pq}" ]]; then
    die "缺少 parquet: ${train_pq} / ${val_pq}
请先: cd ${TIR} && bash scripts/prepare_data.sh"
  fi

  if [[ -n "${rl_yaml}" && "${rl_yaml}" != /* ]]; then
    rl_yaml="${ROOT}/${rl_yaml}"
  fi
  if [[ -n "${rl_yaml}" && ! -f "${rl_yaml}" ]]; then
    die "找不到 --rl-yaml: ${rl_yaml}"
  fi

  export CUDA_VISIBLE_DEVICES="${gpu_opt}"
  export VLLM_USE_V1="${VLLM_USE_V1:-1}"

  local rollout_n="4"
  if [[ "${profile}" == "fast" ]]; then
    rollout_n="2"
  fi
  if [[ -n "${rl_yaml}" ]]; then
    local parsed
    parsed="$("${py}" -c "
import yaml
from pathlib import Path
d = yaml.safe_load(Path(r'''${rl_yaml}''').read_text(encoding='utf-8')) or {}
n = (d.get('actor_rollout_ref') or {}).get('rollout', {})
print(n.get('n') or d.get('rollout_per_gpu') or '')
" 2>/dev/null || true)"
    if [[ -n "${parsed}" ]]; then
      rollout_n="${parsed}"
    fi
  fi

  local argv=(train_tir_agent.py "${profile}" --algo "${algo}" --n-runners "${n_runners}")
  if [[ -n "${rl_yaml}" ]]; then
    argv+=(--rl-yaml "${rl_yaml}")
  fi

  step 1 1 "train_tir_agent.py ${profile} --algo ${algo} --n-runners ${n_runners}"
  echo "说明: LitTirAgent + AGL/VERL；单卡冒烟用 fast + n_runners=1。"
  echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "  n_gpus=${n_sel}  machine_gpus=${gpu_count}"
  echo "  n_runners=${n_runners}  algo=${algo}  profile=${profile}"
  echo "  rollout.n=${rollout_n}"
  if [[ -n "${rl_yaml}" ]]; then
    echo "  rl-yaml=${rl_yaml}"
  fi
  (
    cd "${TIR}"
    export PYTHONPATH="${ROOT}:${TIR}${PYTHONPATH:+:${PYTHONPATH}}"
    export VLLM_USE_V1="${VLLM_USE_V1:-1}"
    run_cmd "${py}" "${argv[@]}"
  )
  echo
  echo "${GREEN}${BOLD}train OK${NC}"
}

ensure_webui() {
  local rebuild="${1:-0}"
  local dist="${ROOT}/webui/dist/index.html"
  if [[ "${rebuild}" != "1" && -f "${dist}" ]]; then
    return 0
  fi
  command -v npm >/dev/null 2>&1 || die "启动 UI 需要 npm 以构建 webui（未找到 npm）"
  step 1 1 "webui npm run build"
  (
    cd "${ROOT}/webui"
    if [[ ! -d node_modules ]]; then
      run_cmd npm install
    fi
    run_cmd npm run build
  )
}

ensure_agl_dashboard() {
  local rebuild="${1:-0}"
  local index="${ROOT}/agent-lightning/agentlightning/dashboard/index.html"
  if [[ "${rebuild}" != "1" && -f "${index}" ]]; then
    return 0
  fi
  command -v npm >/dev/null 2>&1 || die "AGL Metrics 需要构建 dashboard（未找到 npm）: cd agent-lightning/dashboard && npm install && npm run build"
  step 1 1 "AGL dashboard npm run build"
  (
    cd "${ROOT}/agent-lightning/dashboard"
    if [[ ! -d node_modules ]]; then
      run_cmd npm install
    fi
    run_cmd npm run build
  )
  [[ -f "${index}" ]] || die "AGL dashboard 构建失败，缺少 ${index}"
}

ui_health_ok() {
  local port="$1"
  local py
  py="$(resolve_python)"
  "${py}" -c "
import urllib.request
try:
    urllib.request.urlopen('http://127.0.0.1:${port}/api/health', timeout=1.5)
except Exception:
    raise SystemExit(1)
" 2>/dev/null
}

cmd_ui_stop() {
  ensure_out
  if [[ -f "${UI_PID_FILE}" ]]; then
    local pid
    pid="$(cat "${UI_PID_FILE}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "停止 UI pid=${pid}"
      kill "${pid}" 2>/dev/null || true
      sleep 0.4
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${UI_PID_FILE}"
  fi
}

cmd_ui() {
  ensure_out
  ensure_cli
  local host="0.0.0.0"
  local port="${UI_PORT:-8787}"
  local rebuild=0
  local nobuild=0
  local daemon=0
  local stop=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --host)
        host="${2:-}"
        shift 2
        ;;
      --host=*)
        host="${1#*=}"
        shift
        ;;
      --port)
        port="${2:-}"
        shift 2
        ;;
      --port=*)
        port="${1#*=}"
        shift
        ;;
      --rebuild)
        rebuild=1
        shift
        ;;
      --no-build)
        nobuild=1
        shift
        ;;
      --daemon|--bg)
        daemon=1
        shift
        ;;
      --stop)
        stop=1
        shift
        ;;
      *)
        die "未知 ui 参数: $1
用法: ./run.sh ui [--host 0.0.0.0] [--port 8787] [--rebuild|--no-build] [--daemon|--stop]"
        ;;
    esac
  done

  if [[ "${stop}" == "1" ]]; then
    cmd_ui_stop
    echo "${GREEN}${BOLD}ui stopped${NC}"
    return 0
  fi

  if [[ "${nobuild}" != "1" ]]; then
    ensure_webui "${rebuild}"
    ensure_agl_dashboard "${rebuild}"
  elif [[ ! -f "${ROOT}/webui/dist/index.html" ]]; then
    die "未找到 webui/dist。请去掉 --no-build，或先: cd webui && npm run build"
  elif [[ ! -f "${ROOT}/agent-lightning/agentlightning/dashboard/index.html" ]]; then
    die "未找到 AGL dashboard。请去掉 --no-build，或先: cd agent-lightning/dashboard && npm install && npm run build"
  fi

  local py
  py="$(resolve_python)"
  export PYTHONPATH="${ROOT}:${TIR}${PYTHONPATH:+:${PYTHONPATH}}"

  echo
  echo "${CYAN}${BOLD}Science Control UI${NC}"
  echo "  python: ${py}"
  echo "  浏览器: http://127.0.0.1:${port}/"
  echo "  本机网卡: http://${host}:${port}/"
  echo "  API 文档: http://127.0.0.1:${port}/docs"
  echo "  GPU/RL: 顶栏勾选 GPU，RL 页设每题采样 / n_runners，再点 Train"
  echo

  local serve_bin="${ROOT}/.venv/bin/science-infra"
  local serve_cmd=()
  if [[ -x "${serve_bin}" ]]; then
    serve_cmd=("${serve_bin}" serve --host "${host}" --port "${port}")
  else
    serve_cmd=("${py}" -m science_infra.ui.cli serve --host "${host}" --port "${port}")
  fi

  if [[ "${daemon}" == "1" ]]; then
    if ui_health_ok "${port}"; then
      echo "${YELLOW}端口 ${port} 上 UI 已在跑，跳过启动。停掉: ./run.sh ui --stop${NC}"
      return 0
    fi
    cmd_ui_stop
    echo "${YELLOW}+ ${serve_cmd[*]}  (daemon, log=${UI_LOG})${NC}"
    nohup "${serve_cmd[@]}" >"${UI_LOG}" 2>&1 &
    echo $! >"${UI_PID_FILE}"
    local i=0
    while [[ "${i}" -lt 40 ]]; do
      if ui_health_ok "${port}"; then
        echo "${GREEN}${BOLD}ui daemon OK${NC}  pid=$(cat "${UI_PID_FILE}")"
        echo "  log: ${UI_LOG}"
        return 0
      fi
      i=$((i + 1))
      sleep 0.25
    done
    echo "${RED}UI 启动超时，见 ${UI_LOG}${NC}" >&2
    tail -n 40 "${UI_LOG}" >&2 || true
    return 1
  fi

  run_cmd "${serve_cmd[@]}"
}

cmd_ui_test() {
  ensure_out
  ensure_cli
  local port="${UI_PORT:-8787}"
  local no_train=0
  local started=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        port="${2:-}"
        shift 2
        ;;
      --port=*)
        port="${1#*=}"
        shift
        ;;
      --no-train)
        no_train=1
        shift
        ;;
      *)
        die "未知 ui-test 参数: $1
用法: ./run.sh ui-test [--port 8787] [--no-train]"
        ;;
    esac
  done

  local py
  py="$(resolve_python)"
  if [[ ! -x "${py}" ]]; then
    die "需要 uv .venv: ${VENV_PY}"
  fi

  if ! ui_health_ok "${port}"; then
    echo "UI 未在 :${port} 运行，先后台启动…"
    cmd_ui --port "${port}" --daemon
    started=1
  fi

  step 1 1 "scripts/ui_rl_check.py  (uv .venv)"
  local check_args=(--base "http://127.0.0.1:${port}" --experiment demo)
  if [[ "${no_train}" == "1" ]]; then
    check_args+=(--no-train)
  fi
  if ! run_cmd "${py}" "${ROOT}/scripts/ui_rl_check.py" "${check_args[@]}"; then
    if [[ "${started}" == "1" ]]; then
      echo "${YELLOW}检查失败；后台 UI 仍在跑。停掉: ./run.sh ui --stop${NC}"
    fi
    die "ui-test 失败"
  fi
  echo
  echo "${GREEN}${BOLD}ui-test OK${NC}"
  echo "  UI: http://127.0.0.1:${port}/"
  if [[ "${started}" == "1" ]]; then
    echo "  本次拉起的后台服务未关闭。停掉: ./run.sh ui --stop"
  fi
}

cmd_traj_test() {
  ensure_cli
  local webui="${ROOT}/webui"
  if [[ ! -f "${webui}/package.json" ]]; then
    die "缺少 webui/package.json"
  fi
  if [[ ! -d "${webui}/node_modules/vitest" ]]; then
    step 1 2 "npm install (webui，补 vitest)"
    (cd "${webui}" && npm install) || die "webui npm install 失败"
  fi
  step 1 1 "vitest trajectoryGraph (test:traj)"
  if ! (cd "${webui}" && npm run test:traj); then
    die "traj-test 失败"
  fi
  echo
  echo "${GREEN}${BOLD}traj-test OK${NC}"
  echo "  手册: docs/ROLLOUT_SAMPLING_UI_TEST.md"
}

cmd_branch_ui_test() {
  ensure_cli
  local port="${UI_PORT:-8787}"
  local started=0
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        port="${2:-}"
        shift 2
        ;;
      --port=*)
        port="${1#*=}"
        shift
        ;;
      *)
        extra+=("$1")
        shift
        ;;
    esac
  done

  local py
  py="$(resolve_python)"
  if [[ ! -x "${py}" ]]; then
    die "需要 uv .venv: ${VENV_PY}"
  fi

  if ! ui_health_ok "${port}"; then
    echo "UI 未在 :${port} 运行，先后台启动…"
    cmd_ui --port "${port}" --daemon
    started=1
  fi

  step 1 1 "scripts/branch_rollout_ui_test.py  (uv .venv)"
  local check_args=(--base "http://127.0.0.1:${port}" --experiment arpo_e2e)
  if [[ ${#extra[@]} -gt 0 ]]; then
    check_args+=("${extra[@]}")
  fi
  if ! run_cmd "${py}" "${ROOT}/scripts/branch_rollout_ui_test.py" "${check_args[@]}"; then
    if [[ "${started}" == "1" ]]; then
      echo "${YELLOW}检查失败；后台 UI 仍在跑。停掉: ./run.sh ui --stop${NC}"
    fi
    die "branch-ui-test 失败"
  fi
  echo
  echo "${GREEN}${BOLD}branch-ui-test OK${NC}"
  echo "  手册: docs/BRANCH_ROLLOUT_UI_TEST.md / docs/ROLLOUT_SAMPLING_UI_TEST.md"
  echo "  UI: http://127.0.0.1:${port}/"
  if [[ "${started}" == "1" ]]; then
    echo "  本次拉起的后台服务未关闭。停掉: ./run.sh ui --stop"
  fi
}

cmd_arpo_train_test() {
  ensure_cli
  local port="${UI_PORT:-8787}"
  local started=0
  local extra=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --port)
        port="${2:-}"
        shift 2
        ;;
      --port=*)
        port="${1#*=}"
        shift
        ;;
      *)
        extra+=("$1")
        shift
        ;;
    esac
  done

  local py
  py="$(resolve_python)"
  if [[ ! -x "${py}" ]]; then
    die "需要 uv .venv: ${VENV_PY}"
  fi

  if ! ui_health_ok "${port}"; then
    echo "UI 未在 :${port} 运行，先后台启动…"
    cmd_ui --port "${port}" --daemon
    started=1
  fi

  step 1 1 "scripts/arpo_train_verify.py  (ARPO 10 轮采样/reward/loss 验收)"
  local check_args=(--base "http://127.0.0.1:${port}" --experiment arpo_e2e)
  if [[ ${#extra[@]} -gt 0 ]]; then
    check_args+=("${extra[@]}")
  fi
  if ! run_cmd "${py}" "${ROOT}/scripts/arpo_train_verify.py" "${check_args[@]}"; then
    if [[ "${started}" == "1" ]]; then
      echo "${YELLOW}检查失败；后台 UI 仍在跑。停掉: ./run.sh ui --stop${NC}"
    fi
    die "arpo-train-test 失败"
  fi
  echo
  echo "${GREEN}${BOLD}arpo-train-test OK${NC}"
  echo "  手册: docs/ARPO_TRAIN_TEST.md (Phase 4) + docs/MAS_AGENT_FRAMEWORK_TEST.md"
  echo "  UI: http://127.0.0.1:${port}/"
  if [[ "${started}" == "1" ]]; then
    echo "  本次拉起的后台服务未关闭。停掉: ./run.sh ui --stop"
  fi
}

cmd_feature_test() {
  ensure_cli
  local py
  py="$(resolve_python)"
  if [[ ! -x "${py}" ]]; then
    die "需要 uv .venv: ${VENV_PY}"
  fi
  # 无参数 → 打印域列表；--all / 域名透传给 scripts/feature_test.py
  run_cmd "${py}" "${ROOT}/scripts/feature_test.py" "$@"
  local rc=$?
  if [[ ${rc} -ne 0 ]]; then
    die "feature-test 失败"
  fi
  echo
  echo "${GREEN}${BOLD}feature-test OK${NC}"
  echo "  域列表: ${py} ${ROOT}/scripts/feature_test.py --list"
  echo "  手册: docs/MAS_AGENT_FRAMEWORK_TEST.md (§2 测试矩阵)"
}

main() {
  local mode="${1:-smoke}"
  shift || true
  case "${mode}" in
    -h|--help|help) cmd_help ;;
    smoke) cmd_smoke ;;
    tests) cmd_tests ;;
    all) cmd_all ;;
    live-vllm|vllm) cmd_live_vllm ;;
    live-api|api) cmd_live_api ;;
    live-api-data|api-data|data-api) cmd_live_api_data "$@" ;;
    live) cmd_live ;;
    train) cmd_train "$@" ;;
    ui|serve|webui) cmd_ui "$@" ;;
    ui-test|ui-check) cmd_ui_test "$@" ;;
    branch-ui-test|branch-ui|branch-test) cmd_branch_ui_test "$@" ;;
    traj-test|traj|trajectory-test) cmd_traj_test "$@" ;;
    arpo-train-test|arpo-train|arpo-verify) cmd_arpo_train_test "$@" ;;
    feature-test|feature|ftest) cmd_feature_test "$@" ;;
    *)
      die "未知模式: ${mode}
用法: ./run.sh [smoke|tests|all|live-vllm|live-api|live-api-data|live|train|ui|ui-test|branch-ui-test|traj-test|arpo-train-test|feature-test|help]
详见: ./run.sh help"
      ;;
  esac
}

main "$@"
