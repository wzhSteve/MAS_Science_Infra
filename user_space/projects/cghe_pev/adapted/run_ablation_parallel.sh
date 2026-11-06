#!/usr/bin/env bash
# Module ablations for CGHE after accuracy fixes.
# Arms: full | noprune | nomemory | selectonly | forceregen
# Datasets: hotpot + bamboogle in parallel per arm; arms run sequentially
# (at most one arm's two dataset workers) to protect Bing quota.
# forceregen = SoftBing default + CGHE_FORCE_REGEN_AFTER_HARD_PRUNE=1 (ablation only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
ADAPTED="$ROOT/user_space/projects/cghe_pev/adapted"
PROJECT="$ROOT/user_space/projects/cghe_pev"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

LIMIT="${CGHE_LIMIT:-}"
ARMS_CSV="${CGHE_ABLATION_ARMS:-full,noprune,nomemory,selectonly}"
DATASETS_CSV="${CGHE_ABLATION_DATASETS:-hotpot,bamboogle}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${CGHE_ABLATION_OUT:-$PROJECT/artifacts/ablation_${TS}}"
mkdir -p "$OUT_ROOT"

cd "$ROOT"
ENV_EXPORT="$PROJECT/artifacts/.full_eval_env_export.sh"
mkdir -p "$(dirname "$ENV_EXPORT")"
PYTHONPATH="$ROOT:$ROOT/mas${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<PY
import os, shlex, sys
from pathlib import Path
sys.path[:0] = [r"$ROOT", r"$ROOT/mas"]
from workflow.env_load import load_repo_dotenv
load_repo_dotenv()
keys = [
    "OPENAI_API_KEY", "OPENAI_API_BASE", "OPENAI_API_BASE_URL", "MODEL_Name",
    "BRIGHT_DATA_API_KEY_2", "BRIGHT_DATA_SERP_ZONES_2",
]
lines = []
for k in keys:
    v = os.environ.get(k)
    if v is None or str(v).strip() == "":
        continue
    lines.append(f"export {k}={shlex.quote(str(v))}")
Path(r"$ENV_EXPORT").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print("wrote", r"$ENV_EXPORT", "keys", len(lines))
PY
# shellcheck disable=SC1090
set -a
source "$ENV_EXPORT"
set +a

need=(OPENAI_API_KEY OPENAI_API_BASE OPENAI_API_BASE_URL MODEL_Name BRIGHT_DATA_API_KEY_2 BRIGHT_DATA_SERP_ZONES_2)
for k in "${need[@]}"; do
  if [[ -z "${!k:-}" ]]; then
    echo "FATAL: missing required env $k from .env" >&2
    exit 1
  fi
done

export OPENAI_MODEL="$MODEL_Name"
export SERVER_MODEL="$MODEL_Name"
export BRIGHT_DATA_API_KEY="$BRIGHT_DATA_API_KEY_2"
export BRIGHT_DATA_SERP_ZONES="$BRIGHT_DATA_SERP_ZONES_2"
export BRIGHT_DATA_SERP_ZONE="${BRIGHT_DATA_SERP_ZONES_2%%,*}"
export AGENTFLOW_SFT_PROMPTS="${AGENTFLOW_SFT_PROMPTS:-1}"
export AGENTFLOW_TOP_LOGPROBS="${AGENTFLOW_TOP_LOGPROBS:-5}"
export AGENTFLOW_MAX_STEPS="${AGENTFLOW_MAX_STEPS:-10}"
export AGENTFLOW_MAX_TIME="${AGENTFLOW_MAX_TIME:-600}"
export AGENTFLOW_LLM_TIMEOUT="${AGENTFLOW_LLM_TIMEOUT:-180}"
export CGHE_N="${CGHE_N:-4}"
export enable_thinking="${enable_thinking:-false}"

echo "=== ablation out=$OUT_ROOT model=$MODEL_Name steps=$AGENTFLOW_MAX_STEPS arms=$ARMS_CSV ==="

echo "=== Bing preflight ==="
"$PY" - <<'PY'
import os, sys
from pathlib import Path
ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
sys.path[:0] = [str(ROOT), str(ROOT / "mas"), str(ROOT / "user_space/projects/cghe_pev/adapted")]
from tools_local import Bing_Search_Tool, _use_secondary_serp_account
_use_secondary_serp_account()
out = str(Bing_Search_Tool().execute(query="Jefferson County Kentucky population") or "")
print("search_len", len(out))
print(out[:200])
low = out.lower()
if "search_unavailable" in low or "proxyerror" in low or len(out) < 40:
    raise SystemExit("PREFLIGHT_FAIL")
print("PREFLIGHT_OK")
PY

dataset_path() {
  case "$1" in
    hotpot) echo "$ROOT/data/hotpotqa.json" ;;
    bamboogle) echo "$ROOT/data/bamboogle.json" ;;
    *) echo "unknown dataset $1" >&2; exit 1 ;;
  esac
}

apply_arm_env() {
  local arm="$1"
  unset CGHE_NO_PRUNE CGHE_NO_MEMORY CGHE_FORCE_REGEN_AFTER_HARD_PRUNE || true
  case "$arm" in
    full) ;;
    noprune) export CGHE_NO_PRUNE=1 ;;
    nomemory) export CGHE_NO_MEMORY=1 ;;
    selectonly) export CGHE_NO_PRUNE=1; export CGHE_NO_MEMORY=1 ;;
    forceregen) export CGHE_FORCE_REGEN_AFTER_HARD_PRUNE=1 ;;
    *) echo "unknown arm $arm" >&2; exit 1 ;;
  esac
}

run_one() {
  local arm="$1"
  local ds="$2"
  local arm_dir="$OUT_ROOT/$arm"
  mkdir -p "$arm_dir"
  local tag="${arm}_${ds}"
  local out_jsonl="$arm_dir/${ds}.jsonl"
  local out_log="$arm_dir/${ds}.log"
  local sqlite="$arm_dir/${ds}.sqlite"
  apply_arm_env "$arm"
  echo "launch arm=$arm ds=$ds -> $out_jsonl"
  (
    export CGHE_DATA
    CGHE_DATA="$(dataset_path "$ds")"
    export CGHE_OUT="$out_jsonl"
    export CGHE_SQLITE_PATH="$sqlite"
    export CGHE_RUN_TAG="$tag"
    if [[ -n "$LIMIT" ]]; then
      export CGHE_LIMIT="$LIMIT"
      export CGHE_OFFSET=0
    fi
    export PYTHONPATH="$ROOT:$ROOT/mas:$ADAPTED${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ADAPTED"
    exec "$PY" -u run_cghe_eval.py
  ) >"$out_log" 2>&1 &
  echo $! >"$arm_dir/${ds}.pid"
}

IFS=',' read -r -a ARMS <<< "$ARMS_CSV"
IFS=',' read -r -a DATASETS <<< "$DATASETS_CSV"

for arm in "${ARMS[@]}"; do
  arm="$(echo "$arm" | tr -d '[:space:]')"
  [[ -z "$arm" ]] && continue
  echo "=== ARM $arm ==="
  pids=()
  for ds in "${DATASETS[@]}"; do
    ds="$(echo "$ds" | tr -d '[:space:]')"
    [[ -z "$ds" ]] && continue
    run_one "$arm" "$ds"
    pids+=("$(cat "$OUT_ROOT/$arm/${ds}.pid")")
  done
  set +e
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  set -e
  echo "=== ARM $arm done ==="
done

"$PY" - <<PY
import json
from pathlib import Path
out = Path(r"$OUT_ROOT")
arms = [a.strip() for a in "$ARMS_CSV".split(",") if a.strip()]
datasets = [d.strip() for d in "$DATASETS_CSV".split(",") if d.strip()]
summary = {
    "meta": {
        "out_dir": str(out),
        "model": "$MODEL_Name",
        "api_base": "$OPENAI_API_BASE_URL",
        "serp_zone": "$BRIGHT_DATA_SERP_ZONE",
        "max_steps": int("$AGENTFLOW_MAX_STEPS"),
        "limit": "$LIMIT" or None,
        "arms": arms,
        "datasets": datasets,
    },
    "results": {},
}
for arm in arms:
    summary["results"][arm] = {}
    for ds in datasets:
        path = out / arm / f"{ds}.jsonl"
        rows = []
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        n = len(rows)
        hits = sum(1 for r in rows if r.get("hit"))
        correct = sum(1 for r in rows if r.get("correct"))
        nonempty = sum(1 for r in rows if r.get("nonempty") or str(r.get("answer") or "").strip())
        errors = sum(1 for r in rows if r.get("error"))
        summary["results"][arm][ds] = {
            "n": n,
            "hits": hits,
            "em": hits / max(1, n),
            "correct": correct,
            "accuracy": correct / max(1, n),
            "nonempty": nonempty,
            "nonempty_rate": nonempty / max(1, n),
            "errors": errors,
            "regen_sum": sum(int(r.get("regen_count") or 0) for r in rows),
            "pruned_sum": sum(int(r.get("pruned_plan_count") or 0) for r in rows),
            "llm_sum": sum(int(r.get("llm_call_count") or 0) for r in rows),
            "jsonl": str(path),
        }
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

# Append diagnosis table stub
"$PY" - <<PY
from pathlib import Path
import json
summary = json.loads(Path(r"$OUT_ROOT/summary.json").read_text())
diag = Path(r"$PROJECT/ACCURACY_DIAGNOSIS.md")
lines = ["", "## 消融结果", "", f"目录：\`{summary['meta']['out_dir']}\`", "",
         "| 臂 | 数据集 | n | EM | accuracy | nonempty | pruned | regen |",
         "|---|---|---|---|---|---|---|---|"]
for arm, block in summary["results"].items():
    for ds, row in block.items():
        lines.append(
            f"| {arm} | {ds} | {row['n']} | {row['em']:.3f} | {row['accuracy']:.3f} | "
            f"{row['nonempty_rate']:.3f} | {row['pruned_sum']} | {row['regen_sum']} |"
        )
diag.write_text(diag.read_text() + "\\n".join(lines) + "\\n", encoding="utf-8")
print("updated", diag)
PY

echo "DONE $OUT_ROOT/summary.json"
