#!/usr/bin/env bash
# Full CGHE on HotpotQA (100) + Bamboogle (125) in parallel.
# Pins LLM + Bright Data account 2 from repo .env; aborts if preflight fails.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
ADAPTED="$ROOT/user_space/projects/cghe_pev/adapted"
PROJECT="$ROOT/user_space/projects/cghe_pev"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

cd "$ROOT"
ENV_EXPORT="$ROOT/user_space/projects/cghe_pev/artifacts/.full_eval_env_export.sh"
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
# Remap secondary SERP into the keys the shared Bing client reads.
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

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${CGHE_FULL_OUT_DIR:-$PROJECT/artifacts/full_eval_${TS}}"
mkdir -p "$OUT_DIR"

mask() {
  local s="$1"
  local n=${#s}
  if (( n <= 8 )); then echo "***"; else echo "${s:0:4}…${s: -4}"; fi
}

echo "=== CGHE full parallel eval ==="
echo "out_dir=$OUT_DIR"
echo "model=$MODEL_Name"
echo "api_base=$OPENAI_API_BASE_URL"
echo "api_key=$(mask "$OPENAI_API_KEY")"
echo "serp_zone=$BRIGHT_DATA_SERP_ZONE"
echo "serp_key2=$(mask "$BRIGHT_DATA_API_KEY_2")"

echo "=== Bing preflight (account 2) ==="
"$PY" - <<'PY'
import os, sys
from pathlib import Path
ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
sys.path[:0] = [str(ROOT), str(ROOT / "mas"), str(ROOT / "user_space/projects/cghe_pev/adapted")]
from tools_local import Bing_Search_Tool, _use_secondary_serp_account
_use_secondary_serp_account()
assert os.environ.get("BRIGHT_DATA_API_KEY") == os.environ.get("BRIGHT_DATA_API_KEY_2")
assert os.environ.get("BRIGHT_DATA_SERP_ZONES") == os.environ.get("BRIGHT_DATA_SERP_ZONES_2")
out = str(Bing_Search_Tool().execute(query="Jefferson County Kentucky population") or "")
print("search_len", len(out))
print(out[:240])
low = out.lower()
if "search_unavailable" in low or "proxyerror" in low or len(out) < 40:
    raise SystemExit("PREFLIGHT_FAIL")
print("PREFLIGHT_OK")
PY

run_one() {
  local tag="$1"
  local data="$2"
  local out_jsonl="$OUT_DIR/${tag}.jsonl"
  local out_log="$OUT_DIR/${tag}.log"
  local sqlite="$OUT_DIR/${tag}.sqlite"
  echo "launch $tag -> $out_jsonl"
  (
    export CGHE_DATA="$data"
    export CGHE_OUT="$out_jsonl"
    export CGHE_SQLITE_PATH="$sqlite"
    export CGHE_RUN_TAG="$tag"
    export PYTHONPATH="$ROOT:$ROOT/mas:$ADAPTED${PYTHONPATH:+:$PYTHONPATH}"
    cd "$ADAPTED"
    exec "$PY" -u run_cghe_eval.py
  ) >"$out_log" 2>&1 &
  echo $! >"$OUT_DIR/${tag}.pid"
}

run_one hotpot "$ROOT/data/hotpotqa.json"
run_one bamboogle "$ROOT/data/bamboogle.json"

HPID=$(cat "$OUT_DIR/hotpot.pid")
BPID=$(cat "$OUT_DIR/bamboogle.pid")
echo "pids hotpot=$HPID bamboogle=$BPID"
echo "$OUT_DIR" >"$OUT_DIR/OUT_DIR.txt"

set +e
wait "$HPID"
HRC=$?
wait "$BPID"
BRC=$?
set -e

"$PY" - <<PY
import json
from pathlib import Path
out = Path("$OUT_DIR")
meta = {
    "out_dir": str(out),
    "model": "$MODEL_Name",
    "api_base": "$OPENAI_API_BASE_URL",
    "serp_zone": "$BRIGHT_DATA_SERP_ZONE",
    "hotpot_exit": $HRC,
    "bamboogle_exit": $BRC,
}
summary = {"meta": meta, "datasets": {}}
for tag in ("hotpot", "bamboogle"):
    path = out / f"{tag}.jsonl"
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    hits = sum(1 for r in rows if r.get("hit"))
    errors = sum(1 for r in rows if r.get("error"))
    regen = sum(int(r.get("regen_count") or 0) for r in rows)
    pruned = sum(int(r.get("pruned_plan_count") or 0) for r in rows)
    llm = sum(int(r.get("llm_call_count") or 0) for r in rows)
    n = len(rows)
    summary["datasets"][tag] = {
        "n": n,
        "hits": hits,
        "em": hits / max(1, n),
        "errors": errors,
        "regen_sum": regen,
        "pruned_sum": pruned,
        "llm_sum": llm,
        "jsonl": str(path),
        "log": str(out / f"{tag}.log"),
    }
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

if [[ $HRC -ne 0 || $BRC -ne 0 ]]; then
  echo "FATAL: hotpot_exit=$HRC bamboogle_exit=$BRC" >&2
  exit 1
fi
echo "DONE $OUT_DIR/summary.json"
