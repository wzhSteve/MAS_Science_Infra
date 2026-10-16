#!/usr/bin/env bash
# Quick helpers for the running agentflow_4algo job.
# Usage:
#   bash scripts/agentflow_4algo_watch.sh dash     # start HTTP dashboard :8765
#   bash scripts/agentflow_4algo_watch.sh cli      # rich TUI in terminal
#   bash scripts/agentflow_4algo_watch.sh tail     # follow current algo log
#   bash scripts/agentflow_4algo_watch.sh metrics  # watch metrics.jsonl grow
set -euo pipefail
ROOT=/root/autodl-tmp/MAS_Science_Infra
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
LOG_DIR="$ROOT/logs/agentflow_4algo_10step"
CKPT="$ROOT/checkpoints/AgentLightning"
ALGOS=(grpo arpo appo flow_grpo)

current_algo() {
  for a in "${ALGOS[@]}"; do
    if pgrep -af "rl_${a}.yaml" >/dev/null 2>&1; then
      echo "$a"
      return
    fi
  done
  # newest log
  ls -t "$LOG_DIR"/{grpo,arpo,appo,flow_grpo}.log 2>/dev/null | head -1 | xargs -I{} basename {} .log
}

cmd=${1:-dash}
case "$cmd" in
  dash|dashboard|http)
    exec python scripts/agentflow_4algo_monitor.py --host 0.0.0.0 --port "${PORT:-8765}"
    ;;
  cli|tui)
    exec python scripts/agentflow_4algo_monitor.py --cli
    ;;
  once)
    exec python scripts/agentflow_4algo_monitor.py --once
    ;;
  tail|log)
    a=${2:-$(current_algo)}
    echo ">>> following $LOG_DIR/${a}.log"
    exec tail -F "$LOG_DIR/${a}.log"
    ;;
  metrics|m)
    a=${2:-$(current_algo)}
    f="$CKPT/agentflow_4algo_${a}_10step_${a}/metrics.jsonl"
    echo ">>> watching $f"
    while true; do
      clear
      date
      echo "file: $f"
      if [[ -f "$f" ]]; then
        python - <<PY
import json
from pathlib import Path
p=Path("$f")
keys=["step","training/reward","val/reward","actor/pg_loss","actor/ppo_kl","actor/entropy_loss","training/n_triplets","training/n_rollouts"]
rows=[]
for line in p.read_text().splitlines():
    if line.strip():
        rows.append(json.loads(line))
print(f"rows={len(rows)}")
print("\t".join(keys))
for r in rows:
    print("\t".join(str(r.get(k,"—")) if not isinstance(r.get(k),float) else f"{r.get(k):.4f}" for k in keys))
PY
      else
        echo "(not created yet)"
      fi
      sleep 3
    done
    ;;
  *)
    echo "usage: $0 {dash|cli|tail [algo]|metrics [algo]|once}"
    exit 1
    ;;
esac
