#!/usr/bin/env bash
# Default: full ablation on bamboogle pid [53:63)
#
# Usage:
#   ./run.sh                              # evaluation_mode=on  (不写回 offline JSON)
#   ./run.sh full                         # 同上，指定 ablation preset
#   ./run.sh --no_evaluation_mode         # evolve + 写回 tool_*_memory.json
#   ./run.sh full --no_evaluation_mode    # preset + 写回磁盘
#   ./run.sh full --evaluation_mode       # 显式评测模式（默认行为）
#
# Extra flags after the optional preset are forwarded to inference_log.py.
set -euo pipefail

PRESET="full"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  PRESET="${1}"
  shift
fi

# macOS bash 3.2 + set -u: empty "${arr[@]}" is unbound; expand only when set.
python inference_log.py \
  --test_file test/medqa/data/data.json \
  --start_pid 279 --end_pid -1 \
  --n 1 --max_steps 10 \
  --dir_name logs \
  --ablation "${PRESET}" \
  --offline_memory_dir memory \
  ${1+"$@"}
