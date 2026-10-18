#!/bin/bash
# ARPO Qwen3-1.7B-SFT 格式烟测界面
#   bash .../sft_eval_ui.sh           # 前台，:8789
#   bash .../sft_eval_ui.sh bg        # 后台
#   bash .../sft_eval_ui.sh stop
set -euo pipefail

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${_HERE}"
while [ ! -d "${REPO_ROOT}/LLM" ] || [ ! -d "${REPO_ROOT}/agent-lightning" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
    if [ "${REPO_ROOT}" = "/" ]; then
        echo "找不到仓库根" >&2
        exit 1
    fi
done

SFT_ENV="${SFT_ENV:-/root/autodl-tmp/envs/arpo_sft}"
PYTHON="${SFT_ENV}/bin/python"
PORT="${PORT:-8789}"
MODE="${1:-fg}"
OUT_DIR="${REPO_ROOT}/checkpoints/arpo_sft_1.7b"
PID_FILE="${OUT_DIR}/eval_ui.pid"
LOG_FILE="${OUT_DIR}/eval_ui.log"
MODEL="${MODEL:-${REPO_ROOT}/LLM/ARPO/Qwen3-1.7B-SFT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

if [ ! -x "${PYTHON}" ]; then
    echo "找不到 ${PYTHON}" >&2
    exit 1
fi
if [ ! -f "${MODEL}/config.json" ]; then
    echo "找不到模型 ${MODEL}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"

case "${MODE}" in
    stop)
        if [ -f "${PID_FILE}" ]; then
            PID="$(cat "${PID_FILE}")"
            if kill -0 "${PID}" 2>/dev/null; then
                kill "${PID}" 2>/dev/null || true
                sleep 1
                kill -9 "${PID}" 2>/dev/null || true
                echo "已停止 pid=${PID}"
            else
                echo "进程未在运行"
            fi
            rm -f "${PID_FILE}"
        else
            echo "无 pid 文件"
        fi
        ;;
    bg)
        if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
            echo "已在运行 pid=$(cat "${PID_FILE}")  http://127.0.0.1:${PORT}" >&2
            exit 1
        fi
        setsid nohup "${PYTHON}" "${_HERE}/sft_eval_ui.py" \
            --model "${MODEL}" \
            --port "${PORT}" \
            --preload \
            >"${LOG_FILE}" 2>&1 &
        echo $! > "${PID_FILE}"
        echo "pid=$(cat "${PID_FILE}")"
        echo "log=${LOG_FILE}"
        echo "http://127.0.0.1:${PORT}"
        echo "远程: ssh -L ${PORT}:127.0.0.1:${PORT} <host>"
        ;;
    fg|*)
        echo "model=${MODEL}"
        echo "http://127.0.0.1:${PORT}"
        exec "${PYTHON}" "${_HERE}/sft_eval_ui.py" \
            --model "${MODEL}" \
            --port "${PORT}" \
            --preload
        ;;
esac
