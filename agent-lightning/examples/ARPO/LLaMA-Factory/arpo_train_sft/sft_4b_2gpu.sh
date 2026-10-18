#!/bin/bash
# ARPO 冷启动 SFT · Qwen3-4B · 双卡 A800-80GB
#
# 用法（在任意目录）：
#   bash .../arpo_train_sft/sft_4b_2gpu.sh print
#   bash .../arpo_train_sft/sft_4b_2gpu.sh mask
#   bash .../arpo_train_sft/sft_4b_2gpu.sh smoke
#   bash .../arpo_train_sft/sft_4b_2gpu.sh full
#   bash .../arpo_train_sft/sft_4b_2gpu.sh bg     # nohup 正式训练 + 看板 :8790
#
# template=qwen3 + enable_thinking=true，对 <think> 计 loss。
# 官方 8 卡是 batch 1 × grad_accum 2 = 全局 batch 16。
# 两张卡用 batch 1 × grad_accum 8，全局 batch 仍是 16。
# 权重写入 checkpoints/arpo_sft_4b，full 结束再复制到 LLM/ARPO/Qwen3-4B-SFT。
# 不改 LLM/Qwen3-4B。
# 说明：docs/ARPO_Qwen3-4B_SFT_双卡A800.md
set -euo pipefail

MODE="${1:-full}"

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${_HERE}"
while [ ! -d "${REPO_ROOT}/LLM" ] || [ ! -d "${REPO_ROOT}/agent-lightning" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
    if [ "${REPO_ROOT}" = "/" ]; then
        echo "找不到仓库根（需要同时存在 LLM/ 与 agent-lightning/）" >&2
        exit 1
    fi
done

SFT_ENV="${SFT_ENV:-/root/autodl-tmp/envs/arpo_sft}"
PYTHON="${SFT_ENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "找不到 SFT 环境 ${PYTHON}。先创建 conda 环境 arpo_sft。" >&2
    exit 1
fi

LLAMA_DIR="${REPO_ROOT}/agent-lightning/examples/ARPO/LLaMA-Factory"
TRAIN_YAML="${_HERE}/yaml/qwen3_4b_2xa800.yaml"
DATA_JSONL="${_HERE}/dataset/final_5w4_still.jsonl"
MODEL_PATH="${REPO_ROOT}/LLM/Qwen3-4B"
OUT_FULL="${REPO_ROOT}/checkpoints/arpo_sft_4b"
TRAIN_PID_FILE="${OUT_FULL}/train.pid"
DASHBOARD_PID_FILE="${OUT_FULL}/dashboard.pid"
DASHBOARD_PORT="${DASHBOARD_PORT:-8790}"
OUT_SMOKE="${REPO_ROOT}/checkpoints/arpo_sft_4b_smoke"
FINAL_EXPORT="${REPO_ROOT}/LLM/ARPO/Qwen3-4B-SFT"
# 默认 ZeRO-2，便于训练中 LiveGen；可用 DS_CONFIG=.../ds_z3_config.json 覆盖
DS_CONFIG="${DS_CONFIG:-${LLAMA_DIR}/examples/deepspeed/ds_z2_config.json}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/pip-cache}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/cache}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/root/autodl-tmp/cache/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/root/autodl-tmp/cache/torch_extensions}"
# 训练中 LiveGen：周期性 model.generate → output_dir/live_generation_latest.json
# EVERY 默认在 smoke/full 分支里分别设，避免 smoke 被 full 默认值盖住
export ARPO_LIVE_GEN="${ARPO_LIVE_GEN:-1}"
export ARPO_LIVE_GEN_DATA="${ARPO_LIVE_GEN_DATA:-${DATA_JSONL}}"
export ARPO_LIVE_GEN_MAX_NEW="${ARPO_LIVE_GEN_MAX_NEW:-512}"
export ARPO_LIVE_GEN_N_SAMPLES="${ARPO_LIVE_GEN_N_SAMPLES:-2}"
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

if [ ! -f "${DATA_JSONL}" ]; then
    echo "缺少数据 ${DATA_JSONL}" >&2
    exit 1
fi
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "缺少模型 ${MODEL_PATH}" >&2
    exit 1
fi
if [ ! -f "${DS_CONFIG}" ]; then
    echo "缺少 DeepSpeed 配置 ${DS_CONFIG}" >&2
    exit 1
fi

echo "mode=${MODE} gpus=${CUDA_VISIBLE_DEVICES} model=${MODEL_PATH}"
echo "yaml=${TRAIN_YAML}"
echo "data=${DATA_JSONL}"

case "${MODE}" in
    print)
        "${PYTHON}" - << PY
from omegaconf import OmegaConf
cfg = OmegaConf.load("${TRAIN_YAML}")
print(OmegaConf.to_yaml(cfg))
print("---")
print("global_batch = n_gpu * per_device_train_batch_size * gradient_accumulation_steps")
print("expected global_batch = 2 * 1 * 8 = 16")
print("template =", cfg.get("template"), "enable_thinking =", cfg.get("enable_thinking"))
PY
        ;;
    mask)
        CUDA_VISIBLE_DEVICES="" "${PYTHON}" "${_HERE}/check_loss_mask.py" \
            --model "${MODEL_PATH}" \
            --data "${DATA_JSONL}" \
            --template qwen3 \
            --enable-thinking
        ;;
    bg)
        mkdir -p "${OUT_FULL}"
        if [ -f "${TRAIN_PID_FILE}" ]; then
            OLD_PID="$(cat "${TRAIN_PID_FILE}")"
            if kill -0 "${OLD_PID}" 2>/dev/null; then
                echo "训练已在运行 pid=${OLD_PID}，日志 ${OUT_FULL}/training.log" >&2
                exit 1
            fi
        fi
        LAUNCH_LOG="${OUT_FULL}/launcher.log"
        setsid nohup bash "${_HERE}/sft_4b_2gpu.sh" full >"${LAUNCH_LOG}" 2>&1 &
        echo $! > "${TRAIN_PID_FILE}"
        echo "train_pid=$(cat "${TRAIN_PID_FILE}")"
        echo "log=${OUT_FULL}/training.log"
        if [ -f "${DASHBOARD_PID_FILE}" ] && kill -0 "$(cat "${DASHBOARD_PID_FILE}")" 2>/dev/null; then
            echo "dashboard 已在运行 http://127.0.0.1:${DASHBOARD_PORT}"
        else
            setsid nohup "${PYTHON}" "${_HERE}/sft_dashboard.py" \
                --log "${OUT_FULL}/training.log" \
                --pid-file "${TRAIN_PID_FILE}" \
                --port "${DASHBOARD_PORT}" \
                --output-dir "${OUT_FULL}" \
                >"${OUT_FULL}/dashboard.log" 2>&1 &
            echo $! > "${DASHBOARD_PID_FILE}"
            echo "dashboard_pid=$(cat "${DASHBOARD_PID_FILE}")"
            echo "dashboard http://127.0.0.1:${DASHBOARD_PORT}"
        fi
        ;;
    smoke|full)
        if [ "${MODE}" = "smoke" ]; then
            OUT_DIR="${OUT_SMOKE}"
            EXTRA=(
                max_samples=64
                max_steps=20
                save_steps=1000
                overwrite_output_dir=true
                preprocessing_num_workers=4
            )
            export ARPO_LIVE_GEN_EVERY="${ARPO_LIVE_GEN_EVERY:-5}"
        else
            OUT_DIR="${OUT_FULL}"
            EXTRA=()
            export ARPO_LIVE_GEN_EVERY="${ARPO_LIVE_GEN_EVERY:-20}"
        fi
        mkdir -p "${OUT_DIR}"
        LOG="${OUT_DIR}/training.log"
        echo "output=${OUT_DIR}"
        echo "log=${LOG}"
        echo "live_gen every=${ARPO_LIVE_GEN_EVERY} max_new=${ARPO_LIVE_GEN_MAX_NEW} n=${ARPO_LIVE_GEN_N_SAMPLES}"
        cd "${_HERE}"
        export PYTHONPATH="${_HERE}:${PYTHONPATH:-}"
        "${SFT_ENV}/bin/torchrun" \
            --nnodes 1 \
            --node_rank 0 \
            --nproc_per_node 2 \
            --master_addr 127.0.0.1 \
            --master_port "${MASTER_PORT:-29501}" \
            sft_launcher_with_live_gen.py \
            "${TRAIN_YAML}" \
            "model_name_or_path=${MODEL_PATH}" \
            "dataset_dir=${_HERE}/dataset_info" \
            "deepspeed=${DS_CONFIG}" \
            "output_dir=${OUT_DIR}" \
            "${EXTRA[@]}" \
            2>&1 | tee "${LOG}"
        if [ "${MODE}" = "full" ]; then
            SRC="$("${PYTHON}" - << PY
from pathlib import Path
out = Path("${OUT_DIR}")
ckpts = sorted(out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
if ckpts and (ckpts[-1] / "config.json").is_file():
    print(ckpts[-1])
elif (out / "config.json").is_file():
    print(out)
else:
    raise SystemExit(f"full 结束但 {out} 里没有可导出的 HF 权重")
PY
)"
            mkdir -p "$(dirname "${FINAL_EXPORT}")"
            rm -rf "${FINAL_EXPORT}"
            cp -a "${SRC}/." "${FINAL_EXPORT}/"
            echo "exported ${SRC} -> ${FINAL_EXPORT}"
        fi
        ;;
    *)
        echo "未知模式 ${MODE}。可用 print | mask | smoke | full | bg" >&2
        exit 1
        ;;
esac
