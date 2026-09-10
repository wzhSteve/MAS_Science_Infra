# Copyright (c) Microsoft. All rights reserved.

"""用 Agent-lightning + VERL（GRPO）训练 GSM8K 数学 Agent（默认 Qwen3-4B）。

推荐硬件：**2×A800-80GB**（完整超参，不为单卡迁就而砍 batch/序列）。
单卡仅作冒烟或临时回退。

用法:
    python train_math_agent.py a800_2gpu   # 推荐：正式双卡训练
    python train_math_agent.py a800        # 回退：单卡（缩小 batch/显存占用）
    python train_math_agent.py fast        # 冒烟：仅跑 1 个 GRPO step
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

# verl 0.5 + vllm 0.9 的异步 OpenAI server 依赖 V1 引擎；
# 未设置时默认打开，避免 rollout 服务起不来。
os.environ.setdefault("VLLM_USE_V1", "1")

import pandas as pd
from math_agent import LitMathAgent

import agentlightning as agl

# 策略初始化权重路径；可用 CLI --model 覆盖。
DEFAULT_MODEL_PATH = "/root/autodl-tmp/LLM/Qwen3-4B"

# ---------------------------------------------------------------------------
# 正式训练基线配置（面向 2×A800）
#
# 设计原则：
# 1. 单卡不够时优先加卡，而不是先砍 max_prompt/response（否则多轮 tool 轨迹易截断）。
# 2. 结构对齐 VERL Hydra 风格 dict：algorithm / data / actor_rollout_ref / trainer。
# 3. 算法固定 GRPO（无 Critic）；reward 在 LitMathAgent 里用规则给出，此处只配优化器侧。
# ---------------------------------------------------------------------------
RL_TRAINING_CONFIG: Dict[str, Any] = {
    # ---- 算法：优势估计与 KL 是否进 reward ----
    "algorithm": {
        # grpo：同 prompt 采一组响应，用组内相对奖励估 advantage，不需要 value net。
        "adv_estimator": "grpo",
        # False：不把 KL 惩罚揉进标量 reward（更纯的 RLVR 信号）。
        "use_kl_in_reward": False,
    },
    # ---- 数据与序列长度 ----
    "data": {
        "train_files": "data/train.parquet",
        "val_files": "data/val.parquet",
        # 每个训练 step 取多少道题（再 × rollout.n = 实际采样轨迹数上界）。
        "train_batch_size": 32,
        # 多轮 agent + tool 消息会很长；过小会导致 truncation=error 直接失败。
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        # 超长直接报错，避免静默截断污染训练信号。
        "truncation": "error",
    },
    # ---- Actor（训练）/ Rollout（vLLM 采样）/ Ref（参考策略 logprob）----
    "actor_rollout_ref": {
        "rollout": {
            # 4B 模型单卡可放下，TP=1；盲目加大 TP 只会增加通信开销。
            "tensor_model_parallel_size": 1,
            # GRPO group size：同一题采样几条完整轨迹。太小 advantage 噪，太大算力线性涨。
            "n": 4,
            # 重算旧策略 logπ 时的 micro-batch；越大越快越吃显存。
            "log_prob_micro_batch_size_per_gpu": 4,
            # Qwen 系 tool-call 常用 hermes 模板；须与下面 tool_call_parser 一致。
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            # vLLM 预留显存比例。VERL 常把 FSDP 与 vLLM 共置同一 GPU：
            # FSDP 先占一大块后，剩余给 KV cache；比例过高会 OOM，过低则吞吐差。
            # 相对整卡容量；与 FSDP 共置后剩余约 30GB 量级时，0.35 ≈ 28GB 更稳妥。
            "gpu_memory_utilization": 0.35,
            "engine_kwargs": {
                "vllm": {
                    # 允许模型自动选择是否调用工具（Agent 多轮必需）。
                    "enable_auto_tool_choice": True,
                    # 解析 assistant 发出的 tool-call 文本；错 parser → 轨迹坏、reward 噪声大。
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            # 策略更新的 mini-batch；通常与 train_batch_size 对齐或为其因子。
            "ppo_mini_batch_size": 32,
            # 反传 micro-batch：显存不够时优先降这个，而不是先砍学习目标。
            "ppo_micro_batch_size_per_gpu": 4,
            # LLM RL 常用较小 lr，避免一次更新毁掉指令跟随能力。
            "optim": {"lr": 1e-6},
            # 本实验默认关 KL loss：更靠 outcome reward 驱动；风险是格式崩/跑偏。
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            # 0：不额外加熵奖励，探索主要靠采样温度。
            "entropy_coeff": 0,
            # PPO/GRPO clip 带宽（可非对称）：限制 importance ratio 偏离 1 太远。
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                # 参数/优化器状态卸到 CPU，用时间换显存，便于与 vLLM 共卡。
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            # 即使 use_kl_loss=False，框架仍可能保留 ref 路径；开 KL 时必需。
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": DEFAULT_MODEL_PATH,
            # 变长序列去掉无效 padding，省算力。
            "use_remove_padding": True,
            # 激活重计算：用算力换显存，长序列几乎必开。
            "enable_gradient_checkpointing": True,
        },
    },
    # ---- VERL Trainer 进程级配置 ----
    "trainer": {
        "n_gpus_per_node": 2,
        # 训前先打 val 基线，否则无法声称「RL 涨了多少」。
        "val_before_train": True,
        # GRPO 无 Critic，warmup 无意义，保持 0。
        "critic_warmup": 0,
        "logger": ["console", "tensorboard"],
        "project_name": "AgentLightning",
        "experiment_name": "math_gsm",
        "nnodes": 1,
        "test_freq": 16,
        "total_epochs": 2,
        "save_freq": 32,
    },
}


def config_train_fast() -> Dict[str, Any]:
    """构造「冒烟」配置：验证链路能跑通 1 个 GRPO step，不追求效果。

    设计逻辑：
    - 把 batch / group size / 序列长度砍到极小，单卡也能过。
    - total_training_steps=1，跳过训前验证、不存 ckpt，减少等待。
    - experiment_name 带时间戳，避免多次冒烟互相覆盖日志目录。
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 1
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.35
    # group size=2 是 GRPO 能算相对优势的下限附近；仅用于冒烟。
    config["actor_rollout_ref"]["rollout"]["n"] = 2
    config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 1
    config["data"]["train_batch_size"] = 2
    config["data"]["max_prompt_length"] = 2048
    config["data"]["max_response_length"] = 512
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 2
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["test_freq"] = 1
    config["trainer"]["val_before_train"] = False
    config["trainer"]["experiment_name"] = f"math_gsm_fast_{timestamp}"
    config["trainer"].pop("save_freq", None)
    return config


def config_train_a800() -> Dict[str, Any]:
    """单卡 A800 回退配置。

    设计逻辑：
    - 保持完整序列长度与算法目标（仍是 GRPO + 同模型），只缩小
      batch / micro-batch / vLLM 显存占比，让 FSDP 与 vLLM 能共卡。
    - 这不是与 a800_2gpu 对等的实验设置；论文/面试对比应用双卡全量配置。
    """
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 1
    config["data"]["train_batch_size"] = 8
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 8
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.45
    config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 2
    config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 2
    config["trainer"]["experiment_name"] = "math_gsm_a800_1gpu"
    print(
        "WARNING: Using 1x A800 fallback. For full hyperparams without forced "
        "batch/mem downscaling, add a second A800 and run `a800_2gpu` / "
        "scripts/train_2gpu.sh."
    )
    return config


def config_train_a800_2gpu() -> Dict[str, Any]:
    """推荐配置：2×A800，训练 batch/序列长度保持基线。

    注意：``gpu_memory_utilization`` 仍用基线的 0.35——它只限制 vLLM 的
    KV cache 预留，并不等于「训练 batch 被砍了」。VERL 共置 FSDP+vLLM 时
    必须给训练权重留出显存。
    """
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 2
    config["trainer"]["experiment_name"] = "math_gsm_a800_2gpu"
    return config


def train(config: Dict[str, Any], n_runners: int, active_agent: Optional[str]) -> None:
    """组装 Agent-lightning 训练环路并启动 ``Trainer.fit``。

    参数:
        config: VERL 风格超参 dict（由上面三个 config_* 之一生成）。
        n_runners: 并行执行 ``LitMathAgent.rollout`` 的 worker 数。
            越大越能喂饱 GPU 采样，但进程/调度/工具执行开销也越大。
        active_agent: 传给 Adapter 的 ``agent_match``；只训练匹配到的 span
            对应的 agent 片段。默认 None 表示不过滤。

    流程要点:
        1. LitMathAgent：Runner 侧，负责跑 LangGraph + emit_reward。
        2. agl.VERL(config)：Algorithm 侧，管 vLLM / FSDP / GRPO。
        3. Trainer：调度 runners、把 trace 交给 adapter、驱动 fit 循环。
    """
    agent = LitMathAgent()
    # reward_mode 来自 MATH_REWARD_MODE 环境变量（binary|shaped|format_only）。
    print(f"Reward mode: {agent.reward_mode} (override with MATH_REWARD_MODE)")

    from agentlightning.utils.training_metrics_sink import ensure_metrics_env

    project = str(config["trainer"]["project_name"])
    experiment = str(config["trainer"]["experiment_name"])
    local_dir = config["trainer"].get("default_local_dir") or f"checkpoints/{project}/{experiment}"
    metrics_path = ensure_metrics_env(
        project_name=project,
        experiment_name=experiment,
        default_local_dir=str(local_dir),
    )
    tb_dir = os.path.join(str(local_dir), "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    os.environ.setdefault("TENSORBOARD_DIR", os.path.abspath(tb_dir))
    print(f"AGL_METRICS_JSONL={metrics_path}")
    print(f"TENSORBOARD_DIR={os.environ['TENSORBOARD_DIR']}")

    algorithm = agl.VERL(config)
    trainer = agl.Trainer(
        n_runners=n_runners,
        algorithm=algorithm,
        # 仅当指定 active_agent 时才构造 adapter 过滤条件。
        adapter={"agent_match": active_agent} if active_agent else None,
    )
    if active_agent:
        print("Adapter agent match acknowledged:", getattr(trainer.adapter, "agent_match", None))

    train_path = config["data"]["train_files"]
    val_path = config["data"]["val_files"]
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Missing parquet data. Expected {train_path} and {val_path}. "
            "Run: bash scripts/prepare_data.sh"
        )

    # parquet → list[dict]，每条需含 question / answer（及可选 id）。
    train_data = pd.read_parquet(train_path).to_dict(orient="records")
    val_data = pd.read_parquet(val_path).to_dict(orient="records")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"Model: {config['actor_rollout_ref']['model']['path']}")
    print(f"GPUs per node: {config['trainer']['n_gpus_per_node']}")
    print(f"train_batch_size: {config['data']['train_batch_size']}")
    print(f"gpu_memory_utilization: {config['actor_rollout_ref']['rollout']['gpu_memory_utilization']}")
    # fit 内部：采样 → reward → adapter 转 triplet → GRPO 更新 → 权重回灌 vLLM。
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore[arg-type]


def main() -> None:
    """CLI 入口：选择配置变体，解析覆盖项，再调用 ``train``。"""
    parser = argparse.ArgumentParser(description="Train math GSM agent with VERL")
    parser.add_argument(
        "config",
        choices=["fast", "a800", "a800_2gpu"],
        help="Training configuration (recommended: a800_2gpu)",
    )
    parser.add_argument(
        "--active-agent",
        type=str,
        default=None,
        help="Optional adapter agent_match（只训匹配到的 agent span）",
    )
    parser.add_argument(
        "--n-runners",
        type=int,
        default=None,
        help="覆盖并行 rollout runner 数量 "
        "(default: 2 for fast, 4 for a800, 8 for a800_2gpu)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"覆盖模型路径 (default: {DEFAULT_MODEL_PATH})",
    )
    args = parser.parse_args()

    # 配置名 → 工厂函数；统一 deepcopy 基线再改差异项，避免互相污染。
    config_fns = {
        "fast": config_train_fast,
        "a800": config_train_a800,
        "a800_2gpu": config_train_a800_2gpu,
    }
    config = config_fns[args.config]()
    if args.model:
        config["actor_rollout_ref"]["model"]["path"] = args.model

    # runner 默认值按配置「体量」递增：题多/卡多时需要更多并行 agent 进程喂数据。
    if args.n_runners is not None:
        n_runners = args.n_runners
    elif args.config == "fast":
        n_runners = 2
    elif args.config == "a800_2gpu":
        n_runners = 8
    else:
        n_runners = 4

    print(f"Starting training with '{args.config}' configuration...")
    train(config, n_runners=n_runners, active_agent=args.active_agent)


if __name__ == "__main__":
    main()
