"""Train MAS_structagent with Agent-lightning + VERL (GRPO).

CLI:
    python train_mas_agent.py fast
    python train_mas_agent.py a800_2gpu --model /root/autodl-tmp/LLM/Qwen3-4B

LLM calls go through VERL ``main_llm`` (local Qwen3-4B by default), not
``tir_agent/.env`` remote API. Hermes tool-call parsing is disabled because
StructAgent uses JSON chat, not Hermes tool calls.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("MAS_OPENAI_COMPATIBLE", "1")

DEFAULT_MODEL_PATH = "/root/autodl-tmp/LLM/Qwen3-4B"

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "data/train.parquet",
        "val_files": "data/val.parquet",
        "train_batch_size": 32,
        # Qwen3-4B max_position_embeddings=40960; MAS tool dumps often ~18–20k msg tokens.
        # VERL max_model_len ≈ prompt + response → 28672+4096=32768.
        "max_prompt_length": 28672,
        "max_response_length": 4096,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 2,
            "name": "vllm",
            # Longer context (32k) needs more KV; optimizer_offload frees Adam GPU mem.
            "gpu_memory_utilization": 0.45,
            # StructAgent is plain chat JSON — do not enable Hermes tool parser.
            "engine_kwargs": {"vllm": {}},
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            # Longer sequences: keep micro-batch modest to avoid train-step OOM.
            "ppo_micro_batch_size_per_gpu": 2,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                # Params stay on GPU for speed; Adam (m/v) on CPU to free KV budget.
                "param_offload": False,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 4,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": DEFAULT_MODEL_PATH,
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 2,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "tensorboard"],
        "project_name": "AgentLightning",
        "experiment_name": "mas_structagent",
        "nnodes": 1,
        "test_freq": 16,
        "total_epochs": 2,
        "save_freq": 32,
    },
}


def config_train_fast() -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 1
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.35
    config["actor_rollout_ref"]["rollout"]["n"] = 2
    config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 1
    config["data"]["train_batch_size"] = 2
    # Keep enough context for MAS even in fast mode (avoid 5120-style blowups).
    config["data"]["max_prompt_length"] = 8192
    config["data"]["max_response_length"] = 2048
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 2
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["test_freq"] = 1
    config["trainer"]["val_before_train"] = False
    config["trainer"]["experiment_name"] = f"mas_structagent_fast_{timestamp}"
    config["trainer"].pop("save_freq", None)
    return config


def config_train_a800() -> Dict[str, Any]:
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 1
    config["data"]["train_batch_size"] = 8
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 8
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    config["actor_rollout_ref"]["rollout"]["gpu_memory_utilization"] = 0.45
    config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 2
    config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 2
    config["trainer"]["experiment_name"] = "mas_structagent_a800_1gpu"
    print(
        "WARNING: Using 1x A800 fallback. For full hyperparams without forced "
        "batch/mem downscaling, add a second A800 and run `a800_2gpu`."
    )
    return config


def config_train_a800_2gpu() -> Dict[str, Any]:
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 2
    config["trainer"]["experiment_name"] = "mas_structagent_a800_2gpu"
    return config


def config_train_a800_2gpu_smoke() -> Dict[str, Any]:
    """2-GPU smoke: one train step to catch startup / rollout / update errors."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 2
    # Keep base long context (16384+4096); only shrink batch / n for a quick step.
    config["actor_rollout_ref"]["rollout"]["n"] = 2
    config["actor_rollout_ref"]["rollout"]["log_prob_micro_batch_size_per_gpu"] = 1
    config["data"]["train_batch_size"] = 4
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 4
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["test_freq"] = 1
    config["trainer"]["val_before_train"] = False
    config["trainer"]["experiment_name"] = f"mas_structagent_a800_2gpu_smoke_{timestamp}"
    config["trainer"].pop("save_freq", None)
    return config


def train(config: Dict[str, Any], n_runners: int, active_agent: Optional[str], max_steps: int) -> None:
    import pandas as pd
    import agentlightning as agl
    from mas_agent import LitMASAgent, bootstrap_mas_package

    # Tool search proxy only — do NOT set process-wide HTTP_PROXY (breaks VERL LLM).
    proxy = (
        os.environ.get("GOOGLE_CHROME_PROXY")
        or os.environ.get("WEB_SEARCH_PROXY")
        or "http://127.0.0.1:7890"
    )
    os.environ.setdefault("GOOGLE_CHROME_PROXY", proxy)
    os.environ.setdefault("WEB_SEARCH_PROXY", proxy)
    os.environ["MAS_NO_PROCESS_PROXY"] = "1"
    # Clear accidental process proxies from the shell so OpenAI→LiteLLM stays local.
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(k, None)
    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [p.strip() for p in no_proxy.split(",") if p.strip()]
    for host in ("127.0.0.1", "localhost", "0.0.0.0"):
        if host not in parts:
            parts.append(host)
    os.environ["NO_PROXY"] = ",".join(parts)
    os.environ["no_proxy"] = ",".join(parts)
    # Align MAS completion cap with VERL max_response_length.
    max_resp = int(config.get("data", {}).get("max_response_length") or 2048)
    os.environ["MAS_MAX_COMPLETION_TOKENS"] = str(max_resp)

    bootstrap_mas_package()
    agent = LitMASAgent(max_steps=max_steps, max_tokens=max_resp)

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

    train_data = pd.read_parquet(train_path).to_dict(orient="records")
    val_data = pd.read_parquet(val_path).to_dict(orient="records")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"Model: {config['actor_rollout_ref']['model']['path']}")
    print(f"GPUs per node: {config['trainer']['n_gpus_per_node']}")
    print(f"train_batch_size: {config['data']['train_batch_size']}")
    print(f"gpu_memory_utilization: {config['actor_rollout_ref']['rollout']['gpu_memory_utilization']}")
    print(f"MAS max_steps: {max_steps}")
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore[arg-type]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MAS_structagent with VERL (GRPO)")
    parser.add_argument("config", choices=["fast", "a800", "a800_2gpu", "a800_2gpu_smoke"])
    parser.add_argument("--active-agent", type=str, default=None)
    parser.add_argument("--n-runners", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=8, help="StructAgent max planner/actor steps")
    parser.add_argument("--model", type=str, default=None, help=f"Override model path (default: {DEFAULT_MODEL_PATH})")
    args = parser.parse_args()

    config_fns = {
        "fast": config_train_fast,
        "a800": config_train_a800,
        "a800_2gpu": config_train_a800_2gpu,
        "a800_2gpu_smoke": config_train_a800_2gpu_smoke,
    }
    config = config_fns[args.config]()
    if args.model:
        config["actor_rollout_ref"]["model"]["path"] = args.model

    if args.n_runners is not None:
        n_runners = args.n_runners
    elif args.config == "fast":
        n_runners = 2
    elif args.config == "a800_2gpu_smoke":
        n_runners = 4
    elif args.config == "a800_2gpu":
        n_runners = 8
    else:
        n_runners = 4

    # Smoke defaults to fewer StructAgent steps unless overridden.
    max_steps = args.max_steps
    if args.config == "a800_2gpu_smoke" and args.max_steps == 8:
        max_steps = 6

    print(f"Starting MAS training with '{args.config}' ...")
    train(config, n_runners=n_runners, active_agent=args.active_agent, max_steps=max_steps)


if __name__ == "__main__":
    main()
