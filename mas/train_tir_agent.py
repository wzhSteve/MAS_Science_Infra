"""Train the TIR agent with Agent-lightning + VERL.

CLI:
    python train_tir_agent.py a800_2gpu --algo grpo
    python train_tir_agent.py a800_2gpu --algo igpo
    python train_tir_agent.py fast --algo arpo

`algorithm.adv_estimator` is always `grpo` (no Critic). The real algorithm name
lives in `algorithm.tir_algo` and is implemented by TirAgentLightningTrainer /
TirAgentModeDaemon.
"""

from __future__ import annotations

import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

os.environ.setdefault("VLLM_USE_V1", "1")

from rl.hooks.overlay import VALID_ALGOS, apply_sample_policy, apply_train_signal

DEFAULT_MODEL_PATH = "/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B"
# mas/ → MAS_Science_Infra
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = REPO_ROOT / "data"


def _resolve_parquet_path(path: str, default_name: str) -> str:
    """Prefer explicit paths; map relative ``data/*.parquet`` to repo ``data/``."""
    p = Path(str(path)).expanduser()
    if p.is_file():
        return str(p.resolve())
    if not p.is_absolute():
        cand = REPO_DATA_DIR / (p.name if p.name.endswith(".parquet") else default_name)
        if cand.is_file():
            return str(cand.resolve())
        # Still return canonical repo path so error messages point at the right place
        return str(cand)
    return str(p)


RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
        "tir_algo": "grpo",
    },
    "data": {
        "train_files": str(REPO_DATA_DIR / "train.parquet"),
        "val_files": str(REPO_DATA_DIR / "val.parquet"),
        "train_batch_size": 32,
        "max_prompt_length": 8192,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.35,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "ppo_micro_batch_size_per_gpu": 4,
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
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
        "experiment_name": "tir_agent",
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
    config["data"]["max_prompt_length"] = 2048
    config["data"]["max_response_length"] = 512
    config["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 2
    config["actor_rollout_ref"]["actor"]["ppo_micro_batch_size_per_gpu"] = 1
    config["actor_rollout_ref"]["ref"]["log_prob_micro_batch_size_per_gpu"] = 1
    config["trainer"]["total_epochs"] = 1
    config["trainer"]["total_training_steps"] = 1
    config["trainer"]["test_freq"] = 1
    config["trainer"]["val_before_train"] = False
    config["trainer"]["experiment_name"] = f"tir_agent_fast_{timestamp}"
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
    config["trainer"]["experiment_name"] = "tir_agent_a800_1gpu"
    print(
        "WARNING: Using 1x A800 fallback. For full hyperparams without forced "
        "batch/mem downscaling, add a second A800 and run `a800_2gpu`."
    )
    return config


def config_train_a800_2gpu() -> Dict[str, Any]:
    config = deepcopy(RL_TRAINING_CONFIG)
    config["trainer"]["n_gpus_per_node"] = 2
    config["trainer"]["experiment_name"] = "tir_agent_a800_2gpu"
    return config


def train(config: Dict[str, Any], n_runners: int, active_agent: Optional[str]) -> None:
    import pandas as pd
    from tir_agent import LitTirAgent

    import agentlightning as agl

    data = config.get("data") or {}
    max_prompt = int(data.get("max_prompt_length") or 2048)
    max_resp = int(data.get("max_response_length") or 512)
    max_model_len = max_prompt + max_resp
    os.environ["TIR_MAX_MODEL_LEN"] = str(max_model_len)
    print(
        f"TIR context budget: max_model_len={max_model_len} "
        f"(prompt={max_prompt}+response={max_resp}) max_tokens={max_resp}"
    )
    agent = LitTirAgent(max_tokens=max_resp, max_model_len=max_model_len)
    tir_algo = str(config.get("algorithm", {}).get("tir_algo") or "grpo")
    tir_cfg = dict(config.get("algorithm", {}).get("tir") or {})
    os.environ["TIR_ALGO"] = tir_algo
    print(f"tir_algo={tir_algo} (Hydra adv_estimator stays grpo)")

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

    if tir_algo == "grpo":
        algorithm = agl.VERL(config)
    else:
        from rl.hooks.trainer import TirAgentLightningTrainer, bound_daemon_cls

        algorithm = agl.VERL(
            config,
            trainer_cls=TirAgentLightningTrainer,
            daemon_cls=bound_daemon_cls(tir_algo, tir_cfg),
        )
    # MAS agent names (e.g. "hub") live in a different namespace from the langgraph
    # span agent names (langchain.chain.type = langgraph node names: "agent", "tools",
    # "should_continue", "finalize", "react"). TracerTraceToTriplet.agent_name() reads
    # the langgraph namespace, so map the MAS trainable agent onto the langgraph node
    # that owns all LLM calls (the "agent" node), otherwise agent_match filters out
    # every LLM span and the training batch ends up empty.
    agent_match = "agent" if active_agent else None
    trainer = agl.Trainer(
        n_runners=n_runners,
        algorithm=algorithm,
        adapter={"agent_match": agent_match} if agent_match else None,
    )
    if active_agent:
        print(
            f"Adapter agent match: {agent_match!r} (MAS agent {active_agent!r} -> langgraph node)"
        )

    train_path = _resolve_parquet_path(config["data"]["train_files"], "train.parquet")
    val_path = _resolve_parquet_path(config["data"]["val_files"], "val.parquet")
    config["data"]["train_files"] = train_path
    config["data"]["val_files"] = val_path
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Missing parquet data. Expected {train_path} and {val_path} "
            f"(canonical: MAS_Science_Infra/data/). "
            "Run: bash scripts/prepare_data.sh from tir_agent, or place files under repo data/."
        )

    train_data = pd.read_parquet(train_path).to_dict(orient="records")
    val_data = pd.read_parquet(val_path).to_dict(orient="records")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"Model: {config['actor_rollout_ref']['model']['path']}")
    print(f"GPUs per node: {config['trainer']['n_gpus_per_node']}")
    print(f"train_batch_size: {config['data']['train_batch_size']}")
    print(f"gpu_memory_utilization: {config['actor_rollout_ref']['rollout']['gpu_memory_utilization']}")
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore[arg-type]


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    skip = {"profile", "algo", "n_runners", "model_path", "devices", "rollout_per_gpu"}
    for k, v in overlay.items():
        if k in skip:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def load_rl_yaml(path: str) -> Dict[str, Any]:
    import yaml  # type: ignore

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"rl yaml must be a mapping: {path}")
    return raw


def _load_sibling_workflow_sampling(rl_path: str) -> Any:
    """If experiments/<id>/rl.yaml has a sibling workflow.yaml, return its sampling block."""
    import yaml  # type: ignore

    wf = Path(rl_path).expanduser().resolve().parent / "workflow.yaml"
    if not wf.is_file():
        return None
    raw = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    return raw.get("sampling")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TIR agent with VERL + tir_algo overlay")
    parser.add_argument("config", choices=["fast", "a800", "a800_2gpu"])
    parser.add_argument("--algo", type=str, default="grpo", choices=list(VALID_ALGOS))
    parser.add_argument("--active-agent", type=str, default=None)
    parser.add_argument("--n-runners", type=int, default=None)
    parser.add_argument("--model", type=str, default=None, help=f"Override model path (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument(
        "--rl-yaml",
        type=str,
        default=None,
        help="Merge Science Control rl.yaml (Hydra-shaped) onto the profile config",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Alias of --rl-yaml",
    )
    args = parser.parse_args()

    config_fns = {
        "fast": config_train_fast,
        "a800": config_train_a800,
        "a800_2gpu": config_train_a800_2gpu,
    }
    from rl.train_signal import AdvantageSpec, LossSpec, TrainSignal

    rl_path = args.rl_yaml or args.config_yaml
    algo = args.algo
    sampling = None
    if rl_path:
        rl_raw = load_rl_yaml(rl_path)
        if rl_raw.get("algo"):
            algo = str(rl_raw["algo"]).lower()
        if algo not in VALID_ALGOS:
            raise SystemExit(f"Unknown algo in rl yaml: {algo}")
        sampling = _load_sibling_workflow_sampling(rl_path)

    signal = TrainSignal(
        advantage=AdvantageSpec(name=algo, use_critic=False),
        loss=LossSpec(name=algo),
        meta={"algo": algo, "source": "cli" if not rl_path else "rl_yaml"},
    )
    config = apply_train_signal(config_fns[args.config](), signal)
    if rl_path:
        config = _deep_merge(config, load_rl_yaml(rl_path))
        # Re-apply algo overlay fields that may have been overwritten
        config = apply_train_signal(config, signal)
        if sampling is not None:
            config = apply_sample_policy(config, sampling)
            algo = str(config.get("algorithm", {}).get("tir_algo") or algo).lower()
            signal = TrainSignal(
                advantage=AdvantageSpec(name=algo, use_critic=False),
                loss=LossSpec(name=algo),
                meta={"algo": algo, "source": "workflow.sampling"},
            )
            print(
                f"Applied sibling workflow.sampling → tir_algo={algo} "
                f"rollout.n={config.get('actor_rollout_ref', {}).get('rollout', {}).get('n')}"
            )
        print(f"Merged rl yaml: {rl_path}")
    if args.model:
        config["actor_rollout_ref"]["model"]["path"] = args.model
    elif rl_path:
        mp = load_rl_yaml(rl_path).get("model_path")
        if mp:
            config["actor_rollout_ref"]["model"]["path"] = str(mp)

    if args.n_runners is not None:
        n_runners = args.n_runners
    elif rl_path and load_rl_yaml(rl_path).get("n_runners") is not None:
        n_runners = int(load_rl_yaml(rl_path)["n_runners"])
    elif args.config == "fast":
        n_runners = 2
    elif args.config == "a800_2gpu":
        n_runners = 8
    else:
        n_runners = 4

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if vis:
        n_vis = len([x for x in vis.split(",") if x.strip() != ""])
        if n_vis > 0:
            config["trainer"]["n_gpus_per_node"] = n_vis
            print(f"CUDA_VISIBLE_DEVICES={vis} → trainer.n_gpus_per_node={n_vis}")

    print(f"Starting training with '{args.config}' / algo={algo} ...")
    train(config, n_runners=n_runners, active_agent=args.active_agent)


if __name__ == "__main__":
    main()
