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
import logging
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, Optional

os.environ.setdefault("VLLM_USE_V1", "1")

from rl.hooks.overlay import VALID_ALGOS, apply_sample_policy, apply_train_signal, ensure_trainer_horizon
from workflow.agent_tracing import agent_span_pattern

DEFAULT_MODEL_PATH = "/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B"
# mas/ → MAS_Science_Infra
REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_DIR = REPO_ROOT / "data"


def _resolve_parquet_path(path: str, default_name: str) -> str:
    """Keep explicit server paths; resolve only repository-relative data."""
    if PureWindowsPath(str(path)).drive or "\\" in str(path):
        raise ValueError("Training requires Linux server data paths, not Windows paths.")
    p = Path(str(path)).expanduser()
    if p.is_absolute():
        return str(p)
    relative = Path(*p.parts[1:]) if p.parts and p.parts[0] == "data" else p
    return str(REPO_DATA_DIR / (relative if str(relative) != "." else default_name))


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


def resolve_workflow_spec_path(
    rl_path: Optional[str] = None, workflow_path: Optional[str] = None
) -> Optional[str]:
    """Explicit ``--workflow-yaml``, otherwise the sibling of the rl yaml."""
    if workflow_path:
        wf = Path(workflow_path).expanduser().resolve()
        return str(wf) if wf.is_file() else None
    if not rl_path:
        return None
    wf = Path(rl_path).expanduser().resolve().parent / "workflow.yaml"
    return str(wf) if wf.is_file() else None


def trainable_ids_from_spec(spec_path: str) -> list[str]:
    from workflow.compiler import trainable_agents
    from workflow.spec import load_spec

    return trainable_agents(load_spec(spec_path))


def reward_outcome_for(config: Dict[str, Any], tir_algo: str) -> str:
    """``algorithm.reward.outcome`` (written by the overlay) wins; flow_grpo falls back to 0/1."""
    sec = (config.get("algorithm") or {}).get("reward") or {}
    explicit = str(sec.get("outcome") or "").strip().lower() if isinstance(sec, dict) else ""
    if explicit:
        return explicit
    return "binary" if tir_algo == "flow_grpo" else "hierarchical"


def window_agent_match(spec_path: Optional[str], active_agents: list[str]) -> Optional[str]:
    """Planner-only span filter for Mode B-window graphs; Tir hub keeps every span.

    Window LLM calls nest under ``agent__<window id>``; frozen calls made inside a
    trainable window nest under ``agent__frozen__<role>`` and so fall outside the
    match. ``TIR_AGENT_MATCH=0`` disables the filter.
    """
    if not spec_path or not active_agents:
        return None
    if os.environ.get("TIR_AGENT_MATCH", "").strip().lower() in ("0", "false", "no", "off"):
        return None
    from workflow.runtime import agent_window_rollout
    from workflow.spec import load_spec

    try:
        if not agent_window_rollout(load_spec(spec_path)):
            return None
    except Exception:
        return None
    return agent_span_pattern(active_agents)


def train(
    config: Dict[str, Any],
    n_runners: int,
    active_agents: list[str],
    spec_path: Optional[str] = None,
) -> None:
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from science_infra.control.task_files import load_task_rows
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
    agent = LitTirAgent(
        max_tokens=max_resp, max_model_len=max_model_len, spec_path=spec_path
    )
    if spec_path:
        print(f"Workflow spec: {spec_path}")
    tir_algo = str(config.get("algorithm", {}).get("tir_algo") or "grpo")
    tir_cfg = dict(config.get("algorithm", {}).get("tir") or {})
    os.environ["TIR_ALGO"] = tir_algo
    reward_outcome = reward_outcome_for(config, tir_algo)
    os.environ["TIR_REWARD_OUTCOME"] = reward_outcome
    print(f"reward outcome={reward_outcome}")
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

    from rl.hooks.trainer import TirAgentLightningTrainer, bound_daemon_cls

    daemon_cls = bound_daemon_cls(tir_algo, tir_cfg)
    if tir_algo == "grpo":
        algorithm = agl.VERL(config, daemon_cls=daemon_cls)
    else:
        algorithm = agl.VERL(
            config,
            trainer_cls=TirAgentLightningTrainer,
            daemon_cls=daemon_cls,
        )
    agent_match = window_agent_match(spec_path, active_agents)
    # Skip LLM spans with empty prompt/response token_ids so VERL's
    # remove-padding path never sees a 0-length sequence (Qwen3 reshape crash).
    adapter_kwargs: Dict[str, Any] = {"_skip_empty_token_spans": True}
    if agent_match:
        adapter_kwargs["agent_match"] = agent_match
    trainer = agl.Trainer(
        n_runners=n_runners,
        algorithm=algorithm,
        adapter=adapter_kwargs,
    )
    if active_agents:
        print(
            f"Adapter: skip_empty_token_spans=True "
            f"agent_match={agent_match!r} agents={active_agents!r}"
        )

    train_path = _resolve_parquet_path(config["data"]["train_files"], "train.parquet")
    val_path = _resolve_parquet_path(config["data"]["val_files"], "val.parquet")
    config["data"]["train_files"] = train_path
    config["data"]["val_files"] = val_path
    try:
        train_data = load_task_rows(train_path)
        val_data = load_task_rows(val_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing dataset. Expected {train_path} and {val_path} "
            f"(canonical: MAS_Science_Infra/data/). "
            "Run: bash scripts/prepare_data.sh from tir_agent, or place files under repo data/."
        ) from exc
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"Model: {config['actor_rollout_ref']['model']['path']}")
    print(f"GPUs per node: {config['trainer']['n_gpus_per_node']}")
    print(f"train_batch_size: {config['data']['train_batch_size']}")
    print(f"gpu_memory_utilization: {config['actor_rollout_ref']['rollout']['gpu_memory_utilization']}")
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore[arg-type]


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    skip = {"profile", "algo", "n_runners", "model_path", "devices", "rollout_per_gpu", "active_agent"}
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


def _load_workflow_sampling(
    rl_path: str, workflow_path: Optional[str] = None
) -> Any:
    """Load the explicit run snapshot, or the legacy sibling workflow."""
    import yaml  # type: ignore

    wf = (
        Path(workflow_path).expanduser().resolve()
        if workflow_path
        else Path(rl_path).expanduser().resolve().parent / "workflow.yaml"
    )
    if not wf.is_file():
        return None
    raw = yaml.safe_load(wf.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        return None
    return raw.get("sampling")


def _bypass_proxy_for_local_vllm() -> None:
    """Rollout vLLM is on this machine. An ambient HTTP proxy must not intercept it."""
    import socket

    hosts = ["127.0.0.1", "localhost"]
    try:
        hosts.append(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    for raw in (os.environ.get("HOSTNAME", ""),):
        if raw:
            hosts.append(raw)
    try:
        import subprocess

        out = subprocess.check_output(["hostname", "-I"], text=True, timeout=2)
        hosts.extend(part for part in out.split() if part)
    except (OSError, subprocess.SubprocessError):
        pass
    current = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    parts = [item.strip() for item in current.split(",") if item.strip()]
    for host in hosts:
        if host and host not in parts:
            parts.append(host)
    value = ",".join(parts)
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value
    print(f"NO_PROXY={value}")


def _quiet_third_party_logs() -> None:
    """Library warnings are not training failures. Keep errors."""
    import warnings

    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ.setdefault("RAY_DISABLE_DOCKER_CPU_WARNING", "1")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    class _DropWarnings(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno != logging.WARNING

    root = logging.getLogger()
    root.addFilter(_DropWarnings())
    for name in ("vllm", "ray", "agentops", "LiteLLM", "litellm", "httpx", "torch"):
        logging.getLogger(name).addFilter(_DropWarnings())
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> None:
    _quiet_third_party_logs()
    _bypass_proxy_for_local_vllm()
    parser = argparse.ArgumentParser(description="Train TIR agent with VERL + tir_algo overlay")
    parser.add_argument("config", choices=["fast", "a800", "a800_2gpu"])
    parser.add_argument("--algo", type=str, default="grpo", choices=list(VALID_ALGOS))
    parser.add_argument("--active-agent", type=str, default=None)
    parser.add_argument("--active-agents", type=str, default=None)
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
    parser.add_argument(
        "--workflow-yaml",
        type=str,
        default=None,
        help="Workflow snapshot used to resolve Sampling for this run",
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
    explicit_algo = None
    if rl_path:
        rl_raw = load_rl_yaml(rl_path)
        explicit_algo = rl_raw.get("algo") or (rl_raw.get("algorithm") or {}).get("tir_algo")
        if explicit_algo:
            algo = str(explicit_algo).lower()
        if algo not in VALID_ALGOS:
            raise SystemExit(f"Unknown algo in rl yaml: {algo}")
        sampling = _load_workflow_sampling(rl_path, args.workflow_yaml)

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
            mode = str((sampling.get("mode") if isinstance(sampling, dict) else "") or "").lower()
            config = apply_sample_policy(config, sampling, algo=algo if explicit_algo else None)
            if explicit_algo and mode and {"grpo_n": "grpo"}.get(mode, mode) != algo:
                print(f"rl.yaml algo={algo} overrides workflow.sampling.mode={mode} (sampling only sets group_n/sites)")
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

    # a800_2gpu always trains on both cards. Ambient CUDA_VISIBLE_DEVICES=0
    # (common from run.sh defaults / notebook kernels) must not shrink us to 1.
    if args.config == "a800_2gpu":
        vis_raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        n_vis = len([x for x in vis_raw.split(",") if x.strip() != ""]) if vis_raw else 0
        if n_vis < 2:
            was = vis_raw if vis_raw else "<unset>"
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            print(f"a800_2gpu: forcing CUDA_VISIBLE_DEVICES=0,1 (was {was!r})")
        config["trainer"]["n_gpus_per_node"] = 2
        print(
            f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
            f"→ trainer.n_gpus_per_node=2"
        )
    else:
        vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if vis:
            n_vis = len([x for x in vis.split(",") if x.strip() != ""])
            if n_vis > 0:
                config["trainer"]["n_gpus_per_node"] = n_vis
                print(f"CUDA_VISIBLE_DEVICES={vis} → trainer.n_gpus_per_node={n_vis}")

    config = ensure_trainer_horizon(config)
    # Branching guard: ARPO/APPO silently produce branch_local_count=0 when
    # rollout.n collapses the per-root sampling_budget to <=1 (see
    # rl/hooks/daemon.py _preinject_expand_fields). Fail fast instead of a
    # silent no-branch run.
    tir_algo = str(config.get("algorithm", {}).get("tir_algo") or algo).lower()
    if tir_algo in ("arpo", "appo"):
        group_n = int(config.get("actor_rollout_ref", {}).get("rollout", {}).get("n") or 1)
        initial = int(config.get("algorithm", {}).get("tir", {}).get("initial_rollouts") or 1)
        budget = max(1, (group_n + initial - 1) // max(1, initial))
        if budget <= 1:
            raise SystemExit(
                f"{tir_algo} sampling_budget<=1 (rollout.n={group_n}, "
                f"initial_rollouts={initial}); branch expansion would be disabled. "
                "Set actor_rollout_ref.rollout.n >= 4 and initial_rollouts >= 2."
            )
        print(f"{tir_algo} budget check: rollout.n={group_n} initial_rollouts={initial} → sampling_budget={budget}")
    print(
        "trainer horizon: "
        f"total_epochs={config.get('trainer', {}).get('total_epochs')} "
        f"total_training_steps={config.get('trainer', {}).get('total_training_steps')}"
    )
    print(f"Starting training with '{args.config}' / algo={algo} ...")
    active_agents = [
        value.strip()
        for value in str(args.active_agents or args.active_agent or "").split(",")
        if value.strip()
    ]
    spec_path = resolve_workflow_spec_path(rl_path, args.workflow_yaml)
    if not active_agents and spec_path:
        active_agents = trainable_ids_from_spec(spec_path)
        print(f"Trainable agents from workflow: {active_agents}")
    train(config, n_runners=n_runners, active_agents=active_agents, spec_path=spec_path)


if __name__ == "__main__":
    main()
