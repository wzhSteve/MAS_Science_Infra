"""Per-algorithm Hydra/dict overlays. VERL adv_estimator stays 'grpo'."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from rl.train_signal import TrainSignal

VALID_ALGOS = ("grpo", "arpo", "aepo", "igpo", "gigpo", "rae")

DEFAULT_TIR: Dict[str, Any] = {
    "initial_rollouts": 2,
    "beam_size": 2,
    "branch_probability": 0.5,
    "entropy_weight": 0.5,
    "entropy_threshold": 0.15,
    "entropy_tokens": 8,
    "gamma": 1.0,
    "info_gain_type": "log_prob_diff",
    "step_advantage_w": 1.0,
    "gigpo_mode": "mean_std",
    "entropy_adv_coef": 0.1,
    "aepo_beta": 1.0,
    "branch_penalty_slope": 0.25,
    "multi_tool_bonus": 0.1,
    "max_branch_depth": 2,
    "expand_in_runner": True,
    "use_official_arpo_gate": True,
    "sites": None,
    "zero_prefix_adv_for_children": True,
    "ready_batch": False,
    "rae_p_plus": 0.8,
    "rae_k_min": 2,
    "rae_epsilon_f": 1e-3,
    "rae_full_tgt": False,
    "rae_dead_end_backprop": True,
    "rae_dead_end_depth": 1,
}


def apply_sample_policy(config: Dict[str, Any], sampling: Any) -> Dict[str, Any]:
    """Map MAS SamplePolicy onto algorithm.tir / rollout.n (no AGL import)."""
    if sampling is None:
        return config
    cfg = deepcopy(config)
    if hasattr(sampling, "mode"):
        mode = str(getattr(sampling, "mode", None) or "grpo_n")
        group_n = getattr(sampling, "group_n", None)
        beam = getattr(sampling, "beam_size", None)
        init = getattr(sampling, "initial_rollouts", None)
        max_depth = getattr(sampling, "max_branch_depth", None)
        expand = getattr(sampling, "expand_in_runner", None)
    elif isinstance(sampling, dict):
        mode = str(sampling.get("mode") or "grpo_n")
        group_n = sampling.get("group_n")
        beam = sampling.get("beam_size")
        init = sampling.get("initial_rollouts")
        max_depth = sampling.get("max_branch_depth")
        expand = sampling.get("expand_in_runner")
    else:
        return config
    from workflow.contracts import SamplePolicy

    policy = sampling if isinstance(sampling, SamplePolicy) else SamplePolicy.model_validate(sampling)
    mode = mode.lower().strip()
    algo_map = {
        "grpo_n": "grpo",
        "grpo": "grpo",
        "arpo": "arpo",
        "aepo": "aepo",
        "appo": "arpo",
        "rae": "rae",
    }
    algo = algo_map.get(mode, mode if mode in VALID_ALGOS else "grpo")
    cfg = apply_algo_overlay(cfg, algo)
    from workflow.sampling.registry import sampling_adapters

    adapter = sampling_adapters.resolve(algo)
    cfg = adapter.training_overlay(cfg, policy)
    tir = dict(cfg["algorithm"].get("tir") or {})
    if group_n is not None:
        n = max(1, int(group_n))
        roll = dict(cfg.get("actor_rollout_ref") or {})
        r = dict(roll.get("rollout") or {})
        r["n"] = n
        roll["rollout"] = r
        cfg["actor_rollout_ref"] = roll
        cfg["rollout_per_gpu"] = n
    if beam is not None:
        tir["beam_size"] = max(1, int(beam))
    if init is not None:
        tir["initial_rollouts"] = max(1, int(init))
    if max_depth is not None:
        tir["max_branch_depth"] = max(1, int(max_depth))
    if expand is not None:
        tir["expand_in_runner"] = bool(expand)
    tir["sites"] = [site.model_dump(mode="json") for site in policy.resolved_sites()]
    cfg["algorithm"]["tir"] = tir
    return cfg


def apply_algo_overlay(config: Dict[str, Any], algo: str) -> Dict[str, Any]:
    algo = algo.lower().strip()
    if algo not in VALID_ALGOS:
        raise ValueError(f"Unknown --algo {algo!r}. Expected one of {VALID_ALGOS}")
    cfg = deepcopy(config)
    algo_block = dict(cfg.get("algorithm") or {})
    algo_block["adv_estimator"] = "grpo"
    algo_block["use_kl_in_reward"] = False
    algo_block["tir_algo"] = algo
    tir = dict(DEFAULT_TIR)
    tir.update(dict(algo_block.get("tir") or {}))
    if algo in ("arpo", "aepo", "rae"):
        n = int(cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 4))
        tir["initial_rollouts"] = max(1, min(tir["initial_rollouts"], n - 1 if n > 1 else 1))
        tir["ready_batch"] = bool(tir.get("ready_batch", True))
    if algo == "rae":
        tir["rae_full_tgt"] = bool(tir.get("rae_full_tgt", False))
        tir["rae_dead_end_backprop"] = bool(tir.get("rae_dead_end_backprop", True))
    algo_block["tir"] = tir
    cfg["algorithm"] = algo_block
    trainer = dict(cfg.get("trainer") or {})
    base_name = str(trainer.get("experiment_name") or "tir_agent")
    if not base_name.endswith(algo):
        trainer["experiment_name"] = f"{base_name}_{algo}"
    cfg["trainer"] = trainer
    return cfg


def apply_train_signal(config: Dict[str, Any], signal: "TrainSignal") -> Dict[str, Any]:
    """Map RL TrainSignal onto Hydra/VERL dict. Does not import DataProto."""
    algo = str(signal.advantage.name or "grpo").lower()
    cfg = apply_algo_overlay(config, algo)
    actor = dict(cfg.get("actor_rollout_ref", {}).get("actor") or {})
    actor["clip_ratio_low"] = float(signal.loss.clip_ratio_low)
    actor["clip_ratio_high"] = float(signal.loss.clip_ratio_high)
    actor["entropy_coeff"] = float(signal.loss.entropy_coeff)
    actor["kl_loss_coef"] = float(signal.loss.kl_loss_coef)
    roll = dict(cfg.get("actor_rollout_ref") or {})
    roll["actor"] = actor
    cfg["actor_rollout_ref"] = roll
    tir = dict(cfg["algorithm"].get("tir") or {})
    tir["gamma"] = float(signal.advantage.gamma)
    cfg["algorithm"]["tir"] = tir
    return cfg
