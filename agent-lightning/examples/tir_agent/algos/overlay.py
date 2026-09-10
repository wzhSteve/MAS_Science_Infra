"""Per-algorithm Hydra/dict overlays. VERL adv_estimator stays 'grpo'."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from workflow.train_signal import TrainSignal

VALID_ALGOS = ("grpo", "arpo", "aepo", "igpo", "gigpo")

DEFAULT_TIR: Dict[str, Any] = {
    "initial_rollouts": 2,
    "beam_size": 2,
    "branch_probability": 0.5,
    "entropy_weight": 0.2,
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
}


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
    if algo in ("arpo", "aepo"):
        # Keep a fraction of group size for global sampling.
        n = int(cfg.get("actor_rollout_ref", {}).get("rollout", {}).get("n", 4))
        tir["initial_rollouts"] = max(1, min(tir["initial_rollouts"], n - 1 if n > 1 else 1))
    algo_block["tir"] = tir
    cfg["algorithm"] = algo_block
    trainer = dict(cfg.get("trainer") or {})
    base_name = str(trainer.get("experiment_name") or "tir_agent")
    if not base_name.endswith(algo):
        trainer["experiment_name"] = f"{base_name}_{algo}"
    cfg["trainer"] = trainer
    return cfg


def apply_train_signal(config: Dict[str, Any], signal: "TrainSignal") -> Dict[str, Any]:
    """Map data-layer TrainSignal onto Hydra/VERL dict. Does not import DataProto."""
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
