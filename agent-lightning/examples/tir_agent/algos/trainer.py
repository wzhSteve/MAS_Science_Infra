"""Custom VERL trainer: intercept compute_advantage for tir_algo."""

from __future__ import annotations

from typing import Any, Type

from agentlightning.verl.trainer import AgentLightningTrainer

from .advantage import apply_tir_advantages
from .daemon import TirAgentModeDaemon


class TirAgentLightningTrainer(AgentLightningTrainer):
    """Same loop as AgentLightningTrainer; advantage is routed by algorithm.tir_algo."""

    def _train_step(self, batch_dict: dict) -> dict:
        import agentlightning.verl.trainer as trainer_mod

        orig = trainer_mod.compute_advantage
        algo_cfg = self.config.algorithm

        def _wrapped(batch, *args: Any, **kwargs: Any):
            return apply_tir_advantages(
                batch,
                adv_estimator=kwargs.get("adv_estimator", algo_cfg.adv_estimator),
                gamma=float(kwargs.get("gamma", getattr(algo_cfg, "gamma", 1.0))),
                lam=float(kwargs.get("lam", getattr(algo_cfg, "lam", 1.0))),
                num_repeat=int(kwargs.get("num_repeat", self.config.actor_rollout_ref.rollout.n)),
                norm_adv_by_std_in_grpo=bool(
                    kwargs.get("norm_adv_by_std_in_grpo", getattr(algo_cfg, "norm_adv_by_std_in_grpo", True))
                ),
                config=kwargs.get("config", algo_cfg),
            )

        trainer_mod.compute_advantage = _wrapped  # type: ignore[assignment]
        try:
            return super()._train_step(batch_dict)
        finally:
            trainer_mod.compute_advantage = orig


def bound_daemon_cls(tir_algo: str, tir_config: dict) -> Type[TirAgentModeDaemon]:
    class _BoundTirDaemon(TirAgentModeDaemon):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, tir_algo=tir_algo, tir_config=tir_config, **kwargs)

    _BoundTirDaemon.__name__ = f"TirAgentModeDaemon_{tir_algo}"
    return _BoundTirDaemon
