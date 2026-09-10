"""TrainSignal: AdvantageSpec / LossSpec handed to RL (no torch / no AGL)."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from .contracts import TrajectoryBatch


class LossSpec(BaseModel):
    name: str = "grpo"
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.3
    entropy_coeff: float = 0.0
    kl_loss_coef: float = 0.0
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AdvantageSpec(BaseModel):
    """Phase 0: GRPO group-relative (no Critic). use_critic reserved."""

    name: str = "grpo"
    use_critic: bool = False
    gamma: float = 1.0
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class TrainSignal(BaseModel):
    batch: TrajectoryBatch = Field(default_factory=TrajectoryBatch)
    advantage: AdvantageSpec = Field(default_factory=AdvantageSpec)
    loss: LossSpec = Field(default_factory=LossSpec)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


def batch_to_train_signal(batch: TrajectoryBatch, *, algo: str = "grpo") -> TrainSignal:
    return TrainSignal(
        batch=batch,
        advantage=AdvantageSpec(name=algo, use_critic=False),
        loss=LossSpec(name=algo),
        meta={"algo": algo, "n": batch.meta.get("n_trajectories")},
    )
