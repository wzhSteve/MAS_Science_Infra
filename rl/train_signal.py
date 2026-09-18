"""TrainSignal: AdvantageSpec / LossSpec handed to RL overlay (no torch / no AGL)."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field

from rl.loss import AdvantageSpec, LossSpec

# Import submodule directly to avoid workflow/__init__ ↔ rl cycle.
from workflow.contracts import TrajectoryBatch  # noqa: E402


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
