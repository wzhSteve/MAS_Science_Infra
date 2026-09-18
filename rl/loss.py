"""LossSpec / AdvantageSpec — RL-owned training signal fields (no torch / no AGL)."""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


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


class CreditAssignmentSpec(BaseModel):
    """Node-level credit assignment on the RolloutTree (new_framework P2).

    ``k_hop``: node reward = equal-weight mean of ancestor-path rewards within
    k hops (k=0 → leaf outcome only). ``verdict``: RAE validate/invalidate
    written back onto tree nodes for dead-end backprop.
    """

    node_level: bool = False  # off → rollout-level only (legacy behavior)
    k_hop: int = 1  # 1..depth; equal-weight mean over path window
    use_verdict: bool = True  # attach RAE verdicts to nodes when present
    normalize: bool = True  # z-score within same-parent siblings
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}
