"""LossSpec / AdvantageSpec / AdvantagePipelineSpec — RL-owned training signal fields (no torch / no AGL)."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class LossSpec(BaseModel):
    name: str = "grpo"
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.3
    entropy_coeff: float = 0.0
    kl_loss_coef: float = 0.0
    # Mirrored into actor_rollout_ref.actor.policy_loss by apply_train_signal
    # when loss_mode == "future_kl" (read side: dp_actor.py fipo dispatch).
    loss_mode: str = "vanilla"  # vanilla | future_kl
    future_kl_weight: float = 1.0
    decay_rate: float = 128.0
    future_kl_clip_ratio: float = 0.2
    extra: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AdvantageSpec(BaseModel):
    """Phase 0: GRPO group-relative (no Critic). use_critic reserved."""

    name: str = "grpo"
    use_critic: bool = False
    gamma: float = 1.0
    reward_scale_discount: float = 0.9
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


ADVANTAGE_BASES = ("grpo", "igpo", "gigpo")
ADVANTAGE_OPERATORS = (
    "zero_prefix",
    "appo_credit",
    "aepo_entropy",
    "rae_verdict",
    "rae_full_tgt",
)


class AdvantageOperatorSpec(BaseModel):
    """One entry of the ordered post-processing operator stack (RL3).

    ``id`` is constrained to the five-value operator registry
    (zero_prefix/appo_credit/aepo_entropy/rae_verdict/rae_full_tgt);
    ``params`` carries operator kwargs (e.g. discount / entropy_adv_coef /
    p_plus / epsilon_f). Redundant registry-level params (e.g. zero_prefix's
    legacy enable flag, which lives in the tir dict as
    ``zero_prefix_adv_for_children``) are resolved from the tir config during
    pipeline resolution, not stored here.
    """

    id: str
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class AdvantagePipelineSpec(BaseModel):
    """RL3 advantage pipeline: base estimator + ordered operator stack.

    base ∈ {grpo, igpo, gigpo} (RL_design §3.1.3: the non-grpo presets all
    start from the GRPO stage-0 group baseline; igpo/gigpo are the two bases
    that replace the whole group signal). operators is an ordered list — the
    executor applies them left-to-right after the base. Validation rules
    R1/R5 of RL_design §3.1.4 (rae_verdict XOR rae_full_tgt; rae_full_tgt vs
    appo_credit forbidden) are enforced by resolve-time checks in
    rl/hooks/advantage.py (single enforcement point, used by both the v1
    expansion table and direct v2 specs).
    """

    base: str = "grpo"
    operators: List[Any] = Field(default_factory=list)
    base_params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}
