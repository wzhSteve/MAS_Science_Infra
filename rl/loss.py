"""LossSpec / AdvantageSpec / AdvantagePipelineSpec — RL-owned training signal fields (no torch / no AGL)."""

from __future__ import annotations

import math
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


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

    # RL5 V4 (numeric domain): the three future_kl coefficients must be real,
    # finite, non-negative numbers (RL_design §5.2). Positive-only would break
    # legal zero configs (kl_loss_coef=0.0 / entropy_coeff=0.0 are today's
    # defaults), so the rule matches the design text exactly: >= 0.
    @field_validator("future_kl_weight", "decay_rate", "future_kl_clip_ratio")
    @classmethod
    def _finite_non_negative(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"V4 类型错误: expected a real number, got {value!r}")
        fv = float(value)
        if not math.isfinite(fv) or fv < 0.0:
            raise ValueError(
                f"V4 数值域错误: future_kl 系数必须为非负有限数 (>=0, finite), got {value!r}"
            )
        return fv


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

    # RL5 V1 (required/type) + V2 (enum domain) inline validators. The
    # combination rules (V3 XOR / no duplicates / forbidden stacks) stay at
    # the single enforcement point in rl/hooks/advantage.py:validate_pipeline_spec
    # — already locked by RL3's test_operator_composition.py; these field
    # validators only claim what pydantic does best on the save path (fail
    # before anything 是 expanded).
    @field_validator("base")
    @classmethod
    def _base_enum(cls, value: str) -> str:
        if value not in ADVANTAGE_BASES:
            raise ValueError(
                f"V1/V2 枚举域错误: advantage.base 必须是 {ADVANTAGE_BASES} 之一, got {value!r}"
            )
        return value

    @field_validator("operators")
    @classmethod
    def _operators_entries(cls, value: List[Any]) -> List[Any]:
        # Entries may be dicts (yaml) or AdvantageOperatorSpec on the Python
        # side; normalize to AdvantageOperatorSpec so the operator-id enum
        # (V2) and the nesting shape (V1) are enforced right here instead of
        # deep inside pipeline resolution.
        normalized: List[Any] = []
        for entry in value:
            if isinstance(entry, AdvantageOperatorSpec):
                normalized.append(entry)
                continue
            if isinstance(entry, dict):
                normalized.append(
                    AdvantageOperatorSpec(
                        id=str(entry.get("id")),
                        params=dict(entry.get("params") or {}),
                    )
                )
                continue
            raise ValueError(
                f"V1 类型错误: operators 列表项必须是 {{id, params}} 对象, got {entry!r}"
            )
        for op in normalized:
            if op.id not in ADVANTAGE_OPERATORS:
                raise ValueError(
                    f"V2 枚举域错误: 未知 advantage 算子 {op.id!r}; 注册表: {ADVANTAGE_OPERATORS}"
                )
        return normalized


# ---------------------------------------------------------------------------
# RL5: formal preset expansion — RL_design §3.4.4 snapshot table
# ---------------------------------------------------------------------------
# Single authority: the post-RL0-errata §2.3 truth table / §3.4.4 full
# snapshot. This 8-preset expansion is what apply_algo_overlay expands v1
# ``algo`` selections into and what resolve_pipeline_spec (rl/hooks/
# advantage.py) consults for the tensor-side pipeline; both read THIS table
# so the frozen snapshot has exactly one physical copy (advantage.py's
# _PIPELINE_EXPANSION mirrors it as legacy dispatch documentation, kept in
# sync by test_operator_composition + test_expand_preset exact matches).
#
# Sampling/gate params (BRANCHING initial_rollouts收缩 / use_official_arpo_gate /
# aepo_beta / rae_p_plus ...) do NOT belong to the advantage segment — they
# stay in the tir dict / sampling preset segment (§3.1.6 推论 b).
PRESET_EXPANSION_BASE: Dict[str, str] = {
    "grpo": "grpo",
    "arpo": "grpo",
    "appo": "grpo",
    "aepo": "grpo",
    "rae": "grpo",
    "flow_grpo": "grpo",
    "igpo": "igpo",
    "gigpo": "gigpo",
}

PRESET_EXPANSION_OPERATORS: Dict[str, List[str]] = {
    "grpo": ["zero_prefix"],
    "arpo": [],
    "appo": ["appo_credit"],
    "aepo": ["aepo_entropy"],
    "rae": ["zero_prefix", "rae_verdict"],
    "flow_grpo": [],
    "igpo": [],
    "gigpo": [],
}


def expand_preset(algo: str) -> "AdvantagePipelineSpec":
    """Expand a v1 preset name into the frozen AdvantagePipelineSpec snapshot.

    Public API (RL5, formalizing the RL3 internal expansion): 8 VALID_ALGOS
    each expand to ``base + ordered operators`` exactly as the legacy
    if-return dispatch did with today's defaults. ``rae`` keeps its legacy
    fork behaviour — the caller decides the ``rae_full_tgt`` swap at resolve
    time (mirrors advantage.py:92-99); the plain form here is the
    ``rae_full_tgt=False`` default. Operator params stay empty: parameter
    values live in the tir dict read at execution time (param chain
    preserved — expanding them here would fork the param source of truth).

    Unknown names raise ValueError (V1) — the historical silent grpo
    passthrough for ``tir_algo`` typos belonged to the resolve-tolerant
    tensor path; the formal API refuses to invent a pipeline for a name the
    snapshot table does not know, so typos fail at save/preflight instead
    of mid-training. The tensor-side resolver (resolve_pipeline_spec) keeps
    its own容错 fallback (advantage.py:248-253) — see its docstring.
    """
    key = str(algo or "").lower().strip()
    if key not in PRESET_EXPANSION_BASE:
        raise ValueError(
            f"V1/V2 枚举域错误: 未知 preset {key!r};"
            f" 展开表 (RL_design §3.4.4): {sorted(PRESET_EXPANSION_BASE)}"
        )
    return AdvantagePipelineSpec(
        base=PRESET_EXPANSION_BASE[key],
        operators=[
            AdvantageOperatorSpec(id=op_id) for op_id in PRESET_EXPANSION_OPERATORS[key]
        ],
    )
