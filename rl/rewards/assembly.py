"""RewardSpec / RewardComponent — RL4 reward assembly contract (no torch / no AGL).

The hierarchical outcome contract (rl/rewards/outcome.py:compute_outcome_reward)
is frozen; this module gives every hierarchy layer a switchable, weightable
identity so reward ablations (RL_design §1.7 E/F cells) become config instead
of code. compute_outcome_reward keeps its byte-identical literal path when
reward_spec is None (the default for every v1 caller); a provided RewardSpec
routes the same hierarchy through per-component switch/weight application.

Component registry (RL_design §3.2.1):
- format_penalty   — format bad → -1 * weight (may be turned off → no penalty)
- accuracy_reward  — QA token-F1 / GSM8K numeric 0-1 (NOT switchable — the
                     main reward; spec validation rejects enabled=False)
- multi_tool_bonus — both tool families used → +0.1 * weight (switchable)
- cost_penalty     — reserved schema slot ONLY (RL_design §3.2.1 预留位):
                     −λ·n_calls per tool call; implementation deferred to a
                     later stage, the slot is born disabled and the executor
                     no-ops on it. Enabling it here is a validation error.

Lifecycle note (recorded here instead of mas/lit_tir_agent.py — A1 red line
keeps the hot-path trio untouched): the ``multi_tool_bonus`` constructor
param on LitTirAgent (lit_tir_agent.py:52, assigned to self at :61) is
decorative — :247 calls default_reward_fn without passing it and nothing
reads the attribute back for reward computation. As of RL4 the live config
chain is DEFAULT_TIR multi_tool_bonus (overlay.py:29) → apply_algo_overlay →
algorithm.tir.multi_tool_bonus → collector default_reward_fn → RewardSpec;
the LitTirAgent parameter stays untouched and carries no semantics.

Delivery path: RewardSpec travels under ``algorithm.reward`` in the config
tree (the RL5 v2-segment yaml slot); for v1 configs the tir dict's
``multi_tool_bonus`` value feeds the default spec (read-back compatible:
algorithm.tir.multi_tool_bonus keeps its legacy read path in parallel —
dual-read window, no removal this stage).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator, model_validator

OUTCOME_MODES = ("hierarchical", "binary")
FORMAT_PENALTY = "format_penalty"
ACCURACY_REWARD = "accuracy_reward"
MULTI_TOOL_BONUS = "multi_tool_bonus"
COST_PENALTY = "cost_penalty"

# Component ids that may appear in a RewardSpec by default policy. cost_penalty
# is schema-reserved (born disabled; enabling is rejected until implemented).
SWITCHABLE_COMPONENTS = (FORMAT_PENALTY, MULTI_TOOL_BONUS, COST_PENALTY)
RESERVED_COMPONENTS = (COST_PENALTY,)
# The accuracy layer is the main reward — a RewardSpec without it is invalid.
REQUIRED_COMPONENTS = (ACCURACY_REWARD,)

DEFAULT_FORMAT_PENALTY_WEIGHT = 1.0  # penalty magnitude = -1.0 * weight (现状字面量 -1)
DEFAULT_MULTI_TOOL_BONUS_WEIGHT = 0.1  # bonus magnitude = default param 0.1


class RewardComponent(BaseModel):
    """One reward hierarchy layer: id + weight + switch.

    ``weight`` is the component's magnitude scale (format_penalty: base -1
    times weight; multi_tool_bonus: base 0.1 times weight; accuracy_reward:
    weight fixed at 1.0 — weighting the main reward is out of contract).
    ``enabled=False`` switches the component off (accuracy_reward excepted).
    """

    id: str
    weight: float = 1.0
    enabled: bool = True

    model_config = {"extra": "forbid"}

    # RL5 V4 (numeric domain): a component weight must be a real, finite,
    # non-negative number — negative weights would flip a hierarchy layer's
    # sign and turn e.g. a format penalty into a format bonus silently.
    @field_validator("weight")
    @classmethod
    def _finite_non_negative(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"V4 类型错误: weight 必须为实数, got {value!r}")
        fv = float(value)
        if not math.isfinite(fv) or fv < 0.0:
            raise ValueError(
                f"V4 数值域错误: reward 组件 weight 必须为非负有限数 (>=0, finite), got {value!r}"
            )
        return fv


def default_reward_components() -> List[RewardComponent]:
    """Snapshot of the status-quo literal hierarchy as components (byte-equal defaults).

    format_penalty on (weight 1.0 → -1.0), accuracy_reward on (不可关),
    multi_tool_bonus on (weight 0.1 = the hardcoded default param of
    compute_outcome_reward), cost_penalty present but disabled (预留位).
    """
    return [
        RewardComponent(id=FORMAT_PENALTY, weight=DEFAULT_FORMAT_PENALTY_WEIGHT, enabled=True),
        RewardComponent(id=ACCURACY_REWARD, weight=1.0, enabled=True),
        RewardComponent(id=MULTI_TOOL_BONUS, weight=DEFAULT_MULTI_TOOL_BONUS_WEIGHT, enabled=True),
        RewardComponent(id=COST_PENALTY, weight=0.0, enabled=False),
    ]


class RewardSpec(BaseModel):
    """Switchable/weightable composition of the frozen outcome hierarchy.

    ``outcome="hierarchical"`` is the frozen default (层级不变合同, RL_design
    §3.2.1); ``outcome="binary"`` is the Flow-GRPO 0/1 correctness reward and
    ignores every component. ``components`` defaults to the现状 snapshot above; validation
    enforces: no duplicates, known ids only, accuracy_reward present AND
    enabled, reserved slots (cost_penalty) must stay disabled until their
    stage lands.
    """

    outcome: str = "hierarchical"
    components: List[RewardComponent] = Field(default_factory=default_reward_components)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _validate_components(self) -> "RewardSpec":
        known = {FORMAT_PENALTY, ACCURACY_REWARD, MULTI_TOOL_BONUS, COST_PENALTY}
        seen: set = set()
        for comp in self.components:
            if comp.id not in known:
                raise ValueError(f"unknown reward component {comp.id!r}; registry: {sorted(known)}")
            if comp.id in seen:
                raise ValueError(f"duplicate reward component {comp.id!r}")
            seen.add(comp.id)
        for required in REQUIRED_COMPONENTS:
            if required not in seen:
                raise ValueError(f"reward spec must include the main reward component {required!r}")
            comp = next(c for c in self.components if c.id == required)
            if not comp.enabled:
                raise ValueError(f"{required!r} is not switchable (main reward)")
        for reserved in RESERVED_COMPONENTS:
            if reserved in seen:
                comp = next(c for c in self.components if c.id == reserved)
                if comp.enabled:
                    raise ValueError(
                        f"{reserved!r} is a reserved schema slot (implementation deferred); "
                        "it must be created disabled"
                    )
        if self.outcome not in OUTCOME_MODES:
            raise ValueError(f"outcome mode must be one of {OUTCOME_MODES}, got {self.outcome!r}")
        return self

    def component(self, component_id: str) -> RewardComponent:
        for comp in self.components:
            if comp.id == component_id:
                return comp
        return RewardComponent(id=component_id, weight=0.0, enabled=False)

    def is_enabled(self, component_id: str) -> bool:
        return self.component(component_id).enabled

    def weight_of(self, component_id: str) -> float:
        return float(self.component(component_id).weight)


def reward_spec_from_tir(tir: Dict[str, Any]) -> RewardSpec:
    """Derive the live RewardSpec from a v1 algorithm.tir dict.

    Dual-read bridge for the RL4 v1 wiring: the tir dict's ``multi_tool_bonus``
    (DEFAULT_TIR :29 default 0.1, overridable in user yaml) feeds the spec's
    multi_tool_bonus component weight. Absent from the registry, the builtin
    component default (0.1) applies.
    """
    spec = RewardSpec()
    if "multi_tool_bonus" in (tir or {}):
        comps = list(spec.components)
        for i, comp in enumerate(comps):
            if comp.id == MULTI_TOOL_BONUS:
                weight = float(tir["multi_tool_bonus"])
                comps[i] = comp.model_copy(update={"weight": weight, "enabled": weight != 0.0})
        spec = RewardSpec(components=comps)
    return spec


def reward_spec_from_config(config: Any) -> "RewardSpec | None":
    """Read a RewardSpec off a config tree, v2 first then tir fallback; None if neither.

    v2: ``algorithm.reward`` section (the RL5 yaml slot) — validated
    strictly. v1 fallback: ``algorithm.tir.multi_tool_bonus`` (or a bare
    ``tir`` dict) feeds reward_spec_from_tir. Returns None when the config
    carries no reward surface at all so callers keep the byte-identical
    legacy literal path.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        algo_cfg = config.get("algorithm") or {}
        if isinstance(algo_cfg, dict):
            sec = algo_cfg.get("reward")
            if isinstance(sec, dict) and sec:
                return _spec_from_section(sec)
            tir = algo_cfg.get("tir")
            if isinstance(tir, dict) and "multi_tool_bonus" in tir:
                return reward_spec_from_tir(dict(tir))
        tir = config.get("tir")
        if isinstance(tir, dict) and "multi_tool_bonus" in tir:
            return reward_spec_from_tir(dict(tir))
        return None
    # attribute-style config (OmegaConf/dotdict hybrid)
    algo_cfg = getattr(config, "algorithm", None)
    sec = getattr(algo_cfg, "reward", None) if algo_cfg is not None else None
    if sec is not None:
        try:
            from omegaconf import OmegaConf

            as_dict = dict(OmegaConf.to_container(sec, resolve=True) or {})
        except Exception:
            as_dict = dict(sec or {})
        if as_dict:
            return _spec_from_section(as_dict)
    tir_obj = getattr(algo_cfg, "tir", None) if algo_cfg is not None else None
    if tir_obj is not None:
        try:
            from omegaconf import OmegaConf

            tir = dict(OmegaConf.to_container(tir_obj, resolve=True) or {})
        except Exception:
            tir = dict(tir_obj or {})
        if "multi_tool_bonus" in tir:
            return reward_spec_from_tir(tir)
    return None


def _spec_from_section(section: Dict[str, Any]) -> RewardSpec:
    """Normalize a v2 ``algorithm.reward'' section into a RewardSpec."""
    comps_raw = section.get("components")
    components: List[RewardComponent] = []
    if comps_raw is None:
        components = default_reward_components()
    else:
        for comp in comps_raw:
            if isinstance(comp, RewardComponent):
                components.append(comp)
                continue
            if not isinstance(comp, dict):
                comp = dict(comp or {})
            components.append(
                RewardComponent(
                    id=str(comp.get("id")),
                    weight=float(comp.get("weight", 1.0)),
                    enabled=bool(comp.get("enabled", True)),
                )
            )
    return RewardSpec(
        outcome=str(section.get("outcome") or "hierarchical"),
        components=components,
    )


def default_reward_spec() -> RewardSpec:
    """The现状-equivalent spec: handing this to compute_outcome_reward must
    reproduce the literal path value-for-value (locked by
    test_reward_spec_none_path_byte_equal)."""
    return RewardSpec(components=default_reward_components())
