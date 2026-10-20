"""Advantage paths: RL3 pipeline (base + ordered operator stack) with frozen legacy dispatch.

The production entry ``apply_tir_advantages`` resolves an AdvantagePipelineSpec
(base ∈ {grpo, igpo, gigpo} × ordered operators from the five-value registry)
and executes base-then-operators. The pre-RL3 eight-branch if/return dispatch
is preserved verbatim as ``legacy_apply_tir_advantages`` — the frozen comparison
oracle for ``mas/tests/test_preset_equivalence.py`` (kept one version cycle,
RL3 double-implementation window; do not rewrite).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage

from .gigpo_core import broadcast_to_response_tokens, compute_gigpo_advantages
from rl.loss import (
    ADVANTAGE_BASES,
    ADVANTAGE_OPERATORS,
    AdvantageOperatorSpec,
    AdvantagePipelineSpec,
)
from rl.rewards.outcome import discounted_returns, group_zscore

# Process-global no-op degrade counter, drained by the daemon-side
# training/pipeline_noop metric emitter (key pre-wired in RL2 (c)).
_PIPELINE_NOOP_REGISTRY: Optional[Dict[str, Any]] = None


def _tir_algo(config: Any) -> str:
    algo = None
    if hasattr(config, "tir_algo"):
        algo = config.tir_algo
    elif isinstance(config, dict):
        algo = config.get("tir_algo")
    return str(algo or "grpo").lower()


def _tir_cfg(config: Any) -> Dict[str, Any]:
    if hasattr(config, "tir"):
        try:
            from omegaconf import OmegaConf

            return dict(OmegaConf.to_container(config.tir, resolve=True) or {})
        except Exception:
            return dict(getattr(config, "tir", {}) or {})
    if isinstance(config, dict):
        return dict(config.get("tir") or {})
    return {}


def _as_list(arr: Any) -> List[Any]:
    if arr is None:
        return []
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return list(arr)


def apply_tir_advantages(
    batch: DataProto,
    *,
    adv_estimator: Any,
    gamma: float,
    lam: float,
    num_repeat: int,
    norm_adv_by_std_in_grpo: bool,
    config: Any,
) -> DataProto:
    """Replace or reshape GRPO advantages according to algorithm.tir_algo.

    RL3 pipeline implementation: the eight-branch if/return dispatch is frozen
    below as ``legacy_apply_tir_advantages`` (byte-for-byte, same-release
    comparison oracle); this single entry now resolves an
    ``AdvantagePipelineSpec`` (base + ordered operator stack) — v1 callers
    simply pass ``tir_algo`` and the expansion table (RL_design §3.1.6 /
    §3.4.4 snapshot table, the authoritative definition) maps it onto the
    same base × operator composition the legacy branches implemented.
    """
    algo = _tir_algo(config)
    tir = _tir_cfg(config)

    spec = resolve_pipeline_spec(config)

    batch = compute_advantage(
        batch,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    base_fn = _BASES[spec.base]
    batch = base_fn(batch, tir, spec.base_params)
    for op in spec.operators:
        op_fn = _OPERATORS[op.id]
        batch = op_fn(batch, op.params, tir)
    n_flat = count_zero_variance_groups(batch)
    meta = dict(getattr(batch, "meta_info", None) or {})
    meta["n_zero_variance_groups"] = n_flat
    batch.meta_info = meta
    print(f"n_zero_variance_groups={n_flat}", flush=True)
    return batch


def count_zero_variance_groups(batch: DataProto) -> int:
    """Groups whose outcome rewards are all equal (all 0 or all 1) have advantage 0.

    A group of one is excluded: GRPO already forces its std to 1, and there is
    no within-group contrast to count.
    """
    rewards = None
    if "token_level_rewards" in batch.batch:
        rewards = batch.batch["token_level_rewards"]
    elif "token_level_scores" in batch.batch:
        rewards = batch.batch["token_level_scores"]
    if rewards is None:
        return 0
    index = None
    for key in ("uid", "data_id_list"):
        if key in batch.non_tensor_batch:
            index = batch.non_tensor_batch[key]
            break
    if index is None:
        return 0
    scores = rewards.detach().float().sum(dim=-1).cpu()
    buckets: Dict[Any, List[float]] = defaultdict(list)
    for i, idx in enumerate(list(index)):
        buckets[idx].append(float(scores[i]))
    flat = 0
    for vals in buckets.values():
        if len(vals) > 1 and max(vals) == min(vals):
            flat += 1
    return flat


def legacy_apply_tir_advantages(
    batch: DataProto,
    *,
    adv_estimator: Any,
    gamma: float,
    lam: float,
    num_repeat: int,
    norm_adv_by_std_in_grpo: bool,
    config: Any,
) -> DataProto:
    """Frozen comparison oracle: the pre-RL3 eight-branch dispatch, verbatim.

    Kept for one version cycle (RL3 double-implementation window) so
    ``mas/tests/test_preset_equivalence.py`` can assert element-wise equality
    of the pipeline path against the legacy path for all 8 presets. Do not
    modify: any change here defeats the equivalence gate.
    """
    algo = _tir_algo(config)
    tir = _tir_cfg(config)

    batch = compute_advantage(
        batch,
        adv_estimator=adv_estimator,
        gamma=gamma,
        lam=lam,
        num_repeat=num_repeat,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )

    if algo == "flow_grpo":
        # Stock GRPO on the broadcast trajectory reward. No branch credit.
        return batch
    if algo == "arpo":
        return batch
    if algo == "appo":
        return apply_appo_credit(
            batch,
            discount=float(tir.get("reward_scale_discount", 0.9)),
        )
    if algo == "grpo":
        batch = _maybe_zero_prefix(batch, tir)
        return batch
    if algo == "rae":
        from .rae_advantage import (
            apply_rae_full_tgt_advantages,
            apply_rae_verdict_advantages,
            zero_prefix_advantages_for_children,
        )

        batch = zero_prefix_advantages_for_children(batch)
        if bool(tir.get("rae_full_tgt", False)):
            batch = apply_rae_full_tgt_advantages(
                batch,
                epsilon_f=float(tir.get("rae_epsilon_f", 1e-3)),
            )
        else:
            batch = apply_rae_verdict_advantages(
                batch,
                p_plus=float(tir.get("rae_p_plus", 0.8)),
                epsilon_f=float(tir.get("rae_epsilon_f", 1e-3)),
            )
        return batch
    if algo == "aepo":
        return _apply_entropy_aware(batch, float(tir.get("entropy_adv_coef", 0.1)))
    if algo == "igpo":
        return _apply_igpo(batch, float(tir.get("gamma", 1.0)))
    if algo == "gigpo":
        return _apply_gigpo(
            batch,
            step_w=float(tir.get("step_advantage_w", 1.0)),
            gamma=float(tir.get("gamma", 1.0)),
            mode=str(tir.get("gigpo_mode", "mean_std")),
        )
    return batch


# ---------------------------------------------------------------------------
# RL3 pipeline: spec resolution (v1 tir_algo → expansion table) + validation
# ---------------------------------------------------------------------------

# RL_design §3.1.6 / §3.4.4 snapshot table — the authoritative preset→pipeline
# expansion. Every entry is what the matching legacy if-return branch does
# with today's defaults. Params that the legacy branch read from the tir dict
# (reward_scale_discount / gamma / step_advantage_w / gigpo_mode ...) keep
# that same runtime tir-dict read inside the operator/base wrappers below, so
# user yaml overrides keep flowing through unchanged (param chain preserved).
_PIPELINE_EXPANSION: Dict[str, List[str]] = {
    "flow_grpo": [],  # grpo base, operators: [] (KL overlay is config-side)
    "arpo": [],       # advantage passthrough (difference is sampling-side)
    "grpo": [],  # aligned to Flow-GRPO (no MAS-only zero_prefix operator)
    "appo": ["appo_credit"],
    "rae": ["zero_prefix", "rae_verdict"],
    "aepo": ["aepo_entropy"],
    "igpo": [],
    "gigpo": [],
}
_PIPELINE_EXPANSION_BASE = {
    "grpo": "grpo",
    "flow_grpo": "grpo",
    "arpo": "grpo",
    "appo": "grpo",
    "rae": "grpo",
    "aepo": "grpo",
    "igpo": "igpo",
    "gigpo": "gigpo",
}
# NOTE (RL5): _PIPELINE_EXPANSION/_PIPELINE_EXPANSION_BASE above now mirror
# the authoritative PRESET_EXPANSION_* tables in rl/loss.py (via expand_preset);
# they are kept as legacy-dispatch documentation + test anchors only.


# RL5: the expansion tables above are re-exported through the formal
# ``expand_preset`` public API in rl/loss.py (single authority: RL_design
# §3.4.4 snapshot table, post-RL0-errata §2.3 truth table). resolve-time
# checks (validate_pipeline_spec) and the v2 section reader below live here
# (advantage-tensor side); expand_preset itself is import-safe with no
# torch/AGL on the path (rl/loss.py), so the save/preflight layers can call
# it without touching the tensor module.
from rl.loss import expand_preset  # noqa: E402  (re-export; formal public API)


def resolve_pipeline_spec(config: Any) -> "AdvantagePipelineSpec":
    """Resolve a config into an AdvantagePipelineSpec (v1/v2 dual-track).

    v1 (``tir_algo`` single-select, permanent): expand via the snapshot
    table. v2 (RL5 wired): if the config carries a non-empty
    ``algorithm.advantage`` section, its base/operators are validated and
    used directly. rae's ``rae_full_tgt=True`` tir flag swaps the trailing
    verdict operator for rae_full_tgt — mirroring the legacy branch at
    advantage.py:92-99. The legacy ``unknown-tir_algo → grpo passthrough``
    tolerance of the tensor path is preserved here (byte-level compatibility
    with the old dispatch tail; the formal expand_preset API is the one
    that refuses unknown names — hot-path callers keep today's shape).
    """
    algo = _tir_algo(config)
    tir = _tir_cfg(config)

    section = _advantage_section(config)
    if section is not None:
        spec = _spec_from_section(section, tir)
    else:
        # v1 expansion consults the formal public API (rl/loss.py
        # expand_preset) for snapshot fidelity; unknown names keep the
        # historical grpo passthrough so old configs never break the run.
        try:
            spec = expand_preset(algo)
        except ValueError:
            spec = AdvantagePipelineSpec(base="grpo", operators=[])
            spec = AdvantagePipelineSpec(base="grpo", operators=[])
        if algo == "rae" and bool(tir.get("rae_full_tgt", False)):
            # legacy rae fork (advantage.py:92-99, now in legacy_apply_*):
            # full_tgt replaces the verdict operator, not appended.
            spec = AdvantagePipelineSpec(
                base="grpo",
                operators=[
                    AdvantageOperatorSpec(id="zero_prefix"),
                    AdvantageOperatorSpec(id="rae_full_tgt"),
                ],
            )
    validate_pipeline_spec(spec)
    return spec


def _advantage_section(config: Any) -> Optional[Any]:
    """Read a v2-style ``advantage'' section off the config, if present."""
    if config is None:
        return None
    if isinstance(config, dict):
        algo_cfg = config.get("algorithm") or {}
        if isinstance(algo_cfg, dict):
            sec = algo_cfg.get("advantage")
            if isinstance(sec, dict) and sec:
                return sec
        sec = config.get("advantage")
        if isinstance(sec, dict) and sec:
            return sec
        return None
    for attr in ("advantage",):
        candidate = getattr(config, attr, None)
        if candidate is not None:
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(candidate) and not bool(OmegaConf.is_empty(candidate)):
                    return candidate
            except Exception:
                pass
            if isinstance(candidate, dict) and candidate:
                return candidate
    return None


def _spec_from_section(section: Any, tir: Dict[str, Any]) -> "AdvantagePipelineSpec":
    """Normalize a v2 advantage section into an AdvantagePipelineSpec."""
    if not isinstance(section, dict):
        try:
            from omegaconf import OmegaConf

            section = dict(OmegaConf.to_container(section, resolve=True) or {})
        except Exception:
            section = dict(section or {})
    base = str(section.get("base") or "grpo")
    raw_ops = section.get("operators") or []
    ops: List[AdvantageOperatorSpec] = []
    for op in raw_ops:
        if isinstance(op, AdvantageOperatorSpec):
            ops.append(op)
            continue
        if isinstance(op, dict):
            try:
                from omegaconf import OmegaConf

                op = dict(OmegaConf.to_container(op, resolve=True) or {})
            except Exception:
                op = dict(op)
        op_id = str(op.get("id") or "")
        params = dict(op.get("params") or {})
        ops.append(AdvantageOperatorSpec(id=op_id, params=params))
    return AdvantagePipelineSpec(base=base, operators=ops)


def validate_pipeline_spec(spec: "AdvantagePipelineSpec") -> None:
    """Static composition rules (RL_design §3.1.4 R1/R5 + registry checks).

    Fail-fast: raises ValueError on any illegal composition, so a bad config
    cannot even start a training run (enforcement point for both the v1
    expansion table — which can never produce these — and direct v2 specs).
    """
    if spec.base not in ADVANTAGE_BASES:
        raise ValueError(
            f"advantage pipeline base must be one of {ADVANTAGE_BASES}, got {spec.base!r}"
        )
    seen: set = set()
    for op in spec.operators:
        if op.id not in ADVANTAGE_OPERATORS:
            raise ValueError(
                f"unknown advantage operator {op.id!r}; registry: {ADVANTAGE_OPERATORS}"
            )
        if op.id in seen:
            raise ValueError(f"duplicate advantage operator {op.id!r} in the stack")
        seen.add(op.id)
    has_verdict = "rae_verdict" in seen
    has_full_tgt = "rae_full_tgt" in seen
    # R1: rae_verdict XOR rae_full_tgt (both consume verdict_list and reshape
    # rows — stacking them double-counts the adjudication signal).
    if has_verdict and has_full_tgt:
        raise ValueError("rae_verdict and rae_full_tgt are mutually exclusive (R1)")
    # R5: rae_full_tgt writes token-level signal onto branch rows while
    # appo_credit's semantic is "branch rows leave the actor" — contradictory.
    if has_full_tgt and "appo_credit" in seen:
        raise ValueError("rae_full_tgt and appo_credit cannot be stacked (R5)")


# ---------------------------------------------------------------------------
# RL3 pipeline: bases + operator registry (pure DataProto→DataProto)
# ---------------------------------------------------------------------------
# Registry contract (RL_design §3.1.2): every base/operator is a pure
# DataProto→DataProto tensor function; reads non_tensor_batch columns but
# never writes them; on missing consumed columns it degrades to no-op and
# bumps the ``training/pipeline_noop`` counter (daemon pre-wired the metric
# key in RL2) — never raises. Consumed-column declarations are exposed via
# OPERATOR_CONSUMED_COLUMNS for audits/tests.


def _noop_bump(reason: str) -> None:
    """Count a no-op degrade into the daemon-side training/pipeline_noop metric.

    The daemon pre-wires ``training/pipeline_noop`` = 0.0 in its metrics dict
    (rl/hooks/daemon.py, RL2 (c)). Operators run inside the trainer process,
    so the canonical sink is a process-global counter the daemon-side metric
    emitter can drain; here we only increment it (best-effort, no import of
    torch/ray machinery).
    """
    reg = _noop_registry()
    reg["count"] = int(reg.get("count", 0)) + 1
    reg.setdefault("reasons", {}).setdefault(reason, 0)
    reg["reasons"][reason] = int(reg["reasons"][reason]) + 1


def _noop_registry() -> Dict[str, Any]:
    """Global noop-counter registry (same process as the operator stack)."""
    global _PIPELINE_NOOP_REGISTRY
    if _PIPELINE_NOOP_REGISTRY is None:
        _PIPELINE_NOOP_REGISTRY = {"count": 0, "reasons": {}}
    return _PIPELINE_NOOP_REGISTRY


def reset_pipeline_noop_counter() -> Dict[str, Any]:
    """Atomically read-and-clear the noop counter (daemon-side drain hook)."""
    reg = _noop_registry()
    snap = {"count": int(reg.get("count", 0)), "reasons": dict(reg.get("reasons", {}))}
    reg["count"] = 0
    reg["reasons"] = {}
    return snap


def pipeline_noop_count() -> int:
    return int(_noop_registry().get("count", 0))


# --- bases ---------------------------------------------------------------


def _base_grpo(batch: DataProto, tir: Dict[str, Any], params: Dict[str, Any]) -> DataProto:
    """Stock GRPO outcome advantage — nothing beyond the stage-0 compute."""
    return batch


def _base_igpo(batch: DataProto, tir: Dict[str, Any], params: Dict[str, Any]) -> DataProto:
    # Legacy branch read tir gamma; keep exactly that read (default 1.0).
    return _apply_igpo(batch, float(tir.get("gamma", 1.0)))


def _base_gigpo(batch: DataProto, tir: Dict[str, Any], params: Dict[str, Any]) -> DataProto:
    return _apply_gigpo(
        batch,
        step_w=float(tir.get("step_advantage_w", 1.0)),
        gamma=float(tir.get("gamma", 1.0)),
        mode=str(tir.get("gigpo_mode", "mean_std")),
    )


_BASES = {
    "grpo": _base_grpo,
    "igpo": _base_igpo,
    "gigpo": _base_gigpo,
}

BASE_CONSUMED_COLUMNS = {
    "grpo": ("token_level_scores",),
    "igpo": ("ig_logprob_delta", "turn_index_list", "rollout_id_list", "token_level_scores"),
    "gigpo": ("rollout_id_list", "turn_index_list", "anchor_obs", "token_level_scores"),
}


# --- operators ------------------------------------------------------------


def _op_zero_prefix(batch: DataProto, params: Dict[str, Any], tir: Dict[str, Any]) -> DataProto:
    """R0: zero shared-prefix advantage for child/probe rows.

    Legacy enable flag ``zero_prefix_adv_for_children`` lives in the tir dict
    (default True); keep the exact same read so grpo preset equivalence and
    the explicit-False escape hatch behave byte-identically. The try/except
    mirrors the frozen legacy wrapper (_maybe_zero_prefix): any kernel
    failure degrades to a counted no-op, never a raise.
    """
    if not bool(tir.get("zero_prefix_adv_for_children", True)):
        return batch
    try:
        from .rae_advantage import zero_prefix_advantages_for_children

        return zero_prefix_advantages_for_children(batch)
    except Exception:
        _noop_bump("zero_prefix:kernel_noop")
        return batch


def _op_appo_credit(batch: DataProto, params: Dict[str, Any], tir: Dict[str, Any]) -> DataProto:
    """Branch rows leave the actor; outcome written back onto the parent."""
    needed = ("advantages",)
    missing = _missing_batch_keys(batch, tensors=needed)
    if missing:
        _noop_bump(f"appo_credit:missing={','.join(missing)}")
        return batch
    # Legacy branch read tir reward_scale_discount (default 0.9); params
    # override the tir read only when the v2 spec carried an explicit value.
    discount = params.get("discount")
    if discount is None:
        discount = tir.get("reward_scale_discount", 0.9)
    return apply_appo_credit(batch, discount=float(discount))


def _op_aepo_entropy(batch: DataProto, params: Dict[str, Any], tir: Dict[str, Any]) -> DataProto:
    """AEPO-lite entropy reshape Â = Â·(1 + a·z(H)) (scale clamped [0.1, 3.0])."""
    if "entropys" not in batch.batch and "old_log_probs" not in batch.batch:
        _noop_bump("aepo_entropy:missing=entropys|old_log_probs")
        return batch
    coef = params.get("coef")
    if coef is None:
        coef = tir.get("entropy_adv_coef", 0.1)
    return _apply_entropy_aware(batch, float(coef))


def _op_rae_verdict(batch: DataProto, params: Dict[str, Any], tir: Dict[str, Any]) -> DataProto:
    """R1-lite: scale row advantages by validate/invalidate/abstain verdict."""
    if "advantages" not in batch.batch:
        _noop_bump("rae_verdict:missing=advantages")
        return batch
    if not _as_list(batch.non_tensor_batch.get("verdict_list")):
        _noop_bump("rae_verdict:missing=verdict_list")
        return batch
    from .rae_advantage import apply_rae_verdict_advantages

    p_plus = params.get("p_plus")
    if p_plus is None:
        p_plus = tir.get("rae_p_plus", 0.8)
    epsilon_f = params.get("epsilon_f")
    if epsilon_f is None:
        epsilon_f = tir.get("rae_epsilon_f", 1e-3)
    return apply_rae_verdict_advantages(
        batch,
        p_plus=float(p_plus),
        epsilon_f=float(epsilon_f),
    )


def _op_rae_full_tgt(batch: DataProto, params: Dict[str, Any], tir: Dict[str, Any]) -> DataProto:
    """R1-full: A_RAE ≈ log π_tgt − log π_old as response-token bias."""
    if "advantages" not in batch.batch:
        _noop_bump("rae_full_tgt:missing=advantages")
        return batch
    if not _as_list(batch.non_tensor_batch.get("verdict_list")):
        _noop_bump("rae_full_tgt:missing=verdict_list")
        return batch
    from .rae_advantage import apply_rae_full_tgt_advantages

    epsilon_f = params.get("epsilon_f")
    if epsilon_f is None:
        epsilon_f = tir.get("rae_epsilon_f", 1e-3)
    return apply_rae_full_tgt_advantages(batch, epsilon_f=float(epsilon_f))


_OPERATORS = {
    "zero_prefix": _op_zero_prefix,
    "appo_credit": _op_appo_credit,
    "aepo_entropy": _op_aepo_entropy,
    "rae_verdict": _op_rae_verdict,
    "rae_full_tgt": _op_rae_full_tgt,
}

OPERATOR_CONSUMED_COLUMNS = {
    "zero_prefix": ("rollout_role", "resume_boundary"),
    "appo_credit": ("rollout_role", "resume_parent_id"),
    "aepo_entropy": (),
    "rae_verdict": ("verdict_list",),
    "rae_full_tgt": ("verdict_list", "action_key_list"),
}


def _missing_batch_keys(
    batch: DataProto,
    *,
    tensors: Sequence[str] = (),
    ntb: Sequence[str] = (),
) -> List[str]:
    missing: List[str] = []
    for key in tensors:
        if key not in batch.batch:
            missing.append(key)
    for key in ntb:
        if key not in batch.non_tensor_batch:
            missing.append(key)
    return missing


def _is_branch_role(role: Any) -> bool:
    return str(role or "parent").lower() in ("child", "probe", "branch")


def _split_index(raw: Any, width: int) -> int:
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        return width - 1
    if idx < 0:
        return width - 1
    return min(idx, width - 1)


def apply_appo_credit(batch: DataProto, *, discount: float = 0.9) -> DataProto:
    """Dual-group GRPO, then map branch credit onto the parent prefix.

    Init and branch rows are normalized separately with the VERL GRPO rule
    (a group of one keeps ``score / (1 + 1e-6)``). Branch rows are then
    zeroed, including ``loss_mask`` when that column exists, so they do not
    enter the actor. Each branch scalar is averaged onto the parent tokens
    at and before the split index (``appo_branch_token``, else
    ``resume_boundary``).

    ``discount`` is the reward-scale γ inside M, applied before the group
    norm when ``appo_branch_score`` is present. It does not rescale the
    mapped advantage.
    """
    if "advantages" not in batch.batch:
        return batch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    from .appo_branching import average_reward_scales

    template = batch.batch["advantages"]
    n = int(template.shape[0])
    width = int(template.shape[1])
    ntb = batch.non_tensor_batch
    roles = _as_list(ntb.get("rollout_role"))
    data_ids = _as_list(ntb["data_id_list"] if "data_id_list" in ntb else ntb.get("uid"))
    if "rollout_id_list" in ntb:
        rollout_ids = _as_list(ntb["rollout_id_list"])
    else:
        rollout_ids = _as_list(ntb.get("rollout_id"))
    parent_links = _as_list(ntb.get("resume_parent_id"))
    token_idx = _as_list(ntb.get("appo_branch_token"))
    boundaries = _as_list(ntb.get("resume_boundary"))
    branch_scores = _as_list(ntb.get("appo_branch_score"))
    if "response_mask" in batch.batch:
        mask = batch.batch["response_mask"].to(dtype=template.dtype)
    else:
        mask = torch.ones_like(template)
    if "token_level_rewards" in batch.batch:
        rewards = batch.batch["token_level_rewards"].clone()
    elif "token_level_scores" in batch.batch:
        rewards = batch.batch["token_level_scores"].clone()
    else:
        rewards = torch.zeros_like(template)

    branch_rows: List[int] = []
    parent_rows: List[int] = []
    for i in range(n):
        role = roles[i] if i < len(roles) else "parent"
        if _is_branch_role(role):
            branch_rows.append(i)
        else:
            parent_rows.append(i)

    if branch_scores:
        by_parent: Dict[int, List[int]] = defaultdict(list)
        id_to_row = {}
        for i in parent_rows:
            rid = str(rollout_ids[i]) if i < len(rollout_ids) and rollout_ids[i] else ""
            did = str(data_ids[i]) if i < len(data_ids) and data_ids[i] is not None else ""
            if rid:
                id_to_row[(did, rid)] = i
        for i in branch_rows:
            did = str(data_ids[i]) if i < len(data_ids) and data_ids[i] is not None else ""
            parent_id = str(parent_links[i]) if i < len(parent_links) and parent_links[i] else ""
            parent = id_to_row.get((did, parent_id))
            if parent is not None:
                by_parent[parent].append(i)
        for parent, children in by_parent.items():
            splits = []
            scores = []
            for child in children:
                raw_split = token_idx[child] if child < len(token_idx) and token_idx[child] not in ("", None) else (
                    boundaries[child] if child < len(boundaries) else width - 1
                )
                splits.append(_split_index(raw_split, width))
                score = branch_scores[child] if child < len(branch_scores) else 0.0
                try:
                    scores.append(float(score or 0.0))
                except (TypeError, ValueError):
                    scores.append(0.0)
            scaled = average_reward_scales(width, splits, scores, float(discount))
            rewards[parent] = rewards[parent] * rewards.new_tensor(scaled)

    index = np.empty(n, dtype=object)
    for i in range(n):
        did = str(data_ids[i]) if i < len(data_ids) and data_ids[i] is not None else ""
        kind = "branch" if i in set(branch_rows) else "init"
        index[i] = f"{did}|{kind}"
    adv, _returns = compute_grpo_outcome_advantage(
        rewards, mask, index, norm_adv_by_std_in_grpo=True,
    )
    adv = adv.clone()
    scalars: List[float] = []
    for i in range(n):
        valid = mask[i] > 0
        if bool(valid.any()):
            scalars.append(float(adv[i, valid][0]))
        else:
            scalars.append(0.0)
    for i in branch_rows:
        adv[i] = 0

    id_to_row = {}
    for i in parent_rows:
        rid = str(rollout_ids[i]) if i < len(rollout_ids) and rollout_ids[i] else ""
        did = str(data_ids[i]) if i < len(data_ids) and data_ids[i] is not None else ""
        if rid:
            id_to_row[(did, rid)] = i
    buckets: Dict[tuple, List[float]] = defaultdict(list)
    for i in branch_rows:
        did = str(data_ids[i]) if i < len(data_ids) and data_ids[i] is not None else ""
        parent_id = str(parent_links[i]) if i < len(parent_links) and parent_links[i] else ""
        parent = id_to_row.get((did, parent_id))
        if parent is None:
            continue
        raw_split = token_idx[i] if i < len(token_idx) and token_idx[i] not in ("", None) else (
            boundaries[i] if i < len(boundaries) else width - 1
        )
        split = _split_index(raw_split, width)
        for t in range(split + 1):
            if float(mask[parent, t]) == 0.0:
                continue
            buckets[(parent, t)].append(scalars[i])
    for (row, t), vals in buckets.items():
        adv[row, t] = adv[row, t] + (sum(vals) / len(vals))

    batch.batch["advantages"] = adv
    batch.batch["returns"] = adv.clone()
    if "token_level_rewards" in batch.batch:
        batch.batch["token_level_rewards"] = rewards
    if "loss_mask" in batch.batch and branch_rows:
        loss_mask = batch.batch["loss_mask"].clone()
        for i in branch_rows:
            loss_mask[i] = 0
        batch.batch["loss_mask"] = loss_mask
    return batch


def _maybe_zero_prefix(batch: DataProto, tir: Dict[str, Any]) -> DataProto:
    if not bool(tir.get("zero_prefix_adv_for_children", True)):
        return batch
    try:
        from .rae_advantage import zero_prefix_advantages_for_children

        return zero_prefix_advantages_for_children(batch)
    except Exception:
        return batch


def _apply_entropy_aware(batch: DataProto, coef: float) -> DataProto:
    """AEPO-lite: Â = Â_acc * (1 + a * Â_ΔH) using old_log_prob entropy if present."""
    if "entropys" not in batch.batch and "old_log_probs" not in batch.batch:
        return batch
    response_mask = batch.batch["response_mask"]
    advantages = batch.batch["advantages"]
    if "entropys" in batch.batch:
        ent = batch.batch["entropys"]
    else:
        # fallback: token-level |old_log_prob| as uncertainty proxy
        ent = -batch.batch["old_log_probs"]
    mask = response_mask.to(dtype=ent.dtype)
    denom = mask.sum().clamp_min(1.0)
    mean = (ent * mask).sum() / denom
    var = ((ent - mean) ** 2 * mask).sum() / denom
    std = torch.sqrt(var.clamp_min(1e-8))
    ent_z = (ent - mean) / std
    scale = (1.0 + coef * ent_z).clamp(min=0.1, max=3.0)
    batch.batch["advantages"] = advantages * scale * mask + advantages * (1.0 - mask)
    return batch


def _apply_igpo(batch: DataProto, gamma: float) -> DataProto:
    """Turn-level discounted return from IG + outcome, replacing GRPO Â."""
    ntb = batch.non_tensor_batch
    ig_raw = _as_list(ntb.get("ig_logprob_delta"))
    if not ig_raw:
        return batch
    # Latent-bug small fix (RL3 report §5.1): the bare ``or`` fallback was
    # truth-value-ambiguous on the twin-tuple ndarray/list shapes DataProto
    # carries in production (AGL daemon injects data_id_list as np.ndarray,
    # daemon.py:1143). None-ness is the only real fallback condition —
    # data_id_list present-and-non-empty must always win over uid.
    _raw_ids = ntb.get("data_id_list")
    data_ids = _as_list(ntb.get("uid") if _raw_ids is None else _raw_ids)
    rollout_ids = _as_list(ntb.get("rollout_id_list"))
    turn_indices = _as_list(ntb.get("turn_index_list"))
    outcome = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
    response_mask = batch.batch["response_mask"]
    n = response_mask.shape[0]
    if len(ig_raw) != n or len(rollout_ids) != n:
        return batch

    ig_vals = np.array([float(x) if x is not None else 0.0 for x in ig_raw], dtype=np.float64)
    # Separate z-norm within each prompt group
    by_data: Dict[str, List[int]] = defaultdict(list)
    for i, did in enumerate(data_ids):
        by_data[str(did)].append(i)

    ig_z = np.zeros(n, dtype=np.float64)
    out_z = np.zeros(n, dtype=np.float64)
    for idxs in by_data.values():
        ig_z[idxs] = group_zscore([ig_vals[i] for i in idxs])
        # outcome is rollout-level: take unique rollouts
        roll_out: Dict[str, float] = {}
        for i in idxs:
            roll_out[str(rollout_ids[i])] = float(outcome[i])
        z_map = dict(zip(roll_out.keys(), group_zscore(list(roll_out.values()))))
        for i in idxs:
            out_z[i] = z_map[str(rollout_ids[i])]

    # Combined turn reward: IG on non-last turns, outcome on last turn of each rollout
    max_turn: Dict[str, int] = defaultdict(int)
    for i in range(n):
        max_turn[str(rollout_ids[i])] = max(max_turn[str(rollout_ids[i])], int(turn_indices[i]))

    r_turn = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rid = str(rollout_ids[i])
        if int(turn_indices[i]) == max_turn[rid]:
            r_turn[i] = out_z[i]
        else:
            r_turn[i] = ig_z[i]

    # Discount within each rollout (increasing turn order)
    row_adv = np.zeros(n, dtype=np.float64)
    by_roll: Dict[str, List[int]] = defaultdict(list)
    for i in range(n):
        by_roll[str(rollout_ids[i])].append(i)
    for idxs in by_roll.values():
        idxs_sorted = sorted(idxs, key=lambda i: int(turn_indices[i]))
        rets = discounted_returns([r_turn[i] for i in idxs_sorted], gamma)
        for i, ret in zip(idxs_sorted, rets):
            row_adv[i] = ret

    token_adv = broadcast_to_response_tokens(row_adv, response_mask)
    batch.batch["advantages"] = token_adv
    batch.batch["returns"] = token_adv
    return batch


def _apply_gigpo(batch: DataProto, step_w: float, gamma: float, mode: str) -> DataProto:
    ntb = batch.non_tensor_batch
    # Latent-bug small fix (RL3 report §5.1): same bare-``or`` ambiguity as
    # _apply_igpo — the AGL daemon injects data_id_list as np.ndarray
    # (daemon.py:1143), np.ndarray truthiness is undefined and would raise
    # here before gigpo_core ever ran. None-ness is the only fallback.
    _raw_ids = ntb.get("data_id_list")
    data_ids = _as_list(ntb.get("uid") if _raw_ids is None else _raw_ids)
    rollout_ids = _as_list(ntb.get("rollout_id_list"))
    turn_indices = _as_list(ntb.get("turn_index_list"))
    anchors = _as_list(ntb.get("anchor_obs"))
    response_mask = batch.batch["response_mask"]
    n = response_mask.shape[0]
    if not rollout_ids or len(rollout_ids) != n:
        return batch
    if not anchors or len(anchors) != n:
        anchors = [""] * n
    if not turn_indices or len(turn_indices) != n:
        turn_indices = [0] * n
    outcome = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy()
    row_adv = compute_gigpo_advantages(
        outcome,
        data_ids,
        rollout_ids,
        turn_indices,
        [str(a) for a in anchors],
        step_advantage_w=step_w,
        gamma=gamma,
        mode=mode,
    )
    token_adv = broadcast_to_response_tokens(row_adv, response_mask)
    batch.batch["advantages"] = token_adv
    batch.batch["returns"] = token_adv
    return batch
