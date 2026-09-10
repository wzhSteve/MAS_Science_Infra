"""Advantage paths: GRPO (stock), IGPO turn-level return, GiGPO, AEPO entropy reshape."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
from verl import DataProto
from verl.trainer.ppo.ray_trainer import compute_advantage

from .gigpo_core import broadcast_to_response_tokens, compute_gigpo_advantages
from .rewards import discounted_returns, group_zscore


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
    """Replace or reshape GRPO advantages according to algorithm.tir_algo."""
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

    if algo in ("grpo", "arpo"):
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
    data_ids = _as_list(ntb.get("data_id_list") or ntb.get("uid"))
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
    data_ids = _as_list(ntb.get("data_id_list") or ntb.get("uid"))
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
