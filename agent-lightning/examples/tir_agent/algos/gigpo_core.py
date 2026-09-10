"""Group-in-Group advantage: episode A^E + step A^S (GiGPO, NeurIPS 2025)."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _norm(values: np.ndarray, mode: str, eps: float) -> np.ndarray:
    mean = values.mean()
    if mode == "mean_std":
        std = float(values.std())
        return (values - mean) / (std + eps)
    return values - mean


def compute_gigpo_advantages(
    episode_rewards: Sequence[float],
    data_ids: Sequence[str],
    rollout_ids: Sequence[str],
    turn_indices: Sequence[int],
    anchor_obs: Sequence[str],
    *,
    step_advantage_w: float = 1.0,
    gamma: float = 1.0,
    mode: str = "mean_std",
    eps: float = 1e-8,
) -> np.ndarray:
    """Return per-row advantages aligned with transition-level batch rows.

    episode_rewards[i] is the scalar outcome of the row's parent rollout.
    Rows that share (data_id, anchor_obs) form a step group. Discount uses
    remaining turns within the same rollout as a proxy for R_t.
    Isolated anchors yield A^S=0 (GiGPO degrades to GRPO).
    """
    n = len(episode_rewards)
    rewards = np.asarray(episode_rewards, dtype=np.float64)
    a_e = np.zeros(n, dtype=np.float64)

    by_data: Dict[str, List[int]] = defaultdict(list)
    for i, did in enumerate(data_ids):
        by_data[str(did)].append(i)

    for idxs in by_data.values():
        # Unique rollouts in this prompt group
        roll_to_reward: Dict[str, float] = {}
        for i in idxs:
            roll_to_reward[str(rollout_ids[i])] = float(rewards[i])
        vals = np.array(list(roll_to_reward.values()), dtype=np.float64)
        if vals.size == 0:
            continue
        normed = _norm(vals, mode, eps)
        roll_adv = {rid: float(normed[j]) for j, rid in enumerate(roll_to_reward.keys())}
        for i in idxs:
            a_e[i] = roll_adv[str(rollout_ids[i])]

    # Discounted return per (rollout, turn): γ^{T-1-t} * episode_reward
    max_turn: Dict[str, int] = defaultdict(int)
    for i in range(n):
        rid = str(rollout_ids[i])
        max_turn[rid] = max(max_turn[rid], int(turn_indices[i]))

    disc = np.zeros(n, dtype=np.float64)
    for i in range(n):
        rid = str(rollout_ids[i])
        t = int(turn_indices[i])
        t_last = max_turn[rid]
        disc[i] = float(rewards[i]) * (gamma ** max(0, t_last - t))

    a_s = np.zeros(n, dtype=np.float64)
    by_anchor: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i in range(n):
        key = (str(data_ids[i]), str(anchor_obs[i]) if anchor_obs[i] is not None else "")
        if key[1] == "":
            continue
        by_anchor[key].append(i)

    for idxs in by_anchor.values():
        if len(idxs) < 2:
            continue
        vals = disc[idxs]
        normed = _norm(vals, mode, eps)
        for j, i in enumerate(idxs):
            a_s[i] = float(normed[j])

    return a_e + step_advantage_w * a_s


def broadcast_to_response_tokens(
    row_advantages: np.ndarray,
    response_mask: "torch.Tensor",
) -> "torch.Tensor":
    """Copy a per-row scalar onto every valid response token."""
    import torch

    adv = torch.as_tensor(row_advantages, device=response_mask.device, dtype=torch.float32)
    token_adv = torch.zeros_like(response_mask, dtype=torch.float32)
    token_adv = token_adv + adv.unsqueeze(-1)
    return token_adv * response_mask.to(dtype=torch.float32)
