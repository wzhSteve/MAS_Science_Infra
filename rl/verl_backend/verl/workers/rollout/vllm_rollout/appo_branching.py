# APPO: Procedural rollout branching logic
#
# 核心设计：
# 1. 在每条 rollout 内计算 token entropy z-score
# 2. 可选地结合 future value z-score，形成 Branching Score
# 3. 按 Branching Score 选择 top-B 分裂点，直到达到 rollout 上限
# 4. reward 缩放因子 M：基于所选分裂点的 Branching Score
# 5. initial / branch 分组 GRPO advantage 在 trainer 中计算

import logging
import math
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _is_meaningless_token(token_str: str) -> bool:
    stripped = token_str.strip()
    if not stripped:
        return True
    if all(ch in string.punctuation for ch in stripped):
        return True
    return False


def _calc_token_entropy(logprob_dict: dict) -> float:
    if not logprob_dict:
        return 0.0
    log_probs = [v.logprob for v in logprob_dict.values()]
    probs = [math.exp(lp) for lp in log_probs]
    entropy = -sum(p * lp for p, lp in zip(probs, log_probs) if p > 0)
    return max(0.0, entropy)


def _zscore(values: List[float]) -> List[float]:
    if not values:
        return []
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    std_val = math.sqrt(variance) if variance > 1e-16 else 1.0
    return [(v - mean_val) / std_val for v in values]


def compute_future_values_from_logratio(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: float,
    clip_eps: float,
) -> torch.Tensor:
    """
    Compute APPO future values Ω_i = exp(sum_{j>=i} gamma^{j-i} log_ratio_j).

    This helper implements the paper's future-value term. It is used by code paths
    that already have both current-policy and behavior-policy log-probs available.
    Rollout-time branching can only pass this term if such log-probs are available
    before branch generation.
    """
    if log_ratio.dim() != 2:
        raise ValueError(f"log_ratio must be 2-D, got shape {tuple(log_ratio.shape)}")
    batch_size, response_len = log_ratio.shape
    device = log_ratio.device
    dtype = log_ratio.dtype

    positions = torch.arange(response_len, device=device)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)
    causal_decay = torch.where(
        distance >= 0,
        torch.pow(torch.tensor(gamma, dtype=dtype, device=device), distance.clamp(min=0)),
        torch.zeros((), dtype=dtype, device=device),
    )
    masked_log_ratio = log_ratio * response_mask.to(dtype)
    future_log_ratio = torch.matmul(masked_log_ratio, causal_decay.t())
    future_values = torch.exp(torch.clamp(future_log_ratio, min=-20.0, max=20.0))
    if clip_eps is not None and clip_eps > 0:
        future_values = torch.clamp(future_values, min=1.0 - clip_eps, max=1.0 + clip_eps)
    return future_values


class APPOBrancher:
    """
    APPO 的 branching 逻辑封装。

    当前实现说明：
    - 不再基于 segment entropy 变化做 branching，不再 teacher forcing
    - 在 rollout 内计算 entropy/future-value z-score，形成 Branching Score
    - 过滤无意义 token
    - 按 Branching Score 选择 top-B tokens，直到达到 rollout 上限
    """

    def __init__(self, tokenizer, reward_scale_discount: float = 0.9):
        self.tokenizer = tokenizer
        self.reward_scale_discount = reward_scale_discount

    def compute_saliency_scores(
        self,
        response_token_ids: List[int],
        response_logprobs: List[Optional[dict]],
        future_values: Optional[List[float]] = None,
    ) -> List[Dict]:
        """
        计算 rollout 内每个 valid token 的 Branching Score。
        若 future_values 可用，按论文使用 rollout-level Z(entropy) * Z(future_value)；
        否则退化为 entropy z-score，供 rollout-time 无 current-policy log-probs 时使用。
        Returns list of {"response_token_idx", "saliency"}.
        """
        if not response_token_ids:
            return []

        token_indices: List[int] = []
        entropies: List[float] = []

        for token_idx in range(len(response_token_ids)):
            token_str = self.tokenizer.decode([response_token_ids[token_idx]])
            if _is_meaningless_token(token_str):
                continue
            logprob_dict = response_logprobs[token_idx] if token_idx < len(response_logprobs) else None
            entropy = _calc_token_entropy(logprob_dict) if logprob_dict else 0.0
            entropies.append(entropy)
            token_indices.append(token_idx)

        if not token_indices:
            return []

        entropy_z_scores = _zscore(entropies)
        if future_values is not None:
            rollout_future_values = [
                float(future_values[tidx]) if tidx < len(future_values) else 1.0
                for tidx in token_indices
            ]
            future_z_scores = _zscore(rollout_future_values)
            branch_scores = [
                entropy_z * future_z
                for entropy_z, future_z in zip(entropy_z_scores, future_z_scores)
            ]
        else:
            future_z_scores = None
            branch_scores = entropy_z_scores

        all_candidates = []
        for pos, (tidx, score) in enumerate(zip(token_indices, branch_scores)):
            if score <= 0:
                continue
            if future_z_scores is not None and (entropy_z_scores[pos] <= 0 or future_z_scores[pos] <= 0):
                continue
            all_candidates.append({
                "response_token_idx": tidx,
                "saliency": score,
            })

        return all_candidates

    def sample_branch_points(self, candidates: List[Dict], budget: int) -> List[Dict]:
        """按 Branching Score 选择 top-B branching 点。"""
        if not candidates or budget <= 0:
            return []
        return sorted(candidates, key=lambda c: c["saliency"], reverse=True)[:budget]

    def build_branch_rollout(
        self,
        prompt_token_ids: List[int],
        response_token_ids: List[int],
        branch_token_idx: int,
        branch_generated_ids: List[int],
        max_response_len: int,
    ) -> Tuple[List[int], List[int]]:
        """
        在 branch_token_idx 位置（response 中的索引）分裂，
        拼接 prompt + response[:branch_token_idx+1] + branch_generated_ids。
        result_mask 全部为 0。按 APPO 论文，branch rollout 只用于 reward/advantage
        contrastive signal，不直接进入 actor loss。
        """
        response_prefix = response_token_ids[:branch_token_idx + 1]
        remaining_len = max_response_len - len(response_prefix)
        if remaining_len <= 0:
            branch_generated_ids = []
        else:
            branch_generated_ids = branch_generated_ids[:remaining_len]

        response_part = (response_prefix + branch_generated_ids)[:max_response_len]
        full_token_ids = list(prompt_token_ids) + response_part
        result_mask = [0] * len(response_part)
        return full_token_ids, result_mask


# ============================================================
# reward 缩放因子 M 计算
# ============================================================

def compute_reward_scale_M(
    branch_token_response_idx: int,
    branch_score: float,
    max_branch_score: float,
    response_length: int,
    discount: float = 0.9,
) -> np.ndarray:
    """
    计算 reward 缩放因子 M，作用于 response 中分裂点 token 及其之前的所有 token。

    M_base = branch_score / max_branch_score
    对于 response 中第 t 个 token（0-indexed）：
    - 若 t > branch_token_response_idx：M[t] = 1.0（不缩放）
    - 若 t <= branch_token_response_idx：
        distance = branch_token_response_idx - t
        M[t] = 1.0 + M_base * (discount ** distance)
    """
    M = np.ones(response_length, dtype=np.float32)
    if max_branch_score <= 0 or branch_score <= 0:
        return M

    M_base = float(branch_score) / float(max_branch_score)
    for t in range(min(branch_token_response_idx + 1, response_length)):
        distance = branch_token_response_idx - t
        M[t] = 1.0 + M_base * (discount ** distance)

    return M
