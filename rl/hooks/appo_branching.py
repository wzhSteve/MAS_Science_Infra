"""APPO token selection and reward-scale M, matched to the official fork.

Source: ``ref_Rep/APPO-main/APPO`` ``appo_branching.py`` / ``ray_trainer.py``.

- Branching score on one response: ``Z(entropy) * Z(future value)``.
  Tokens with both z-scores positive are preferred. If future value is
  absent the score falls back to the entropy z-score.
- Reward scale ``M`` multiplies token rewards on the initial trajectory::

    M_base = branch_score / max_branch_score
    t <= split: M[t] = 1 + M_base * discount ** (split - t)
    t >  split: M[t] = 1

  Several branches of one parent are averaged token-wise.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def _zscore(values: Sequence[float]) -> List[float]:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    std = math.sqrt(var)
    if std < 1e-12:
        return [0.0] * n
    return [(v - mean) / std for v in vals]


def branching_scores(
    entropy: Sequence[float],
    future_value: Optional[Sequence[float]] = None,
) -> List[float]:
    """Per-token branch score. Length follows ``entropy``."""
    zh = _zscore(entropy)
    if not future_value or len(future_value) != len(entropy):
        return zh
    zf = _zscore(future_value)
    return [h * f for h, f in zip(zh, zf)]


def select_branch_tokens(
    entropy: Sequence[float],
    future_value: Optional[Sequence[float]],
    k: int,
) -> List[int]:
    """Pick up to ``k`` token indexes, preferring jointly positive z-scores.

    Shortfall is filled from the remaining tokens by score, then by repeating
    the chosen indexes. Official ``build_appo_branch_request`` does the same
    so a prompt still reaches ``max_branches_per_prompt``.
    """
    n = len(entropy)
    want = max(0, int(k))
    if n == 0 or want == 0:
        return []
    zh = _zscore(entropy)
    if future_value and len(future_value) == n:
        zf = _zscore(future_value)
        scores = [h * f for h, f in zip(zh, zf)]
        preferred = [i for i in range(n) if zh[i] > 0 and zf[i] > 0]
    else:
        scores = zh
        preferred = [i for i in range(n) if zh[i] > 0]
    rest = [i for i in range(n) if i not in set(preferred)]
    preferred.sort(key=lambda i: scores[i], reverse=True)
    rest.sort(key=lambda i: scores[i], reverse=True)
    ordered = preferred + rest
    if not ordered:
        return []
    picked = ordered[:want]
    base = list(picked)
    cursor = 0
    while len(picked) < want:
        picked.append(base[cursor % len(base)])
        cursor += 1
    return picked[:want]


def reward_scale_row(
    length: int,
    split: int,
    branch_score: float,
    max_branch_score: float,
    discount: float,
) -> List[float]:
    """One branch's M vector. Positions after ``split`` stay 1."""
    n = max(0, int(length))
    row = [1.0] * n
    if n == 0 or max_branch_score <= 0:
        return row
    base = float(branch_score) / float(max_branch_score)
    last = min(max(int(split), 0), n - 1)
    gamma = float(discount)
    for t in range(last + 1):
        row[t] = 1.0 + base * (gamma ** (last - t))
    return row


def average_reward_scales(
    length: int,
    splits: Sequence[int],
    scores: Sequence[float],
    discount: float,
) -> List[float]:
    """Mean M across the branches that share one parent trajectory."""
    n = max(0, int(length))
    if n == 0 or not scores:
        return [1.0] * n
    max_score = max(float(s) for s in scores)
    acc = [0.0] * n
    for split, score in zip(splits, scores):
        row = reward_scale_row(n, int(split), float(score), max_score, discount)
        for t, value in enumerate(row):
            acc[t] += value
    count = float(len(list(scores)))
    return [value / count for value in acc]
