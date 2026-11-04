"""Small import-safe numerical helpers for FCR-GRPO."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Optional, Sequence, Tuple

import torch


def binary_brier_from_scores(
    satisfy_score: torch.Tensor,
    unsatisfy_score: torch.Tensor,
    label: bool,
    *,
    pos_weight: float = 1.0,
):
    """Differentiable binary softmax and Brier loss for complete label sequences.

    ``pos_weight`` multiplies the loss when ``label`` is True (Y=1), used to
    counter failure-dominated batches that otherwise collapse to always-NO.
    """
    probability = torch.softmax(
        torch.stack((satisfy_score, unsatisfy_score)), dim=0
    )[0]
    target = probability.new_tensor(1.0 if label else 0.0)
    loss = (probability - target).square()
    if bool(label) and float(pos_weight) != 1.0:
        loss = loss * loss.new_tensor(float(pos_weight))
    return loss, probability


def binary_bce_from_scores(
    satisfy_score: torch.Tensor,
    unsatisfy_score: torch.Tensor,
    label: bool,
    *,
    pos_weight: float = 1.0,
):
    """Binary cross-entropy on the YES/NO two-score softmax probability.

    Gradient w.r.t. logit z=s_+-s_- is (p-Y), which stays O(1) when p→0 and Y=1,
    unlike Brier's 2(p-Y)p(1-p) which vanishes.
    """
    probability = torch.softmax(
        torch.stack((satisfy_score, unsatisfy_score)), dim=0
    )[0]
    target = probability.new_tensor(1.0 if label else 0.0)
    eps = probability.new_tensor(1e-8)
    p = probability.clamp(eps, 1.0 - eps)
    loss = -(target * torch.log(p) + (1.0 - target) * torch.log(1.0 - p))
    if bool(label) and float(pos_weight) != 1.0:
        loss = loss * loss.new_tensor(float(pos_weight))
    return loss, probability


def combine_pred_losses(
    bce: Optional[torch.Tensor],
    brier: Optional[torch.Tensor],
    *,
    bce_weight: float = 1.0,
    brier_weight: float = 0.1,
) -> Optional[torch.Tensor]:
    """λ_BCE * BCE + λ_Brier * Brier; None if both missing."""
    terms: List[torch.Tensor] = []
    if bce is not None and float(bce_weight) != 0.0:
        terms.append(float(bce_weight) * bce)
    if brier is not None and float(brier_weight) != 0.0:
        terms.append(float(brier_weight) * brier)
    if not terms:
        return None
    out = terms[0]
    for term in terms[1:]:
        out = out + term
    return out


@dataclass
class PredReplayItem:
    """Frozen prediction supervision row (prefix token ids + YES/NO ids + Y)."""

    prefix_ids: List[int]
    satisfy_ids: List[int]
    unsatisfy_ids: List[int]
    label: bool
    policy_version: int = 0


class PersistentPredReplay:
    """Class-balanced replay for prediction loss only (never enters GRPO groups)."""

    def __init__(self, capacity: int = 2048):
        self.capacity = max(1, int(capacity))
        self.pos: Deque[PredReplayItem] = deque(maxlen=self.capacity)
        self.neg: Deque[PredReplayItem] = deque(maxlen=self.capacity)

    def add(self, item: PredReplayItem) -> None:
        if bool(item.label):
            self.pos.append(item)
        else:
            self.neg.append(item)

    def add_many(self, items: Iterable[PredReplayItem]) -> None:
        for item in items:
            self.add(item)

    def sample_balanced(self, k_pos: int = 2, k_neg: int = 2) -> List[PredReplayItem]:
        import random

        out: List[PredReplayItem] = []
        if self.pos and k_pos > 0:
            out.extend(random.choices(list(self.pos), k=int(k_pos)))
        if self.neg and k_neg > 0:
            out.extend(random.choices(list(self.neg), k=int(k_neg)))
        return out

    def sizes(self) -> Tuple[int, int]:
        return len(self.pos), len(self.neg)
