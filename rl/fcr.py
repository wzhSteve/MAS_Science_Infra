"""Small import-safe numerical helpers for FCR-GRPO."""

from __future__ import annotations

import torch


def binary_brier_from_scores(
    satisfy_score: torch.Tensor,
    unsatisfy_score: torch.Tensor,
    label: bool,
):
    """Differentiable binary softmax and Brier loss for complete label sequences."""
    probability = torch.softmax(
        torch.stack((satisfy_score, unsatisfy_score)), dim=0
    )[0]
    target = probability.new_tensor(1.0 if label else 0.0)
    return (probability - target).square(), probability
