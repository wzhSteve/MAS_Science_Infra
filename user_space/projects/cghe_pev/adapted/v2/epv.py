"""Expected pruning value from an explicit feedback partition.

EPV(a) = N - sum_j n_j^2 / N for an equal-weight, deterministic partition.
A prediction never authorizes pruning. Illegal probabilities are rejected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class FeedbackPrediction:
    """One candidate action and the plans that fall into each predicted outcome."""

    action_id: str
    outcomes: Mapping[str, Sequence[str]]
    probabilities: Optional[Mapping[str, float]] = None
    estimated_cost: float = 1.0
    estimated_risk: float = 0.0
    evidence_basis: str = "explicit_premise"
    uncertainty: str = "low"


@dataclass(frozen=True)
class ScoreBreakdown:
    action_id: str
    epv: Optional[float]
    cost: float
    risk: float
    score: Optional[float]
    fallback: bool
    reason: str
    partition: dict[str, list[str]] = field(default_factory=dict)


def _counts(outcomes: Mapping[str, Sequence[str]]) -> tuple[list[str], dict[str, list[str]]]:
    seen: list[str] = []
    partition: dict[str, list[str]] = {}
    for label, plans in outcomes.items():
        bucket = []
        for plan_id in plans:
            name = str(plan_id)
            if name in seen:
                continue
            seen.append(name)
            bucket.append(name)
        if bucket:
            partition[str(label)] = bucket
    return seen, partition


def probabilities_are_valid(probabilities: Optional[Mapping[str, float]], labels: Sequence[str]) -> bool:
    if probabilities is None:
        return True
    if set(probabilities) != set(labels):
        return False
    total = 0.0
    for label in labels:
        value = float(probabilities[label])
        if value < 0.0:
            return False
        total += value
    return abs(total - 1.0) <= 1e-6


def expected_pruning_value(
    outcomes: Mapping[str, Sequence[str]],
    *,
    probabilities: Optional[Mapping[str, float]] = None,
    priors: Optional[Mapping[str, float]] = None,
) -> Optional[float]:
    """Return EPV, or None when the partition or the probabilities are unusable."""
    plans, partition = _counts(outcomes)
    if len(partition) == 0:
        return None
    if not probabilities_are_valid(probabilities, list(partition)):
        return None
    if priors is None:
        n = len(plans)
        if n <= 0:
            return None
        return float(n - sum(len(bucket) ** 2 for bucket in partition.values()) / n)
    weights = []
    for plan_id in plans:
        try:
            weight = float(priors[plan_id])
        except KeyError:
            return None
        if weight < 0.0:
            return None
        weights.append(weight)
    total = sum(weights)
    if total <= 0.0:
        return None
    # Weighted Gini form: 1 - sum_j (P_j^2) / P, scaled by plan mass.
    mass = {label: 0.0 for label in partition}
    for plan_id, weight in zip(plans, weights):
        for label, bucket in partition.items():
            if plan_id in bucket:
                mass[label] += weight
    return float(total - sum(value * value for value in mass.values()) / total)


def score_action(
    prediction: FeedbackPrediction,
    *,
    lambda_cost: float = 0.0,
    mu_risk: float = 0.0,
) -> ScoreBreakdown:
    plans, partition = _counts(prediction.outcomes)
    epv = expected_pruning_value(prediction.outcomes, probabilities=prediction.probabilities)
    fallback = epv is None
    score = None if epv is None else epv - float(lambda_cost) * prediction.estimated_cost - float(mu_risk) * prediction.estimated_risk
    reason = "scored" if not fallback else "rejected prediction; caller must use the shared-node baseline"
    return ScoreBreakdown(
        action_id=prediction.action_id,
        epv=epv,
        cost=float(prediction.estimated_cost),
        risk=float(prediction.estimated_risk),
        score=score,
        fallback=fallback,
        reason=reason,
        partition=partition,
    )


def select_scored(rows: Sequence[ScoreBreakdown]) -> Optional[ScoreBreakdown]:
    """Highest J, then lower cost, then stable action id. None if every row fell back."""
    usable = [row for row in rows if row.score is not None]
    if not usable:
        return None
    usable.sort(key=lambda row: (-float(row.score), row.cost, row.action_id))
    return usable[0]
