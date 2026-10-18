"""Top-k token entropy used by ARPO's post-tool gate.

Official rollout asks vLLM for ``logprobs=10`` and turns that distribution
into a normalized entropy. A completion that only carries the chosen-token
logprob falls back to ``-mean(log π)``.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence


def distribution_entropy(logprobs: Sequence[float]) -> float:
    """Entropy of a top-k log-probability list, in nats, after renormalizing."""
    vals = [float(x) for x in logprobs if x is not None and math.isfinite(float(x))]
    if not vals:
        return 0.0
    if len(vals) == 1:
        return max(0.0, -vals[0])
    peak = max(vals)
    weights = [math.exp(x - peak) for x in vals]
    total = sum(weights)
    if total <= 0:
        return 0.0
    ent = 0.0
    for weight in weights:
        p = weight / total
        if p > 0:
            ent -= p * math.log(p)
    return ent


def mean_token_entropy(per_token: Iterable[Sequence[float]]) -> Optional[float]:
    rows = [distribution_entropy(row) for row in per_token if row]
    if not rows:
        return None
    return sum(rows) / len(rows)


def entropy_from_trace(trace: object) -> Optional[float]:
    """Planner window trace: explicit ``entropy`` or the last planner call."""
    if not isinstance(trace, dict):
        return None
    raw = trace.get("entropy")
    if isinstance(raw, (int, float)):
        return float(raw)
    llm = trace.get("llm_calls")
    calls = llm.get("calls") if isinstance(llm, dict) else None
    if not isinstance(calls, list):
        return None
    for call in reversed(calls):
        if not isinstance(call, dict) or not isinstance(call.get("entropy"), (int, float)):
            continue
        role = call.get("role")
        if role in (None, "planner", "generate_next_step"):
            return float(call["entropy"])
    return None


def _rows_from_choice(tokens: Sequence[dict]) -> List[List[float]]:
    rows: List[List[float]] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        top = token.get("top_logprobs") or []
        values: List[float] = []
        if isinstance(top, list):
            for item in top:
                if isinstance(item, dict) and item.get("logprob") is not None:
                    values.append(float(item["logprob"]))
        if not values and token.get("logprob") is not None:
            values.append(float(token["logprob"]))
        if values:
            rows.append(values)
    return rows


def token_entropies_from_choice(tokens: Sequence[dict]) -> List[float]:
    return [distribution_entropy(row) for row in _rows_from_choice(tokens)]


def entropy_from_choice_logprobs(tokens: Sequence[dict]) -> Optional[float]:
    """Read OpenAI/vLLM ``choice.logprobs.content`` entries."""
    return mean_token_entropy(_rows_from_choice(tokens))


def token_entropies_from_trace(trace: object) -> List[float]:
    if not isinstance(trace, dict):
        return []
    llm = trace.get("llm_calls")
    calls = llm.get("calls") if isinstance(llm, dict) else None
    if not isinstance(calls, list):
        return []
    for call in reversed(calls):
        if not isinstance(call, dict):
            continue
        raw = call.get("token_entropies")
        role = call.get("role")
        if role in (None, "planner", "generate_next_step") and isinstance(raw, list) and raw:
            try:
                return [float(x) for x in raw]
            except (TypeError, ValueError):
                return []
    return []
