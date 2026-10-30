"""Two-way probability of the planner labels 满足 and 不满足.

``满足`` is one Qwen3 token. ``不满足`` is two tokens, so the score of a label
is the sum of its token logprobs under the same prefix. Softmax is only over
these two strings. That reduces to a next-token softmax when both labels are
one token.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

SATISFY = "满足"
UNSATISFY = "不满足"
LABELS = (SATISFY, UNSATISFY)

_PREDICTION_QUOTE = re.compile(r'"prediction"\s*:\s*"')


def written_prediction(completion: str) -> str:
    match = re.search(r'"prediction"\s*:\s*"([^"]*)"', str(completion or ""))
    return match.group(1).strip() if match else ""


def prediction_prefix(completion: str) -> Optional[str]:
    """Completion text up to and including the opening quote of ``prediction``."""
    match = _PREDICTION_QUOTE.search(str(completion or ""))
    if not match:
        return None
    return str(completion)[: match.end()]


def binary_softmax(score_satisfy: float, score_unsatisfy: float) -> tuple[float, float]:
    """Return ``(p_satisfy, p_unsatisfy)``."""
    shift = max(float(score_satisfy), float(score_unsatisfy))
    satisfy = math.exp(float(score_satisfy) - shift)
    unsatisfy = math.exp(float(score_unsatisfy) - shift)
    total = satisfy + unsatisfy
    if total == 0.0 or math.isnan(total):
        return 0.5, 0.5
    return satisfy / total, unsatisfy / total


def sum_suffix_logprobs(token_logprobs: Sequence[Any], n_prefix: int, n_label: int) -> float:
    """Sum echo logprobs of the label tokens.

    OpenAI echo puts ``None`` on the first prompt token. Index ``i`` is the
    logprob of prompt token ``i``.
    """
    if n_label <= 0:
        raise ValueError("label has no tokens")
    start = int(n_prefix)
    stop = start + int(n_label)
    if start < 0 or stop > len(token_logprobs):
        raise ValueError("label span is outside the echo logprobs")
    chunk = list(token_logprobs[start:stop])
    if any(item is None for item in chunk):
        raise ValueError("label logprob is missing")
    return float(sum(float(item) for item in chunk))


def prediction_record(label: str, score_satisfy: float, score_unsatisfy: float) -> Dict[str, Any]:
    """Probabilities plus whether the written word disagrees with ``p >= 1/2``."""
    p_satisfy, p_unsatisfy = binary_softmax(score_satisfy, score_unsatisfy)
    text = str(label or "").strip().strip("\"'")
    mismatch = False
    if text == SATISFY:
        mismatch = p_satisfy < 0.5
    elif text == UNSATISFY:
        mismatch = p_satisfy >= 0.5
    return {
        "p_satisfy": p_satisfy,
        "p_unsatisfy": p_unsatisfy,
        "s_satisfy": float(score_satisfy),
        "s_unsatisfy": float(score_unsatisfy),
        "prediction_mismatch": mismatch,
    }
