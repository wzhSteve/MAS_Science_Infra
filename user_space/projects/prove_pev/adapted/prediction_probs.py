"""Two-way probability of the planner labels YES and NO.

These match the verifier line ``Satisfied: YES|NO``. Both strings are single
Qwen3 tokens, so each score is one token logprob. Softmax is only over the
two strings.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Sequence

SATISFY = "YES"
UNSATISFY = "NO"
LABELS = (SATISFY, UNSATISFY)

_PREDICTION_QUOTE = re.compile(r'"prediction"\s*:\s*"')


def written_prediction(completion: str) -> str:
    match = re.search(r'"prediction"\s*:\s*"([^"]*)"', str(completion or ""))
    raw = match.group(1).strip() if match else ""
    return canonical_prediction(raw)


def canonical_prediction(value: Any) -> str:
    token = str(value or "").strip().strip("\"'").strip()
    key = token.upper() if token.isascii() else token
    aliases = {
        "YES": SATISFY,
        "NO": UNSATISFY,
        "满足": SATISFY,
        "不满足": UNSATISFY,
        "Y": SATISFY,
        "N": UNSATISFY,
    }
    return aliases.get(key, aliases.get(token, ""))


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
    text = canonical_prediction(label)
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
