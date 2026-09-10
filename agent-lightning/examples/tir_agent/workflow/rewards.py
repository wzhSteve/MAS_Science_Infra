"""Canonical MAS data-layer rewards (no agentlightning / verl)."""

from __future__ import annotations

import json
import re
import string
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_PLACEHOLDERS = {"", "answer", "<answer>", "number", "<number>"}


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def extract_answer_text(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = strip_thinking(text) or text
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        ans = match.group(1).strip()
        if ans.lower() not in _PLACEHOLDERS:
            return ans
    match = re.search(r"(?<!#)###\s*(.+?)(\s*###|$)", cleaned, flags=re.DOTALL)
    if match:
        ans = match.group(1).strip()
        if ans.lower() not in _PLACEHOLDERS:
            return ans
    match = re.search(r"####\s*(.+)", cleaned)
    if match:
        return match.group(1).split("\n")[0].strip()
    return None


def has_answer_format(text: Optional[str]) -> bool:
    if not text:
        return False
    cleaned = strip_thinking(text) or text
    return bool(
        re.search(r"<answer>\s*.+?\s*</answer>", cleaned, flags=re.DOTALL | re.IGNORECASE)
        or re.search(r"(?<!#)###\s*.+?\s*###", cleaned)
    )


def normalize_qa(s: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    exclude = set(string.punctuation)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in exclude)
    s = remove_articles(s)
    return " ".join(s.split())


def f1_score(prediction: str, golds: Sequence[str]) -> float:
    pred_toks = normalize_qa(prediction).split()
    best = 0.0
    for gold in golds:
        gold_toks = normalize_qa(gold).split()
        if not pred_toks and not gold_toks:
            best = max(best, 1.0)
            continue
        common = set(pred_toks) & set(gold_toks)
        if not common:
            continue
        prec = len(common) / max(1, len(pred_toks))
        rec = len(common) / max(1, len(gold_toks))
        if prec + rec == 0:
            continue
        best = max(best, 2 * prec * rec / (prec + rec))
    return float(best)


def numeric_match(prediction: str, gold: str) -> bool:
    pred = prediction.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    gt = gold.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    m = re.match(r"(-?\d+(?:\.\d+)?)", pred)
    if m:
        pred = m.group(1)
    try:
        return bool(np.isclose(float(pred), float(gt), rtol=1e-5, atol=1e-8))
    except (TypeError, ValueError):
        return pred.lower() == gt.lower()


def accuracy(
    prediction: Optional[str],
    gold: str,
    *,
    source: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    if prediction is None:
        return 0.0
    golds: List[str] = [gold]
    if aliases:
        golds.extend([str(a) for a in aliases if str(a)])
    if source == "gsm8k":
        return 1.0 if any(numeric_match(prediction, g) for g in golds) else 0.0
    return f1_score(prediction, golds)


def compute_outcome_reward(
    prediction: Optional[str],
    gold: str,
    *,
    source: str,
    aliases: Optional[Sequence[str]] = None,
    format_ok: bool,
    n_search: int,
    n_python: int,
    multi_tool_bonus: float = 0.1,
) -> float:
    """Hierarchical outcome used by all tir_algo variants.

    Format bad → -1; format ok & Acc=0 → 0; Acc>0 → Acc [+ r_M if both tools used].
    """
    if not format_ok:
        return -1.0
    acc = accuracy(prediction, gold, source=source, aliases=aliases)
    if acc <= 0:
        return 0.0
    bonus = multi_tool_bonus if (n_search > 0 and n_python > 0) else 0.0
    return float(acc + bonus)


def compute_mas_outcome_reward(
    prediction: Optional[str],
    task: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
) -> float:
    """Outcome reward for MAS_structagent (no <answer> format penalty).

    Empty / missing prediction → 0. gsm8k → numeric 0/1; QA → token F1.
    ``result`` is accepted for future shaping but unused in v1.
    """
    del result  # reserved for completed/audit shaping
    if prediction is None or not str(prediction).strip():
        return 0.0
    gold = str(task.get("answer") or "")
    source = str(task.get("source") or "gsm8k")
    aliases = parse_alias_field(task.get("answers"))
    return float(accuracy(str(prediction).strip(), gold, source=source, aliases=aliases))


def discounted_returns(rewards: Sequence[float], gamma: float) -> List[float]:
    out = [0.0] * len(rewards)
    acc = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[t]) + gamma * acc
        out[t] = acc
    return out


def group_zscore(values: Sequence[float], eps: float = 1e-8) -> List[float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return []
    std = float(arr.std())
    if std < eps:
        return [0.0] * len(values)
    mean = float(arr.mean())
    return [float((v - mean) / (std + eps)) for v in values]


def parse_alias_field(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        if raw.startswith("[") or raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, (list, tuple)):
                    return [str(x) for x in parsed]
            except Exception:
                try:
                    import ast

                    parsed = ast.literal_eval(raw)
                    if isinstance(parsed, (list, tuple)):
                        return [str(x) for x in parsed]
                except Exception:
                    pass
        return [raw]
    if isinstance(raw, np.ndarray):
        return [str(x) for x in raw.tolist()]
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    return [str(raw)]
