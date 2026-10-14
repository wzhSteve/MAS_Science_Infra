"""Canonical outcome rewards (no agentlightning / verl). Owned by the RL layer."""


from __future__ import annotations

import json
import re
import string
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # RewardSpec lives in the same package (rl.rewards.assembly);
    # imported lazily to keep outcome.py import-order trivial for hot paths.
    from rl.rewards.assembly import RewardSpec

_PLACEHOLDERS = {"", "answer", "<answer>", "number", "<number>"}


def strip_thinking(text: str) -> str:
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Truncation fallback: an unclosed <think> (max_response_length cut mid-
    # reasoning) means no usable answer text remains; blank it so the caller
    # falls back to format/accuracy failure instead of scoring the reasoning.
    out = re.sub(r"<think>.*\Z", "", out, flags=re.DOTALL | re.IGNORECASE)
    return out.strip()


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
    """Match prediction to a numeric gold.

    The prediction may be a long CoT-wrapped answer such as
    ``"<answer>Jason has 16 cousins. ... **Answer:** 16</answer>"``; we try the
    first number, the last number, and the inner ``<answer>...</answer>`` body
    (in that order) before falling back to a literal comparison.
    """
    pred = str(prediction or "").strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    gt = gold.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")
    candidates: List[str] = [pred]
    # 1. Try the inner <answer>...</answer> body if any.
    m_ans = re.search(r"<answer>(.*?)</answer>", pred, flags=re.DOTALL | re.IGNORECASE)
    if m_ans:
        candidates.append(m_ans.group(1).strip())
    # 2. Try the last number anywhere in the prediction (CoT often says
    #    "Answer: 16" at the very end).
    nums = re.findall(r"-?\d+(?:\.\d+)?", pred)
    if nums:
        candidates.append(nums[-1])
    # 3. Try the first number for the common "16 cousins" pattern.
    if nums:
        candidates.append(nums[0])
    for cand in candidates:
        c = cand.strip()
        if not c:
            continue
        m = re.match(r"(-?\d+(?:\.\d+)?)", c)
        if m:
            c = m.group(1)
        try:
            if bool(np.isclose(float(c), float(gt), rtol=1e-5, atol=1e-8)):
                return True
        except (TypeError, ValueError):
            if c.lower() == gt.lower():
                return True
    return False


def accuracy(
    prediction: Optional[str],
    gold: str,
    *,
    source: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    if prediction is None:
        return 0.0
    text = str(prediction)
    extracted = extract_answer_text(text)
    if extracted:
        text = extracted
    golds: List[str] = [gold]
    if aliases:
        golds.extend([str(a) for a in aliases if str(a)])
    if source == "gsm8k":
        return 1.0 if any(numeric_match(text, g) for g in golds) else 0.0
    return f1_score(text, golds)


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
    reward_spec: Optional["RewardSpec"] = None,
) -> float:
    """Hierarchical outcome used by all tir_algo variants.

    Format bad → -1; format ok & Acc=0 → 0; Acc>0 → Acc [+ r_M if both tools used].

    RL4: ``reward_spec`` (an optional RewardSpec from rl.rewards.assembly)
    routes the SAME hierarchy through per-component switch/weight application.
    ``reward_spec=None`` (the default for every v1 caller) keeps the literal
    path below byte-identical — no behavioral change for existing callers.
    When a spec is provided, the ``multi_tool_bonus`` keyword is IGNORED (the
    spec's multi_tool_bonus component owns the magnitude); format_penalty
    becomes switchable (off → a format failure falls through to the accuracy
    layer instead of returning -1); accuracy_reward is not switchable (spec
    validation enforces enabled=True upstream).
    """
    if reward_spec is not None:
        # --- component path (switchable / weightable) --------------------
        spec = reward_spec
        # format_penalty: gate BEFORE accuracy unless switched off.
        if format_ok:
            gained = accuracy(prediction, gold, source=source, aliases=aliases)
        else:
            if not spec.is_enabled("format_penalty"):
                # switched off: no penalty, still evaluate accuracy on the
                # raw prediction (the hierarchy's second layer).
                gained = accuracy(prediction, gold, source=source, aliases=aliases)
            else:
                return float(-1.0 * spec.weight_of("format_penalty"))
        if gained <= 0:
            return 0.0
        # cost_penalty: reserved slot — validation guarantees disabled, and
        # the executor no-ops it even if a future caller hand-builds one.
        bonus = 0.0
        if spec.is_enabled("multi_tool_bonus") and (n_search > 0 and n_python > 0):
            bonus = spec.weight_of("multi_tool_bonus")
        return float(gained + bonus)

    # --- literal path (v1 default; byte-identical to pre-RL4) -------------
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
    """Outcome reward for the StructAgent stack (legacy/structagent/, no <answer> format penalty).

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
