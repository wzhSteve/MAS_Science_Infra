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

_PLACEHOLDERS = {
    "",
    "answer",
    "<answer>",
    "</answer>",
    "<answer></answer>",
    "number",
    "<number>",
    "short answer",
    "think",
    "<think>",
    "</think>",
    "none",
    "n/a",
    "na",
    "unknown",
    "null",
}
_CONTENT_STOPS = {"a", "an", "the", "of", "and"}
_LETTER_RE = re.compile(r"^\$*\s*\(?([A-E])\)?\s*\$*$", re.IGNORECASE)
_MCQ_TEXTBF_RE = re.compile(
    r"\\textbf\s*\{\s*\(?([A-E])\)?\s*\}\s*(.*?)(?=\\qquad|\\textbf|$)",
    re.IGNORECASE | re.DOTALL,
)
_MCQ_PAREN_RE = re.compile(
    r"(?:^|\s)[\(（]([A-E])[\)）]\s*([^\n]{1,80})",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    out = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Truncation fallback: an unclosed <think> (max_response_length cut mid-
    # reasoning) means no usable answer text remains; blank it so the caller
    # falls back to format/accuracy failure instead of scoring the reasoning.
    out = re.sub(r"<think>.*\Z", "", out, flags=re.DOTALL | re.IGNORECASE)
    return out.strip()


_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_BOXED_SPACE_RE = re.compile(r"\\boxed\s+([^\n$]+)")


def unwrap_boxed(text: str) -> str:
    """Peel ``\\boxed{...}`` so Hotpot F1 / binary EM score the answer, not the wrapper.

    Qwen-style MAS finals often emit ``<answer>\\boxed{Harry F. Sinclair}</answer>``.
    Leaving the wrapper in place makes ``normalize_qa`` glue ``boxed`` onto the
    first token (``boxedharry``), so a correct answer gets F1≈2/3 and binary 0.
    """
    if not text:
        return text
    match = _BOXED_RE.search(text)
    if match:
        return match.group(1).strip()
    match = _BOXED_SPACE_RE.search(text)
    if match:
        return match.group(1).strip().rstrip("$").strip()
    return text.strip()


def _usable_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    ans = unwrap_boxed(strip_thinking(ans) or ans)
    ans = unwrap_boxed(ans)
    if not ans or ans.lower() in _PLACEHOLDERS:
        return None
    return ans


def extract_answer_text(text: str) -> Optional[str]:
    """Answer used for scoring. Thinking, including a boxed span inside it, is dropped."""
    if not text:
        return None
    cleaned = strip_thinking(text)
    if not cleaned:
        return None
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match:
        ans = _usable_answer(match.group(1).strip())
        if ans is not None:
            return ans
    # Only a boxed span that survives think-stripping counts. A formula inside
    # an unclosed <think> must not become the scored answer.
    boxed = _usable_answer(unwrap_boxed(cleaned))
    if boxed is not None and (_BOXED_RE.search(cleaned) or _BOXED_SPACE_RE.search(cleaned)):
        return boxed
    match = re.search(r"(?<!#)###\s*(.+?)(\s*###|$)", cleaned, flags=re.DOTALL)
    if match:
        ans = _usable_answer(match.group(1).strip())
        if ans is not None:
            return ans
    match = re.search(r"####\s*(.+)", cleaned)
    if match:
        return _usable_answer(match.group(1).split("\n")[0].strip())
    return None


def answer_for_reward(text: str) -> Optional[str]:
    """Short answer for llm_binary. Empty tags and placeholders are not answers."""
    return extract_answer_text(text)


def _pure_number(gold: str) -> bool:
    s = str(gold or "").strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
    return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", s))


def _norm_eq(pred: str, gold: str) -> bool:
    left = normalize_qa(pred)
    right = normalize_qa(gold)
    return bool(left) and left == right


def _option_letter(answer: str) -> Optional[str]:
    match = _LETTER_RE.match(str(answer or "").strip())
    if not match:
        return None
    return match.group(1).upper()


def _mcq_options(question: str) -> Dict[str, str]:
    found: Dict[str, str] = {}
    for match in _MCQ_TEXTBF_RE.finditer(question or ""):
        found[match.group(1).upper()] = match.group(2).strip()
    if found:
        return found
    for match in _MCQ_PAREN_RE.finditer(question or ""):
        found.setdefault(match.group(1).upper(), match.group(2).strip())
    return found


def _option_matches(question: str, letter: str, gold: str) -> bool:
    option = _mcq_options(question).get(letter, "")
    if not option:
        return False
    if _norm_eq(option, gold):
        return True
    opt_digits = re.sub(r"\D", "", option)
    gold_digits = re.sub(r"\D", "", gold)
    return bool(gold_digits) and opt_digits == gold_digits


def _content_tokens(text: str, min_len: int) -> set:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {word for word in words if len(word) >= min_len and word not in _CONTENT_STOPS}


def _gold_supported(answer: str, gold: str) -> bool:
    """True when the answer contains a real gold token, not just a shared short word."""
    blob = re.sub(r"[^a-z0-9]+", "", (answer or "").lower())
    long = _content_tokens(gold, 5)
    if long:
        return any(token in blob for token in long)
    short = _content_tokens(gold, 4)
    if not short:
        return True
    return any(token in blob for token in short)


def score_llm_binary(
    prediction: Optional[str],
    gold: str,
    *,
    question: str = "",
    aliases: Optional[Sequence[str]] = None,
) -> float:
    """0/1 outcome. The judge only sees a short answer the local checks could not settle."""
    from rl.rewards.llm_judge import llm_judge_correct

    extracted = answer_for_reward("" if prediction is None else str(prediction))
    if not extracted:
        return 0.0
    golds = [str(gold or "")] + [str(item) for item in (aliases or []) if str(item)]
    letter = _option_letter(extracted)
    any_text_gold = False
    for item in golds:
        if _norm_eq(extracted, item):
            return 1.0
        if len(extracted) <= 80 and numeric_match(extracted, item):
            return 1.0
        if letter and _option_matches(question, letter, item):
            return 1.0
        if not _pure_number(item):
            any_text_gold = True
    if letter or not any_text_gold:
        return 0.0
    if len(extracted) > 300:
        return 0.0
    best = 0.0
    for item in golds:
        if _pure_number(item) or not _gold_supported(extracted, item):
            continue
        best = max(best, llm_judge_correct(question, item, extracted))
        if best >= 1.0 and _gold_supported(extracted, item):
            return 1.0
    return 0.0


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


def binary_correct(
    prediction: Optional[str],
    gold: str,
    *,
    source: str,
    aliases: Optional[Sequence[str]] = None,
) -> float:
    """0/1 correctness (Flow-GRPO outcome). gsm8k → numeric match; QA → normalized
    exact match or the gold appearing as a whole phrase in the extracted answer."""
    if prediction is None:
        return 0.0
    text = extract_answer_text(str(prediction)) or strip_thinking(str(prediction))
    golds = [gold] + [str(a) for a in (aliases or []) if str(a)]
    if source == "gsm8k":
        return 1.0 if any(numeric_match(text, g) for g in golds) else 0.0
    pred = normalize_qa(text)
    if not pred:
        return 0.0
    for g in golds:
        ng = normalize_qa(g)
        if ng and (pred == ng or f" {ng} " in f" {pred} "):
            return 1.0
    return 0.0


def compute_outcome_reward(
    prediction: Optional[str],
    gold: str,
    *,
    source: str,
    aliases: Optional[Sequence[str]] = None,
    format_ok: bool,
    n_search: int,
    n_python: int,
    n_fetch: int = 0,
    multi_tool_bonus: float = 0.1,
    reward_spec: Optional["RewardSpec"] = None,
    question: str = "",
) -> float:
    """Outcome reward for tir_algo variants.

    Modes (via ``reward_spec.outcome``):
    - ``hierarchical`` — format/-1 + F1 + multi_tool (legacy MAS)
    - ``binary`` — local 0/1 EM (legacy Flow without LLM)
    - ``llm_binary`` — Flow-GRPO gpt judge 0/1 (no format / multi_tool)
    - ``deep_research_llm`` — ARPO/APPO soft format + LLM 0/1 + penalties

    ``reward_spec=None`` keeps the literal hierarchical path byte-identical.
    """
    if reward_spec is not None:
        spec = reward_spec
        outcome = str(spec.outcome or "hierarchical").strip().lower()
        if outcome == "binary":
            return binary_correct(prediction, gold, source=source, aliases=aliases)
        if outcome == "llm_binary":
            return score_llm_binary(
                prediction,
                gold,
                question=question,
                aliases=aliases,
            )
        if outcome == "deep_research_llm":
            from rl.rewards.deep_research import compute_deep_research_llm_score

            bonus = (
                spec.weight_of("multi_tool_bonus")
                if spec.is_enabled("multi_tool_bonus")
                else 0.0
            )
            scored = compute_deep_research_llm_score(
                str(prediction or ""),
                gold,
                question=question,
                n_search=n_search,
                n_python=n_python,
                n_fetch=n_fetch,
                multi_tool_bonus=bonus if bonus > 0 else multi_tool_bonus,
                aliases=aliases,
            )
            return float(scored["score"])
        from rl.rewards.deep_research import multi_tool_ok

        # format_penalty: gate BEFORE accuracy unless switched off.
        if format_ok:
            gained = accuracy(prediction, gold, source=source, aliases=aliases)
        else:
            if not spec.is_enabled("format_penalty"):
                gained = accuracy(prediction, gold, source=source, aliases=aliases)
            else:
                return float(-1.0 * spec.weight_of("format_penalty"))
        if gained <= 0:
            return 0.0
        bonus = 0.0
        if spec.is_enabled("multi_tool_bonus") and multi_tool_ok(
            n_search=n_search, n_python=n_python, n_fetch=n_fetch
        ):
            bonus = spec.weight_of("multi_tool_bonus")
        return float(gained + bonus)

    # --- literal path (v1 default; byte-identical when n_fetch=0) -------------
    from rl.rewards.deep_research import multi_tool_ok

    if not format_ok:
        return -1.0
    acc = accuracy(prediction, gold, source=source, aliases=aliases)
    if acc <= 0:
        return 0.0
    bonus = (
        multi_tool_bonus
        if multi_tool_ok(n_search=n_search, n_python=n_python, n_fetch=n_fetch)
        else 0.0
    )
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
