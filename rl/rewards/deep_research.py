"""ARPO/APPO deep_research format scaffold with LLM accuracy (F1 replaced).

Format constants and soft/strict validation mirror
``ref_Rep/APPO-main/ARPO/.../reward_score/deep_research.py``. Accuracy layer is
Flow-style LLM 0/1 via ``rl.rewards.llm_judge`` instead of token F1.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rl.rewards.llm_judge import llm_judge_correct

SOFT_UNPARSEABLE_SCORE = -0.3
MISSING_BOXED_PENALTY = 0.05
STRICT_MISMATCH_PENALTY = 0.05

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:final answer|the answer is|answer is)\s*[:\-]?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def sanitize_response(text: str) -> str:
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(</answer>\s*)+</answer>", "</answer>", text, flags=re.IGNORECASE)
    return text


def get_format_mode(extra_info: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(extra_info, dict) and extra_info.get("format_mode"):
        return str(extra_info["format_mode"]).lower()
    return os.getenv("APPO_FORMAT_MODE", "soft").lower()


def validate_format(text: str) -> Tuple[bool, str]:
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return False, "<answer> or </answer> not found"

    current_pos = 0
    while True:
        search_pos = text.find("<search>", current_pos)
        if search_pos == -1:
            break
        result_pos = text.find("<result>", search_pos)
        search_end_pos = text.find("</search>", search_pos)
        result_end_pos = text.find("</result>", result_pos)
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"
        current_pos = result_end_pos

    current_pos = 0
    while True:
        python_pos = text.find("<python>", current_pos)
        if python_pos == -1:
            break
        result_pos = text.find("<result>", python_pos)
        python_end_pos = text.find("</python>", python_pos)
        result_end_pos = text.find("</result>", result_pos)
        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"
        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"
        current_pos = result_end_pos

    answer_start = text.find("<answer>")
    answer_end = text.find("</answer>")
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if "\\boxed{" not in answer_content or "}" not in answer_content:
        return False, "answer is missing \\boxed{} format"
    return True, "format is correct"


def extract_boxed_content(text: str) -> Optional[str]:
    match = _BOXED_RE.search(text)
    if match:
        boxed = match.group(1).strip()
        boxed = boxed.replace("\\,", " ").replace("\\ ", " ")
        return boxed
    return None


def extract_answer(text: str) -> Optional[str]:
    text = text.strip()
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1)
    open_m = re.search(r"<answer>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    if open_m:
        return open_m.group(1)
    return None


def _clean_answer_text(block: str) -> str:
    block = block.strip()
    boxed = extract_boxed_content(block)
    if boxed:
        return boxed
    match = _ANSWER_IS_RE.search(block)
    if match:
        block = match.group(1).strip()
    block = re.sub(r"^[\s\$\\\[\(]+", "", block)
    block = re.sub(r"[\s\$\\\]\)]+$", "", block)
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    if lines:
        block = lines[-1]
    block = block.strip(" \t\n\"'`.,;:")
    return block


def extract_final_answer(text: str) -> Tuple[Optional[str], str]:
    block = extract_answer(text)
    if block:
        boxed = extract_boxed_content(block)
        if boxed:
            return boxed, "answer_boxed"
        cleaned = _clean_answer_text(block)
        if cleaned:
            return cleaned, "answer_text"
    boxed = extract_boxed_content(text)
    if boxed:
        return boxed, "global_boxed"
    return None, "none"


def validate_format_soft(text: str) -> Tuple[bool, str, List[str]]:
    issues: List[str] = []
    n_ans_open = len(re.findall(r"<answer>", text, flags=re.IGNORECASE))
    n_ans_close = len(re.findall(r"</answer>", text, flags=re.IGNORECASE))
    if n_ans_open == 0 and n_ans_close == 0:
        issues.append("no_answer_tag")
    elif n_ans_open != n_ans_close:
        issues.append("answer_unpaired")
    elif n_ans_open > 1:
        issues.append("multiple_answer")
    if "<search>" in text and "</search>" not in text:
        issues.append("search_unclosed")
    if "<python>" in text and "</python>" not in text:
        issues.append("python_unclosed")
    answer, parse_path = extract_final_answer(text)
    if answer is None:
        return False, "cannot extract answer", issues
    if "boxed" not in (parse_path or ""):
        issues.append("missing_boxed")
    return True, "soft format ok", issues


def multi_tool_ok(
    *,
    n_search: int = 0,
    n_python: int = 0,
    n_fetch: int = 0,
    response: str = "",
) -> bool:
    """Process bonus for MAS fact QA (search∧fetch) and ARPO-style / arithmetic paths."""
    text = str(response or "")
    if n_search > 0 and n_fetch > 0:
        return True
    if n_search > 0 and n_python > 0:
        return True
    if "</search>" in text and "</python>" in text:
        return True
    return False


def compute_deep_research_llm_score(
    solution_str: str,
    ground_truth: Any,
    *,
    question: str = "",
    n_search: int = 0,
    n_python: int = 0,
    n_fetch: int = 0,
    multi_tool_bonus: float = 0.1,
    extra_info: Optional[Dict[str, Any]] = None,
    aliases: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Soft/strict format + LLM 0/1 accuracy + multi-tool / format penalties."""
    result: Dict[str, Any] = {
        "score": 0.0,
        "reason": "",
        "answer": "",
        "llm_score": 0.0,
        "format_strict": False,
        "format_soft": False,
        "parse_path": "none",
        "format_issues": [],
    }
    response = sanitize_response(solution_str or "")
    mode = get_format_mode(extra_info)
    strict_ok, strict_reason = validate_format(response)
    soft_ok, soft_reason, issues = validate_format_soft(response)
    answer, parse_path = extract_final_answer(response)

    result["format_strict"] = bool(strict_ok)
    result["format_soft"] = bool(soft_ok)
    result["parse_path"] = parse_path
    result["format_issues"] = issues
    result["answer"] = answer or ""

    if mode == "strict":
        if not strict_ok:
            result["score"] = -1.0
            result["reason"] = f"bad format: {strict_reason}"
            return result
        if answer is None:
            result["score"] = -1.0
            result["reason"] = "cannot extract answer"
            return result
    else:
        if answer is None:
            result["score"] = SOFT_UNPARSEABLE_SCORE
            result["reason"] = f"unparseable: {soft_reason}; issues={issues}"
            return result

    golds = [str(ground_truth)]
    if aliases:
        golds.extend([str(a) for a in aliases if str(a)])
    # Prefer extracted answer for the judge; include question like AgentFlow.
    judge_answer = answer or response
    llm_score = 0.0
    for g in golds:
        llm_score = max(llm_score, llm_judge_correct(question, g, judge_answer))
        if llm_score >= 1.0:
            break
    result["llm_score"] = llm_score

    penalty = 0.0
    if not strict_ok:
        penalty += STRICT_MISMATCH_PENALTY
    if parse_path != "answer_boxed":
        penalty += MISSING_BOXED_PENALTY

    multi_ok = multi_tool_ok(
        n_search=n_search, n_python=n_python, n_fetch=n_fetch, response=response
    )
    if llm_score > 0 and multi_ok:
        score = llm_score + float(multi_tool_bonus) - penalty
        result["reason"] = (
            f"correct answer with multi-tool, llm={llm_score:.3f}, penalty={penalty:.3f}"
        )
    elif llm_score > 0:
        score = llm_score - penalty
        result["reason"] = (
            f"correct answer, llm={llm_score:.3f}, path={parse_path}, penalty={penalty:.3f}"
        )
    else:
        score = 0.0
        result["reason"] = f"wrong answer but parseable: {answer}; path={parse_path}"

    result["score"] = float(max(-1.0, score))
    return result
