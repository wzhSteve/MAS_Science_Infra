import os
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union


SOFT_UNPARSEABLE_SCORE = -0.3
MISSING_BOXED_PENALTY = 0.05
STRICT_MISMATCH_PENALTY = 0.05

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:final answer|the answer is|answer is)\s*[:\-]?\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)


def _verbose() -> bool:
    return os.getenv("APPO_REWARD_VERBOSE", "0").lower() in ("1", "true", "yes")


def _log(msg: str) -> None:
    if _verbose():
        print(msg)


def sanitize_response(text: str) -> str:
    """Drop noisy extra closes that Qwen3 emits after tool results."""
    text = re.sub(r"</result>\s*</think>", "</result>", text, flags=re.IGNORECASE)
    text = re.sub(r"(</answer>\s*)+</answer>", "</answer>", text, flags=re.IGNORECASE)
    return text


def get_format_mode(extra_info: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(extra_info, dict) and extra_info.get("format_mode"):
        return str(extra_info["format_mode"]).lower()
    return os.getenv("APPO_FORMAT_MODE", "soft").lower()


def validate_format(text: str) -> Tuple[bool, str]:
    """Strict protocol check used by paper-style evaluation."""
    if text.count("<think>") != text.count("</think>"):
        return False, "<think> </think> not paired"

    if text.count("<think>") == 0 or text.count("</think>") == 0:
        return False, "<think> or </think> not found"

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


def validate_format_python(text: str) -> Tuple[bool, str]:
    return validate_format(text)


def validate_format_soft(text: str) -> Tuple[bool, str, List[str]]:
    """Tolerant check: extra </think> and missing boxed are warnings, not hard fails."""
    issues: List[str] = []
    n_open = text.count("<think>")
    n_close = text.count("</think>")
    if n_open == 0:
        issues.append("no_think")
    elif n_close < n_open:
        issues.append("think_unclosed")
    elif n_close > n_open:
        issues.append("extra_think_close")

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

    extractable = extract_final_answer(text)[0] is not None
    if not extractable:
        return False, "cannot extract answer", issues
    if "boxed" not in (extract_final_answer(text)[1] or ""):
        issues.append("missing_boxed")
    return True, "soft format ok", issues


def extract_answer(text: str) -> Optional[str]:
    """Last complete <answer>...</answer> block (or trailing unclosed <answer>)."""
    text = text.strip()
    matches = list(_ANSWER_RE.finditer(text))
    if matches:
        return matches[-1].group(1)
    open_m = re.search(r"<answer>(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
    if open_m:
        return open_m.group(1)
    return None


def extract_boxed_content(text: str) -> Optional[str]:
    boxed = last_boxed_only_string(text)
    if boxed:
        try:
            return remove_boxed(boxed).strip()
        except Exception:
            pass
    match = _BOXED_RE.search(text)
    if match:
        boxed = match.group(1).strip()
        boxed = boxed.replace("\\,", " ").replace("\\ ", " ")
        return boxed
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
    """Waterfall: boxed-in-answer → answer text → global boxed."""
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


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]

    left = "\\boxed{"
    if not s.startswith(left):
        match = _BOXED_RE.search(s)
        if match:
            return match.group(1)
        raise ValueError("string does not start with \\boxed{")
    if not s.endswith("}"):
        raise ValueError("string does not end with }")
    return s[len(left) : -1]


def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]) -> float:
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif isinstance(ground_truths, dict) and "ground_truth" in ground_truths:
        ground_truths = ground_truths["ground_truth"]
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(str(prediction))
        normalized_ground_truth = normalize_answer(str(ground_truth))
        if not normalized_prediction or not normalized_ground_truth:
            continue

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        if normalized_prediction == normalized_ground_truth:
            return 1.0
        if normalized_ground_truth in normalized_prediction or normalized_prediction in normalized_ground_truth:
            shorter, longer = (
                (normalized_ground_truth, normalized_prediction)
                if len(normalized_ground_truth) <= len(normalized_prediction)
                else (normalized_prediction, normalized_ground_truth)
            )
            if shorter and shorter in longer:
                final_metric["f1"] = max(final_metric["f1"], len(shorter) / len(longer))

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        if not prediction_tokens or not ground_truth_tokens:
            continue
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue

        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])

    return final_metric["f1"]


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "score": 0.0,
        "reason": "",
        "answer": "",
        "f1_score": 0.0,
        "format_strict": False,
        "format_soft": False,
        "parse_path": "none",
        "format_issues": [],
    }

    response = sanitize_response(solution_str or "")
    if extra_info is not None and "tokenizer" in extra_info:
        tok = extra_info["tokenizer"]
        eos = getattr(tok, "eos_token", None)
        if eos and response.endswith(eos):
            response = response[: -len(eos)]

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
            _log(f"bad format: {strict_reason}")
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

    f1_score = get_f1_score(answer, ground_truth)
    result["f1_score"] = f1_score

    penalty = 0.0
    if not strict_ok:
        penalty += STRICT_MISMATCH_PENALTY
    if parse_path != "answer_boxed":
        penalty += MISSING_BOXED_PENALTY

    if f1_score > 0 and "</search>" in response and "</python>" in response:
        score = f1_score + 0.1 - penalty
        result["reason"] = f"correct answer with multi-tool, f1={f1_score:.3f}, penalty={penalty:.3f}"
    elif f1_score > 0:
        score = f1_score - penalty
        result["reason"] = f"correct answer, f1={f1_score:.3f}, path={parse_path}, penalty={penalty:.3f}"
    else:
        score = 0.0
        result["reason"] = f"wrong answer but parseable: {answer}; path={parse_path}"

    result["score"] = float(max(-1.0, score))
    _log(result["reason"])
    return result
