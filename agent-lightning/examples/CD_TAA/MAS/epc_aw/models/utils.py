import json
import re
from typing import Any, Dict, Optional


def _extract_outer_json_object(text: str) -> Optional[str]:
    """Extract outermost {...}; safe when JSON strings contain ``` backticks."""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return None


def _strip_think_blocks(text: str) -> str:
    """Remove closed and unclosed Qwen-style <think> blocks."""
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*\Z", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _strip_markdown_fence(text: str) -> str:
    clean = text.strip()
    if clean.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if match:
            return match.group(1).strip()
        # Truncated fence: drop opening line
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
    return clean.strip()


def _try_loads(text: str) -> Optional[Any]:
    if not text or not str(text).strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _repair_truncated_json(text: str) -> Optional[Any]:
    """Best-effort repair for truncated / trailing-comma JSON objects."""
    if not text:
        return None
    candidate = text.rstrip()
    # Drop trailing commas before } or ]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    parsed = _try_loads(candidate)
    if parsed is not None:
        return parsed

    # Close open strings / braces / brackets from truncation.
    in_string = False
    escape = False
    stack = []
    for ch in candidate:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()

    repaired = candidate
    if in_string:
        repaired += '"'
    # Drop dangling trailing comma after closing the string
    repaired = re.sub(r",\s*$", "", repaired)
    while stack:
        repaired += stack.pop()

    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return _try_loads(repaired)


def _extract_query_analysis_partial(text: str) -> Optional[Dict[str, Any]]:
    """If QueryAnalysis-shaped JSON is damaged, salvage analysis and empty outline."""
    if not text:
        return None
    lower = text.lower()
    if "analysis" not in lower and "execution_outline" not in lower and "executionoutline" not in lower:
        return None

    # Prefer snake_case then PascalCase quoted string values
    patterns = [
        r'"analysis"\s*:\s*"((?:[^"\\]|\\.)*)"',
        r'"Analysis"\s*:\s*"((?:[^"\\]|\\.)*)"',
        # Truncated analysis string (no closing quote)
        r'"analysis"\s*:\s*"([^"]*)',
        r'"Analysis"\s*:\s*"([^"]*)',
    ]
    analysis = ""
    for pat in patterns:
        m = re.search(pat, text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            analysis = m.group(1)
            # Unescape common sequences
            try:
                analysis = json.loads(f'"{analysis}"')
            except Exception:
                analysis = analysis.replace('\\"', '"').replace("\\n", "\n")
            if analysis.strip():
                break

    if not analysis.strip() and ("execution_outline" in text or "ExecutionOutline" in text):
        # Still return empty analysis so caller can apply schema fallback outline
        return {"analysis": "", "execution_outline": {}}

    if not analysis.strip():
        return None
    return {"analysis": analysis, "execution_outline": {}}


def parse_json_from_llm_response(raw: Any) -> Any:
    """
    Parse JSON from LLM text that may be wrapped in markdown fences (```json ... ```).
    Matches the cleaning logic used in planner.analyze_query.

    When the engine uses structured output (e.g. FinalAnswer), ``raw`` may be a
    Pydantic model whose JSON payload lives in ``final_answer`` — unwrap that
    before parsing.

    Also strips Qwen ``<think>`` blocks and best-effort repairs truncated JSON.
    """
    if isinstance(raw, (dict, list)):
        return raw
    if hasattr(raw, "model_dump"):
        try:
            return raw.model_dump(by_alias=True)
        except TypeError:
            return raw.model_dump()
    # Chat completions.parse returns FinalAnswer(final_answer="...markdown json...")
    if hasattr(raw, "final_answer"):
        inner = getattr(raw, "final_answer", None)
        if isinstance(inner, str) and inner.strip():
            raw = inner
    if not isinstance(raw, str):
        raw = str(raw)

    clean = _strip_think_blocks(raw)
    clean = _strip_markdown_fence(clean)
    clean = clean.replace("\\'", "'")

    brace_chunk = _extract_outer_json_object(clean)
    for candidate in (brace_chunk, clean):
        if not candidate:
            continue
        parsed = _try_loads(candidate)
        if parsed is not None:
            return parsed
        repaired = _repair_truncated_json(candidate)
        if repaired is not None:
            return repaired

    # Damaged QueryAnalysis: keep analysis text, empty outline for planner fallback
    partial = _extract_query_analysis_partial(clean) or _extract_query_analysis_partial(raw)
    if partial is not None:
        return partial

    # Last resort: original brace extract then raise
    fallback = _extract_outer_json_object(clean) or _extract_outer_json_object(raw)
    if fallback:
        repaired = _repair_truncated_json(fallback)
        if repaired is not None:
            return repaired
        return json.loads(fallback)
    raise json.JSONDecodeError("Expecting value", clean or raw, 0)


def make_json_serializable(obj):
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {make_json_serializable(key): make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(element) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        return str(obj)
    

def make_json_serializable_truncated(obj, max_length: int = 100000):
    if isinstance(obj, (int, float, bool, type(None))):
        if isinstance(obj, (int, float)) and len(str(obj)) > max_length:
            return str(obj)[:max_length - 3] + "..."
        return obj
    elif isinstance(obj, str):
        return obj if len(obj) <= max_length else obj[:max_length - 3] + "..."
    elif isinstance(obj, dict):
        return {make_json_serializable_truncated(key, max_length): make_json_serializable_truncated(value, max_length) 
                for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable_truncated(element, max_length) for element in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable_truncated(obj.__dict__, max_length)
    else:
        result = str(obj)
        return result if len(result) <= max_length else result[:max_length - 3] + "..."
