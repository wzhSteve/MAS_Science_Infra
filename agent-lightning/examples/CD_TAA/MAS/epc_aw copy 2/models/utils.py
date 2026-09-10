import json
import re
from typing import Any


def parse_json_from_llm_response(raw: Any) -> Any:
    """
    Parse JSON from LLM text that may be wrapped in markdown fences (```json ... ```).
    Matches the cleaning logic used in planner.analyze_query.

    When the engine uses structured output (e.g. FinalAnswer), ``raw`` may be a
    Pydantic model whose JSON payload lives in ``final_answer`` — unwrap that
    before parsing.
    """
    if isinstance(raw, (dict, list)):
        return raw
    # Chat completions.parse returns FinalAnswer(final_answer="...markdown json...")
    if hasattr(raw, "final_answer"):
        inner = getattr(raw, "final_answer", None)
        if isinstance(inner, str) and inner.strip():
            raw = inner
    if not isinstance(raw, str):
        raw = str(raw)
    clean = raw.strip()
    if clean.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if match:
            clean = match.group(1).strip()
    clean = clean.replace("\\'", "'")
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        start, end = clean.find("{"), clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(clean[start : end + 1])
        raise


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