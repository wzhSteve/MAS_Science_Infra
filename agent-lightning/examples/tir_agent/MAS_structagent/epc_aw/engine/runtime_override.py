"""Process-wide LLM runtime override for RL / local vLLM.

MAS_structagent normally reads ``MODEL_Name`` / ``OPENAI_API_BASE_URL`` from ``.env``.
During Agent-lightning training those must be replaced by the VERL OpenAI
endpoint + policy model for the duration of one full ``solver.solve()``.

Priority (highest first):
1. Values set via ``set_llm_runtime_override``
2. Explicit ``ChatOpenAI(..., base_url=..., model_string=...)`` kwargs
3. Process env / dotenv
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator

_RUNTIME: Dict[str, Any] = {}


def set_llm_runtime_override(
    *,
    base_url: str,
    model: str,
    api_key: str = "dummy",
    openai_compatible: bool = True,
) -> None:
    """Force all subsequent ChatOpenAI clients onto this endpoint/model."""
    global _RUNTIME
    if not base_url:
        raise ValueError("base_url is required for LLM runtime override")
    if not model:
        raise ValueError("model is required for LLM runtime override")
    _RUNTIME = {
        "base_url": str(base_url).rstrip("/"),
        "model": str(model),
        "api_key": api_key or "dummy",
        "openai_compatible": bool(openai_compatible),
    }


def clear_llm_runtime_override() -> None:
    global _RUNTIME
    _RUNTIME = {}


def get_llm_runtime_override() -> Dict[str, Any]:
    return dict(_RUNTIME)


@contextmanager
def llm_runtime_override(
    *,
    base_url: str,
    model: str,
    api_key: str = "dummy",
    openai_compatible: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Context manager: apply override for one full MAS solve / rollout."""
    prev = get_llm_runtime_override()
    set_llm_runtime_override(
        base_url=base_url,
        model=model,
        api_key=api_key,
        openai_compatible=openai_compatible,
    )
    try:
        yield get_llm_runtime_override()
    finally:
        clear_llm_runtime_override()
        if prev:
            set_llm_runtime_override(
                base_url=prev["base_url"],
                model=prev["model"],
                api_key=prev.get("api_key", "dummy"),
                openai_compatible=bool(prev.get("openai_compatible", True)),
            )
