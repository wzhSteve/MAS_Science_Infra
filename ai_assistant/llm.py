"""Resolve the assistant LLM endpoint without exposing credentials.

Moved verbatim from science_infra/control/assistant.py so the assistant
lives in its own top-level package (ai_assistant/).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from science_infra.env import load_science_env  # noqa: F401  (patch target for tests)


@dataclass(frozen=True)
class AssistantLlm:
    model: str
    base_url: str
    api_key: str
    source: str  # assistant_env | experiment | none

    def __repr__(self) -> str:
        return f"AssistantLlm(model={self.model!r}, base_url={self.base_url!r}, source={self.source!r})"


def resolve_assistant_llm(experiment_id: Optional[str] = None) -> AssistantLlm:
    """Prefer AI_ASSISTANT_* from repo .env. Never log the key."""
    from science_infra.control.llm_config import resolve_llm_config

    load_science_env()
    key = os.environ.get("AI_ASSISTANT_API_KEY", "").strip()
    base = (os.environ.get("AI_ASSISTANT_API_BASE") or os.environ.get("AI_ASSISTANT_BASE_URL") or "").strip()
    model = os.environ.get("AI_ASSISTANT_MODEL", "").strip()
    if base and model:
        return AssistantLlm(model=model, base_url=base.rstrip("/"), api_key=key, source="assistant_env")
    if experiment_id:
        try:
            cfg = resolve_llm_config(experiment_id)
            if cfg.base_url and cfg.model:
                return AssistantLlm(
                    model=cfg.model,
                    base_url=str(cfg.base_url).rstrip("/"),
                    api_key=cfg.api_key or "",
                    source="experiment",
                )
        except Exception:
            pass
    return AssistantLlm("", "", "", "none")
