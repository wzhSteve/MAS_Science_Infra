"""Entropy / branch helpers for ARPO and AEPO-lite (agent-side approximation)."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

RESUME_DIR = os.getenv("TIR_RESUME_DIR", os.path.join(os.path.dirname(__file__), "..", ".resume_cache"))


def obs_hash(tool_name: str, observation: str, n_chars: int = 512) -> str:
    blob = f"{tool_name}\n{(observation or '')[:n_chars]}"
    return hashlib.sha1(blob.encode("utf-8", errors="ignore")).hexdigest()[:16]


def mean_token_entropy_from_logprobs(token_logprobs: Sequence[float]) -> float:
    """Approximate entropy from per-token log π(chosen). H ≈ -mean log p."""
    if not token_logprobs:
        return 0.0
    return float(-sum(token_logprobs) / max(1, len(token_logprobs)))


def sigmoid(x: float) -> float:
    x = max(-20.0, min(20.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def aepo_global_budget(total_m: int, h_root: float, h_tool: float, beta: float = 1.0) -> int:
    """m = M * σ(β (H_root - H_tool))."""
    m = total_m * sigmoid(beta * (h_root - h_tool))
    m_int = int(round(m))
    return max(1, min(total_m, m_int))


def branch_probability(
    delta_h: float,
    *,
    alpha: float = 0.5,
    gamma: float = 0.2,
    consecutive_high: int = 0,
    penalty_slope: float = 0.25,
) -> float:
    p_hat = min(0.9, penalty_slope * max(0, consecutive_high))
    return max(0.0, (alpha + gamma * delta_h) * (1.0 - p_hat))


def should_branch(prob: float, tau: float) -> bool:
    return prob > tau


def dump_resume(rollout_id: str, payload: Dict[str, Any]) -> str:
    path = Path(RESUME_DIR)
    path.mkdir(parents=True, exist_ok=True)
    dest = path / f"{rollout_id}.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(dest)


def load_resume(rollout_id: str) -> Optional[Dict[str, Any]]:
    dest = Path(RESUME_DIR) / f"{rollout_id}.json"
    if not dest.is_file():
        return None
    try:
        return json.loads(dest.read_text(encoding="utf-8"))
    except Exception:
        return None


def serialize_messages(messages: Sequence[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = getattr(msg, "type", None) or getattr(msg, "role", None) or msg.__class__.__name__
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
            "HumanMessage": "user",
            "AIMessage": "assistant",
            "SystemMessage": "system",
            "ToolMessage": "tool",
        }
        role = role_map.get(str(role), str(role))
        content = getattr(msg, "content", "") or ""
        item: Dict[str, Any] = {"role": role, "content": str(content)}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = _jsonable_tool_calls(tool_calls)
        tool_call_id = getattr(msg, "tool_call_id", None)
        if tool_call_id:
            item["tool_call_id"] = tool_call_id
        name = getattr(msg, "name", None)
        if name:
            item["name"] = name
        out.append(item)
    return out


def _jsonable_tool_calls(tool_calls: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for call in tool_calls or []:
        if isinstance(call, dict):
            out.append(
                {
                    "name": call.get("name"),
                    "args": call.get("args") or call.get("arguments") or {},
                    "id": call.get("id") or "",
                    "type": call.get("type") or "tool_call",
                }
            )
        else:
            out.append(
                {
                    "name": getattr(call, "name", None),
                    "args": getattr(call, "args", {}) or {},
                    "id": getattr(call, "id", "") or "",
                    "type": getattr(call, "type", "tool_call"),
                }
            )
    return out


def deserialize_messages(items: Sequence[Dict[str, Any]]) -> List[Any]:
    """Rebuild LangChain messages from serialize_messages output."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    out: List[Any] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = str(item.get("content") or "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role in ("user", "human"):
            out.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            kwargs: Dict[str, Any] = {"content": content}
            if item.get("tool_calls"):
                kwargs["tool_calls"] = item["tool_calls"]
            out.append(AIMessage(**kwargs))
        elif role == "tool":
            out.append(
                ToolMessage(
                    content=content,
                    tool_call_id=str(item.get("tool_call_id") or item.get("name") or "tool"),
                    **({"name": item["name"]} if item.get("name") else {}),
                )
            )
        else:
            out.append(HumanMessage(content=content))
    return out


def extract_token_logprobs(ai_message: Any, max_tokens: int = 8) -> List[float]:
    """Pull chosen-token logprobs from a LangChain AIMessage if the server returned them."""
    meta = getattr(ai_message, "response_metadata", None) or {}
    additional = getattr(ai_message, "additional_kwargs", None) or {}
    lp = meta.get("logprobs") or additional.get("logprobs") or {}
    if isinstance(lp, dict):
        content = lp.get("content") or lp.get("token_logprobs") or []
    else:
        content = lp
    out: List[float] = []
    for tok in content or []:
        if isinstance(tok, dict) and tok.get("logprob") is not None:
            out.append(float(tok["logprob"]))
        elif isinstance(tok, (int, float)):
            out.append(float(tok))
        if len(out) >= max_tokens:
            break
    return out


def estimate_turn_entropy(ai_message: Any, max_tokens: int = 8) -> float:
    lps = extract_token_logprobs(ai_message, max_tokens=max_tokens)
    if lps:
        return mean_token_entropy_from_logprobs(lps)
    if getattr(ai_message, "tool_calls", None):
        return 1.0
    return 0.3
