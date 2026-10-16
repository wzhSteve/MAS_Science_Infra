"""History compaction: keep recent turns verbatim, summarize the overflow."""

from __future__ import annotations

from typing import Dict, List

HISTORY_LIMIT = 24          # same constant the old module used
COMPACT_TRIGGER = 24        # when kept history reaches this, compress the oldest half
COMPACT_CHUNK = 12          # how many oldest messages to fold into the summary
MAX_SUMMARY_CHARS = 1500


def _render(message: Dict[str, str]) -> str:
    role = "用户" if message.get("role") == "user" else "助手"
    return f"{role}: {message.get('content') or ''}".strip()


def compress_history(
    history: List[Dict[str, str]],
    *,
    previous_summary: str = "",
) -> tuple[List[Dict[str, str]], str]:
    """Return (kept_history, running_summary).

    When history stays under COMPACT_TRIGGER the input passes through unchanged.
    Otherwise the oldest COMPACT_CHUNK messages are folded into a running
    summary string; the caller injects it as a system message so long
    conversations stop "forgetting" their beginnings.
    """
    kept = [m for m in history or [] if isinstance(m, dict) and (m.get("content") or "").strip()]
    if len(kept) < COMPACT_TRIGGER:
        return kept, previous_summary
    overflow, retained = kept[:COMPACT_CHUNK], kept[COMPACT_CHUNK:]
    lines = [_render(m) for m in overflow]
    if previous_summary:
        lines.insert(0, f"此前摘要: {previous_summary}")
    summary = "\n".join(line for line in lines if line)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[-MAX_SUMMARY_CHARS:]
    return retained, summary


def summary_system_message(summary: str) -> Dict[str, str] | None:
    text = (summary or "").strip()
    if not text:
        return None
    return {"role": "system", "content": "更早对话的压缩摘要（供参考，不要复述）：\n" + text}
