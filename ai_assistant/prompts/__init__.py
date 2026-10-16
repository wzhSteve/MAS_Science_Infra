"""System prompt assembly: base.md + one-line skill index + trigger hints."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from ai_assistant.skills.registry import match_skills, skill_index_lines

_PROMPT_PATH = Path(__file__).resolve().parent / "base.md"


@lru_cache(maxsize=1)
def load_base_prompt() -> str:
    """Read (and cache) the base system prompt shipped next to this module."""
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_system_prompt(message: str = "") -> str:
    """base.md + skill index (one line per skill) + trigger hint when matched.

    The resident cost stays small: the index is one line per skill; full
    SKILL.md bodies load on demand via the load_skill tool.
    """
    lines = [load_base_prompt()]
    index = skill_index_lines()
    if index:
        lines.append("可用技能（需要时先 load_skill 再使用）：")
        lines.extend(index)
        matched = match_skills(message)
        if matched:
            names = "、".join(meta.name for meta in matched)
            lines.append(f"用户消息疑似涉及技能：{names}，优先 load_skill 获取详细步骤。")
    return "\n".join(lines)
