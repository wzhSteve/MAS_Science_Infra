"""Cursor-style progressive skill discovery.

Each skill is a directory under ai_assistant/skills/<name>/SKILL.md with a
YAML frontmatter (name / description / triggers). The system prompt only
carries the one-line index; the full SKILL.md body is loaded on demand via
the load_skill tool (progressive disclosure keeps the context small).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_SKILLS_DIR = Path(__file__).resolve().parent
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class SkillMeta:
    name: str
    description: str
    triggers: tuple[str, ...] = ()
    path: Path = field(default_factory=Path)

    def index_line(self) -> str:
        return f"- {self.name}: {self.description}"


def _parse_frontmatter(text: str) -> tuple[Dict[str, str], List[str], str]:
    """Return (meta, triggers, body). Minimal YAML subset parser (flat keys + inline list)."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, [], text
    meta: Dict[str, str] = {}
    triggers: List[str] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "triggers":
            triggers = [item.strip() for item in value.strip("[]").split(",") if item.strip()]
        elif key:
            meta[key] = value
    body = text[match.end():]
    return meta, triggers, body


def _cache() -> Dict[str, SkillMeta]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is None:
        _REGISTRY_CACHE = discover_skills()
    return _REGISTRY_CACHE


_REGISTRY_CACHE: Optional[Dict[str, SkillMeta]] = None


def discover_skills(skills_dir: Optional[Path] = None) -> Dict[str, SkillMeta]:
    """Scan <skills_dir>/*/SKILL.md and parse frontmatter metadata."""
    root = Path(skills_dir) if skills_dir else _SKILLS_DIR
    found: Dict[str, SkillMeta] = {}
    if not root.is_dir():
        return found
    for skill_md in sorted(root.glob("*/SKILL.md")):
        try:
            text = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue
        meta, triggers, _ = _parse_frontmatter(text)
        name = str(meta.get("name") or skill_md.parent.name).strip()
        found[name] = SkillMeta(
            name=name,
            description=str(meta.get("description") or "").strip(),
            triggers=tuple(triggers),
            path=skill_md,
        )
    return found


def list_skills() -> List[SkillMeta]:
    return list(_cache().values())


def skill_index_lines() -> List[str]:
    return [meta.index_line() for meta in list_skills()]


def match_skills(message: str) -> List[SkillMeta]:
    """Skills whose triggers appear in the user message (hint only)."""
    lowered = str(message or "").lower()
    matched: List[SkillMeta] = []
    for meta in list_skills():
        if any(trigger.lower() in lowered for trigger in meta.triggers):
            matched.append(meta)
    return matched


def load_skill(name: str) -> str:
    meta = _cache().get(str(name or "").strip())
    if meta is None:
        available = ", ".join(sorted(_cache())) or "(none)"
        raise ValueError(f"未知技能 {name!r}。可用技能：{available}")
    body = meta.path.read_text(encoding="utf-8")
    return f"{body}\n\n（本技能可用查询函数见 query_* 工具说明；回答需引用查询返回的真实数据。）"
