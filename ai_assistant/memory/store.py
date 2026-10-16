"""Durable assistant memory, scoped per experiment/project (Cursor agent-store style)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

_LOCK = threading.Lock()
MAX_FACTS = 64
MAX_FACT_CHARS = 500
_SCOPE_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def memory_root() -> Path:
    """Memory lives under user_space so the three-zone rule still holds."""
    override = os.environ.get("SCIENCE_USER_SPACE_DIR", "").strip()
    user_root = Path(override).expanduser() if override else Path(__file__).resolve().parents[2] / "user_space"
    return user_root / "assistant_memory"


def resolve_scope(experiment_id: Optional[str] = None, project_id: Optional[str] = None) -> str:
    if experiment_id:
        return f"experiment:{experiment_id}"
    if project_id:
        return f"project:{project_id}"
    return "global"


class MemoryStore:
    """JSON-file facts store: user_space/assistant_memory/<scope>.json."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = Path(root) if root else memory_root()

    def _path(self, scope: str) -> Path:
        return self.root / f"{scope}.json"

    @staticmethod
    def _validate_scope(scope: str) -> str:
        scope = str(scope or "global").strip()
        if not _SCOPE_RE.match(scope):
            raise ValueError(f"非法记忆作用域 {scope!r}")
        return scope

    def read(self, scope: str) -> Dict[str, Any]:
        scope = self._validate_scope(scope)
        path = self._path(scope)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"scope": scope, "facts": [], "sessions": []}
        if not isinstance(data, dict):
            return {"scope": scope, "facts": [], "sessions": []}
        data.setdefault("scope", scope)
        data.setdefault("facts", [])
        data.setdefault("sessions", [])
        return data

    def _write(self, scope: str, data: Dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(scope)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        temporary.replace(path)

    def facts(self, scope: str, pinned_only: bool = False) -> List[Dict[str, Any]]:
        items = [f for f in self.read(scope).get("facts", []) if isinstance(f, dict)]
        if pinned_only:
            items = [f for f in items if f.get("pinned")]
        return sorted(items, key=lambda f: (not f.get("pinned"), str(f.get("updated_at") or "")))

    def write_fact(self, scope: str, text: str, *, pinned: bool = False) -> str:
        """Append (or update a fact with identical text) and enforce MAX_FACTS."""
        scope = self._validate_scope(scope)
        text = str(text or "").strip()[:MAX_FACT_CHARS]
        if not text:
            raise ValueError("记忆内容为空。")
        with _LOCK:
            data = self.read(scope)
            now = time.time()
            facts: List[Dict[str, Any]] = [f for f in data.get("facts", []) if isinstance(f, dict)]
            for fact in facts:
                if str(fact.get("text") or "").strip() == text:
                    fact["updated_at"] = now
                    fact["pinned"] = bool(pinned) or bool(fact.get("pinned"))
                    self._write(scope, data)
                    return str(fact.get("id"))
            facts.append({"id": uuid.uuid4().hex[:12], "text": text, "created_at": now, "updated_at": now, "pinned": bool(pinned)})
            overflow = len(facts) - MAX_FACTS
            if overflow > 0:
                facts.sort(key=lambda f: (f.get("pinned"), f.get("updated_at") or 0), reverse=True)
                evicted = facts[MAX_FACTS:]
                facts = facts[:MAX_FACTS]
            else:
                evicted = []
            data["facts"] = facts
            self._write(scope, data)
        return "ok"

    def delete_fact(self, scope: str, fact_id: str) -> bool:
        scope = self._validate_scope(scope)
        with _LOCK:
            data = self.read(scope)
            before = data.get("facts") or []
            kept = [f for f in before if str(f.get("id")) != str(fact_id)]
            if len(kept) == len(before):
                return False
            data["facts"] = kept
            self._write(scope, data)
        return True

    def save_session_summary(self, scope: str, summary: str, *, experiment_id: Optional[str] = None) -> None:
        scope = self._validate_scope(scope)
        summary = str(summary or "").strip()[:4000]
        if not summary:
            return
        with _LOCK:
            data = self.read(scope)
            sessions: List[Dict[str, Any]] = [s for s in data.get("sessions", []) if isinstance(s, dict)]
            sessions.append({"id": uuid.uuid4().hex[:12], "summary": summary, "experiment_id": experiment_id or "", "updated_at": time.time()})
            data["sessions"] = sessions[-8:]
            self._write(scope, data)

    def latest_session_summary(self, scope: str) -> str:
        sessions = [s for s in self.read(scope).get("sessions", []) if isinstance(s, dict)]
        if not sessions:
            return ""
        return str(sessions[-1].get("summary") or "")


def pinned_memory_block(store: MemoryStore, scope: str, char_limit: int = 2000) -> str:
    """Render pinned facts for system-prompt injection (~500 tokens cap)."""
    facts = store.facts(scope, pinned_only=True)
    if not facts:
        return ""
    lines: List[str] = []
    used = 0
    for fact in facts:
        line = f"- {fact.get('text')}"
        if used + len(line) > char_limit:
            break
        lines.append(line)
        used += len(line)
    if not lines:
        return ""
    return "已知用户偏好与机器事实（来自长期记忆，可被新信息覆盖）：\n" + "\n".join(lines)
