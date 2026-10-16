"""Repo read tools and user-space write jail (moved from control.assistant)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SKIP_DIR_PARTS = {
    ".venv",
    "node_modules",
    "agent-lightning",
    "LLM",
    ".git",
    "__pycache__",
    "dist",
    ".local_expansion",
}


def ensure_mas_path() -> None:
    import sys

    from science_infra.control.paths import mas_root

    mas = str(mas_root())
    if mas not in sys.path:
        sys.path.insert(0, mas)


def build_repo_tools(
    project_id: Optional[str],
    session: Any,
) -> List[Dict[str, Any]]:
    """read/grep/list_dir (repo-wide, read-only) + write_file (user-space jail)."""
    ensure_mas_path()
    from science_infra.env import repo_root
    from workflow.user_gateway import paths
    from workflow.user_gateway.paths import assert_readable_repo_path, assert_writable_user_path

    root = repo_root()

    def _resolve_read(path: str) -> Path:
        raw = Path(path) if path else root
        target = raw if raw.is_absolute() else root / raw
        return assert_readable_repo_path(target)

    def read_file(path: str, offset: int = 1, limit: int = 0) -> str:
        target = _resolve_read(path)
        if target.is_dir():
            names = sorted(p.name for p in target.iterdir())[:80]
            return "\n".join(names)
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(int(offset or 1) - 1, 0)
        chunk = lines[start : start + int(limit)] if int(limit or 0) > 0 else lines[start:]
        blob = "\n".join(chunk)
        if len(blob) > 12000:
            return blob[:12000] + "\n...[truncated]"
        return blob

    def grep(query: str, path: str = "", glob: str = "*.py", max_hits: int = 40) -> str:
        if not str(query or "").strip():
            return "(empty query)"
        start = _resolve_read(path) if path else root
        if start.is_file():
            files = [start]
        else:
            files = [p for p in start.rglob(glob or "*.py") if p.is_file()]
        hits: List[str] = []
        for file in files:
            if any(part in SKIP_DIR_PARTS for part in file.parts):
                continue
            try:
                assert_readable_repo_path(file)
                blob = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = file.relative_to(root).as_posix() if str(file).startswith(str(root)) else file.as_posix()
            for idx, line in enumerate(blob.splitlines(), 1):
                if query in line:
                    hits.append(f"{rel}:{idx}:{line.strip()[:200]}")
                    if len(hits) >= max(1, int(max_hits or 40)):
                        return "\n".join(hits)
        return "\n".join(hits) or "(no hits)"

    def search_code(query: str, glob: str = "*.py") -> str:
        return grep(query, path="", glob=glob)

    def list_dir(path: str = ".") -> str:
        target = _resolve_read(path)
        if target.is_file():
            return target.name
        names: List[str] = []
        for child in sorted(target.iterdir()):
            if child.name in SKIP_DIR_PARTS:
                continue
            suffix = "/" if child.is_dir() else ""
            names.append(child.name + suffix)
            if len(names) >= 80:
                break
        return "\n".join(names) or "(empty)"

    def write_file(path: str, content: str) -> str:
        if not project_id:
            raise PermissionError("请先选择或创建用户项目再写文件。")
        rel = path
        if not rel.startswith("adapted/") and not rel.startswith("contracts/") and not rel.startswith("artifacts/"):
            rel = f"adapted/{Path(path).name}"
        target = paths.project_dir(project_id) / rel
        locked = assert_writable_user_path(target, project_id=project_id)
        locked.parent.mkdir(parents=True, exist_ok=True)
        before = locked.read_text(encoding="utf-8") if locked.exists() else ""
        locked.write_text(content, encoding="utf-8")
        session.wrote_files = True
        return json.dumps({"path": rel, "bytes": len(content), "changed": before != content}, ensure_ascii=False)

    return [
        {"name": "read_file", "fn": read_file},
        {"name": "grep", "fn": grep},
        {"name": "search_code", "fn": search_code},
        {"name": "list_dir", "fn": list_dir},
        {"name": "write_file", "fn": write_file},
    ]
