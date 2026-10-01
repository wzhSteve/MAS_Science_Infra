"""Filesystem jail for the single user-code tree ``user_space/``.

Management code may read the repo. Writes from the assistant and user uploads
must resolve inside this tree. ``upload/`` is ingest-once (read-only afterwards).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

PROJECT_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,47}$")
USER_AGENT_PREFIX = "u_"
WRITABLE_REL_DIRS = ("adapted", "contracts", "artifacts")
READ_ONLY_REL_DIRS = ("upload",)
SECRET_NAMES = {".env", ".secrets.env"}
FORBIDDEN_WRITE_ROOTS = (
    "mas",
    "science_infra",
    "rl",
    "webui",
    "experiments",
    "agent-lightning",
    ".venv",
    "LLM",
    "data",
)


def repo_root() -> Path:
    """``user_gateway/paths.py`` → workflow → mas → repo. No science_infra import."""
    return Path(__file__).resolve().parents[3]


def user_space_root() -> Path:
    override = os.environ.get("SCIENCE_USER_SPACE_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (repo_root() / "user_space").resolve()


def registry_path() -> Path:
    return user_space_root() / "registry.yaml"


def projects_root() -> Path:
    return user_space_root() / "projects"


def ensure_user_space() -> Path:
    root = user_space_root()
    (root / "projects").mkdir(parents=True, exist_ok=True)
    if not registry_path().is_file():
        registry_path().write_text("version: 1\nprojects: []\n", encoding="utf-8")
    return root


def validate_project_id(project_id: str) -> str:
    pid = str(project_id or "").strip().lower()
    if not PROJECT_ID_RE.match(pid):
        raise ValueError("project id 必须是小写字母开头、仅含字母数字_-，最长 48。")
    return pid


def slugify_project_id(title: str, fallback: str = "project") -> str:
    raw = re.sub(r"[^a-z0-9_-]+", "-", str(title or "").strip().lower()).strip("-")
    if not raw or not raw[0].isalpha():
        raw = f"{fallback}-{raw}".strip("-") if raw else fallback
    raw = re.sub(r"-{2,}", "-", raw)[:48].strip("-")
    if not PROJECT_ID_RE.match(raw):
        raw = fallback
    return raw


def project_dir(project_id: str) -> Path:
    return projects_root() / validate_project_id(project_id)


def user_agent_id(project_id: str, local_id: str) -> str:
    pid = validate_project_id(project_id)
    local = re.sub(r"[^a-zA-Z0-9_]+", "_", str(local_id or "agent").strip()) or "agent"
    return f"{USER_AGENT_PREFIX}{pid}__{local}"


def parse_user_agent_id(agent_id: str) -> Optional[Tuple[str, str]]:
    text = str(agent_id or "")
    if not text.startswith(USER_AGENT_PREFIX) or "__" not in text:
        return None
    body = text[len(USER_AGENT_PREFIX) :]
    pid, _, local = body.partition("__")
    if not PROJECT_ID_RE.match(pid) or not local:
        return None
    return pid, local


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_in_user_space(path: Path, *, must_exist: bool = False) -> Path:
    root = user_space_root()
    resolved = Path(path).expanduser()
    if resolved.is_symlink():
        raise ValueError("拒绝符号链接路径。")
    resolved = resolved.resolve()
    if not _is_relative_to(resolved, root):
        raise PermissionError(f"路径不在用户区: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def assert_readable_repo_path(path: Path) -> Path:
    """Assistant read jail: repo (or user_space override) minus secrets and weights."""
    resolved = Path(path).expanduser().resolve()
    roots = [repo_root().resolve(), user_space_root()]
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise PermissionError(f"不能读取仓库外路径: {resolved}")
    name = resolved.name
    if name in SECRET_NAMES or name.endswith((".safetensors", ".gguf", ".bin")):
        raise PermissionError(f"拒绝读取密钥或权重: {name}")
    parts = {p.lower() for p in resolved.parts}
    if ".venv" in parts or "node_modules" in parts:
        raise PermissionError("拒绝读取运行时依赖目录。")
    return resolved


def assert_writable_user_path(path: Path, *, project_id: Optional[str] = None) -> Path:
    """Write jail: only ``adapted/`` ``contracts/`` ``artifacts/`` under a project."""
    resolved = assert_in_user_space(path)
    rel = resolved.relative_to(user_space_root())
    parts = rel.parts
    if len(parts) < 3 or parts[0] != "projects":
        raise PermissionError("只能写入 user_space/projects/<id>/{adapted,contracts,artifacts}/")
    pid = parts[1]
    validate_project_id(pid)
    if project_id and validate_project_id(project_id) != pid:
        raise PermissionError("不能写入其他用户项目。")
    if parts[2] not in WRITABLE_REL_DIRS:
        raise PermissionError(f"{parts[2]!r} 对辅助 AI 只读（upload 在 ingest 后锁定）。")
    for root in FORBIDDEN_WRITE_ROOTS:
        if resolved == repo_root() / root or _is_relative_to(resolved, repo_root() / root):
            raise PermissionError(f"禁止写入管理区 {root}/")
    return resolved


def project_writable_dirs(project_id: str) -> Iterable[Path]:
    root = project_dir(project_id)
    for name in WRITABLE_REL_DIRS:
        yield root / name


def ensure_project_layout(project_id: str) -> Path:
    root = project_dir(project_id)
    for name in ("upload",) + WRITABLE_REL_DIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
        if name == "adapted":
            (root / name / "agents").mkdir(exist_ok=True)
            (root / name / "tools").mkdir(exist_ok=True)
    return root
