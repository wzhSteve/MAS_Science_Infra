"""Load repo-root ``.env`` into ``os.environ`` (no python-dotenv dependency).

Priority: existing non-empty process env wins unless ``override=True``.
Looks for ``Agent_Science_Infra/.env`` (or ``SCIENCE_INFRA_ENV`` path).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED_FROM: Optional[Path] = None


def repo_root() -> Path:
    """``science_infra/env.py`` → package → Agent_Science_Infra."""
    return Path(__file__).resolve().parents[1]


def default_env_path() -> Path:
    override = os.environ.get("SCIENCE_INFRA_ENV", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / ".env"


def _strip_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = rest.strip()
        if val and val[0] not in ("'", '"') and " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        out[key] = _strip_quotes(val)
    return out


def load_science_env(*, override: bool = False, path: Optional[Path] = None) -> Optional[Path]:
    """Load ``.env``. Returns the path loaded, or None if missing."""
    global _LOADED_FROM
    env_path = Path(path) if path is not None else default_env_path()
    if not env_path.is_file():
        return None
    if (
        _LOADED_FROM is not None
        and _LOADED_FROM.resolve() == env_path.resolve()
        and not override
    ):
        return _LOADED_FROM
    data = parse_env_file(env_path)
    for key, value in data.items():
        if override:
            os.environ[key] = value
        elif key not in os.environ or os.environ.get(key, "") == "":
            os.environ[key] = value
    _LOADED_FROM = env_path
    return env_path


def env_loaded_from() -> Optional[Path]:
    return _LOADED_FROM
