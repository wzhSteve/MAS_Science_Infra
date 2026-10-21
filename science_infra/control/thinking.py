"""Read and write the repo ``.env`` key ``enable_thinking``.

The value is the global switch for Qwen thinking. Other lines in the file stay
as they are. The current process env is updated so the next training launch
inherits it.
"""

from __future__ import annotations

import os
from pathlib import Path

from science_infra.env import default_env_path, parse_env_file

_KEY = "enable_thinking"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}


def parse_enable_thinking(raw: str | None) -> bool:
    text = str(raw or "").strip().lower()
    if text in _TRUE:
        return True
    return False


def read_enable_thinking(path: Path | None = None) -> bool:
    env_path = path or default_env_path()
    file_value = parse_env_file(env_path).get(_KEY)
    if file_value is not None:
        enabled = parse_enable_thinking(file_value)
    else:
        enabled = parse_enable_thinking(os.environ.get(_KEY) or os.environ.get("ENABLE_THINKING"))
    os.environ[_KEY] = "true" if enabled else "false"
    return enabled


def write_enable_thinking(enabled: bool, path: Path | None = None) -> bool:
    env_path = path or default_env_path()
    value = "true" if enabled else "false"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.is_file() else []
    found = False
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        body = stripped[len("export ") :].strip() if stripped.startswith("export ") else stripped
        key = body.split("=", 1)[0].strip() if "=" in body and not stripped.startswith("#") else ""
        if key == _KEY:
            updated.append(f"{_KEY}={value}")
            found = True
        else:
            updated.append(line)
    if not found:
        if updated and updated[-1] != "":
            updated.append("")
        updated.append(f"{_KEY}={value}")
    text = "\n".join(updated)
    if not text.endswith("\n"):
        text += "\n"
    env_path.write_text(text, encoding="utf-8")
    os.environ[_KEY] = value
    return enabled
