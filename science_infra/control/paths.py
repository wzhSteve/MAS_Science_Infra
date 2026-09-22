"""Repo / experiments path helpers."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath

from science_infra.env import repo_root


def experiments_root() -> Path:
    override = os.environ.get("SCIENCE_EXPERIMENTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "experiments"


def model_resources_db() -> Path:
    override = os.environ.get("SCIENCE_MODEL_RESOURCES_DB", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "artifacts" / "control" / "model_resources.sqlite3"


def tir_agent_root() -> Path:
    """MAS layer root (formerly agent-lightning/examples/tir_agent)."""
    return repo_root() / "mas"


def mas_root() -> Path:
    return tir_agent_root()


def rl_root() -> Path:
    """Thin RL hooks package (reward / loss / algo overlays)."""
    return repo_root() / "rl"


def repo_data_dir() -> Path:
    """Canonical parquet dataset root: ``MAS_Science_Infra/data``."""
    return repo_root() / "data"


SERVER_DATA_DIR = PurePosixPath("/root/autodl-tmp/MAS_Science_Infra/data")


def server_data_path(value: str) -> str:
    """Keep persisted dataset paths independent of the developer's OS."""
    value = value.strip()
    if PureWindowsPath(value).drive or "\\" in value:
        raise ValueError("数据路径必须是 Linux 服务器路径，不能使用 Windows 本地路径。")
    path = PurePosixPath(value)
    if path.is_absolute():
        return str(path)
    if path.parts and path.parts[0] == "data":
        path = PurePosixPath(*path.parts[1:])
    return str(SERVER_DATA_DIR / path)


def webui_dist() -> Path:
    return repo_root() / "webui" / "dist"


def webui_src() -> Path:
    return repo_root() / "webui"
