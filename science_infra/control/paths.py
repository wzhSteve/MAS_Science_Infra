"""Repo / experiments path helpers."""

from __future__ import annotations

import os
from pathlib import Path

from science_infra.env import repo_root


def experiments_root() -> Path:
    override = os.environ.get("SCIENCE_EXPERIMENTS_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / "experiments"


def model_resources_db() -> Path:
    """Private workspace store; SCIENCE_MODEL_RESOURCES_DB may override its path."""
    override = os.environ.get("SCIENCE_MODEL_RESOURCES_DB", "").strip()
    path = Path(override).expanduser().resolve() if override else repo_root() / "resources" / "model_resources.sqlite3"
    if path == experiments_root() or experiments_root() in path.parents:
        raise ValueError("模型资源目录必须位于实验目录之外。")
    return path


def tir_agent_root() -> Path:
    return repo_root() / "agent-lightning" / "examples" / "tir_agent"


def webui_dist() -> Path:
    return repo_root() / "webui" / "dist"


def webui_src() -> Path:
    return repo_root() / "webui"
