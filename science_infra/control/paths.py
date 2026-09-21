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


def webui_dist() -> Path:
    return repo_root() / "webui" / "dist"


def webui_src() -> Path:
    return repo_root() / "webui"
