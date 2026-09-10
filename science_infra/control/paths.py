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


def tir_agent_root() -> Path:
    return repo_root() / "agent-lightning" / "examples" / "tir_agent"


def webui_dist() -> Path:
    return repo_root() / "webui" / "dist"


def webui_src() -> Path:
    return repo_root() / "webui"
