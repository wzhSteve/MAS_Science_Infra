"""Frozen roles must not request a 4096-token completion."""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "user_space/projects/simple_pev/adapted"))

from engine import frozen_generation_cap  # noqa: E402


def test_trainable_planner_keeps_its_budget():
    assert frozen_generation_cap(4096, trainable=True) == 4096


def test_frozen_role_is_capped_at_1024_by_default(monkeypatch):
    monkeypatch.delenv("AGENTFLOW_FROZEN_MAX_TOKENS", raising=False)
    assert frozen_generation_cap(4096, trainable=False) == 1024
    assert frozen_generation_cap(128, trainable=False) == 128


def test_frozen_cap_can_be_overridden(monkeypatch):
    monkeypatch.setenv("AGENTFLOW_FROZEN_MAX_TOKENS", "512")
    assert frozen_generation_cap(4096, trainable=False) == 512
    os.environ.pop("AGENTFLOW_FROZEN_MAX_TOKENS", None)
