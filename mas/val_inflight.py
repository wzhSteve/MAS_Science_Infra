"""Cap how many validation planner generations run at once.

Training keeps every runner. Validation used to start one 4096-token planner
call per runner at the same moment, and those calls exceeded the 300s client
timeout. Eight file locks match the val-only run that finished without
timeouts. The lock covers that generation only: search, fetch, and Python
release it. A process-local semaphore cannot see the other runners.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, TextIO

_DEFAULT_SLOTS = 8


def inflight_limit() -> int:
    raw = os.environ.get("AGENTFLOW_VAL_INFLIGHT", "").strip()
    if not raw:
        return _DEFAULT_SLOTS
    return max(1, int(raw))


def _slot_dir() -> Path:
    raw = os.environ.get("AGENTFLOW_VAL_INFLIGHT_DIR", "").strip()
    directory = Path(raw) if raw else Path("/tmp/fcr_val_inflight")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class ValInflight:
    """Hold one of ``slots`` exclusive file locks until released."""

    def __init__(self, slots: int, directory: Optional[Path] = None) -> None:
        self.slots = max(1, int(slots))
        self.directory = directory or _slot_dir()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[TextIO] = None

    def acquire(self, poll_s: float = 0.05) -> None:
        while self._fh is None:
            for index in range(self.slots):
                handle = open(self.directory / f"slot_{index}", "a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                self._fh = handle
                return
            time.sleep(poll_s)

    def release(self) -> None:
        handle = self._fh
        self._fh = None
        if handle is None:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


@contextmanager
def val_inflight(slots: Optional[int] = None, directory: Optional[Path] = None) -> Iterator[None]:
    gate = ValInflight(inflight_limit() if slots is None else slots, directory)
    gate.acquire()
    try:
        yield
    finally:
        gate.release()


_VAL_MODE = "AGENTFLOW_VAL_INFLIGHT_ON"


@contextmanager
def bind_val_mode(mode: str) -> Iterator[None]:
    """Mark this process's current rollout as validation. Training is unchanged."""
    if mode == "train":
        yield
        return
    previous = os.environ.get(_VAL_MODE)
    os.environ[_VAL_MODE] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_VAL_MODE, None)
        else:
            os.environ[_VAL_MODE] = previous


def val_generation_active() -> bool:
    return os.environ.get(_VAL_MODE) == "1"


@contextmanager
def hold_planner_generation(slots: Optional[int] = None, directory: Optional[Path] = None) -> Iterator[None]:
    """Hold one slot for a trainable planner call. Tool time and training do not."""
    if not val_generation_active():
        yield
        return
    with val_inflight(slots=slots, directory=directory):
        yield


@contextmanager
def hold_val_slot(mode: str, slots: Optional[int] = None, directory: Optional[Path] = None) -> Iterator[None]:
    """Validation waits for a free slot. Training does not."""
    if mode == "train":
        yield
        return
    with val_inflight(slots=slots, directory=directory):
        yield
