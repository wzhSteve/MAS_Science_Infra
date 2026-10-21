"""Read the VERL tqdm line from a training stdout tail."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# A real update looks like: 2/5 [31:39<45:44, 914.77s/it]
# The opening 0/5 [00:00<?, ?it/s] has no estimate and is ignored.
_PROGRESS = re.compile(
    r"(\d+)\s*/\s*(\d+)\s*\[(\d+(?::\d+){0,2})<(\d+(?::\d+){0,2}),\s*([0-9.]+)(s/it|it/s)\]"
)
_OPEN = re.compile(r"(\d+)\s*/\s*(\d+)\s*\[(\d+(?::\d+){0,2})<\?,")


def _clock_seconds(text: str) -> int:
    parts = [int(part) for part in text.split(":") if part != ""]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def parse_train_progress(text: str) -> Optional[Dict[str, Any]]:
    """Return the latest numeric tqdm update, or None when no step has finished."""
    cleaned = _ANSI.sub("", text or "").replace("\r", "\n")
    match = None
    opened = None
    for line in cleaned.splitlines():
        if "Training Progress" not in line:
            continue
        numeric = None
        for numeric in _PROGRESS.finditer(line):
            pass
        if numeric is not None:
            match = numeric
            opened = None
            continue
        pending = None
        for pending in _OPEN.finditer(line):
            pass
        if pending is not None:
            match = None
            opened = pending
    if match is None and opened is not None:
        step, total, elapsed = opened.groups()
        step_i = int(step)
        total_i = int(total)
        # The opening bar counts finished steps. fit() has already moved
        # global_steps forward, so the step in flight is the next one.
        if 0 <= step_i < total_i:
            step_i += 1
        return {
            "step": step_i,
            "total": total_i,
            "elapsed_seconds": _clock_seconds(elapsed),
            "eta_seconds": None,
            "seconds_per_step": None,
        }
    if match is None:
        return None
    step, total, elapsed, eta, rate, unit = match.groups()
    per_step = float(rate)
    if unit == "it/s":
        per_step = 0.0 if per_step <= 0 else 1.0 / per_step
    return {
        "step": int(step),
        "total": int(total),
        "elapsed_seconds": _clock_seconds(elapsed),
        "eta_seconds": _clock_seconds(eta),
        "seconds_per_step": per_step,
    }


def read_train_progress(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 262144))
            raw = handle.read()
    except OSError:
        return None
    return parse_train_progress(raw.decode("utf-8", errors="replace"))
