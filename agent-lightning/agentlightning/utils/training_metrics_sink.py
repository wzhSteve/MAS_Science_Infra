# Copyright (c) Microsoft. All rights reserved.

"""Persist VERL / Agent-lightning training scalars as JSONL for the dashboard."""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()

# Keep common training curves + anything that looks like a loss / reward / val metric.
_ALWAYS_KEYS = (
    "training/reward",
    "training/global_step",
    "training/epoch",
    "training/n_rollouts",
    "actor/pg_loss",
    "actor/entropy_loss",
    "actor/grad_norm",
    "actor/lr",
    "critic/rewards/mean",
    "critic/rewards/mean_before_processing",
    "critic/rewards/mean_after_processing",
)


def resolve_metrics_jsonl_path(
    *,
    project_name: str,
    experiment_name: str,
    default_local_dir: Optional[str] = None,
) -> str:
    """Resolve metrics.jsonl path; honors ``AGL_METRICS_JSONL`` if set."""
    env_path = os.environ.get("AGL_METRICS_JSONL", "").strip()
    if env_path:
        return os.path.abspath(env_path)
    base = (default_local_dir or "").strip() or os.path.join("checkpoints", project_name, experiment_name)
    return os.path.abspath(os.path.join(base, "metrics.jsonl"))


def ensure_metrics_env(
    *,
    project_name: str,
    experiment_name: str,
    default_local_dir: Optional[str] = None,
) -> str:
    """Ensure ``AGL_METRICS_JSONL`` points at the experiment metrics file."""
    path = resolve_metrics_jsonl_path(
        project_name=project_name,
        experiment_name=experiment_name,
        default_local_dir=default_local_dir,
    )
    os.environ["AGL_METRICS_JSONL"] = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


def _is_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    # numpy / torch scalars
    item = getattr(value, "item", None)
    if callable(item):
        try:
            v = item()
            return isinstance(v, (int, float, bool))
        except Exception:
            return False
    return False


def _to_python_scalar(value: Any) -> float | int | bool:
    item = getattr(value, "item", None)
    if callable(item):
        value = item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    return float(value)


def _keep_key(key: str) -> bool:
    if key in _ALWAYS_KEYS:
        return True
    lower = key.lower()
    if "loss" in lower or "reward" in lower:
        return True
    if lower.startswith("val") or "/val" in lower or lower.startswith("val-"):
        return True
    if lower.startswith("training/"):
        return True
    if lower.startswith("actor/") or lower.startswith("critic/"):
        return True
    return False


def filter_metric_scalars(data: Mapping[str, Any]) -> Dict[str, float | int | bool]:
    """Keep dashboard-relevant scalar metrics from a VERL metrics dict."""
    out: Dict[str, float | int | bool] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not _keep_key(key):
            continue
        if not _is_scalar(value):
            continue
        try:
            out[key] = _to_python_scalar(value)
        except Exception:
            continue
    return out


def append_metrics(
    path: str,
    *,
    step: int,
    data: Mapping[str, Any],
) -> None:
    """Append one training step of scalars to ``metrics.jsonl`` (thread-safe)."""
    scalars = filter_metric_scalars(data)
    record = {"step": int(step), **scalars}
    line = json.dumps(record, ensure_ascii=False, sort_keys=True)
    directory = os.path.dirname(path) or "."
    with _LOCK:
        os.makedirs(directory, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_metrics_jsonl(path: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Read metrics.jsonl; optionally keep only the last ``limit`` rows."""
    if not path or not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError as exc:
        logger.warning("Failed to read metrics jsonl %s: %s", path, exc)
        return []
    if limit is not None and limit > 0 and len(rows) > limit:
        return rows[-limit:]
    return rows


def discover_metrics_jsonl_candidates(extra: Optional[Iterable[str]] = None) -> List[str]:
    """Candidate paths when ``AGL_METRICS_JSONL`` is unset."""
    candidates: List[str] = []
    env_path = os.environ.get("AGL_METRICS_JSONL", "").strip()
    if env_path:
        candidates.append(os.path.abspath(env_path))
    if extra:
        for p in extra:
            if p:
                candidates.append(os.path.abspath(p))
    # Common experiment roots under cwd
    ckpt = os.path.abspath("checkpoints")
    if os.path.isdir(ckpt):
        for root, _dirs, files in os.walk(ckpt):
            if "metrics.jsonl" in files:
                candidates.append(os.path.join(root, "metrics.jsonl"))
    # Dedup preserve order
    seen = set()
    out: List[str] = []
    for p in candidates:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out
