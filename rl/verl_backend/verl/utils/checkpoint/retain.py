"""Keep the latest checkpoint and the one with the best val/reward."""

from __future__ import annotations

import json
import os
import re
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

BEST_VAL_FILENAME = "best_val_checkpoint.json"
STEP_VAL_FILENAME = "val_reward.json"
_STEP_DIR = re.compile(r"^global_step_(\d+)$")


def parse_global_step_name(name: str) -> Optional[int]:
    match = _STEP_DIR.fullmatch(name)
    return int(match.group(1)) if match else None


def list_global_step_dirs(root: str) -> List[Tuple[int, str]]:
    if not root or not os.path.isdir(root):
        return []
    out: List[Tuple[int, str]] = []
    for name in os.listdir(root):
        step = parse_global_step_name(name)
        if step is None:
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            out.append((step, os.path.abspath(path)))
    out.sort(key=lambda item: item[0])
    return out


def load_best_val(root: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(root, BEST_VAL_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_best_val(root: str, step: int, reward: float, folder: str) -> Dict[str, Any]:
    payload = {
        "step": int(step),
        "val/reward": float(reward),
        "path": os.path.abspath(folder),
    }
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, BEST_VAL_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)
    return payload


def write_step_val_reward(folder: str, step: int, reward: Optional[float]) -> None:
    if reward is None:
        return
    payload = {"step": int(step), "val/reward": float(reward)}
    path = os.path.join(folder, STEP_VAL_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
        handle.write("\n")


def best_from_metrics_jsonl(metrics_path: str, existing_steps: Iterable[int]) -> Optional[Tuple[int, float]]:
    have = set(int(s) for s in existing_steps)
    if not have or not os.path.isfile(metrics_path):
        return None
    best_step: Optional[int] = None
    best_reward: Optional[float] = None
    with open(metrics_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("training/global_step", row.get("step"))
            reward = row.get("val/reward")
            if step is None or reward is None:
                continue
            try:
                step_i = int(step)
                reward_f = float(reward)
            except (TypeError, ValueError):
                continue
            if step_i not in have:
                continue
            if best_reward is None or reward_f > best_reward:
                best_reward = reward_f
                best_step = step_i
    if best_step is None or best_reward is None:
        return None
    return best_step, best_reward


def resolve_best_step(root: str, current_step: int, current_reward: Optional[float]) -> Optional[int]:
    """Return the step number whose weights should be kept as best-val."""
    existing = list_global_step_dirs(root)
    existing_steps = {step for step, _ in existing}
    recorded = load_best_val(root)
    best_step: Optional[int] = None
    best_reward: Optional[float] = None
    if recorded is not None:
        try:
            rec_step = int(recorded.get("step"))
            rec_reward = float(recorded.get("val/reward"))
        except (TypeError, ValueError):
            rec_step = None
            rec_reward = None
        if rec_step in existing_steps and rec_reward is not None:
            best_step, best_reward = rec_step, rec_reward
    if best_step is None:
        seeded = best_from_metrics_jsonl(os.path.join(root, "metrics.jsonl"), existing_steps)
        if seeded is not None:
            best_step, best_reward = seeded
    if current_reward is not None:
        if best_reward is None or float(current_reward) > best_reward:
            best_step = int(current_step)
            best_reward = float(current_reward)
            folder = os.path.join(root, f"global_step_{current_step}")
            write_best_val(root, current_step, best_reward, folder)
            return best_step
    if best_step is not None and best_reward is not None:
        folder = os.path.join(root, f"global_step_{best_step}")
        if os.path.isdir(folder):
            write_best_val(root, best_step, best_reward, folder)
    return best_step


def dirs_to_keep(
    step_dirs: Sequence[Tuple[int, str]],
    *,
    latest_step: int,
    best_step: Optional[int],
    max_keep: Optional[int],
) -> Set[str]:
    keep: Set[str] = set()
    by_step = {step: path for step, path in step_dirs}
    if latest_step in by_step:
        keep.add(by_step[latest_step])
    if best_step is not None and best_step in by_step:
        keep.add(by_step[best_step])
    if max_keep is None:
        return {path for _, path in step_dirs}
    newest = sorted(step_dirs, key=lambda item: item[0], reverse=True)
    for step, path in newest[: max(0, int(max_keep))]:
        keep.add(path)
    return keep


def drop_unkept_global_steps(root: str, keep_paths: Iterable[str]) -> List[str]:
    keep_abs = {os.path.abspath(path) for path in keep_paths}
    removed: List[str] = []
    for step, path in list_global_step_dirs(root):
        if path in keep_abs:
            continue
        print(f"Removing older checkpoint {path}")
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed
