#!/usr/bin/env python3
"""Task-level deltas of frozen branch gates against no-branch."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
EVENTS = ROOT / "experiments" / "fcr_short" / "events"
CKPT = ROOT / "mas" / "checkpoints" / "AgentLightning"
OUT = ROOT / "experiments" / "fcr_short" / "BRANCH_PAIRED.md"
BASE = "fcr_eval_nobranch"
ARMS = ("fcr_eval_fail", "fcr_eval_confirmed", "fcr_eval_random")


def _load(name: str):
    path = EVENTS / f"{name}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _task_success(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (int(row.get("global_step") or 0), str(row.get("data_id") or ""))
        answer = row.get("answer")
        if answer is None:
            continue
        grouped[key].append(float(answer))
    return {key: (1.0 if max(vals) > 0 else 0.0) for key, vals in grouped.items() if key[1]}


def _metric(name: str, key: str):
    path = CKPT / name / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals), sum(vals), len(vals)


def main() -> int:
    base = _task_success(_load(BASE))
    lines = ["# Branch paired evaluation (lr=0)", "", "Baseline: no-branch. Delta is task success minus no-branch on the same (step, data_id).", ""]
    base_forks = _metric(BASE, "fcr/branch_count")
    lines.append(f"- no-branch tasks: {len(base)}")
    if base_forks:
        lines.append(f"- no-branch mean branch_count: {base_forks[0]:.4g}")
    lines.append("")
    for arm in ARMS:
        rows = _load(arm)
        if not rows:
            lines.append(f"## {arm}")
            lines.append("- missing")
            lines.append("")
            continue
        scores = _task_success(rows)
        keys = sorted(set(base) & set(scores))
        deltas = [scores[key] - base[key] for key in keys]
        mean = sum(deltas) / len(deltas) if deltas else None
        pos = sum(1 for value in deltas if value > 0) / len(deltas) if deltas else None
        forks = _metric(arm, "fcr/branch_count")
        budget = _metric(arm, "fcr/budget_violation_count")
        lines.append(f"## {arm}")
        lines.append(f"- paired tasks: {len(keys)}")
        lines.append(f"- mean delta: {mean if mean is not None else 'n/a'}")
        lines.append(f"- positive delta fraction: {pos if pos is not None else 'n/a'}")
        if forks:
            lines.append(f"- mean branch_count: {forks[0]:.4g} (sum {forks[1]:.4g})")
        if budget:
            lines.append(f"- mean budget_violation: {budget[0]:.4g}")
        if mean is not None and forks and forks[0] > 0:
            lines.append(f"- delta per mean fork: {mean / forks[0]:.4g}")
        lines.append("")
    lines.append("旧 BR-Strict 的 uplift 不进入本表。")
    lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
