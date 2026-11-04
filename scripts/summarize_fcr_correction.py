#!/usr/bin/env python3
"""Summarize correction-first pilot metrics."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "mas" / "checkpoints" / "AgentLightning"
OUT = ROOT / "experiments" / "fcr_short" / "CORRECTION_PILOT_SUMMARY.md"
ARMS = (
    "fcr_br_strict_smoke",
    "fcr_br_strict",
    "fcr_grpo_base",
    "fcr_b_only",
    "fcr_fail_only",
    "fcr_br_weak",
    "fcr_random",
    "fcr_r_strict",
)
KEYS = (
    "training/reward",
    "fcr/R_answer_mean",
    "fcr/R_correct_mean",
    "fcr/strict_event_count",
    "fcr/weak_event_count",
    "fcr/mean_p",
    "fcr/quad_unexpected",
    "fcr/bss",
    "fcr/budget_violation_count",
    "fcr/branch_uplift",
    "fcr/branch_count",
    "fcr/failure_confirmed_rate",
    "fcr/branch_starvation",
    "fcr/corr_correct_answer",
    "fcr/hack_highN_wrong_frac",
)


def _rows(name: str):
    path = CKPT / name / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    lines = ["# Correction-first pilot", ""]
    for name in ARMS:
        rows = _rows(name)
        lines.append(f"## {name}")
        if not rows:
            lines.append("- missing")
            lines.append("")
            continue
        lines.append(f"- steps: {len(rows)}")
        for key in KEYS:
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            if not vals:
                continue
            lines.append(f"- `{key}` last={vals[-1]:.6g} mean={sum(vals)/len(vals):.6g} sum={sum(vals):.6g}")
        lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
