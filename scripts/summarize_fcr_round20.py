#!/usr/bin/env python3
"""Summarize 20h round A/B/C into experiments/fcr_short/ROUND20_SUMMARY.md."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "experiments" / "fcr_short"
CKPT = ROOT / "mas" / "checkpoints" / "AgentLightning"
OUT = SPEC / "ROUND20_SUMMARY.md"


def _offline(name: str) -> dict:
    path = SPEC / "offline_pred" / f"{name}_metrics.jsonl"
    if not path.exists():
        return {"missing": True}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    last = rows[-1]
    return {
        "steps": len(rows),
        "mean_p": last.get("mean_p"),
        "bss": last.get("bss"),
        "pos_recall": last.get("positive_recall"),
        "mean_p_pos": last.get("mean_p_given_pos"),
        "acc": last.get("accuracy"),
        "elapsed_s": last.get("elapsed_s"),
    }


def _online(name: str) -> dict:
    path = CKPT / name / "metrics.jsonl"
    if not path.exists():
        return {"missing": True}
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        return {"missing": False, "steps": 0, "empty": True}

    def mean(key):
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    last = rows[-1]
    return {
        "steps": len(rows),
        "reward": mean("training/reward"),
        "brier": last.get("fcr/brier"),
        "acc": last.get("fcr/accuracy"),
        "bss": last.get("fcr/bss"),
        "mean_p": last.get("fcr/mean_p"),
        "unexpected": last.get("fcr/quad_unexpected"),
        "budget": last.get("fcr/budget_violation_count"),
        "repair_hits_meta": last.get("fcr/n_corr_hits"),
    }


def main() -> int:
    lines = ["# Round20 summary (≤20h)", ""]
    lines.append("## H1 offline")
    for name in ("fcr_r20_A1", "fcr_r20_A5"):
        lines.append(f"- `{name}`: `{json.dumps(_offline(name), ensure_ascii=False)}`")
    lines.append("")
    lines.append("## H2/H3 online")
    for name in ("fcr_r20_B1", "fcr_r20_B3", "fcr_r20_C1", "fcr_r20_C2"):
        lines.append(f"- `{name}`: `{json.dumps(_online(name), ensure_ascii=False)}`")
    lines.append("")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
