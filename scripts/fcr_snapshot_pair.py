#!/usr/bin/env python3
"""Pair logged failure snapshots: continue (parent) versus FCR children.

random-K is reported only when a sibling outcomes file is passed. This does
not claim a counterfactual action value.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _pair(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    by_id = {str(row.get("rollout_id")): row for row in rows}
    deltas = []
    costs = []
    for row in rows:
        role = str(row.get("role") or "").lower()
        if role not in ("child", "probe", "branch"):
            continue
        parent = by_id.get(str(row.get("parent_id") or ""))
        if not parent or parent.get("outcome") is None or row.get("outcome") is None:
            continue
        deltas.append(float(row["outcome"]) - float(parent["outcome"]))
    snaps = list(Path(rows[0]["_dir"]).glob("*.json")) if rows and rows[0].get("_dir") else []
    return {
        "n_pairs": float(len(deltas)),
        "n_snapshots": float(len(list(snaps))),
        "mean_delta_fcr_minus_continue": float(sum(deltas) / len(deltas)) if deltas else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--random-outcomes", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = _load(args.snapshot_dir / "outcomes.jsonl")
    for row in rows:
        row["_dir"] = str(args.snapshot_dir)
    report = _pair(rows)
    snaps = sorted(args.snapshot_dir.glob("*.json"))
    report["n_snapshot_files"] = float(len(snaps))
    report["used_snapshots"] = float(min(16, len(snaps)))
    if args.random_outcomes and args.random_outcomes.is_file():
        random_rows = _load(args.random_outcomes)
        random_mean = [
            float(row["outcome"]) for row in random_rows
            if row.get("outcome") is not None and str(row.get("role") or "").lower() in ("child", "probe", "branch")
        ]
        fcr_mean = [
            float(row["outcome"]) for row in rows
            if row.get("outcome") is not None and str(row.get("role") or "").lower() in ("child", "probe", "branch")
        ]
        if random_mean and fcr_mean:
            report["mean_delta_fcr_minus_random"] = (
                sum(fcr_mean) / len(fcr_mean) - sum(random_mean) / len(random_mean)
            )
    report["note"] = (
        "parent outcome is the continued trajectory; child outcome is the "
        "failure-conditioned continuation. Not a do(a) counterfactual."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
