#!/usr/bin/env python3
"""Aggregate paper ablation predictions + product layer telemetry.

Example:
  python scripts/aggregate_dual_track.py \\
    --runs_dir experiments/ablation/runs \\
    --out experiments/ablation/aggregate
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _score_row(row: Dict[str, Any], gold: Optional[Dict[str, str]]) -> Optional[bool]:
    if not gold:
        return None
    qid = str(row.get("id"))
    if qid not in gold:
        return None
    pred = str(row.get("direct_output") or row.get("final_output") or "").strip()
    ans = str(gold[qid]).strip()
    if not pred or not ans:
        return False
    return pred.lower() == ans.lower()


def aggregate_layer_telemetry(telem_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(r.get("event") for r in telem_rows)
    l2a_enter = counts.get("l2a_enter", 0)
    l2a_ok = counts.get("l2a_complete", 0)
    l2b_enter = counts.get("l2b_enter", 0)
    l2b_ok = counts.get("l2b_complete", 0)
    l3_enter = counts.get("l3_enter", 0)
    return {
        "event_counts": dict(counts),
        "l2a_rescue_rate": (l2a_ok / l2a_enter) if l2a_enter else None,
        "l2b_rescue_rate": (l2b_ok / l2b_enter) if l2b_enter else None,
        "l3_enter_count": l3_enter,
        "intervention_trigger_count": counts.get("intervention_trigger", 0),
        "intervention_disabled_count": counts.get("intervention_disabled", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--gold",
        default=None,
        help="Optional JSONL with id + answer for EM scoring",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold: Dict[str, str] = {}
    if args.gold:
        for row in _iter_jsonl(Path(args.gold)):
            gold[str(row.get("id"))] = str(row.get("answer") or row.get("Final answer") or "")

    paper_rows = []
    product_rows = []

    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        meta_path = run_dir / "run_meta.json"
        preset = run_dir.name
        ablation = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            preset = meta.get("preset") or preset
            ablation = meta.get("ablation") or {}

        preds = list(_iter_jsonl(run_dir / "predictions.jsonl"))
        telem = list(_iter_jsonl(run_dir / "layer_telemetry.jsonl"))
        correct = 0
        scored = 0
        for row in preds:
            hit = _score_row(row, gold or None)
            if hit is not None:
                scored += 1
                correct += int(hit)

        layer_stats = aggregate_layer_telemetry(telem)
        summary = {
            "run": run_dir.name,
            "preset": preset,
            "n": len(preds),
            "em": (correct / scored) if scored else None,
            "n_scored": scored,
            "mean_elapsed_s": (
                sum(float(r.get("elapsed_s") or 0) for r in preds) / len(preds)
                if preds
                else None
            ),
            "ablation": ablation,
            "layer": layer_stats,
        }
        paper_rows.append(summary)
        product_rows.append(
            {
                "run": run_dir.name,
                "preset": preset,
                **layer_stats,
            }
        )

    paper_md = ["# Paper track ablation", "", "| run | preset | n | EM | mean_s |", "|---|---|---|---|---|"]
    for s in paper_rows:
        em = f"{s['em']:.3f}" if s["em"] is not None else "n/a"
        mean_s = f"{s['mean_elapsed_s']:.1f}" if s["mean_elapsed_s"] is not None else "n/a"
        paper_md.append(f"| {s['run']} | {s['preset']} | {s['n']} | {em} | {mean_s} |")

    product_md = [
        "# Product track layer report",
        "",
        "| run | preset | L2a rescue | L2b rescue | L3 enter | triggers |",
        "|---|---|---|---|---|---|",
    ]
    for s in product_rows:
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, float) else "n/a"

        product_md.append(
            f"| {s['run']} | {s['preset']} | {fmt(s.get('l2a_rescue_rate'))} | "
            f"{fmt(s.get('l2b_rescue_rate'))} | {s.get('l3_enter_count')} | "
            f"{s.get('intervention_trigger_count')} |"
        )

    (out_dir / "paper_table.md").write_text("\n".join(paper_md) + "\n", encoding="utf-8")
    (out_dir / "product_layer_report.md").write_text(
        "\n".join(product_md) + "\n", encoding="utf-8"
    )
    (out_dir / "summary.json").write_text(
        json.dumps({"paper": paper_rows, "product": product_rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out_dir / 'paper_table.md'} and {out_dir / 'product_layer_report.md'}")


if __name__ == "__main__":
    main()
