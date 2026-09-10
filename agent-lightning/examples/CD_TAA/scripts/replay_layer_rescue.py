#!/usr/bin/env python3
"""Offline product-track analysis: layer rescue rates from Full-run telemetry.

Does not re-call LLMs — aggregates `layer_telemetry.jsonl` produced by
`run_ablation.py` under preset=full (or any run with emit_layer_telemetry).

For true replay of L2a/L2b operators on frozen failure contexts, extend this
script to load online causal_graph.json failure nodes and invoke mixin
helpers; that requires a live Executor/Diagnoser and is left as a follow-up.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


def load_telemetry(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter(r.get("event") for r in rows)
    l2a_enter = counts["l2a_enter"]
    l2b_enter = counts["l2b_enter"]
    return {
        "events": dict(counts),
        "l2a_rescue_rate": (counts["l2a_complete"] / l2a_enter) if l2a_enter else None,
        "l2b_rescue_rate": (counts["l2b_complete"] / l2b_enter) if l2b_enter else None,
        "l2a_skip_rate": (counts["l2a_skip"] / max(1, counts["intervention_trigger"])),
        "l3_enter_rate": (counts["l3_enter"] / max(1, counts["intervention_trigger"])),
        "notes": (
            "Rescue rates are computed on Full-system traces (product diagnostic). "
            "They are not leave-one-layer-out EM ablations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--telemetry",
        required=True,
        help="Path to layer_telemetry.jsonl from a Full (or product) run",
    )
    parser.add_argument("--out", required=True, help="Output JSON report path")
    args = parser.parse_args()

    rows = load_telemetry(Path(args.telemetry))
    report = summarize(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
