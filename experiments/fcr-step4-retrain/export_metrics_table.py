#!/usr/bin/env python3
"""Export fcr_step4_retrain step metrics as Markdown (train + val + val-only override)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parents[2]
TRAIN_METRICS = REPO / "mas/checkpoints/AgentLightning/fcr_step4_retrain/metrics.jsonl"
VALONLY_METRICS = REPO / "mas/checkpoints/AgentLightning/fcr_step4_valonly/metrics.jsonl"


def _last_by_step(path: Path) -> Dict[int, Dict[str, Any]]:
    by: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return by
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        step = row.get("step", row.get("global_step"))
        if step is not None:
            by[int(step)] = row
    return by


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # nan
        return "—"
    return f"{v:.{nd}f}"


def build_table(
    train: Dict[int, Dict[str, Any]],
    valonly: Dict[int, Dict[str, Any]],
    *,
    max_step: int = 17,
) -> str:
    lines = [
        "| step | 训练 Ra | 训练 Rc | 验证 Ra | 验证 Rc | ppo_kl | mean_n_search | 备注 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    notes = {
        1: "metrics 已丢",
        2: "metrics 已丢",
        3: "metrics 已丢",
        4: "metrics 已丢",
        5: "权重在；验证=val-only",
        8: "工具塌缩",
    }
    for step in range(1, max_step + 1):
        tr = train.get(step, {})
        vo = valonly.get(step, {})
        tr_ra = tr.get("fcr/R_answer_mean")
        tr_rc = tr.get("fcr/R_correct_mean")
        if vo:
            val_ra = vo.get("val/fcr/R_answer_mean", vo.get("val/reward"))
            val_rc = vo.get("val/fcr/R_correct_mean")
            note = notes.get(step, "验证=val-only 重跑") if step in vo else notes.get(step, "")
        else:
            val_ra = tr.get("val/fcr/R_answer_mean")
            val_rc = tr.get("val/fcr/R_correct_mean")
            note = notes.get(step, "")
            if step in (7, 9, 11, 13) and val_ra is None and tr_ra is not None:
                note = (note + "; " if note else "") + "当时未跑 valid"
        kl = tr.get("actor/ppo_kl")
        if kl is None:
            kl = tr.get("actor/kl_loss")
        search = tr.get("cost/mean_n_search")
        if not tr and step <= 4:
            lines.append(f"| {step} | — | — | — | — | — | — | metrics 已丢，不估算 |")
            continue
        if not tr and step > 14:
            lines.append(f"| {step} | — | — | — | — | — | — | 未写入 metrics |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    str(step),
                    _fmt(tr_ra),
                    _fmt(tr_rc),
                    _fmt(val_ra),
                    _fmt(val_rc),
                    _fmt(kl),
                    _fmt(search, 2),
                    note or "—",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-metrics",
        type=Path,
        default=TRAIN_METRICS,
    )
    parser.add_argument(
        "--valonly-metrics",
        type=Path,
        default=VALONLY_METRICS,
    )
    parser.add_argument("--max-step", type=int, default=17)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write Markdown to file; default stdout",
    )
    args = parser.parse_args()
    train = _last_by_step(args.train_metrics)
    valonly = _last_by_step(args.valonly_metrics)
  # step 5 only in valonly
    md = build_table(train, valonly, max_step=args.max_step)
    if args.output:
        args.output.write_text(md, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(md, end="")


if __name__ == "__main__":
    main()