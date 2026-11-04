#!/usr/bin/env python3
"""Follow a training metrics.jsonl and print FCR PASS/WARN/STOP gates."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

TRACKED = (
    "fcr/prediction_coverage",
    "fcr/label_coverage",
    "fcr/leakage_count",
    "fcr/cross_chain_pair_count",
    "fcr/budget_violation_count",
    "fcr/infra_drop_rate",
    "fcr/failure_confirmed_rate",
    "fcr/error_step_repair_rate",
    "fcr/correction_rate",
    "fcr/pair_valid_rate",
    "fcr/brier",
    "fcr/ece_10",
    "fcr/accuracy",
    "fcr/nonzero_token_reward_frac",
    "fcr/mean_abs_score_nonfinal",
    "fcr/n_corr_hits",
    "actor/fcr_brier_loss",
    "actor/fcr_grad_norm_ratio",
    "actor/fcr_grad_cosine",
    "training/pipeline_noop",
    "critic/score/mean",
    "val/reward",
    "cost/mean_wall_time",
)


def _num(row: Dict[str, Any], key: str) -> Optional[float]:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return float("nan")
    return number


def evaluate_step(
    row: Dict[str, Any],
    *,
    coverage_misses: int = 0,
    high_grad_streak: int = 0,
) -> Tuple[str, List[str], int, int]:
    """Return gate, reasons, and updated streaks.

    STOP: NaN, leakage, cross-chain, infra drop >0.2, budget violation, two consecutive coverage misses.
    WARN: no failure triggers, no valid pairs, Brier gradient dominates.
    """
    reasons: List[str] = []
    coverage = _num(row, "fcr/prediction_coverage")
    labels = _num(row, "fcr/label_coverage")
    leakage = _num(row, "fcr/leakage_count")
    cross = _num(row, "fcr/cross_chain_pair_count")
    budget = _num(row, "fcr/budget_violation_count")
    infra = _num(row, "fcr/infra_drop_rate")
    fail_rate = _num(row, "fcr/failure_confirmed_rate")
    pair_rate = _num(row, "fcr/pair_valid_rate")
    grad_ratio = _num(row, "actor/fcr_grad_norm_ratio")
    reward = _num(row, "critic/score/mean")

    finite_keys = TRACKED + ("step",)
    for key in finite_keys:
        value = _num(row, key) if key != "step" else _num(row, "step")
        if value is not None and not math.isfinite(value):
            return "STOP", [f"nan:{key}"], 0, 0

    if reward is not None and not math.isfinite(reward):
        return "STOP", ["reward_nan"], 0, 0
    if leakage is not None and leakage > 0:
        reasons.append(f"leakage={leakage:g}")
    if cross is not None and cross > 0:
        reasons.append(f"cross_chain={cross:g}")
    if budget is not None and budget > 0:
        reasons.append(f"budget_violation={budget:g}")
    if infra is not None and infra > 0.2:
        reasons.append(f"infra_drop_rate={infra:.3f}")
    if grad_ratio is not None and grad_ratio > 50:
        reasons.append(f"grad_explosion={grad_ratio:.3g}")
    if reasons:
        return "STOP", reasons, 0, 0

    next_coverage_misses = coverage_misses
    covered = True
    if coverage is not None and coverage < 0.99:
        covered = False
        reasons.append(f"prediction_coverage={coverage:.3f}")
    if labels is not None and labels < 0.99:
        covered = False
        reasons.append(f"label_coverage={labels:.3f}")
    next_coverage_misses = 0 if covered else coverage_misses + 1
    if next_coverage_misses >= 2:
        return "STOP", reasons + ["coverage_miss_streak=2"], next_coverage_misses, 0

    warn: List[str] = list(reasons)
    starvation = _num(row, "fcr/branch_starvation")
    if starvation is not None and starvation >= 1:
        warn.append("branch_starvation")
    if fail_rate is not None and fail_rate <= 0:
        warn.append("failure_confirmed_rate=0")
    if pair_rate is not None and pair_rate <= 0:
        warn.append("pair_valid_rate=0")
    next_high_grad = high_grad_streak
    if grad_ratio is not None and grad_ratio > 1.0:
        next_high_grad = high_grad_streak + 1
        if next_high_grad >= 2:
            warn.append(f"fcr_grad_norm_ratio={grad_ratio:.3g}")
    else:
        next_high_grad = 0
    if warn:
        return "WARN", warn, next_coverage_misses, next_high_grad
    return "PASS", [], next_coverage_misses, next_high_grad


def format_line(row: Dict[str, Any], gate: str, reasons: Iterable[str]) -> str:
    step = row.get("step", "?")
    bits = [f"step={step}", f"gate={gate}"]
    for key in (
        "critic/score/mean",
        "fcr/prediction_coverage",
        "fcr/label_coverage",
        "fcr/failure_confirmed_rate",
        "fcr/error_step_repair_rate",
        "fcr/accuracy",
        "fcr/brier",
        "fcr/nonzero_token_reward_frac",
        "fcr/n_corr_hits",
        "fcr/strict_event_count",
        "fcr/weak_event_count",
        "fcr/R_answer_mean",
        "fcr/R_correct_mean",
        "fcr/R_total_mean",
        "fcr/mean_p",
        "fcr/bss",
        "fcr/quad_unexpected",
        "fcr/budget_violation_count",
        "fcr/branch_uplift",
        "fcr/branch_starvation",
        "fcr/corr_correct_answer",
        "training/reward",
        "val/reward",
        "fcr/infra_drop_rate",
        "actor/fcr_brier_loss",
        "actor/fcr_grad_norm_ratio",
    ):
        value = _num(row, key)
        if value is not None:
            bits.append(f"{key.split('/')[-1]}={value:.4g}")
    if reasons:
        bits.append("reasons=" + ",".join(reasons))
    return " ".join(bits)


def iter_new_lines(path: Path, *, follow: bool, poll: float) -> Iterable[str]:
    offset = 0
    while True:
        if path.is_file():
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            for line in chunk.splitlines():
                if line.strip():
                    yield line
        if not follow:
            return
        time.sleep(poll)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_jsonl", type=Path)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--stop-on-stop", action="store_true")
    args = parser.parse_args()
    coverage_misses = 0
    high_grad = 0
    seen_stop = False
    try:
        for line in iter_new_lines(args.metrics_jsonl, follow=args.follow, poll=args.poll):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            gate, reasons, coverage_misses, high_grad = evaluate_step(
                row, coverage_misses=coverage_misses, high_grad_streak=high_grad
            )
            print(format_line(row, gate, reasons), flush=True)
            if gate == "STOP":
                seen_stop = True
                if args.stop_on_stop:
                    return 2
    except KeyboardInterrupt:
        return 2 if seen_stop else 0
    return 2 if seen_stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
