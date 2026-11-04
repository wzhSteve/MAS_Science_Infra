#!/usr/bin/env python3
"""Conditional recovery rates from dumped prediction chains. No training."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


def _chains(events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        chain = str(event.get("recovery_chain_id") or f"row-{index}")
        grouped[chain].append(event)
    for chain, rows in grouped.items():
        rows.sort(key=lambda item: (int(item.get("planner_step") or 0),))
        grouped[chain] = rows
    return grouped


def audit_records(records: List[Dict[str, Any]], *, threshold: float = 0.5, window: int = 3) -> Dict[str, Any]:
    pred_fail = exec_fail = labeled = 0
    pred_rec_n = pred_rec_d = 0
    exec_rec_n = exec_rec_d = 0
    strict_n = strict_d = 0
    adjacent = windowed = lineage = false_corr = p_cross_pos = 0
    pairs = 0
    for record in records:
        events = record.get("events") or []
        if not isinstance(events, list):
            continue
        for chain in _chains(events).values():
            usable = []
            for event in chain:
                probability = _float(event.get("p_satisfy"))
                label = _bool(event.get("actual_satisfied"))
                if probability is None or label is None:
                    continue
                usable.append(event)
                labeled += 1
                if probability < threshold:
                    pred_fail += 1
                if not label:
                    exec_fail += 1
            for prev, cur in zip(usable, usable[1:]):
                pairs += 1
                p_prev = float(prev["p_satisfy"])
                p_cur = float(cur["p_satisfy"])
                y_prev = bool(prev["actual_satisfied"])
                y_cur = bool(cur["actual_satisfied"])
                if p_prev < threshold:
                    pred_rec_d += 1
                    if p_cur >= threshold:
                        pred_rec_n += 1
                if not y_prev:
                    exec_rec_d += 1
                    if y_cur:
                        exec_rec_n += 1
                if p_prev < threshold and not y_prev:
                    strict_d += 1
                    if p_cur >= threshold and y_cur:
                        strict_n += 1
                        adjacent += 1
                if p_prev < threshold <= p_cur and y_cur:
                    p_cross_pos += 1
                    if y_prev:
                        false_corr += 1
            for index, cur in enumerate(usable):
                p_cur = float(cur["p_satisfy"])
                y_cur = bool(cur["actual_satisfied"])
                if p_cur < threshold or not y_cur:
                    continue
                earlier = usable[max(0, index - window):index]
                if any(float(item["p_satisfy"]) < threshold and not bool(item["actual_satisfied"]) for item in earlier):
                    windowed += 1
                failures = [item for item in usable[:index] if not bool(item["actual_satisfied"])]
                if failures and str(cur.get("command") or "") != str(failures[-1].get("command") or ""):
                    lineage += 1
    def rate(num: int, den: int) -> Optional[float]:
        return (num / den) if den else None

    pred_rate = rate(pred_rec_n, pred_rec_d)
    exec_rate = rate(exec_rec_n, exec_rec_d)
    joint = rate(strict_n, strict_d)
    if exec_rate is not None and exec_rate < 0.02:
        source = "execution_recovery_near_zero"
    elif pred_rate is not None and pred_rate < 0.02:
        source = "prediction_recovery_near_zero"
    elif (
        pred_rate is not None and exec_rate is not None and joint is not None
        and pred_rate >= 0.02 and exec_rate >= 0.02 and joint < 0.02
    ):
        source = "joint_too_rare"
    elif strict_d == 0:
        source = "no_strict_conditioning_pairs"
    else:
        source = "events_present"
    return {
        "rollouts": len(records),
        "labeled_steps": labeled,
        "pairs": pairs,
        "prediction_failure_rate": rate(pred_fail, labeled),
        "execution_failure_rate": rate(exec_fail, labeled),
        "prediction_failure_count": pred_fail,
        "execution_failure_count": exec_fail,
        "prediction_recovery": {"num": pred_rec_n, "den": pred_rec_d, "rate": pred_rate},
        "execution_recovery": {"num": exec_rec_n, "den": exec_rec_d, "rate": exec_rate},
        "strict_adjacent": {"num": strict_n, "den": strict_d, "rate": joint, "count": adjacent},
        "window_k": {"k": window, "count": windowed},
        "lineage": {"count": lineage},
        "false_correction": false_corr,
        "p_cross_positive": p_cross_pos,
        "sparsity_source": source,
    }


def flat_rates(stats: Dict[str, Any], *, prefix: str = "fcr") -> Dict[str, float]:
    """Seven recovery rates plus numerators, safe when a denominator is zero."""
    def pack(name: str, rate: Any, num: int, den: int) -> Dict[str, float]:
        return {
            f"{prefix}/{name}": float(rate or 0.0),
            f"{prefix}/{name}_num": float(num),
            f"{prefix}/{name}_den": float(den),
        }

    pred = stats["prediction_recovery"]
    exe = stats["execution_recovery"]
    strict = stats["strict_adjacent"]
    exec_fail_n = int(stats["execution_failure_count"])
    lineage_n = int(stats["lineage"]["count"])
    false_n = int(stats["false_correction"])
    cross_n = int(stats.get("p_cross_positive") or 0)
    out: Dict[str, float] = {}
    out.update(pack(
        "prediction_failure_rate",
        stats["prediction_failure_rate"],
        int(stats["prediction_failure_count"]),
        int(stats["labeled_steps"]),
    ))
    out.update(pack(
        "execution_failure_rate",
        stats["execution_failure_rate"],
        exec_fail_n,
        int(stats["labeled_steps"]),
    ))
    out.update(pack("prediction_recovery_rate", pred["rate"], int(pred["num"]), int(pred["den"])))
    out.update(pack("execution_recovery_rate", exe["rate"], int(exe["num"]), int(exe["den"])))
    out.update(pack("strict_adjacent_rate", strict["rate"], int(strict["num"]), int(strict["den"])))
    out.update(pack(
        "lineage_recovery_rate",
        (lineage_n / exec_fail_n) if exec_fail_n else None,
        lineage_n,
        exec_fail_n,
    ))
    out.update(pack(
        "false_correction_rate",
        (false_n / cross_n) if cross_n else None,
        false_n,
        cross_n,
    ))
    out[f"{prefix}/strict_event_count"] = float(strict["count"])
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def render(name: str, stats: Dict[str, Any]) -> str:
    def fmt(rate: Any, num: Any, den: Any) -> str:
        if rate is None:
            return f"n/a ({num}/{den})"
        return f"{rate:.4f} ({num}/{den})"

    pred = stats["prediction_recovery"]
    exe = stats["execution_recovery"]
    strict = stats["strict_adjacent"]
    lines = [
        f"# Event audit: {name}",
        "",
        f"- rollouts: {stats['rollouts']}",
        f"- labeled steps: {stats['labeled_steps']}",
        f"- adjacent pairs: {stats['pairs']}",
        f"- prediction failure rate: {fmt(stats['prediction_failure_rate'], stats['prediction_failure_count'], stats['labeled_steps'])}",
        f"- execution failure rate: {fmt(stats['execution_failure_rate'], stats['execution_failure_count'], stats['labeled_steps'])}",
        f"- P(p recovers | p failed): {fmt(pred['rate'], pred['num'], pred['den'])}",
        f"- P(Y=1 | Y=0): {fmt(exe['rate'], exe['num'], exe['den'])}",
        f"- P(strict | prev pred-fail and exec-fail): {fmt(strict['rate'], strict['num'], strict['den'])}",
        f"- adjacent strict count: {strict['count']}",
        f"- window-{stats['window_k']['k']} count: {stats['window_k']['count']}",
        f"- lineage count: {stats['lineage']['count']}",
        f"- false correction count: {stats['false_correction']}",
        f"- sparsity source: `{stats['sparsity_source']}`",
        "",
    ]
    if stats["sparsity_source"] == "execution_recovery_near_zero":
        lines.append("执行恢复接近 0。后续 Branch uplift 不能解释成认知修正。")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", default="audit")
    args = parser.parse_args()
    stats = audit_records(load_jsonl(args.events))
    text = render(args.name, stats)
    args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
