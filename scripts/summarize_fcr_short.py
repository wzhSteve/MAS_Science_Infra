#!/usr/bin/env python3
"""Summarize fcr_short metrics.jsonl runs into observation notes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

KEYS = (
    "critic/score/mean",
    "fcr/prediction_coverage",
    "fcr/label_coverage",
    "fcr/leakage_count",
    "fcr/cross_chain_pair_count",
    "fcr/budget_violation_count",
    "fcr/infra_drop_rate",
    "fcr/failure_confirmed_rate",
    "fcr/error_step_repair_rate",
    "fcr/recovery_rate",
    "fcr/branch_opportunity_coverage",
    "fcr/correction_rate",
    "fcr/brier",
    "fcr/ece_10",
    "fcr/accuracy",
    "fcr/nonzero_token_reward_frac",
    "fcr/mean_abs_score_nonfinal",
    "fcr/n_corr_hits",
    "val/reward",
    "actor/fcr_brier_loss",
    "actor/fcr_grad_norm_ratio",
    "cost/mean_wall_time",
    "training/pipeline_noop",
)


def load_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _mean(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = []
    for row in rows:
        try:
            number = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    if not values:
        return None
    return sum(values) / len(values)


def _last(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    for row in reversed(rows):
        try:
            number = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def summarize(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run": name, "n_steps": len(rows)}
    for key in KEYS:
        out[key] = {
            "mean": _mean(rows, key),
            "last": _last(rows, key),
        }
    leakage = _mean(rows, "fcr/leakage_count") or 0.0
    cross = _mean(rows, "fcr/cross_chain_pair_count") or 0.0
    cov = _mean(rows, "fcr/prediction_coverage")
    lab = _mean(rows, "fcr/label_coverage")
    out["integrity_ok"] = leakage <= 0 and cross <= 0 and (cov is None or cov >= 0.99) and (
        lab is None or lab >= 0.99
    )
    return out


def fmt(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{value:.4g}"


def compare(base: Dict[str, Any], other: Dict[str, Any], key: str) -> Optional[float]:
    left = (base.get(key) or {}).get("mean")
    right = (other.get(key) or {}).get("mean")
    if left is None or right is None:
        return None
    return right - left


def write_obs1(path: Path, grpo: Dict[str, Any], fcr: Dict[str, Any]) -> None:
    reward_delta = compare(grpo, fcr, "critic/score/mean")
    repair_delta = compare(grpo, fcr, "fcr/error_step_repair_rate")
    brier_delta = compare(grpo, fcr, "fcr/brier")
    fail_rate = (fcr.get("fcr/failure_confirmed_rate") or {}).get("mean")
    infra = (fcr.get("fcr/infra_drop_rate") or {}).get("mean")
    integrity = bool(grpo.get("integrity_ok") and fcr.get("integrity_ok"))
    verdict = "证据不足"
    if not integrity:
        verdict = "证伪（完整性失败）"
    elif fail_rate is not None and fail_rate <= 0:
        verdict = "证据不足（失败确认触发率为 0）"
    elif infra is not None and infra > 0.3:
        verdict = "证据不足（infra_drop 过高）"
    else:
        reward_ok = reward_delta is not None and reward_delta >= -0.02
        repair_up = repair_delta is not None and repair_delta > 0
        brier_down = brier_delta is not None and brier_delta < -0.01
        reward_down = reward_delta is not None and reward_delta < -0.02
        repair_down = repair_delta is not None and repair_delta < 0
        brier_not_down = brier_delta is None or brier_delta >= 0
        if (reward_ok and repair_up) or (brier_down and not reward_down):
            verdict = "支持继续（pilot，单 seed）"
        elif reward_down and repair_down and brier_not_down:
            verdict = "证伪 idea 方向（pilot）"
    lines = [
        "# 观察 1：沿 idea 优化是否有效",
        "",
        "范围：8-step / seed=42 / 等 group 预算 GRPO vs FCR-full。短程 on-policy，不宣称 held-out 认知纠正。",
        "",
        f"- 判定：**{verdict}**",
        f"- GRPO steps={grpo.get('n_steps')} integrity={grpo.get('integrity_ok')}",
        f"- FCR steps={fcr.get('n_steps')} integrity={fcr.get('integrity_ok')}",
        f"- reward mean GRPO={fmt((grpo.get('critic/score/mean') or {}).get('mean'))} FCR={fmt((fcr.get('critic/score/mean') or {}).get('mean'))} Δ={fmt(reward_delta)}",
        f"- repair mean GRPO={fmt((grpo.get('fcr/error_step_repair_rate') or {}).get('mean'))} FCR={fmt((fcr.get('fcr/error_step_repair_rate') or {}).get('mean'))} Δ={fmt(repair_delta)}",
        f"- brier mean GRPO={fmt((grpo.get('fcr/brier') or {}).get('mean'))} FCR={fmt((fcr.get('fcr/brier') or {}).get('mean'))} Δ={fmt(brier_delta)}",
        f"- failure_confirmed_rate FCR={fmt(fail_rate)}",
        f"- infra_drop_rate FCR={fmt(infra)}",
        "",
        "下一步旋钮：触发率为 0 则先修标签/门控；infra 高则先修搜索；否则进入 Wave2 拆 branch/reward/loss。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_obs2(path: Path, runs: Dict[str, Dict[str, Any]]) -> None:
    base = runs.get("grpo") or {}
    lines = [
        "# 观察 2：branch / R_corr / Brier 是否有正面价值",
        "",
        "正面价值：打开该组件后对应机制指标改善，且任务 reward 不劣于关闭时。仅 correction_rate 升而任务不升记为过程奖励错位。",
        "",
    ]
    for name, label, metric in (
        ("branch_only", "branch（纯再分配）", "fcr/error_step_repair_rate"),
        ("corr_only", "R_corr（配对奖励）", "fcr/correction_rate"),
        ("brier_only", "Brier（预测监督）", "fcr/brier"),
        ("full", "完整方法", "critic/score/mean"),
    ):
        summary = runs.get(name) or {}
        delta_task = compare(base, summary, "critic/score/mean")
        delta_metric = compare(base, summary, metric)
        positive = False
        if name == "brier_only":
            positive = (delta_metric is not None and delta_metric < 0) and (
                delta_task is None or delta_task >= -0.02
            )
        else:
            positive = (delta_metric is not None and delta_metric > 0) and (
                delta_task is None or delta_task >= -0.02
            )
        if name == "corr_only" and positive and (delta_task is None or delta_task <= 0):
            result = "过程奖励错位，不算任务有效"
        elif not summary:
            result = "未跑"
        else:
            result = "有正面价值（pilot）" if positive else "未见正面价值（pilot）"
        lines.append(
            f"- **{label}**: {result}; reward Δ={fmt(delta_task)}; {metric} Δ={fmt(delta_metric)}; steps={summary.get('n_steps')}"
        )
    lines.extend(["", "短程单 seed，不能替代 5-seed 等预算终局。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("experiments/fcr_short"))
    parser.add_argument("--grpo", type=Path)
    parser.add_argument("--full", type=Path)
    parser.add_argument("--branch", type=Path)
    parser.add_argument("--corr", type=Path)
    parser.add_argument("--brier", type=Path)
    args = parser.parse_args()
    runs = {
        "grpo": summarize("grpo", load_rows(args.grpo)) if args.grpo else {},
        "full": summarize("full", load_rows(args.full)) if args.full else {},
        "branch_only": summarize("branch_only", load_rows(args.branch)) if args.branch else {},
        "corr_only": summarize("corr_only", load_rows(args.corr)) if args.corr else {},
        "brier_only": summarize("brier_only", load_rows(args.brier)) if args.brier else {},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    if args.grpo and args.full:
        write_obs1(args.out / "OBS1_idea_direction.md", runs["grpo"], runs["full"])
    if args.branch or args.corr or args.brier or args.full:
        write_obs2(args.out / "OBS2_component_value.md", runs)
    print(json.dumps({k: {"n_steps": v.get("n_steps"), "integrity_ok": v.get("integrity_ok")} for k, v in runs.items() if v}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
