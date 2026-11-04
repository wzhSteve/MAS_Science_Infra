#!/usr/bin/env python3
"""Per-step recovery rates for the two correct-reward arms."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
CKPT = ROOT / "mas" / "checkpoints" / "AgentLightning"
OUT = ROOT / "experiments" / "fcr_short" / "CORRECT_5STEP.md"
ARMS = ("fcr_grpo_corr", "fcr_branch_corr")
RATES = (
    "prediction_failure_rate",
    "execution_failure_rate",
    "prediction_recovery_rate",
    "execution_recovery_rate",
    "strict_adjacent_rate",
    "lineage_recovery_rate",
    "false_correction_rate",
)


def _rows(name: str):
    path = CKPT / name / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _fmt(row: dict, prefix: str, key: str) -> str:
    rate = row.get(f"{prefix}/{key}")
    num = row.get(f"{prefix}/{key}_num")
    den = row.get(f"{prefix}/{key}_den")
    if rate is None:
        return "n/a"
    extra = ""
    if num is not None and den is not None:
        extra = f" ({int(num)}/{int(den)})"
    return f"{float(rate):.4f}{extra}"


def main() -> int:
    lines = [
        "# GRPO+Correct vs Branch+Correct (5 steps)",
        "",
        "## Setup",
        "",
        "- `rollout.n=8` / `group_n=8`（每样本 8 条轨迹，一步 `train_batch_size=2` → 16 条）",
        "- 训练 `data/rl_datasets/train_10k.parquet`；每步验证完整 `data/rl_datasets/valid.parquet`（180 题）",
        "- **Correct pattern（奖励）**：相邻步 prev `(p≥τ)≠Y`（预测 vs 执行失配），cur `p>τ` 且 `Y=1`；"
        "同 query 内 `R_correct=N_i/N_max`，再加终局 answer 0/1",
        "- **Branch 门控**（仅 `fcr_branch_corr`）：`failure_confirmed`（`p<0.5` 且 `Y=0`，非 tool_fault）",
        "- 下表七个恢复率为 **事件链诊断**（`audit_fcr_events`）。其中 `strict_adjacent` 仍是旧基线"
        "（`p`低∧`Y`失败 → 双成功），**不等于** Correct 奖励命中；奖励命中看 `R_correct_mean` / "
        "`strict_event_count`（现与新 pattern 对齐）",
        "",
    ]
    for name in ARMS:
        rows = _rows(name)
        lines.append(f"## {name}")
        if not rows:
            lines.append("- missing")
            lines.append("")
            continue
        for row in rows:
            step = row.get("training/global_step", "?")
            reward = row.get("training/reward")
            val_reward = row.get("val/reward")
            pattern_hits = row.get("fcr/strict_event_count")
            lines.append(f"### step {step}")
            lines.append(f"- train reward: {reward}")
            lines.append(f"- val reward: {val_reward}")
            lines.append(f"- train Correct pattern hits (new design): {pattern_hits}")
            if pattern_hits == 0 or pattern_hits == 0.0:
                lines.append("- 该步训练集 Correct pattern 命中为 0，Correct 奖励没有进入更新。")
            lines.append("")
            lines.append("| rate | train | val |")
            lines.append("|---|---|---|")
            for key in RATES:
                lines.append(f"| {key} | {_fmt(row, 'fcr', key)} | {_fmt(row, 'val/fcr', key)} |")
            lines.append("")
            lines.append(
                f"- R_correct_mean: {row.get('fcr/R_correct_mean')} | "
                f"frac_Nmax_zero: {row.get('fcr/frac_Nmax_zero')} | "
                f"budget_violation: {row.get('fcr/budget_violation_count')}"
            )
            lines.append("")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
