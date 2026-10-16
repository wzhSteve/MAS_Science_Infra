#!/usr/bin/env python3
"""Read 4 metrics.jsonl (grpo/arpo/appo/flow_grpo) and emit a 10-step comparison.

Usage:
    python scripts/agentflow_4algo_summarize.py --out artifacts/agentflow_4algo/summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOTS = [
    ROOT / "checkpoints" / "AgentLightning",
    ROOT / "mas" / "checkpoints" / "AgentLightning",
]
ALGOS = ["grpo", "arpo", "appo", "flow_grpo"]
FIELDS = [
    "training/reward",
    "val/reward",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/entropy_loss",
    "training/branch_local_count",
    "training/n_rollouts",
    "training/n_triplets",
    "training/n_with_anchor",
]


def _metrics_path(algo: str) -> Path | None:
    candidates = [
        f"agentflow_4algo_{algo}_10step_{algo}/metrics.jsonl",
        f"agentflow_4algo_{algo}_10step/metrics.jsonl",
    ]
    for root in CKPT_ROOTS:
        for rel in candidates:
            path = root / rel
            if path.is_file():
                return path
    return None


def load(algo: str) -> list[dict]:
    path = _metrics_path(algo)
    if path is None:
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "step" not in row:
            row["step"] = row.get("training/global_step")
        out.append(row)
    return out


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "agentflow_4algo" / "summary.md")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = {a: load(a) for a in ALGOS}
    n_steps = max((len(r) for r in rows.values()), default=0)

    lines: list[str] = []
    lines.append("# AgentFlow 4 算法 10 步训练对比")
    lines.append("")
    lines.append(f"训练对象：Qwen3-4B（{ROOT}/LLM/Qwen3-4B）")
    lines.append(f"Workflow: experiments/agentflow_4algo/workflow.yaml（alpha 模板，3 节点 planner / executor / verifier，5 个 AgentFlow 工具）")
    lines.append(f"GPU: 2×A800-80GB（`CUDA_VISIBLE_DEVICES=0,1`）")
    lines.append(f"n_runners: 2")
    lines.append(f"Data: data/{{train,val}}.parquet（小样本 GSM8K+HotpotQA，train_batch_size=2，2 卡可整除）")
    lines.append(f"每个 algo 各 10 步，串行执行")
    lines.append("")

    lines.append("## 1. 10 步全部指标")
    lines.append("")
    header = ["algo", "step"] + FIELDS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for algo in ALGOS:
        traj = rows[algo]
        if not traj:
            lines.append(f"| {algo} | — | " + " | ".join(["(empty)"] * len(FIELDS)) + " |")
            continue
        for row in traj:
            step = row.get("step", "?")
            cells = [fmt(row.get(f)) for f in FIELDS]
            lines.append("| " + " | ".join([algo, str(step)] + cells) + " |")
    lines.append("")

    lines.append("## 2. 末步快照")
    lines.append("")
    snap_fields = [
        "training/reward",
        "val/reward",
        "actor/pg_loss",
        "actor/ppo_kl",
        "actor/entropy_loss",
        "training/n_triplets",
        "training/branch_local_count",
        "training/n_with_anchor",
    ]
    lines.append("| algo | steps | " + " | ".join(snap_fields) + " |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(snap_fields)) + " |")
    for algo in ALGOS:
        traj = [r for r in rows[algo] if (r.get("step") or 0) > 0] or rows[algo]
        if not traj:
            lines.append(f"| {algo} | 0 | " + " | ".join(["—"] * len(snap_fields)) + " |")
            continue
        last = traj[-1]
        cells = [fmt(last.get(f)) for f in snap_fields]
        lines.append("| " + " | ".join([algo, str(len(traj))] + cells) + " |")
    lines.append("")

    lines.append("## 3. 观察与判读")
    lines.append("")
    for algo in ALGOS:
        traj = rows[algo]
        train_traj = [r for r in traj if (r.get("step") or 0) > 0]
        if not train_traj:
            lines.append(f"- **{algo}**：`metrics.jsonl` 缺失或无训练 step。")
            continue
        first, last = train_traj[0], train_traj[-1]
        path = _metrics_path(algo)
        branch_max = max((r.get("training/branch_local_count") or 0 for r in train_traj), default=0)
        anchor_max = max((r.get("training/n_with_anchor") or 0 for r in train_traj), default=0)
        trip_mean = sum((r.get("training/n_triplets") or 0) for r in train_traj) / len(train_traj)
        lines.append(
            f"- **{algo}**（{len(train_traj)} 步，`{path}`）："
            f"training/reward {first.get('training/reward')} → {last.get('training/reward')}，"
            f"val/reward {last.get('val/reward')}，"
            f"actor/pg_loss {first.get('actor/pg_loss')} → {last.get('actor/pg_loss')}，"
            f"actor/ppo_kl {first.get('actor/ppo_kl')} → {last.get('actor/ppo_kl')}，"
            f"n_triplets mean={trip_mean:.1f}，"
            f"branch_local_count max={branch_max}，n_with_anchor max={anchor_max}。"
        )
    lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
