#!/usr/bin/env python3
"""Run the FCR probe matrix one job at a time and write the P4 decision."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON") or str(ROOT / ".venv" / "bin" / "python")
if not Path(PYTHON).is_file():
    PYTHON = sys.executable
SPEC = ROOT / "experiments" / "fcr_short"
CKPT = ROOT / "mas" / "checkpoints" / "AgentLightning"


def _rows(name: str) -> list:
    path = CKPT / name / "metrics.jsonl"
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _last(rows: list, key: str) -> float:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _mean(rows: list, key: str) -> float:
    vals = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else 0.0


def _run(name: str) -> int:
    yaml_path = SPEC / f"{name}.yaml"
    print(f"=== start {name} ===", flush=True)
    proc = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / "run_fcr_short_job.py"), "--rl-yaml", str(yaml_path), "--name", name, "--wait-gpu-idle"],
        cwd=str(ROOT),
    )
    print(f"=== end {name} exit={proc.returncode} ===", flush=True)
    for path in (CKPT / name).glob("global_step_*"):
        if path.is_dir():
            subprocess.check_call(["rm", "-rf", str(path)])
            print(f"removed weights {path}", flush=True)
    return int(proc.returncode)


def _summary(name: str) -> dict:
    rows = _rows(name)
    return {
        "name": name,
        "steps": len(rows),
        "exit_metrics": bool(rows),
        "reward": _mean(rows, "training/reward"),
        "adv_std": _last(rows, "fcr/adv_within_response_std"),
        "adv_max_std": _last(rows, "fcr/adv_within_response_max_std"),
        "false_correction_rate": _last(rows, "fcr/false_correction_rate"),
        "budget": _last(rows, "fcr/budget_violation_count"),
        "coverage": _last(rows, "fcr/prediction_coverage"),
        "label_coverage": _last(rows, "fcr/label_coverage"),
        "leakage": _last(rows, "fcr/leakage_count"),
        "cross": _last(rows, "fcr/cross_chain_pair_count"),
        "infra": _last(rows, "fcr/infra_drop_rate"),
        "uplift": _mean(rows, "fcr/branch_uplift"),
        "recovery": _mean(rows, "fcr/recovery_success_rate"),
        "brier": _mean(rows, "fcr/brier"),
        "ece": _mean(rows, "fcr/ece_10"),
        "quad_pred_fail": _last(rows, "fcr/quad_pred_fail"),
        "quad_unexpected": _last(rows, "fcr/quad_unexpected"),
        "quad_false_alarm": _last(rows, "fcr/quad_false_alarm"),
        "quad_pred_success": _last(rows, "fcr/quad_pred_success"),
        "gate_precision": _last(rows, "fcr/gate_precision"),
        "gate_recall": _last(rows, "fcr/gate_recall"),
        "tool_calls": _mean(rows, "cost/mean_n_search") + _mean(rows, "cost/mean_n_fetch") + _mean(rows, "cost/mean_n_python"),
    }


def _intact(item: dict) -> bool:
    return (
        item["coverage"] >= 0.99
        and item["label_coverage"] >= 0.99
        and item["leakage"] == 0
        and item["cross"] == 0
        and item["budget"] == 0
        and item["infra"] <= 0.2
        and item["steps"] > 0
    )


def _decide_p1(p1: list) -> tuple:
    base = next(row for row in p1 if row["name"] == "fcr_p1_none")
    branched = [row for row in p1 if row["name"] != "fcr_p1_none"]
    cap = base["tool_calls"] * 1.15 if base["tool_calls"] else float("inf")
    winner = None
    best = None
    for row in branched:
        score = row["reward"] + row["uplift"]
        if row["tool_calls"] <= cap and (best is None or score > best):
            best = score
            winner = row["name"]
    pred = next(row for row in p1 if row["name"] == "fcr_p1_pred_fail")
    if winner == "fcr_p1_pred_fail":
        winner_gate = "failure_confirmed"
    else:
        others = [row for row in branched if row["name"] != "fcr_p1_pred_fail"]
        if not any((pred["reward"] + pred["uplift"]) > (row["reward"] + row["uplift"]) for row in others):
            winner_gate = "failure_confirmed"
        else:
            winner_gate = {
                "fcr_p1_all_fail": "all_fail",
                "fcr_p1_unexpected": "unexpected_fail",
                "fcr_p1_pred_fail": "failure_confirmed",
            }.get(winner or "", "failure_confirmed")
    return winner, winner_gate, pred


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", default="p0", choices=["p0", "p2"])
    args = parser.parse_args()
    report = []
    if args.from_stage == "p2":
        p1_names = ["fcr_p1_none", "fcr_p1_all_fail", "fcr_p1_pred_fail", "fcr_p1_unexpected"]
        p1 = [_summary(name) for name in p1_names]
        for row in p1:
            row["exit"] = 0
        (SPEC / "P1_gates.json").write_text(json.dumps(p1, indent=2), encoding="utf-8")
        winner, winner_gate, pred = _decide_p1(p1)
        (SPEC / "P1_winner.txt").write_text(
            f"winner={winner}\ngate={winner_gate}\nreward_key=training/reward\n",
            encoding="utf-8",
        )
        item = _summary("fcr_p0_diag")
        strict = pred["false_correction_rate"] > 0.3
        report.extend(p1)
    else:
        subprocess.check_call([PYTHON, str(ROOT / "scripts" / "build_fcr_probe.py")], cwd=str(ROOT))
        p0 = "fcr_p0_diag"
        code = _run(p0)
        item = _summary(p0)
        item["exit"] = code
        report.append(item)
        (SPEC / "P0_diag.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        if item["steps"] < 1:
            (SPEC / "P4_decision.md").write_text(
                "P0 没有写出 metrics.jsonl，后续臂不启动。\n", encoding="utf-8",
            )
            return code or 1

        p1_names = ["fcr_p1_none", "fcr_p1_all_fail", "fcr_p1_pred_fail", "fcr_p1_unexpected"]
        p1 = []
        for name in p1_names:
            code = _run(name)
            row = _summary(name)
            row["exit"] = code
            p1.append(row)
            report.append(row)
        (SPEC / "P1_gates.json").write_text(json.dumps(p1, indent=2), encoding="utf-8")
        winner, winner_gate, pred = _decide_p1(p1)
        (SPEC / "P1_winner.txt").write_text(
            f"winner={winner}\ngate={winner_gate}\nreward_key=training/reward\n",
            encoding="utf-8",
        )
        if pred["budget"] > 0:
            _run("fcr_p2_hardcap")
            report.append({**_summary("fcr_p2_hardcap"), "exit": 0})
        strict = pred["false_correction_rate"] > 0.3

    p2_names = [
        "fcr_p2_grpo", "fcr_p2_B", "fcr_p2_R", "fcr_p2_P",
        "fcr_p2_BR", "fcr_p2_BP", "fcr_p2_full", "fcr_p2_bp_adapt",
    ]
    if strict:
        p2_names.append("fcr_p2_full_strict")
    if winner_gate != "failure_confirmed":
        import yaml
        for name in p2_names:
            path = SPEC / f"{name}.yaml"
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            sites = ((doc.get("algorithm") or {}).get("tir") or {}).get("sites") or []
            for site in sites:
                if isinstance(site, dict) and isinstance(site.get("gate"), dict):
                    site["gate"]["type"] = winner_gate
            path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    p2 = []
    for name in p2_names:
        code = _run(name)
        row = _summary(name)
        row["exit"] = code
        p2.append(row)
        report.append(row)
    (SPEC / "P2_ablation.json").write_text(json.dumps(p2, indent=2), encoding="utf-8")

    snap = SPEC / "snapshots" / "fcr_p1_pred_fail"
    subprocess.call([
        PYTHON, str(ROOT / "scripts" / "fcr_snapshot_pair.py"),
        "--snapshot-dir", str(snap),
        "--out", str(SPEC / "P3_pairs.json"),
    ])

    candidates = [row for row in p2 if row["name"] != "fcr_p2_grpo" and _intact(row)]
    grpo = next((row for row in p2 if row["name"] == "fcr_p2_grpo"), None)
    chosen = None
    if grpo and candidates:
        for row in candidates:
            if row["reward"] >= grpo["reward"]:
                if chosen is None or row["reward"] > chosen["reward"]:
                    chosen = row
    lines = [
        "# P4 decision",
        "",
        f"P0 adv_std={item.get('adv_std')} max_std={item.get('adv_max_std')} false_correction={item.get('false_correction_rate')} budget={item.get('budget')}",
        f"P1 winner={winner} gate={winner_gate}",
        "",
    ]
    if chosen is None:
        lines.append("没有臂同时通过完整性门槛且终局 reward 不低于 GRPO。不上 10k。")
        (SPEC / "P4_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        lines.append(f"赢家 {chosen['name']} reward={chosen['reward']}。启动 10k 子集验证。")
        (SPEC / "P4_decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        import yaml
        src = yaml.safe_load((SPEC / f"{chosen['name']}.yaml").read_text(encoding="utf-8"))
        src["data"]["train_files"] = str(ROOT / "data" / "rl_datasets" / "train_10k.parquet")
        src["data"]["val_files"] = str(ROOT / "data" / "rl_datasets" / "fcr_probe_val64.parquet")
        src["trainer"]["experiment_name"] = "fcr_p4_scale"
        src["trainer"]["default_local_dir"] = str(CKPT / "fcr_p4_scale")
        src["trainer"]["test_freq"] = 5
        src["trainer"]["total_training_steps"] = 8
        src["trainer"]["save_freq"] = 1000
        path = SPEC / "fcr_p4_scale.yaml"
        path.write_text(yaml.safe_dump(src, sort_keys=False, allow_unicode=True), encoding="utf-8")
        grpo_src = yaml.safe_load((SPEC / "fcr_p2_grpo.yaml").read_text(encoding="utf-8"))
        grpo_src["data"]["train_files"] = str(ROOT / "data" / "rl_datasets" / "train_10k.parquet")
        grpo_src["data"]["val_files"] = str(ROOT / "data" / "rl_datasets" / "fcr_probe_val64.parquet")
        grpo_src["trainer"]["experiment_name"] = "fcr_p4_grpo"
        grpo_src["trainer"]["default_local_dir"] = str(CKPT / "fcr_p4_grpo")
        grpo_src["trainer"]["test_freq"] = 5
        grpo_src["trainer"]["total_training_steps"] = 8
        grpo_src["trainer"]["save_freq"] = 1000
        (SPEC / "fcr_p4_grpo.yaml").write_text(
            yaml.safe_dump(grpo_src, sort_keys=False, allow_unicode=True), encoding="utf-8",
        )
        _run("fcr_p4_grpo")
        _run("fcr_p4_scale")
        report.append(_summary("fcr_p4_grpo"))
        report.append(_summary("fcr_p4_scale"))
    (SPEC / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
