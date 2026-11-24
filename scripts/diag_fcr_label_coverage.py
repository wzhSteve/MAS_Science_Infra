"""Diagnose Rc collapse: verifier label loss vs p-drift, per training-step window.

Scans rollout-tree execution JSONs (``.../runs/<run>/rollout-trees/executions/*/*.json``),
pulls every ``agentflow_state.actions`` step record and buckets records by the
``predicted_at`` epoch into training-step windows inferred from metrics land times.

For each window it reports:
- label rate: share of prediction events carrying a boolean ``actual_satisfied``
  (missing == verifier UNKNOWN/timeout, the suspected Rc substrate loss);
- p distribution: p>0.9 / 0.5<p<0.9 / p<0.5 (the ``p_cur > tau`` leg of the
  Correct pattern);
- Correct-pattern availability: adjacent (t, t+1) labeled pairs whose prev step
  mismatches and cur step succeeds, and how many are killed ONLY by missing
  labels vs only by p drift (mechanism attribution).

Usage:
  python3 scripts/diag_fcr_label_coverage.py [--run-id ID] [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import glob
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

ROOT_DEFAULT = "/root/autodl-tmp/MAS_Science_Infra"
RUN_DEFAULT = "9a45bbd456ad"
THRESHOLD = 0.5

# Step landing times (epoch seconds, local land times from finalize log):
# run start 17:37; step5 20:20; step6 00:58(+1d); step7 05:28(+1d); stop 10:46(+1d) 2026-11-08/09 +0800.
STEP_WINDOWS = [
    # (label, start_epoch, end_epoch)
    ("step5", 1794130635, 1794140400),   # 17:37 -> 20:20 local
    ("step6", 1794140400, 1794157080),   # 20:20 -> 00:58+1d
    ("step7", 1794157080, 1794173280),   # 00:58+1d -> 05:28+1d
    ("step8_partial", 1794173280, 1794192414),  # 05:28+1d -> cancelled 10:46+1d
]


def window_of(ts: Optional[float]) -> Optional[str]:
    if ts is None:
        return None
    for name, lo, hi in STEP_WINDOWS:
        if lo <= ts < hi:
            return name
    return None


def as_bool(value: Any) -> bool:
    return isinstance(value, bool)


def step_records(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    state = (doc.get("output") or {}).get("agentflow_state") or {}
    actions = state.get("actions") or {}
    rows: List[Dict[str, Any]] = []
    if isinstance(actions, dict):
        for key, act in actions.items():
            if isinstance(act, dict) and act.get("prediction_event_id"):
                rows.append(act)
    rows.sort(key=lambda a: (a.get("planner_step") if isinstance(a.get("planner_step"), int) else 0, str(a.get("prediction_event_id"))))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=RUN_DEFAULT)
    ap.add_argument("--root", default=ROOT_DEFAULT)
    args = ap.parse_args()

    base = os.path.join(args.root, "experiments", "fcr-branch-corr", "artifacts", "runs", args.run_id, "rollout-trees")
    files = glob.glob(os.path.join(base, "executions", "*", "*.json"))
    print(f"executions: {len(files)} files under {base}")

    stats: Dict[str, Dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("status") not in (None, "succeeded", "failed", "running", "cancelled"):
            pass
        rows = step_records(doc)
        for row in rows:
            ts = row.get("predicted_at")
            win = window_of(float(ts)) if isinstance(ts, (int, float)) else None
            if win is None:
                continue
            s = stats[win]
            p = row.get("p_satisfy")
            label = row.get("actual_satisfied")
            s["events"] += 1
            has_p = isinstance(p, (int, float)) and not isinstance(p, bool)
            has_y = as_bool(label)
            if has_p:
                s["has_p"] += 1
                pv = float(p)
                if pv > 0.9:
                    s["p_gt_09"] += 1
                elif pv >= 0.5:
                    s["p_05_09"] += 1
                else:
                    s["p_lt_05"] += 1
            if has_y:
                s["has_y"] += 1
                if bool(label):
                    s["y_pos"] += 1
                    if has_p and float(p) > 0.5:
                        s["y_pos_p_gt_05"] += 1
            if has_p and not has_y:
                s["p_no_y"] += 1
            # verifier parse state hints for missing labels
            if not has_y:
                concl = str(row.get("conclusion") or "")
                if concl.upper() not in ("STOP", "CONTINUE"):
                    s["no_y_concl_odd"] += 1
                if not str(row.get("verifier_reason") or "").strip():
                    s["no_y_reason_blank"] += 1

    # adjacent-pair Correct availability
    pair: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        rows = step_records(doc)
        for prev, cur in zip(rows, rows[1:]):
            ts = cur.get("predicted_at")
            win = window_of(float(ts)) if isinstance(ts, (int, float)) else None
            if win is None:
                continue
            pm, py = prev.get("p_satisfy"), prev.get("actual_satisfied")
            cm, cy = cur.get("p_satisfy"), cur.get("actual_satisfied")
            p_lab = as_bool(py)
            c_lab = as_bool(cy)
            p_float = isinstance(pm, (int, float)) and not isinstance(pm, bool)
            c_float = isinstance(cm, (int, float)) and not isinstance(cm, bool)
            q = pair[win]
            q["pairs"] += 1
            if not (p_float and c_float):
                q["pair_no_p"] += 1
                continue
            if not (p_lab and c_lab):
                q["pair_no_y"] += 1
                # Counterfactual upper bound: assume the MISSING side's label
                # existed and was consistent with the observable signal.
                poss_mismatch = True if not p_lab else ((float(pm) >= THRESHOLD) != bool(py))
                if c_lab:
                    poss_hit = float(cm) > THRESHOLD and bool(cy)
                else:
                    poss_hit = float(cm) > THRESHOLD
                if poss_mismatch and poss_hit:
                    q["counterfactual_hits"] += 1
                continue
            mismatch = (float(pm) >= THRESHOLD) != bool(py)
            if not mismatch:
                q["pair_no_mismatch"] += 1
                continue
            if float(cm) > THRESHOLD and bool(cy):
                q["pair_hits"] += 1
            elif float(cm) <= THRESHOLD and bool(cy):
                q["pair_miss_pdrift"] += 1  # success but p<=tau: p-drift kills Rc
            else:
                q["pair_miss_notrecovered"] += 1

    def pct(a: float, b: float) -> str:
        return f"{100.0 * a / max(1.0, b):5.1f}%"

    print(f"\n{'window':14s} {'events':>7s} {'has_y':>7s} {'y_rate':>7s} {'p>0.9':>7s} {'p.5-.9':>7s} {'p<0.5':>7s} {'p_no_y':>7s} {'y_pos_rate':>10s}")
    for name, _lo, _hi in STEP_WINDOWS:
        s = stats.get(name)
        if not s:
            continue
        ev = s["events"]
        print(
            f"{name:14s} {ev:7.0f} {s['has_y']:7.0f} {pct(s['has_y'], ev):>7s} "
            f"{pct(s['p_gt_09'], s['has_p']):>7s} {pct(s['p_05_09'], s['has_p']):>7s} "
            f"{pct(s['p_lt_05'], s['has_p']):>7s} {s['p_no_y']:7.0f} "
            f"{pct(s['y_pos'], s['has_y']):>10s}"
        )
    print("\n(blank verifier_reason among unlabeled: " +
          ", ".join(f"{k}={stats[k]['no_y_reason_blank']:.0f}/{stats[k]['p_no_y']:.0f}" for k, _l, _h in STEP_WINDOWS if stats.get(k)) + ")")

    print(f"\n{'window':14s} {'pairs':>8s} {'hits':>6s} {'miss_pdrift':>11s} {'miss_norec':>10s} {'no_mismatch':>11s} {'pair_no_y':>9s} {'cf_hits_if_labeled':>16s}")
    for name, _lo, _hi in STEP_WINDOWS:
        q = pair.get(name)
        if not q:
            continue
        print(
            f"{name:14s} {q['pairs']:8.0f} {q['pair_hits']:6.0f} {q['pair_miss_pdrift']:11.0f} "
            f"{q['pair_miss_notrecovered']:10.0f} {q['pair_no_mismatch']:11.0f} {q['pair_no_y']:9.0f} "
            f"{q.get('counterfactual_hits', 0):16.0f}"
        )
    print(
        "\nhits = actual Correct-pattern events; miss_pdrift = cur step succeeded but p<=tau"
        " (policy drift); pair_no_y = candidate pair dropped for missing verifier label;"
        " cf_hits_if_labeled = pairs that WOULD hit if the missing label had existed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
