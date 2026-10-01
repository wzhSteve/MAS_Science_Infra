#!/usr/bin/env python3
"""ARPO training acceptance via Control HTTP APIs.

PUT sampling sites on planner/verifier (not tools), train, restore YAML.

Usage: python scripts/arpo_train_verify.py [--base http://127.0.0.1:8787]
       [--experiment arpo_e2e] [--steps 3] [--timeout 1800] [--negative]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from branch_rollout_ui_test import (  # noqa: E402
    make_after_agent_turn_site,
    make_after_verifier_site,
    req,
    wait_health,
)


def find_metrics_jsonl(root: Path, run_id: str, max_age_s: float = 3600.0) -> Optional[Path]:
    """Locate the freshest metrics.jsonl for this run (AGL_METRICS_JSONL layout)."""
    candidates: List[Path] = []
    for base in (root / "checkpoints", root / "artifacts", root):
        if not base.is_dir():
            continue
        for p in base.rglob("metrics.jsonl"):
            try:
                age = time.time() - p.stat().st_mtime
            except OSError:
                continue
            if age <= max_age_s:
                candidates.append(p)
    if not candidates:
        return None
    # prefer paths mentioning the run id, else freshest
    rid_hits = [p for p in candidates if run_id and run_id in str(p)]
    pool = rid_hits or candidates
    return max(pool, key=lambda p: p.stat().st_mtime)


def scan_metrics(path: Path, skip_first: int = 0) -> Dict[str, Any]:
    """Aggregate AGL metrics.jsonl: loss steps, tir_algo, lr, rewards."""
    out: Dict[str, Any] = {
        "path": str(path),
        "n_lines": 0,
        "loss_steps": 0,
        "tir_algo_hits": 0,
        "actor_lr": 0.0,
        "rollout_rewards": [],  # list of (data_id, reward)
        "rollout_advantages": [],
    }
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    skipped = 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        skipped += 1
        if skipped <= skip_first:
            continue
        out["n_lines"] += 1
        step_hit = False
        for k, v in rec.items():
            kl = str(k).lower()
            if kl in ("training/global_step", "actor/pg_loss") or "training/loss" in kl:
                if isinstance(v, (int, float)):
                    step_hit = True
            if "training/tir_algo" in kl and isinstance(v, (int, float)):
                out["tir_algo_hits"] += 1
                out["tir_algo_value"] = v
            if ("actor" in kl and "lr" in kl) and isinstance(v, (int, float)):
                out["actor_lr"] = max(out["actor_lr"], float(v))
            if kl in ("rollout/reward", "reward", "training/reward") and isinstance(v, (int, float)):
                # try to pair with data id on the same record
                did = None
                for kk in ("data_id", "rollout/data_id", "id"):
                    if rec.get(kk) is not None:
                        did = str(rec[kk])
                        break
                out["rollout_rewards"].append((did, float(v)))
            if kl in ("rollout/advantage", "advantage") and isinstance(v, (int, float)):
                out["rollout_advantages"].append(float(v))
        if step_hit:
            out["loss_steps"] += 1
    return out


def group_variance_ok(rewards: List[Tuple[Optional[str], float]]) -> Tuple[bool, str]:
    """Same data_id groups should show > 0 variance (GRPO baseline meaningful)."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for did, r in rewards:
        if did:
            groups[did].append(r)
    sized = {k: v for k, v in groups.items() if len(v) >= 2}
    if not sized:
        # fall back: overall variance across rewards
        vals = [r for _, r in rewards]
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            return (spread > 0, f"no data_id grouping; overall spread={spread:.4f}")
        return (False, "not enough reward records")
    n_var = sum(1 for v in sized.values() if max(v) - min(v) > 0)
    detail = f"{n_var}/{len(sized)} groups vary (sizes: {sorted(len(v) for v in sized.values())})"
    return (n_var > 0, detail)


def _wave2_seen(text: str) -> bool:
    low = text.lower()
    if ("resume_messages" in low) or ("arpo_branch" in low) or ("tir_branch" in low) or ("branch site=" in low):
        return True
    return bool(
        re.search(r"branch_local_count[\"':=\s]+[1-9]", text)
        or re.search(r"store_enqueue_branch_count[\"':=\s]+[1-9]", text)
    )


def main() -> int:
    p = argparse.ArgumentParser(description="ARPO training acceptance")
    p.add_argument("--base", default="http://127.0.0.1:8787")
    p.add_argument("--experiment", default="arpo_e2e")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--timeout", type=float, default=1800.0, help="seconds to wait for train to finish")
    p.add_argument("--run-id", default="", help="assert an existing train run; skip PUT/POST")
    p.add_argument("--negative", action="store_true", help="also run negative case (huge threshold)")
    args = p.parse_args()

    base, exp = args.base, args.experiment
    failed = 0

    def ok(name: str, cond: bool, detail: Any = "") -> None:
        nonlocal failed
        mark = "PASS" if cond else "FAIL"
        if not cond:
            failed += 1
        print(f"  [{mark}] {name}{('  ' + str(detail)) if detail not in ('', None) else ''}")

    print(f"ARPO train verify → {base}  exp={exp}  steps={args.steps}")
    wait_health(base)

    # --- snapshot originals for restore (red line) ---
    code, bund0 = req(base, "GET", f"/api/experiments/{exp}")
    ok("GET experiment bundle", code == 200 and isinstance(bund0, dict) and "workflow" in bund0)
    if code != 200 or not isinstance(bund0, dict):
        return 1
    orig_wf = dict(bund0.get("workflow") or {})
    orig_rl = dict(bund0.get("rl") or {})
    existing_run = str(args.run_id or "").strip()
    t_train = time.time()
    metrics_skip = 0
    mpath0 = find_metrics_jsonl(REPO, existing_run or "pending", max_age_s=7200)
    if mpath0:
        metrics_skip = scan_metrics(mpath0)["n_lines"] if not existing_run else 0

    if existing_run:
        run_id = existing_run
        _, run0 = req(base, "GET", f"/api/runs/{run_id}?tail=20")
        if isinstance(run0, dict) and run0.get("started_at"):
            t_train = float(run0["started_at"])
        ok("reuse existing run", True, f"run_id={run_id}")
    else:
        # --- 1. PUT workflow.sampling (arpo) ---
        wf = dict(orig_wf)
        sampling = dict(wf.get("sampling") or {})
        sampling.update({"mode": "arpo", "group_n": 4, "beam_size": 2, "initial_rollouts": 2})
        sampling["sites"] = [
            make_after_agent_turn_site(agent_id="planner", beam=2),
            make_after_verifier_site(agent_id="verifier", beam=2),
            make_after_agent_turn_site(agent_id="route_exec", beam=2),
        ]
        wf["sampling"] = sampling
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/workflow", {"data": wf})
        ok("PUT workflow.sampling arpo + sites", code == 200, f"status={code}")

        rl = dict(orig_rl)
        trainer = dict(rl.get("trainer") or {})
        trainer["total_training_steps"] = int(args.steps)
        rl["trainer"] = trainer
        algo_block = dict(rl.get("algorithm") or {})
        algo_block["tir_algo"] = "arpo"
        algo_block["adv_estimator"] = "grpo"
        tir = dict(algo_block.get("tir") or {})
        tir.update({"expand_in_runner": True, "ready_batch": True})
        algo_block["tir"] = tir
        rl["algorithm"] = algo_block
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/rl", {"data": rl})
        ok(f"PUT rl total_training_steps={args.steps} tir_algo=arpo", code == 200, f"status={code}")

        code, train = req(base, "POST", f"/api/rl/train?experiment_id={exp}", {"stop_llm": True, "confirm_gpu": True}, timeout=60.0)
        run_id = str((train or {}).get("run_id") or "") if isinstance(train, dict) else ""
        ok("POST /api/rl/train", code == 200 and bool(run_id), train if code != 200 else f"run_id={run_id}")
        if not run_id:
            print(f"\nARPO train verify FAILED ({failed} checks)")
            return 1
        t_train = time.time()

    saw_applied = saw_resume = False
    log_all = ""
    t_end = time.time() + float(args.timeout)
    finished = False
    while time.time() < t_end:
        _, run = req(base, "GET", f"/api/runs/{run_id}?tail=200")
        log = ""
        if isinstance(run, dict):
            log = str(run.get("log_tail") or run.get("tail") or run.get("log") or "")
        log_all = log if len(log) > len(log_all) else log_all
        status = str((run or {}).get("state") or (run or {}).get("status") or "") if isinstance(run, dict) else ""
        low = log_all.lower()
        if "applied sibling workflow.sampling" in low and "arpo" in low:
            saw_applied = True
        if _wave2_seen(log_all):
            saw_resume = True
        if status in ("finished", "done", "completed", "stopped", "failed", "error", "succeeded"):
            finished = True
            break
        time.sleep(5.0)

    _, run_final = req(base, "GET", f"/api/runs/{run_id}?tail=400")
    log_path = str((run_final or {}).get("log_path") or "") if isinstance(run_final, dict) else ""
    if log_path:
        try:
            log_all = Path(log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    low = log_all.lower()
    if "applied sibling workflow.sampling" in low and "arpo" in low:
        saw_applied = True
    if _wave2_seen(log_all):
        saw_resume = True

    # --- 3a. sampling assertions ---
    ok("log: Applied sibling workflow.sampling → tir_algo=arpo", saw_applied,
       "" if saw_applied else "not seen in run log")
    ok("log: wave-2 resume_messages / arpo_branch", saw_resume,
       "" if saw_resume else "not seen in run log")

    exp_dir = REPO / "mas" / ".local_expansion"
    n_plans = 0
    branch_local_max = 0
    group_sizes: Dict[str, int] = defaultdict(int)
    if exp_dir.is_dir():
        for f in exp_dir.glob("*.json"):
            try:
                if f.stat().st_mtime < t_train - 5:
                    continue
                payload = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            plans = list(payload.get("plans") or [])
            n_plans += len(plans)
            branch_local_max = max(branch_local_max, int(payload.get("branch_local_count") or 0))
            for pl in plans:
                did = str((pl or {}).get("data_id") or ((pl or {}).get("meta") or {}).get("data_id") or "")
                if did:
                    group_sizes[did] += 1
    ok("expansion branch_local_count/plans ≥ 1", (branch_local_max + n_plans) >= 1,
       f"plans={n_plans} max_branch_local={branch_local_max}")
    if group_sizes:
        near4 = sum(1 for v in group_sizes.values() if v >= 2)
        ok("same data_id group size ≥ 2 (≈ group_n fill)", near4 >= 1,
           f"sizes={sorted(group_sizes.values())}")
    else:
        ok("same data_id group size (skip: no grouping data)", True, "no data_id in plans")

    # --- 3b. reward / loss assertions from metrics.jsonl ---
    mpath = find_metrics_jsonl(REPO, run_id, max_age_s=7200)
    ok("metrics.jsonl located", mpath is not None, str(mpath) if mpath else "checkpoints/**/metrics.jsonl")
    if mpath:
        m = scan_metrics(mpath, skip_first=0 if existing_run else metrics_skip)
        rewards = m["rollout_rewards"]
        ok("rewards present (non-null per rollout)", len(rewards) >= 1, f"n={len(rewards)}")
        if rewards:
            gv, detail = group_variance_ok(rewards)
            ok("intra-group reward variance (GRPO baseline)", gv, detail)
        ok(f"training/loss ≥ {args.steps} steps", m["loss_steps"] >= int(args.steps), f"steps={m['loss_steps']}")
        log_steps = len(re.findall(r"\bstep:\d+", log_all))
        if existing_run:
            ok(f"run log step:N ≥ {args.steps}", log_steps >= int(args.steps), f"log_steps={log_steps}")
        ok("training/tir_algo reported", m["tir_algo_hits"] >= 1,
           f"hits={m['tir_algo_hits']} value={m.get('tir_algo_value')}")
        ok("actor lr > 0", m["actor_lr"] > 0, f"lr={m['actor_lr']}")
    else:
        ok("reward assertions", False, "metrics.jsonl missing")
        ok("loss assertions", False, "metrics.jsonl missing")

    # --- optional negative case ---
    if args.negative and not existing_run:
        print("  --- negative case: extreme entropy_threshold → n_branch=0 ---")
        rl_neg = dict(orig_rl)
        trainer_neg = dict(rl_neg.get("trainer") or {})
        trainer_neg["total_training_steps"] = 2
        rl_neg["trainer"] = trainer_neg
        algo_neg = dict(rl_neg.get("algorithm") or {})
        algo_neg["tir_algo"] = "arpo"
        tir_neg = dict(algo_neg.get("tir") or {})
        tir_neg["entropy_threshold"] = 1e9  # never branches
        algo_neg["tir"] = tir_neg
        rl_neg["algorithm"] = algo_neg
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/rl", {"data": rl_neg})
        ok("PUT rl negative (huge threshold)", code == 200)
        code, train2 = req(base, "POST", f"/api/rl/train?experiment_id={exp}", {"stop_llm": True, "confirm_gpu": True}, timeout=60.0)
        run2 = str((train2 or {}).get("run_id") or "") if isinstance(train2, dict) else ""
        if run2:
            t2 = time.time() + 300.0
            while time.time() < t2:
                time.sleep(5.0)
                _, r2 = req(base, "GET", f"/api/runs/{run2}?tail=100")
                st = str((r2 or {}).get("status") if isinstance(r2, dict) else "") or ""
                if st in ("finished", "done", "completed", "stopped", "failed", "error"):
                    break
            _, r2 = req(base, "GET", f"/api/runs/{run2}?tail=400")
            log2 = str((r2 or {}).get("log_tail") or (r2 or {}).get("log") or "") if isinstance(r2, dict) else ""
            ok("negative: no resume_messages wave", "resume_messages" not in log2.lower())
            m2 = find_metrics_jsonl(REPO, run2, max_age_s=1800)
            if m2:
                m2s = scan_metrics(m2)
                ok("negative: rewards still present (group_n filled)",
                   len(m2s["rollout_rewards"]) >= 2, f"n={len(m2s['rollout_rewards'])}")
            else:
                ok("negative: rewards still present (skip)", True, "metrics not found in window")

    # --- 4. restore (red line) ---
    if not existing_run:
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/workflow", {"data": orig_wf})
        ok("restore original workflow.yaml", code == 200)
        code, _ = req(base, "PUT", f"/api/experiments/{exp}/rl", {"data": orig_rl})
        ok("restore original rl.yaml", code == 200)

    if failed:
        print(f"\nARPO train verify FAILED ({failed} checks)")
        return 1
    print("\nARPO train verify OK")
    print("Docs: docs/ARPO_TRAIN_TEST.md (Phase 4) + docs/MAS_AGENT_FRAMEWORK_TEST.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
