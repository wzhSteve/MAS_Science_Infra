#!/usr/bin/env python3
"""Smoke the Science Control UI RL/GPU APIs (uses repo .venv).

Does not wait for a full GRPO step. Optionally starts the train subprocess
and stops it immediately to verify Control wiring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


def req(
    base: str,
    method: str,
    path: str,
    body: Optional[Dict[str, Any]] = None,
    timeout: float = 30.0,
) -> tuple[int, Any]:
    url = f"{base.rstrip('/')}{path}"
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = raw
            return resp.status, parsed
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        return e.code, parsed


def wait_health(base: str, attempts: int = 40) -> None:
    last = ""
    for i in range(attempts):
        try:
            code, body = req(base, "GET", "/api/health", timeout=2.0)
            if code == 200 and isinstance(body, dict) and body.get("ok"):
                return
            last = f"{code} {body}"
        except Exception as e:
            last = str(e)
        time.sleep(0.25)
    raise SystemExit(f"UI 未就绪: {base}/api/health  ({last})")


def main() -> int:
    p = argparse.ArgumentParser(description="Check Science Control GPU/RL APIs")
    p.add_argument("--base", default="http://127.0.0.1:8787")
    p.add_argument("--experiment", default="demo")
    p.add_argument("--algo", default="grpo", help="algo for rl patch + mock collect (grpo|arpo|...)")
    p.add_argument("--no-train", action="store_true", help="skip start_train / stop_train")
    args = p.parse_args()
    base = args.base
    exp = args.experiment
    algo = str(args.algo or "grpo").lower()
    failed = 0

    def ok(name: str, cond: bool, detail: Any = "") -> None:
        nonlocal failed
        mark = "PASS" if cond else "FAIL"
        if not cond:
            failed += 1
        print(f"  [{mark}] {name}{('  ' + str(detail)) if detail not in ('', None) else ''}")

    print(f"Science Control UI check → {base}  exp={exp}  algo={algo}")
    wait_health(base)

    code, health = req(base, "GET", "/api/health")
    ok("GET /api/health", code == 200 and isinstance(health, dict) and health.get("ok"))

    code, page = req(base, "GET", "/")
    html = page if isinstance(page, str) else json.dumps(page)
    ok("GET /  (WebUI)", code == 200 and ("root" in html or "assets/" in html or "Science" in html))

    code, gpus = req(base, "GET", "/api/gpus")
    ok("GET /api/gpus", code == 200 and isinstance(gpus, dict) and "gpus" in gpus and "recommend" in gpus)
    if isinstance(gpus, dict):
        print(f"       gpus.count={gpus.get('count')} recommend.profile={((gpus.get('recommend') or {}).get('profile'))}")

    code, bundle = req(base, "GET", f"/api/experiments/{exp}")
    ok(f"GET /api/experiments/{exp}", code == 200 and isinstance(bundle, dict) and "rl" in bundle)

    rec = (gpus.get("recommend") if isinstance(gpus, dict) else None) or {}
    ids = list((rec.get("devices") or {}).get("ids") or [0])
    rl_patch = {
        "profile": rec.get("profile") or "fast",
        "algo": algo,
        "n_runners": int(rec.get("n_runners") or 1),
        "rollout_per_gpu": int(rec.get("rollout_per_gpu") or 2),
        "devices": {"ids": ids},
    }
    if isinstance(bundle, dict) and isinstance(bundle.get("rl"), dict):
        merged = dict(bundle["rl"])
        merged.update(rl_patch)
        merged["devices"] = {"ids": ids}
        merged["algo"] = algo
        algo_block = dict(merged.get("algorithm") or {})
        algo_block["tir_algo"] = algo
        algo_block["adv_estimator"] = "grpo"
        merged["algorithm"] = algo_block
        merged["trainer"] = {
            **(merged.get("trainer") or {}),
            "n_gpus_per_node": len(ids),
        }
        if merged.get("rollout_per_gpu") is not None:
            arr = dict(merged.get("actor_rollout_ref") or {})
            rollout = dict(arr.get("rollout") or {})
            rollout["n"] = int(merged["rollout_per_gpu"])
            arr["rollout"] = rollout
            merged["actor_rollout_ref"] = arr
        rl_patch = merged

    code, saved = req(base, "PUT", f"/api/experiments/{exp}/rl", {"data": rl_patch})
    ok("PUT /api/experiments/{id}/rl  (GPU + n_runners)", code == 200, f"status={code}")
    if code == 200 and isinstance(saved, dict):
        rl = saved.get("rl") or {}
        got_ids = (rl.get("devices") or {}).get("ids")
        n_gpus = (rl.get("trainer") or {}).get("n_gpus_per_node")
        ok("rl.yaml devices.ids persisted", list(got_ids or []) == [int(x) for x in ids], got_ids)
        ok("rl.yaml trainer.n_gpus_per_node", int(n_gpus or 0) == len(ids), n_gpus)
        ok("rl.yaml n_runners", int(rl.get("n_runners") or 0) >= 1, rl.get("n_runners"))
        ok("rl.yaml algo", str(rl.get("algo") or "").lower() == algo, rl.get("algo"))

    code, col = req(
        base,
        "POST",
        f"/api/mas/collect?experiment_id={exp}",
        {"mock": True, "n": 1, "algo": algo},
        timeout=60.0,
    )
    ok("POST /api/mas/collect mock", code == 200 and isinstance(col, dict) and int(col.get("n") or 0) >= 1)
    if code == 200 and isinstance(col, dict):
        sig = col.get("train_signal") or {}
        adv = (sig.get("advantage") or {}) if isinstance(sig, dict) else {}
        name = str(adv.get("name") or "")
        ok(f"collect TrainSignal.advantage.name={algo}", name.lower() == algo, name)

    if not args.no_train:
        code, train = req(
            base,
            "POST",
            f"/api/rl/train?experiment_id={exp}",
            {"stop_llm": True, "confirm_gpu": True},
            timeout=30.0,
        )
        ok("POST /api/rl/train", code == 200 and isinstance(train, dict) and train.get("run_id"), train if code != 200 else "")
        if code == 200 and isinstance(train, dict):
            argv = [str(x) for x in (train.get("argv") or [])]
            ok("train CUDA_VISIBLE_DEVICES", bool(train.get("cuda_visible_devices")), train.get("cuda_visible_devices"))
            ok("train n_gpus", int(train.get("n_gpus") or 0) == len(ids), train.get("n_gpus"))
            ok("train argv --rl-yaml", any(a.endswith("rl.yaml") or a == "--rl-yaml" for a in argv))
            ok("train argv --n-runners", "--n-runners" in argv)
            ok("train argv --algo", "--algo" in argv and algo in argv, argv)
            ok("train argv uses uv .venv", ".venv" in argv[0], argv[0] if argv else "")
            run_id = str(train.get("run_id"))
            time.sleep(1.0)
            code, run = req(base, "GET", f"/api/runs/{run_id}?tail=40")
            ok("GET /api/runs/{id}", code == 200, f"running={isinstance(run, dict) and run.get('running')}")
            code, stopped = req(base, "POST", f"/api/rl/stop?experiment_id={exp}", {})
            ok("POST /api/rl/stop", code == 200, stopped)
            time.sleep(0.5)
            code, runs = req(base, "GET", f"/api/runs?experiment_id={exp}")
            ok("GET /api/runs after stop", code == 200)

    if failed:
        print(f"\nUI RL check FAILED ({failed} checks)")
        return 1
    print("\nUI RL check OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
