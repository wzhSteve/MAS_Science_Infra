#!/usr/bin/env python3
"""Launch one fcr_short a800_2gpu job and follow its metrics.jsonl gates."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get("PYTHON") or str(ROOT / ".venv" / "bin" / "python")


def gpu_busy() -> bool:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return False
    return bool(out.strip())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-yaml", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--wait-gpu-idle", action="store_true")
    args = parser.parse_args()
    if args.wait_gpu_idle:
        for _ in range(60):
            if not gpu_busy():
                break
            time.sleep(10)
        else:
            print("GPU still busy; refusing to overlap", flush=True)
            return 3
    elif gpu_busy():
        print("GPU busy; pass --wait-gpu-idle to queue", flush=True)
        return 3

    rl = Path(args.rl_yaml).resolve()
    log_dir = ROOT / "experiments" / "fcr_short"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{args.name}.log"
    metrics = ROOT / "mas" / "checkpoints" / "AgentLightning" / args.name / "metrics.jsonl"
    metrics.parent.mkdir(parents=True, exist_ok=True)
    if metrics.exists():
        metrics.write_text("", encoding="utf-8")
    env = os.environ.copy()
    dotenv = ROOT / ".env"
    if dotenv.is_file():
        for raw in dotenv.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES") or "0,1"
    env["PYTHONUNBUFFERED"] = "1"
    # vLLM CuMemAllocator rejects expandable_segments; do not set it here.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("VLLM_USE_V1", "1")
    env.setdefault("TIR_MAS_HEALTH", "0")
    env.setdefault("FCR_GRAD_DIAGNOSTICS", "0")
    env.setdefault("FCR_BRIER_MAX_ROWS", "0")
    env.setdefault("SCIENCE_INFRA_PROXY", env.get("WEB_SEARCH_PROXY", "http://127.0.0.1:17890"))
    env.setdefault("WEB_SEARCH_PROXY", env["SCIENCE_INFRA_PROXY"])
    env.pop("TIR_OFFLINE_SEARCH", None)
    # Keep LiteLLM group name aligned with VERL's last-two-path-components default.
    env.pop("MODEL_NAME", None)
    env.pop("OPENAI_MODEL", None)
    env.pop("MODEL", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "rl" / "verl_backend"), str(ROOT / "mas"), env.get("PYTHONPATH", "")]
    )
    with rl.open(encoding="utf-8") as fh:
        rl_raw = yaml.safe_load(fh) or {}
    for key, value in (rl_raw.get("env") or {}).items():
        env[str(key)] = str(value)
    algo = str(rl_raw.get("algo") or (rl_raw.get("algorithm") or {}).get("tir_algo") or "grpo").lower()
    cmd = [
        PYTHON,
        str(ROOT / "mas" / "train_tir_agent.py"),
        "a800_2gpu",
        "--algo",
        algo,
        "--rl-yaml",
        str(rl),
        "--workflow-yaml",
        str(ROOT / "experiments" / "fcr_short" / "workflow.yaml"),
        "--n-runners",
        str(int(rl_raw.get("n_runners") or 2)),
    ]
    print("launch", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(ROOT / "mas"), env=env, stdout=log, stderr=subprocess.STDOUT)
    watch_log = log_dir / f"{args.name}.watch.log"
    watch = subprocess.Popen(
        [PYTHON, str(ROOT / "scripts" / "watch_fcr_metrics.py"), str(metrics), "--follow", "--poll", "8"],
        cwd=str(ROOT),
        stdout=watch_log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        code = proc.wait()
    time.sleep(2)
    if watch.poll() is None:
        watch.send_signal(signal.SIGINT)
        try:
            watch.wait(timeout=10)
        except subprocess.TimeoutExpired:
            watch.kill()
            watch.wait(timeout=5)
    print(f"job {args.name} exit={code} log={log_path} metrics={metrics}", flush=True)
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
