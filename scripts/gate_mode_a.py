#!/usr/bin/env python3
"""Mode A live gate: run a user_space window graph (no training) on real questions.

    python scripts/gate_mode_a.py --workflow experiments/agentflow_flow_grpo_4b/workflow.yaml \
        --base-url http://127.0.0.1:8001/v1 --model Qwen3-4B --n 3 --out artifacts/gate_4b/af_mode_a.json

Writes per-question answer, rewards (hierarchical and 0/1), tool calls and BTS
evidence; exits non-zero when any episode raises or no episode yields an answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "mas")):
    if p not in sys.path:
        sys.path.insert(0, p)

GSM_TASK = {
    "id": "gate-gsm8k",
    "question": "Jason bought 4 dozen cupcakes. He plans to give 3 cupcakes each to his cousins. "
    "How many cousins does Jason have?",
    "answer": "16",
    "answers": ["16"],
    "source": "gsm8k",
}


def _tasks(n: int, split: str):
    import pandas as pd

    rows = pd.read_parquet(ROOT / "data" / f"{split}.parquet").head(max(0, n - 1)).to_dict("records")
    for r in rows:
        r["answers"] = [str(a) for a in (r.get("answers") if r.get("answers") is not None else [])]
    return [GSM_TASK] + rows


def _evidence(messages):
    tools, bts, kinds = [], [], []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        kinds.append(str(m.get("kind") or ""))
        payload = m.get("payload") or {}
        trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
        if isinstance(trace.get("bts"), dict):
            bts.append(trace["bts"])
        if m.get("kind") == "tool_result" or "output" in payload:
            tools.append({
                "tool": payload.get("tool") or payload.get("tool_name") or "",
                "ok": payload.get("ok"),
                "snippet": str(payload.get("output") or payload.get("content") or "")[:160],
            })
    return kinds, tools, bts


def run_one(task, cfg, spec):
    from mas_agent import bind_training_llm_env
    from rl.rewards import binary_correct, parse_alias_field
    from workflow.archive import Archive, register_archive
    from workflow.collector import default_reward_fn
    from workflow.memory import MemoryStore
    from workflow.runtime import episode_to_trajectory, run_compiled_episode

    arch = register_archive(Archive())
    mem = MemoryStore()
    t0 = time.time()
    with bind_training_llm_env(endpoint=cfg.endpoint, model=cfg.model, api_key="EMPTY"):
        raw = run_compiled_episode(task, cfg, arch, spec, mem, None)
    traj = episode_to_trajectory(task, raw, arch, collector="gate_mode_a", spec=spec, memory=mem)
    kinds, tools, bts = _evidence(raw.messages)
    answer = str(raw.final_answer or "")
    aliases = parse_alias_field(task.get("answers"))
    return {
        "id": task.get("id"),
        "source": task.get("source"),
        "gold": task.get("answer"),
        "final_answer": answer[:400],
        "reward_hierarchical": float(default_reward_fn(traj, task)),
        "reward_binary": binary_correct(answer, str(task.get("answer") or ""), source=str(task.get("source")),
                                        aliases=aliases),
        "n_messages": len(raw.messages or []),
        "kinds": kinds,
        "tools": tools,
        "bts": bts,
        "error": getattr(raw, "error", None),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workflow", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--n", type=int, default=3, help="questions: 1 gsm8k + n-1 from data/<split>")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--require-bts", type=int, default=0, help="fail unless some BTS trace has n >= this")
    args = ap.parse_args()

    from workflow.env_load import load_repo_dotenv
    from workflow.runtime import LLMConfig
    from workflow.spec import load_spec

    load_repo_dotenv()
    spec = load_spec(str(Path(args.workflow).resolve()))
    cfg = LLMConfig(endpoint=args.base_url, model=args.model, temperature=0.7, max_turns=8,
                    max_tokens=args.max_tokens, max_model_len=16384, enabled_tools=list(spec.tools))
    results = []
    for task in _tasks(args.n, args.split):
        print(f"=== {task['id']} ===", flush=True)
        try:
            r = run_one(task, cfg, spec)
        except Exception as exc:  # noqa: BLE001
            r = {"id": task["id"], "exception": f"{type(exc).__name__}: {exc}"}
        print(json.dumps({k: v for k, v in r.items() if k not in ("tools", "kinds")}, ensure_ascii=False), flush=True)
        results.append(r)

    max_bts = max([int(b.get("n") or 0) for r in results for b in r.get("bts") or []] or [0])
    report = {
        "workflow": args.workflow,
        "model": args.model,
        "no_exception": all("exception" not in r for r in results),
        "answered": sum(1 for r in results if r.get("final_answer")),
        "tool_results": sum(len(r.get("tools") or []) for r in results),
        "acc_binary": sum(float(r.get("reward_binary") or 0) for r in results) / max(1, len(results)),
        "max_bts_n": max_bts,
    }
    report["pass"] = bool(report["no_exception"] and report["answered"] > 0
                          and (args.require_bts <= 0 or max_bts >= args.require_bts))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("REPORT", json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # timed-out user windows leave non-daemon worker threads behind
    os._exit(code)
