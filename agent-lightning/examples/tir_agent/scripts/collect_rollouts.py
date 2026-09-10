#!/usr/bin/env python3
"""Collect TIR rollouts + rewards without VERL / training GPU.

Examples:
  python scripts/collect_rollouts.py --mock --out /tmp/traj.json
  python scripts/collect_rollouts.py --endpoint $OPENAI_API_BASE --model $OPENAI_MODEL
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_tasks(path: Optional[str]) -> List[Dict[str, Any]]:
    if not path:
        return [
            {"id": "demo-1", "question": "What is 1+1?", "answer": "2", "source": "gsm8k"},
            {"id": "demo-2", "question": "What is 2+2?", "answer": "4", "source": "gsm8k", "_mock_answer": "4"},
        ]
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON must be a list of tasks")
        return data
    if p.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(p).to_dict(orient="records")
    raise ValueError(f"Unsupported tasks file: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="TIR workflow collect (no RL)")
    parser.add_argument("--mock", action="store_true", help="CPU mock rollouts (no LLM)")
    parser.add_argument("--endpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument("--n", type=int, default=1, help="samples per task")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    from workflow.env_load import load_repo_dotenv
    from workflow import Collector, batch_to_train_signal

    load_repo_dotenv()

    tasks = _load_tasks(args.tasks)
    # For default mock demos, first task uses gold as answer via mock
    if args.mock and not args.tasks:
        tasks[0]["_mock_answer"] = "2"

    collector = Collector(
        mock=args.mock,
        endpoint=args.endpoint,
        model=args.model,
        temperature=args.temperature,
        n=args.n,
    )
    batch = collector.collect(tasks)
    signal = batch_to_train_signal(batch, algo="grpo")
    payload = {
        "batch": batch.model_dump(mode="json"),
        "train_signal": {
            "advantage": signal.advantage.model_dump(mode="json"),
            "loss": signal.loss.model_dump(mode="json"),
            "meta": signal.meta,
        },
    }

    # Prove we did not need agentlightning for --mock
    if args.mock and "agentlightning" in sys.modules:
        print("WARNING: agentlightning was imported during --mock collect", file=sys.stderr)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"wrote {out} n={batch.meta.get('n_trajectories')} "
            f"mean_reward={batch.meta.get('mean_reward')}"
        )
    else:
        for t in batch.trajectories:
            print(
                f"reward={t.final_reward} answer={t.final_answer!r} "
                f"events={len(t.events)} snapshots={t.archive.n_snapshots if t.archive else 0}"
            )
        print(f"mean_reward={batch.meta.get('mean_reward')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
