"""20-question check of the updated simple_PEV loop. No weight update.

Scores each answer with score_llm_binary against ground_truth. Empty
tags, unfinished think traces, and placeholders are 0 before any judge
call. Format success is not accuracy. At most one correct answer in
twenty is treated as the old near-zero regime, and RL stays off.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
PROJECT = ROOT / "user_space" / "projects" / "prove_pev"
ADAPTED = PROJECT / "adapted"
REPORT = PROJECT / "artifacts" / "mas20_accuracy.json"

for path in (str(ROOT), str(ROOT / "mas"), str(ADAPTED)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _one(ctx: Any, item: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    from workflow.user_gateway.sandbox import bind_episode_id, reset_episode_id

    from rl.rewards.outcome import extract_answer_text, score_llm_binary

    token = bind_episode_id(f"mas20-{item['index']}")
    payload = {"question": item["question"], "source": "hotpot"}
    answer = ""
    error = ""
    try:
        for _ in range(max_steps):
            plan = ctx.plan_window({"payload": payload})
            body = plan.get("payload") or {}
            if body.get("done"):
                answer = str(body.get("answer") or "")
                break
            ctx.exec_window({"payload": payload})
            verify = ctx.verify_window({"payload": payload})
            vbody = verify.get("payload") or {}
            if vbody.get("ready_to_stop"):
                answer = str(vbody.get("answer") or "")
                break
        episode = ctx.episode(payload)
        obtained = list(getattr(episode.memory, "obtained_information", None) or [])
        outline = list(getattr(episode.memory, "outline", None) or [])
    except Exception as exc:  # noqa: BLE001
        obtained = []
        outline = []
        error = f"{type(exc).__name__}: {exc}"
    finally:
        reset_episode_id(token)
    extracted = extract_answer_text(answer) or ""
    reward = 0.0 if error and not extracted else score_llm_binary(
        answer, item["ground_truth"], question=item["question"]
    )
    try:
        outline_text = json.dumps(outline, ensure_ascii=False, default=str)
    except TypeError:
        outline_text = str(outline)
    return {
        "index": item["index"],
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "answer": extracted,
        "raw_answer": answer[:500],
        "think_only": "<think" in answer.lower() and not extracted,
        "placeholder_outline": '"subgoal": "task"' in outline_text or "[pending] task" in outline_text,
        "reward": reward,
        "obtained_empty": not obtained,
        "obtained": obtained[:8],
        "outline": outline,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_URL") or "http://127.0.0.1:8000/v1")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()
    os.environ["AGENTFLOW_SFT_PROMPTS"] = "1"
    os.environ["AGENTFLOW_MAX_STEPS"] = str(args.max_steps)
    os.environ["OPENAI_API_BASE"] = args.base_url.rstrip("/")
    os.environ["OPENAI_API_BASE_URL"] = args.base_url.rstrip("/")
    os.environ.setdefault("AGENTFLOW_LLM_TIMEOUT", "120")

    from probe_temperature import _load_questions
    import context as ctx_mod

    ctx = ctx_mod.get_episode_context()
    ctx.components()
    rows: List[Dict[str, Any]] = []
    for item in _load_questions(args.n):
        t0 = time.time()
        row = _one(ctx, item, args.max_steps)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        print(
            f"q={row['index']} reward={row['reward']} answer={row['answer'][:80]!r} "
            f"obtained_empty={row['obtained_empty']}",
            flush=True,
        )
    correct = sum(1 for row in rows if float(row["reward"]) >= 1.0)
    report = {
        "n": len(rows),
        "max_steps": args.max_steps,
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "obtained_empty": sum(1 for row in rows if row["obtained_empty"]),
        "think_only": sum(1 for row in rows if row["think_only"]),
        "placeholder_outline": sum(1 for row in rows if row["placeholder_outline"]),
        "reward_one_bad_answer": sum(
            1
            for row in rows
            if float(row["reward"]) >= 1.0
            and (
                row["think_only"]
                or row["answer"].strip().lower() in {"short answer", "short", "think"}
                or "distance_between_centers" in row["answer"]
            )
        ),
        "rewards": [row["reward"] for row in rows],
        # One hit in twenty is still the old near-zero regime.
        "run_short_train": correct > 1,
        "rows": rows,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"accuracy={report['accuracy']} correct={correct}/{len(rows)} "
        f"obtained_empty={report['obtained_empty']} run_short_train={report['run_short_train']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
