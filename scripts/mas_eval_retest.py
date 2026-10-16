#!/usr/bin/env python3
"""Offline replica of the UI 'MAS 测试' eval run (demo experiment).

Uses the exact same execution path as science_infra/control/rollout_runs.py
(_execute_question → ExecutionService → user_space AgentFlow windows), but
synchronously, so fixes can be verified without driving the FastAPI app.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mas"))

os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")


def main() -> int:
    from science_infra.control.experiments import load_bundle
    from science_infra.control.task_files import load_task_rows
    from workflow.runtime import LLMConfig

    bundle = load_bundle("agentflow_4algo")
    workflow = bundle.get("workflow") or {}
    tasks = load_task_rows(ROOT / "data" / "hotpotqa.json")[:2]

    llm = LLMConfig(
        endpoint=os.environ["OPENAI_BASE_URL"],
        model="Qwen3-4B",
        api_key="EMPTY",
    )

    from science_infra.control.rollout_runs import _execute_question

    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n=== 题目 {i}/2: {task.get('question', '')[:80]}...", flush=True)
        lines: list[str] = []

        def _on_window(line: str) -> None:
            text = str(line or "")
            lines.append(text)
            print(text, flush=True)

        summary, _traj = _execute_question(
            "demo", workflow, task, llm=llm, model_summary=None, model_public=None,
            on_window=_on_window,
        )
        answer = str(summary.get("final_answer") or "")[:200]
        gold = str(task.get("answer") or "")
        print(f"--- 答案: {answer}", flush=True)
        print(f"--- 标准: {gold}", flush=True)
        results.append({
            "id": task.get("id"),
            "answer": answer,
            "gold": gold,
            "log": lines,
        })

    out = ROOT / "artifacts/agentflow_tool_rl_verification/mas_eval_retest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
