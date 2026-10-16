#!/usr/bin/env python3
"""Live E2E retest: pure AgentFlow (PEV) user_space wrap, 2 HotpotQA questions.

The 'agentflow' project lives in user_space (mode=native_mas, wraps=AgentFlow.PEV).
To keep this retest hermetic and independent of experiment-file drift, this script
hand-builds the AgentFlow spec — mirroring experiments/agentflow_4algo/workflow.yaml:
user_project=agentflow, wraps AgentFlow.PEV, executor id u_agentflow__executor.

Same execution path as science_infra/control/rollout_runs.py
(_execute_question → ExecutionService → user_space AgentFlow windows), run
synchronously against a local vLLM endpoint. The AgentFlow window used at
runtime is the pure-PEV path (no BTS/belief_prediction).

Verification targets:
  a) loader logs contain '模型  N 次调用' + '提示词' lines
  b) search→fetch hard rule (planning round right after a search emits a
     forced fetch / fetch-url command; N/A if no SERP at all)
  c) final answer is not a raw SERP dump
  d) verify-window trace carries llm_calls
  e) trajectory free of BTS/belief_prediction (pure PEV)
  f) window payload trace shows update_outline/memory fields
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

EXPERIMENT_ID = "agentflow_4algo"  # artifacts land under its rollout-runs dir

WORKFLOW = {
    "schema_version": "0.3",
    "topology": "centralized",
    "entry_agent": "planner",
    "hub": {
        "role": "planner",
        "skills": [],
        "max_feedback_hops": 6,
        "system_prompt": "AgentFlow planner window. Output JSON next/args/sub_goal/done.",
    },
    "tools": [],
    "llm": {
        "kind": "local",
        "model": "Qwen3-4B",
        "base_url": "http://127.0.0.1:8000/v1",
    },
    "memory": {"agent": "messages", "system": "none"},
    "archive": {"window": "post_first_tool"},
    "agents": [
        {
            "id": "planner",
            "kind": "planner",
            "role": "planner",
            "label": "agentflow_planner",
            "trainable": True,
            "system_prompt": "Wrap AgentFlow Planner.generate_next_step. next is the executor window.",
            "tools": [],
            "profile": {"backend": "user_space", "user_project": "agentflow"},
            "meta": {"origin": "user", "user_project": "agentflow", "wraps": "AgentFlow.Planner"},
        },
        {
            "id": "u_agentflow__executor",
            "kind": "blank",
            "role": "executor",
            "label": "agentflow_executor",
            "trainable": False,
            "system_prompt": "Wrap AgentFlow Executor window (tool command + execute).",
            "profile": {"backend": "user_space", "user_project": "agentflow", "llm_required": False},
            "meta": {"origin": "user", "user_project": "agentflow", "wraps": "AgentFlow.Executor"},
        },
        {
            "id": "verifier",
            "kind": "verifier",
            "role": "verifier",
            "label": "agentflow_verifier",
            "trainable": False,
            "system_prompt": "Wrap AgentFlow Diagnoser.verificate_context.",
            "profile": {"backend": "user_space", "user_project": "agentflow"},
            "meta": {"origin": "user", "user_project": "agentflow", "wraps": "AgentFlow.Diagnoser"},
        },
    ],
    "edges": [
        {"from": "planner", "to": "u_agentflow__executor", "kind": "message"},
        {"from": "u_agentflow__executor", "to": "verifier", "kind": "message"},
        {"from": "verifier", "to": "planner", "kind": "feedback"},
    ],
    "routers": [],
    "sampling": {
        "mode": "arpo",
        "group_n": 4,
        "beam_size": 2,
        "sites": [
            {
                "id": "after_planner",
                "enabled": True,
                "anchor": {"kind": "after_agent_turn", "agent_id": "planner"},
                "gate": {"type": "always"},
                "fork": {"beam_size": 2, "resume_mode": "messages"},
            },
            {
                "id": "after_u_agentflow__executor",
                "enabled": True,
                "anchor": {"kind": "after_agent_turn", "agent_id": "u_agentflow__executor"},
                "gate": {"type": "always"},
                "fork": {"beam_size": 2, "resume_mode": "messages"},
            },
            {
                "id": "after_verifier",
                "enabled": True,
                "anchor": {"kind": "after_verifier", "agent_id": "verifier"},
                "gate": {"type": "always"},
                "fork": {"beam_size": 2, "resume_mode": "messages"},
            },
        ],
    },
}


def main() -> int:
    from science_infra.control.task_files import load_task_rows
    from workflow.runtime import LLMConfig

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

        summary, traj = _execute_question(
            EXPERIMENT_ID, WORKFLOW, task, llm=llm, model_summary=None, model_public=None,
            on_window=_on_window,
        )
        answer = str(summary.get("final_answer") or "")[:200]
        gold = str(task.get("answer") or "")
        events = []
        if traj is not None:
            for ev in traj.events or []:
                events.append({
                    "kind": ev.kind.value if hasattr(ev.kind, "value") else str(ev.kind),
                    "payload": ev.payload,
                })
        results.append({
            "id": task.get("id"),
            "answer": answer,
            "gold": gold,
            "status": summary.get("status"),
            "log": lines,
            "events": events,
        })

    out = ROOT / "artifacts/agentflow_tool_rl_verification/mas_eval_retest_af.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
