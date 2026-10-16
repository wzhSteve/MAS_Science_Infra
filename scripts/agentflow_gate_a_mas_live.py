#!/usr/bin/env python3
"""Gate A: AgentFlow MAS live episode on 1 GSM8K + 1 HotpotQA."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mas"))

os.environ.setdefault("TIR_MAX_MODEL_LEN", "2560")
os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
os.environ.setdefault("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")


def _tool_kinds(messages):
    kinds = []
    tools = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        kind = str(m.get("kind") or "")
        kinds.append(kind)
        if kind in ("tool_invoke", "tool_result"):
            payload = m.get("payload") or {}
            tools.append(
                {
                    "kind": kind,
                    "tool": payload.get("tool") or payload.get("tool_name") or "",
                    "ok": payload.get("ok"),
                    "snippet": str(payload.get("output") or payload.get("content") or "")[:160],
                }
            )
    return kinds, tools


def run_one(task, cfg, spec):
    from mas.mas_agent import bind_training_llm_env
    from workflow.archive import Archive, register_archive
    from workflow.memory import MemoryStore
    from workflow.runtime import run_compiled_episode
    from rl.rewards import compute_mas_outcome_reward

    arch = register_archive(Archive())
    mem = MemoryStore()
    with bind_training_llm_env(
        endpoint=cfg.endpoint, model=cfg.model, api_key="EMPTY"
    ):
        raw = run_compiled_episode(task, cfg, arch, spec, mem, None)
    kinds, tools = _tool_kinds(raw.messages)
    answer = str(raw.final_answer or "")
    try:
        reward = float(compute_mas_outcome_reward(answer, task))
    except Exception as exc:
        reward = None
        reward_err = str(exc)
    else:
        reward_err = ""
    return {
        "id": task.get("id"),
        "source": task.get("source"),
        "n_messages": len(raw.messages or []),
        "kinds": kinds,
        "tools": tools,
        "final_answer": answer[:500],
        "has_answer_tag": "<answer>" in answer and "</answer>" in answer,
        "reward": reward,
        "reward_err": reward_err,
        "error": getattr(raw, "error", None),
    }


def main() -> int:
    from workflow.runtime import LLMConfig
    from workflow.spec import load_spec

    cfg = LLMConfig(
        endpoint="http://127.0.0.1:8000/v1",
        model="Qwen3-4B",
        temperature=0.0,
        max_turns=8,
        max_tokens=512,
        max_model_len=2560,
        enabled_tools=[
            "Python_Coder_Tool",
            "Wikipedia_Search_Tool",
            "Google_Search_Tool",
            "Web_Fetch_Tool",
            "Base_Generator_Tool",
        ],
    )
    spec = load_spec(str(ROOT / "experiments/agentflow_4algo/workflow.yaml"))
    tasks = [
        {
            "id": "gate-a-gsm8k",
            "question": (
                "Jason bought 4 dozen cupcakes. He plans to give 3 cupcakes each "
                "to his cousins. How many cousins does Jason have?"
            ),
            "answer": "16",
            "source": "gsm8k",
        },
        {
            "id": "gate-a-hotpot",
            "question": (
                "What nationality was Oliver Reed, the actor who played the title role "
                "in the 1975 film Royal Flash? Use Wikipedia or web search before answering."
            ),
            "answer": "British",
            "answers": ["British", "English"],
            "source": "hotpot",
        },
    ]

    results = []
    hard_fail = False
    for task in tasks:
        print(f"\n=== RUN {task['id']} ===", flush=True)
        try:
            r = run_one(task, cfg, spec)
        except Exception as exc:
            hard_fail = True
            r = {"id": task["id"], "exception": f"{type(exc).__name__}: {exc}"}
            print("EXCEPTION:", r["exception"], flush=True)
        else:
            print(json.dumps({k: v for k, v in r.items() if k != "tools"}, ensure_ascii=False, indent=2), flush=True)
            print("tools:", json.dumps(r.get("tools") or [], ensure_ascii=False)[:800], flush=True)
        results.append(r)

    # Gate criteria
    ok_no_exception = all("exception" not in r for r in results)
    ok_answer_tag = any(r.get("has_answer_tag") for r in results)
    tool_hits = []
    for r in results:
        for t in r.get("tools") or []:
            snippet = str(t.get("snippet") or "")
            if "AuthenticationError" in snippet or "401" in snippet:
                continue
            if t.get("kind") == "tool_result" and snippet.strip():
                tool_hits.append(t)
    ok_tool = len(tool_hits) > 0
    gsm = next((r for r in results if r.get("source") == "gsm8k"), {})
    ok_reward = gsm.get("reward") is not None and not gsm.get("reward_err")

    report = {
        "ok_no_exception": ok_no_exception,
        "ok_tool": ok_tool,
        "ok_answer_tag": ok_answer_tag,
        "ok_reward_computable": ok_reward,
        "tool_hits_n": len(tool_hits),
        "hard_fail": hard_fail,
        "pass": bool(ok_no_exception and ok_tool and ok_answer_tag and ok_reward and not hard_fail),
    }
    print("\n=== GATE A REPORT ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    out = ROOT / "logs/agentflow_gate_a/mas_live_report.json"
    out.write_text(json.dumps({"report": report, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
