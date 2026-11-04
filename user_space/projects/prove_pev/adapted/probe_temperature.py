"""Planner-temperature probe. No weight update.

Eight Hotpot questions, temperatures 0.5 / 0.7 / 1.0, four samples each,
max_steps 3. A group counts only when llm_binary contains both 0 and 1.
Unique answer strings are recorded and are not the selection rule.
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
PARQUET = ROOT / "ref_Rep" / "APPO-main" / "APPO" / "rl_datasets" / "valid.parquet"
REPORT = PROJECT / "artifacts" / "temperature_probe.json"

for path in (str(ROOT), str(ROOT / "mas"), str(ADAPTED)):
    if path not in sys.path:
        sys.path.insert(0, path)


def choose_temperature(by_temp: Dict[str, Dict[str, Any]]) -> float:
    """Keep 0.7 unless another temperature creates more groups with both 0 and 1."""
    base = int((by_temp.get("0.7") or {}).get("contrast_groups") or 0)
    chosen = 0.7
    best = base
    for label in ("0.5", "1.0"):
        count = int((by_temp.get(label) or {}).get("contrast_groups") or 0)
        if count > best:
            chosen = float(label)
            best = count
    return chosen


def summarize_group(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    answers = [str(row.get("answer") or "") for row in rows]
    tools = [tuple(row.get("tools") or []) for row in rows]
    contrast = any(r <= 0.0 for r in rewards) and any(r >= 1.0 for r in rewards)
    return {
        "rewards": rewards,
        "answers": answers,
        "tool_sequences": ["|".join(seq) for seq in tools],
        "unique_answers": len({a for a in answers if a}),
        "unique_tool_sequences": len(set(tools)),
        "contrast": contrast,
    }


def _load_questions(n: int) -> List[Dict[str, Any]]:
    import pandas as pd

    frame = pd.read_parquet(PARQUET)
    rows = []
    for i, row in frame.head(n).iterrows():
        extra = row.get("extra_info") or {}
        if not isinstance(extra, dict):
            extra = {}
        question = str(extra.get("question") or "")
        if not question:
            prompt = row.get("prompt")
            if hasattr(prompt, "tolist"):
                prompt = prompt.tolist()
            if isinstance(prompt, list):
                for message in prompt:
                    if isinstance(message, dict) and message.get("role") == "user":
                        question = str(message.get("content") or "")
        reward = row.get("reward_model") or {}
        gold = reward.get("ground_truth") if isinstance(reward, dict) else ""
        rows.append({
            "index": int(extra.get("index") if isinstance(extra, dict) and extra.get("index") is not None else len(rows)),
            "question": question,
            "ground_truth": str(gold or ""),
        })
    return rows


def _tool_names(episode: Any) -> List[str]:
    actions = episode.memory.get_actions() if episode is not None else {}
    names = []
    for _key, action in (actions or {}).items():
        if isinstance(action, dict):
            names.append(str(action.get("tool_name") or ""))
    return names


def _one_sample(ctx: Any, item: Dict[str, Any], sample_id: str, max_steps: int) -> Dict[str, Any]:
    from workflow.user_gateway.sandbox import bind_episode_id, reset_episode_id

    from rl.rewards.llm_judge import llm_judge_correct
    from rl.rewards.outcome import extract_answer_text

    token = bind_episode_id(sample_id)
    payload = {"question": item["question"], "source": "hotpot"}
    answer = ""
    error = ""
    tools: List[str] = []
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
        tools = _tool_names(ctx.episode(payload))
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        reset_episode_id(token)
    extracted = extract_answer_text(answer) or ""
    reward = 0.0 if error and not extracted else llm_judge_correct(item["question"], item["ground_truth"], extracted)
    return {
        "answer": extracted,
        "raw_answer": answer[:500],
        "tools": tools,
        "reward": reward,
        "error": error,
    }


def run_probe(temperatures: List[float], n_questions: int, group: int, max_steps: int, base_url: str) -> Dict[str, Any]:
    os.environ["AGENTFLOW_SFT_PROMPTS"] = "1"
    os.environ["AGENTFLOW_MAX_STEPS"] = str(max_steps)
    os.environ["OPENAI_API_BASE"] = base_url
    os.environ["OPENAI_API_BASE_URL"] = base_url
    os.environ.setdefault("AGENTFLOW_LLM_TIMEOUT", "180")

    import context as ctx_mod

    ctx = ctx_mod.get_episode_context()
    questions = _load_questions(n_questions)
    by_temp: Dict[str, Any] = {}
    for temperature in temperatures:
        label = f"{temperature:.1f}"
        os.environ["AGENTFLOW_TEMPERATURE"] = label
        comps = ctx.components()
        comps.planner.llm_engine.temperature = float(temperature)
        groups = []
        for item in questions:
            rows = []
            for k in range(group):
                sample_id = f"probe-{label}-{item['index']}-{k}"
                t0 = time.time()
                row = _one_sample(ctx, item, sample_id, max_steps)
                row["seconds"] = round(time.time() - t0, 1)
                rows.append(row)
                print(
                    f"t={label} q={item['index']} k={k} reward={row['reward']} "
                    f"answer={row['answer'][:80]!r} tools={row['tools']}",
                    flush=True,
                )
            summary = summarize_group(rows)
            summary["index"] = item["index"]
            summary["question"] = item["question"]
            groups.append(summary)
        contrast = sum(1 for g in groups if g["contrast"])
        by_temp[label] = {
            "contrast_groups": contrast,
            "n_groups": len(groups),
            "groups": groups,
        }
        print(f"temperature {label}: contrast_groups={contrast}/{len(groups)}", flush=True)
    chosen = choose_temperature(by_temp)
    return {
        "n_questions": n_questions,
        "group": group,
        "max_steps": max_steps,
        "by_temperature": by_temp,
        "recommended_temperature": chosen,
        "contrast_groups": int(by_temp.get(f"{chosen:.1f}", {}).get("contrast_groups") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE_URL") or "http://127.0.0.1:8000/v1")
    parser.add_argument("--n-questions", type=int, default=8)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--temperatures", default="0.5,0.7,1.0")
    args = parser.parse_args()
    temps = [float(x) for x in args.temperatures.split(",") if x.strip()]
    report = run_probe(temps, args.n_questions, args.group, args.max_steps, args.base_url.rstrip("/"))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"recommended_temperature={report['recommended_temperature']} "
        f"contrast_groups={report['contrast_groups']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
