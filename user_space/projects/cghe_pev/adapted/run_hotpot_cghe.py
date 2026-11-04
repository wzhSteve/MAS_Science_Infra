"""Run CGHE on data/hotpotqa.json. One planner call per DAG, Bing via the second account."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "mas"), str(ROOT / "user_space" / "projects" / "cghe_pev" / "adapted")]

from workflow.env_load import load_repo_dotenv

load_repo_dotenv()
os.environ.setdefault("AGENTFLOW_TOP_LOGPROBS", "5")
os.environ.setdefault("AGENTFLOW_MAX_STEPS", "3")
os.environ.setdefault("AGENTFLOW_MAX_TIME", "180")
os.environ.setdefault("CGHE_N", "4")


def _gold(row: dict) -> str:
    answer = row.get("answer")
    if isinstance(answer, list) and answer:
        return str(answer[0])
    return str(answer or "")


def _hit(prediction: str, gold: str) -> bool:
    pred = " ".join(str(prediction or "").casefold().split())
    needle = " ".join(gold.casefold().split())
    return bool(needle) and needle in pred


def main() -> None:
    import context as context_mod

    data_path = ROOT / "data" / "hotpotqa.json"
    rows = json.loads(data_path.read_text())
    limit = int(os.environ.get("CGHE_HOTPOT_LIMIT") or len(rows))
    offset = int(os.environ.get("CGHE_HOTPOT_OFFSET") or 0)
    rows = rows[offset:offset + limit]
    tag = os.environ.get("CGHE_RUN_TAG") or "full"
    out = Path(os.environ.get("CGHE_OUT") or f"/tmp/cghe_hotpot_{tag}.jsonl")
    ctx = context_mod.EpisodeContext()
    hits = 0
    with out.open("w") as handle:
        for index, row in enumerate(rows):
            question = str(row["question"]).split("When ready")[0].strip()
            payload = {"question": question, "source": f"{tag}-{offset + index}"}
            record = {"index": offset + index, "question": question, "gold": _gold(row), "error": ""}
            all_events: list = []
            role_counts: dict = {}
            try:
                for _ in range(int(os.environ.get("AGENTFLOW_MAX_STEPS") or 3)):
                    plan = ctx.plan_window({"dst": "planner", "kind": "plan_step", "payload": payload})
                    body = plan.get("payload") or {}
                    for item in ((body.get("trace") or {}).get("llm_calls") or {}).get("calls") or []:
                        role = str(item.get("role") or "llm")
                        role_counts[role] = role_counts.get(role, 0) + 1
                    if body.get("done"):
                        record["answer"] = str(body.get("answer") or "")
                        break
                    exec_out = ctx.exec_window({"dst": "executor", "payload": payload})
                    ebody = exec_out.get("payload") or {}
                    for item in ((ebody.get("trace") or {}).get("llm_calls") or {}).get("calls") or []:
                        role = str(item.get("role") or "llm")
                        role_counts[role] = role_counts.get(role, 0) + 1
                    verified = ctx.verify_window({"dst": "verifier", "kind": "verify", "payload": payload})
                    vbody = verified.get("payload") or {}
                    for item in ((vbody.get("trace") or {}).get("llm_calls") or {}).get("calls") or []:
                        role = str(item.get("role") or "llm")
                        role_counts[role] = role_counts.get(role, 0) + 1
                    events = list((vbody.get("trace") or {}).get("cghe_events") or [])
                    all_events.extend(events)
                    record["answer"] = str(vbody.get("answer") or "")
                    record["trace"] = {
                        "decision": (vbody.get("trace") or {}).get("cghe_decision"),
                        "reason": (vbody.get("trace") or {}).get("cghe_reason"),
                        "events": all_events,
                    }
                    if vbody.get("ready_to_stop"):
                        break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["trace_error"] = traceback.format_exc()[-800:]
            try:
                ep = ctx.episode(payload)
                session = getattr(getattr(ep, "_cghe", None), "cghe_session", None)
                if session is not None and getattr(session, "events", None):
                    all_events = list(session.events)
            except Exception:
                pass
            record["hit"] = _hit(record.get("answer") or "", record["gold"])
            if all_events:
                trace = dict(record.get("trace") or {})
                trace["events"] = all_events
                record["trace"] = trace
            events = list(((record.get("trace") or {}).get("events")) or all_events)
            record["regen_count"] = sum(1 for item in events if item.get("event") == "dag_regenerated")
            record["pruned_plan_count"] = sum(
                len(item.get("pruned_plan_ids") or [])
                for item in events
                if item.get("event") == "plans_pruned"
            )
            record["step1_exec_count"] = sum(1 for item in events if item.get("event") == "step1_executed")
            record["llm_calls"] = role_counts
            record["llm_call_count"] = sum(role_counts.values())
            hits += int(record["hit"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                tag,
                record["index"],
                "hit" if record["hit"] else "miss",
                f"regen={record['regen_count']}",
                f"pruned={record['pruned_plan_count']}",
                f"llm={record['llm_call_count']}",
                flush=True,
            )
    print(json.dumps({"tag": tag, "n": len(rows), "hits": hits, "em": hits / max(1, len(rows))}))


if __name__ == "__main__":
    main()
