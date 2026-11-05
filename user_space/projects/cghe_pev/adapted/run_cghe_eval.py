"""Run Full CGHE on a JSON QA dataset (HotpotQA / Bamboogle / ...).

Env:
  CGHE_DATA          path to json list (default: data/hotpotqa.json)
  CGHE_OUT           output jsonl path
  CGHE_RUN_TAG       tag for logs / source
  CGHE_OFFSET / CGHE_LIMIT   (also accept CGHE_HOTPOT_*)
  CGHE_SQLITE_PATH   per-process evidence db (set by launcher)
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
ADAPTED = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "mas"), str(ADAPTED)]

from workflow.env_load import load_repo_dotenv

load_repo_dotenv()
# Do not override LLM / SERP_2 keys already pinned by the launcher.
os.environ.setdefault("AGENTFLOW_TOP_LOGPROBS", "5")
os.environ.setdefault("AGENTFLOW_MAX_STEPS", "10")
os.environ.setdefault("AGENTFLOW_MAX_TIME", "600")
os.environ.setdefault("CGHE_N", "4")
if os.environ.get("MODEL_Name"):
    os.environ.setdefault("OPENAI_MODEL", os.environ["MODEL_Name"])
    os.environ.setdefault("SERVER_MODEL", os.environ["MODEL_Name"])


def _gold(row: dict) -> str:
    answer = row.get("answer")
    if isinstance(answer, list) and answer:
        return str(answer[0])
    return str(answer or "")


def _golds(row: dict) -> list[str]:
    answer = row.get("answer")
    if isinstance(answer, list):
        return [str(item) for item in answer if str(item or "").strip()]
    text = str(answer or "").strip()
    return [text] if text else []


def _hit(prediction: str, gold: str) -> bool:
    pred = " ".join(str(prediction or "").casefold().split())
    needle = " ".join(gold.casefold().split())
    return bool(needle) and needle in pred


def _local_correct(prediction: str, golds: list[str]) -> bool:
    try:
        from rl.rewards.outcome import extract_answer_text, normalize_qa, numeric_match
    except Exception:
        return any(_hit(prediction, gold) for gold in golds)
    extracted = extract_answer_text(prediction or "") or ""
    if not extracted:
        return False
    for gold in golds:
        if normalize_qa(extracted) and normalize_qa(extracted) == normalize_qa(str(gold)):
            return True
        if len(extracted) <= 80 and numeric_match(extracted, str(gold)):
            return True
    return False


def _backfill_answer(ctx: Any, payload: dict, current: str) -> str:
    if str(current or "").strip():
        return str(current)
    try:
        ep = ctx.episode(payload)
    except Exception:
        return str(current or "")
    for key in ("direct_output", "final_output", "answer"):
        value = getattr(ep, key, None)
        if value is None and hasattr(ep, "json_data"):
            value = (ep.json_data or {}).get(key)
        text = str(value or "").strip()
        if text:
            return text
    return str(current or "")


def _int_env(*names: str, default: int) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw is not None and str(raw).strip() != "":
            return int(raw)
    return default


def main() -> None:
    import context as context_mod

    data_path = Path(
        os.environ.get("CGHE_DATA")
        or (ROOT / "data" / "hotpotqa.json")
    )
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    rows = json.loads(data_path.read_text())
    offset = _int_env("CGHE_OFFSET", "CGHE_HOTPOT_OFFSET", default=0)
    limit = _int_env("CGHE_LIMIT", "CGHE_HOTPOT_LIMIT", default=len(rows))
    rows = rows[offset:offset + limit]
    tag = os.environ.get("CGHE_RUN_TAG") or data_path.stem
    out = Path(os.environ.get("CGHE_OUT") or f"/tmp/cghe_{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    sqlite = os.environ.get("CGHE_SQLITE_PATH", "").strip()
    if sqlite:
        Path(sqlite).parent.mkdir(parents=True, exist_ok=True)

    ctx = context_mod.EpisodeContext()
    hits = 0
    corrects = 0
    nonempty = 0
    errors = 0
    max_steps = int(os.environ.get("AGENTFLOW_MAX_STEPS") or 10)
    with out.open("w") as handle:
        for index, row in enumerate(rows):
            question = str(row.get("question") or row.get("query") or "").split("When ready")[0].strip()
            payload = {"question": question, "source": f"{tag}-{offset + index}"}
            golds = _golds(row)
            record = {
                "index": offset + index,
                "dataset": data_path.name,
                "question": question,
                "gold": _gold(row),
                "golds": golds,
                "error": "",
            }
            all_events: list = []
            role_counts: dict = {}
            try:
                for _ in range(max_steps):
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
                        "budget_empty_stop": (vbody.get("trace") or {}).get("budget_empty_stop"),
                    }
                    if vbody.get("ready_to_stop"):
                        break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["trace_error"] = traceback.format_exc()[-800:]
                errors += 1
            record["answer"] = _backfill_answer(ctx, payload, record.get("answer") or "")
            try:
                ep = ctx.episode(payload)
                session = getattr(getattr(ep, "_cghe", None), "cghe_session", None)
                if session is not None and getattr(session, "events", None):
                    all_events = list(session.events)
            except Exception:
                pass
            record["hit"] = _hit(record.get("answer") or "", record["gold"])
            record["correct"] = _local_correct(record.get("answer") or "", golds)
            record["nonempty"] = bool(str(record.get("answer") or "").strip())
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
            corrects += int(record["correct"])
            nonempty += int(record["nonempty"])
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                tag,
                record["index"],
                "hit" if record["hit"] else "miss",
                f"correct={int(record['correct'])}",
                f"nonempty={int(record['nonempty'])}",
                f"regen={record['regen_count']}",
                f"pruned={record['pruned_plan_count']}",
                f"llm={record['llm_call_count']}",
                flush=True,
            )
    summary = {
        "tag": tag,
        "dataset": str(data_path),
        "n": len(rows),
        "hits": hits,
        "correct": corrects,
        "nonempty": nonempty,
        "nonempty_rate": nonempty / max(1, len(rows)),
        "errors": errors,
        "em": hits / max(1, len(rows)),
        "accuracy": corrects / max(1, len(rows)),
        "max_steps": max_steps,
        "model": os.environ.get("MODEL_Name") or os.environ.get("OPENAI_MODEL"),
        "api_base": os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        "serp_zone": os.environ.get("BRIGHT_DATA_SERP_ZONE") or os.environ.get("BRIGHT_DATA_SERP_ZONES"),
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
