"""Paired HotpotQA runner for frozen pure_pev and cghe_pev.

Temperature is forced to 0. Each question stops at ready_to_stop, done, or
6 planner steps. Scoring uses local normalized match only.

Examples
--------
python experiments/cghe_hotpot/run_pair.py worker --system pure_pev --split dev --out experiments/cghe_hotpot/rounds/round_00/pure_pev.json
python experiments/cghe_hotpot/run_pair.py compare --pure experiments/cghe_hotpot/rounds/round_00/pure_pev.json --cghe experiments/cghe_hotpot/rounds/round_00/cghe_full.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
DATA = ROOT / "data" / "hotpotqa.json"
EXP = ROOT / "experiments" / "cghe_hotpot"
PROJECTS = {
    "pure_pev": ROOT / "user_space" / "projects" / "pure_pev",
    "cghe_pev": ROOT / "user_space" / "projects" / "cghe_pev",
}
SPLITS = {"dev": (0, 19), "heldout": (20, 39)}
ABLATONS = (
    "full",
    "graph_only",
    "no_memory",
    "no_shared",
    "shared_only",
    "constraint_inference_only",
    "epv_only",
    "counterfactual_only",
    "epv_plus_counterfactual",
    "full_cghe_v2",
)


def _load_items(split: str) -> List[Dict[str, Any]]:
    start, end = SPLITS[split]
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    return [row for row in rows if start <= int(row["idx"]) <= end]


def _local_correct(prediction: str, golds: List[str]) -> bool:
    mas = str(ROOT / "mas")
    if mas not in sys.path:
        sys.path.insert(0, mas)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from rl.rewards.outcome import extract_answer_text, normalize_qa, numeric_match

    extracted = extract_answer_text(prediction or "") or ""
    if not extracted:
        return False
    for gold in golds:
        if normalize_qa(extracted) and normalize_qa(extracted) == normalize_qa(str(gold)):
            return True
        if len(extracted) <= 80 and numeric_match(extracted, str(gold)):
            return True
    return False


def _shrink_trace(trace: Any) -> Dict[str, Any]:
    if not isinstance(trace, dict):
        return {}
    calls = trace.get("llm_calls") if isinstance(trace.get("llm_calls"), dict) else {}
    kept = {
        "cghe_decision": trace.get("cghe_decision", ""),
        "cghe_reason": str(trace.get("cghe_reason") or "")[:300],
        "cghe_mode": trace.get("cghe_mode", ""),
        "cghe_node_id": trace.get("cghe_node_id", ""),
        "cghe_outlines": trace.get("cghe_outlines") or [],
        "cghe_frontier": trace.get("cghe_frontier") or [],
        "conclusion": trace.get("conclusion", ""),
        "tool": trace.get("tool", ""),
        "llm_calls": int(calls.get("count") or 0),
    }
    return {key: value for key, value in kept.items() if value not in ("", 0, None)}


def _run_question(ctx: Any, item: Dict[str, Any], system: str, ablation: str, max_steps: int) -> Dict[str, Any]:
    from workflow.user_gateway.sandbox import bind_episode_id, reset_episode_id

    token = bind_episode_id(f"hotpot-{system}-{ablation}-{item['idx']}")
    payload = {"question": item["question"], "source": "hotpot"}
    steps: List[Dict[str, Any]] = []
    answer = ""
    error = ""
    started = time.time()
    try:
        for _ in range(max_steps):
            print(f"idx={item['idx']} plan", flush=True)
            plan = ctx.plan_window({"payload": payload})
            body = plan.get("payload") or {}
            args = body.get("args") if isinstance(body.get("args"), dict) else {}
            print(
                f"idx={item['idx']} planned tool={args.get('tool_name','')} "
                f"mode={(body.get('trace') or {}).get('cghe_mode','')}",
                flush=True,
            )
            steps.append({
                "phase": "plan",
                "done": bool(body.get("done")),
                "tool": args.get("tool_name", ""),
                "sub_goal": str(body.get("sub_goal") or "")[:240],
                "trace": _shrink_trace(body.get("trace")),
            })
            if body.get("done") or body.get("ready_to_stop"):
                answer = str(body.get("answer") or "")
                break
            executed = ctx.exec_window({"payload": payload})
            ebody = executed.get("payload") or {}
            steps.append({
                "phase": "exec",
                "tool": ebody.get("tool_name") or ebody.get("tool") or "",
                "ok": bool(ebody.get("ok")),
                "output": str(ebody.get("output") or "")[:400],
            })
            verified = ctx.verify_window({"payload": payload})
            vbody = verified.get("payload") or {}
            steps.append({
                "phase": "verify",
                "conclusion": vbody.get("conclusion", ""),
                "reason": str(vbody.get("reason") or "")[:400],
                "trace": _shrink_trace(vbody.get("trace")),
            })
            if vbody.get("ready_to_stop"):
                answer = str(vbody.get("answer") or "")
                break
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        reset_episode_id(token)
    golds = [str(piece) for piece in (item.get("answer") or [])]
    tool_calls = sum(1 for step in steps if step["phase"] == "exec")
    recovery = any(
        str((step.get("trace") or {}).get("cghe_decision") or "") in {"need_universe", "continue_execution"}
        for step in steps
    )
    return {
        "idx": int(item["idx"]),
        "question": item["question"],
        "gold": golds,
        "answer": answer,
        "correct": _local_correct(answer, golds),
        "tool_calls": tool_calls,
        "seconds": round(time.time() - started, 1),
        "recovery": recovery,
        "error": error,
        "steps": steps,
    }


def _prepare_imports(system: str) -> None:
    project = PROJECTS[system]
    adapted = project / "adapted"
    for path in (str(ROOT), str(ROOT / "mas"), str(adapted)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["AGENTFLOW_TEMPERATURE"] = "0"
    os.environ["AGENTFLOW_MAX_STEPS"] = os.environ.get("AGENTFLOW_MAX_STEPS", "6")
    os.environ["AGENTFLOW_TOP_LOGPROBS"] = "5"
    os.environ.setdefault("AGENTFLOW_LLM_TIMEOUT", "120")
    from workflow.env_load import load_repo_dotenv

    load_repo_dotenv()


def worker(args: argparse.Namespace) -> None:
    if args.system == "pure_pev" and args.ablation != "full":
        raise SystemExit("pure_pev has no CGHE ablation")
    os.environ["CGHE_ABLATION"] = args.ablation
    if args.system == "cghe_pev":
        os.environ["CGHE_SQLITE_PATH"] = str(Path(args.out).with_suffix(".sqlite"))
    _prepare_imports(args.system)
    import context as ctx_mod

    if args.system == "cghe_pev" and args.reset_memory and args.ablation != "no_memory":
        db = Path(ctx_mod.cghe_settings()["sqlite_path"])
        if db.exists():
            db.unlink()
    ctx = ctx_mod.get_episode_context()
    ctx.components()
    rows = []
    items = _load_items(args.split)
    if args.limit:
        items = items[: args.limit]
    for item in items:
        row = _run_question(ctx, item, args.system, args.ablation, args.max_steps)
        rows.append(row)
        print(
            f"{args.system}:{args.ablation} idx={row['idx']} correct={row['correct']} "
            f"tools={row['tool_calls']} recovery={row['recovery']} seconds={row['seconds']}",
            flush=True,
        )
    correct = sum(1 for row in rows if row["correct"])
    calls = [row["tool_calls"] for row in rows]
    report = {
        "system": args.system,
        "ablation": args.ablation,
        "split": args.split,
        "n": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "mean_tool_calls": round(sum(calls) / len(calls), 4) if calls else 0.0,
        "recovery_questions": sum(1 for row in rows if row["recovery"]),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} correct={correct}/{len(rows)} mean_tools={report['mean_tool_calls']}", flush=True)


def _index(report: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    return {int(row["idx"]): row for row in report["rows"]}


def compare(args: argparse.Namespace) -> None:
    pure = json.loads(Path(args.pure).read_text(encoding="utf-8"))
    cghe = json.loads(Path(args.cghe).read_text(encoding="utf-8"))
    left = _index(pure)
    right = _index(cghe)
    shared = sorted(set(left) & set(right))
    gain = sum(1 for idx in shared if right[idx]["correct"]) - sum(1 for idx in shared if left[idx]["correct"])
    pure_calls = sum(left[idx]["tool_calls"] for idx in shared) / len(shared) if shared else 0.0
    cghe_calls = sum(right[idx]["tool_calls"] for idx in shared) / len(shared) if shared else 0.0
    worse = [idx for idx in shared if left[idx]["correct"] and not right[idx]["correct"]]
    better = [idx for idx in shared if right[idx]["correct"] and not left[idx]["correct"]]
    summary = {
        "pure": args.pure,
        "cghe": args.cghe,
        "ablation": cghe.get("ablation", ""),
        "n": len(shared),
        "pure_correct": sum(1 for idx in shared if left[idx]["correct"]),
        "cghe_correct": sum(1 for idx in shared if right[idx]["correct"]),
        "gain": gain,
        "pure_mean_tool_calls": round(pure_calls, 4),
        "cghe_mean_tool_calls": round(cghe_calls, 4),
        "tool_call_ratio": round(cghe_calls / pure_calls, 4) if pure_calls else None,
        "better": better,
        "worse": worse,
        "cghe_recovery": sum(1 for idx in shared if right[idx].get("recovery")),
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


def ledger_append(args: argparse.Namespace) -> None:
    path = EXP / "ledger.json"
    rows: List[Dict[str, Any]] = []
    if path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
    entry = json.loads(Path(args.entry).read_text(encoding="utf-8"))
    rows = [row for row in rows if int(row.get("round", -1)) != int(entry["round"]) or row.get("ablation") != entry.get("ablation")]
    rows.append(entry)
    rows.sort(key=lambda row: (int(row.get("round", 0)), str(row.get("ablation") or "")))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("worker")
    run.add_argument("--system", required=True, choices=sorted(PROJECTS))
    run.add_argument("--split", required=True, choices=sorted(SPLITS))
    run.add_argument("--ablation", default="full", choices=ABLATONS)
    run.add_argument("--max-steps", type=int, default=6)
    run.add_argument("--out", required=True)
    run.add_argument("--reset-memory", action="store_true")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=worker)

    diff = sub.add_parser("compare")
    diff.add_argument("--pure", required=True)
    diff.add_argument("--cghe", required=True)
    diff.add_argument("--summary", default="")
    diff.set_defaults(func=compare)

    book = sub.add_parser("ledger")
    book.add_argument("--entry", required=True)
    book.set_defaults(func=ledger_append)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
