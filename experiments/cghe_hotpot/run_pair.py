"""Paired HotpotQA runner for frozen pure_pev, prove_pev, and cghe_pev.

Temperature is forced to 0 for the global eval stack. prove_pev still samples
plan candidates at ``PROVE_CANDIDATE_TEMPERATURE`` (default 0.7) inside its
Best-of-N path. Each question stops at ready_to_stop, done, or 6 planner steps.
Scoring uses local normalized match only.

Examples
--------
python experiments/cghe_hotpot/run_pair.py worker --system pure_pev --split dev --out experiments/cghe_hotpot/rounds/round_00/pure_pev.json
python experiments/cghe_hotpot/run_pair.py worker --system prove_pev --split dev --out experiments/cghe_hotpot/rounds/round_00/prove_pev.json
python experiments/cghe_hotpot/run_pair.py compare --pure .../pure_pev.json --cghe .../cghe_full.json
python experiments/cghe_hotpot/run_pair.py compare3 --pure .../pure.json --prove .../prove.json --cghe .../cghe.json
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
    "prove_pev": ROOT / "user_space" / "projects" / "prove_pev",
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
        "cghe_feedback": trace.get("cghe_feedback") or [],
        "cghe_metrics": trace.get("cghe_metrics") or {},
        "cghe_task_id": trace.get("cghe_task_id", ""),
        "cghe_select_policy": trace.get("cghe_select_policy", ""),
        "cghe_round_id": trace.get("cghe_round_id", ""),
        "prove_decision": trace.get("prove_decision", ""),
        "prove_reason": str(trace.get("prove_reason") or "")[:300],
        "prove_n": trace.get("prove_n", 0),
        "prove_best_idx": trace.get("prove_best_idx", ""),
        "prove_rankings": trace.get("prove_rankings") or [],
        "prove_site": trace.get("prove_site", ""),
        "prove_judge_head": str(trace.get("prove_judge_head") or "")[:200],
        "conclusion": trace.get("conclusion", ""),
        "tool": trace.get("tool", ""),
        "llm_calls": int(calls.get("count") or 0),
    }
    return {key: value for key, value in kept.items() if value not in ("", 0, None, [], "")}


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
            trace = body.get("trace") if isinstance(body.get("trace"), dict) else {}
            print(
                f"idx={item['idx']} planned tool={args.get('tool_name','')} "
                f"mode={trace.get('cghe_mode','')} prove={trace.get('prove_decision','')}",
                flush=True,
            )
            steps.append({
                "phase": "plan",
                "done": bool(body.get("done")),
                "tool": args.get("tool_name", ""),
                "sub_goal": str(body.get("sub_goal") or "")[:240],
                "trace": _shrink_trace(trace),
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
                "trace": _shrink_trace(ebody.get("trace")),
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
    llm_calls = sum(int((step.get("trace") or {}).get("llm_calls") or 0) for step in steps)
    recovery = any(
        str((step.get("trace") or {}).get("cghe_decision") or "") in {"need_universe", "continue_execution"}
        for step in steps
    )
    prove_selected = any(
        str((step.get("trace") or {}).get("prove_decision") or "") == "selected"
        for step in steps
    )
    metrics: Dict[str, Any] = {}
    for step in reversed(steps):
        trace = step.get("trace") if isinstance(step.get("trace"), dict) else {}
        if isinstance(trace.get("cghe_metrics"), dict) and trace.get("cghe_metrics"):
            metrics = dict(trace.get("cghe_metrics") or {})
            break
    return {
        "idx": int(item["idx"]),
        "question": item["question"],
        "gold": golds,
        "answer": answer,
        "correct": _local_correct(answer, golds),
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "seconds": round(time.time() - started, 1),
        "recovery": recovery,
        "prove_selected": prove_selected,
        "cghe_metrics": metrics,
        "error": error,
        "steps": steps,
    }


def _prepare_imports(system: str) -> None:
    # Drop previous project adapted/ from path so context imports stay isolated.
    drop = []
    for path in list(sys.path):
        if path.endswith("/adapted") and "/user_space/projects/" in path.replace("\\", "/"):
            drop.append(path)
    for path in drop:
        while path in sys.path:
            sys.path.remove(path)
    for name in list(sys.modules):
        if name == "context" or name.startswith("prove_") or name in {
            "engine", "sft_prompts", "tools_local", "prediction_probs",
        }:
            del sys.modules[name]

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
    if args.system == "prove_pev" and args.ablation != "full":
        raise SystemExit("prove_pev has no CGHE ablation; use --prove-n / --prove-granularity")
    os.environ["CGHE_ABLATION"] = args.ablation
    if args.system == "cghe_pev":
        os.environ["CGHE_SQLITE_PATH"] = str(Path(args.out).with_suffix(".sqlite"))
    if args.system == "prove_pev":
        if args.prove_n:
            os.environ["PROVE_N"] = str(args.prove_n)
        if args.prove_granularity:
            os.environ["PROVE_GRANULARITY"] = str(args.prove_granularity)
        if args.prove_disabled:
            os.environ["PROVE_ENABLED"] = "0"
        os.environ.setdefault("PROVE_CANDIDATE_TEMPERATURE", "0.7")
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
            f"tools={row['tool_calls']} llm={row['llm_calls']} "
            f"recovery={row['recovery']} prove={row['prove_selected']} seconds={row['seconds']}",
            flush=True,
        )
    correct = sum(1 for row in rows if row["correct"])
    calls = [row["tool_calls"] for row in rows]
    llm_calls = [int(row.get("llm_calls") or 0) for row in rows]
    metric_rows = [row.get("cghe_metrics") or {} for row in rows if isinstance(row.get("cghe_metrics"), dict)]
    def _mean_metric(key: str) -> float:
        vals = [float(item.get(key) or 0) for item in metric_rows]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    sufficient_correct = sum(
        1 for row in rows
        if row.get("correct") and int((row.get("cghe_metrics") or {}).get("answer_correct_given_sufficient") or 0)
    )
    sufficient_n = sum(
        1 for row in rows
        if int((row.get("cghe_metrics") or {}).get("evidence_sufficient_steps") or 0) > 0
        or int((row.get("cghe_metrics") or {}).get("answer_correct_given_sufficient") or 0)
    )
    report = {
        "system": args.system,
        "ablation": args.ablation,
        "split": args.split,
        "n": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "mean_tool_calls": round(sum(calls) / len(calls), 4) if calls else 0.0,
        "mean_llm_calls": round(sum(llm_calls) / len(llm_calls), 4) if llm_calls else 0.0,
        "recovery_questions": sum(1 for row in rows if row["recovery"]),
        "prove_selected_questions": sum(1 for row in rows if row.get("prove_selected")),
        "prove_n": int(os.environ.get("PROVE_N") or 0) if args.system == "prove_pev" else None,
        "prove_granularity": os.environ.get("PROVE_GRANULARITY") if args.system == "prove_pev" else None,
        "p0_metrics": {
            "mean_dependency_ready_rate": _mean_metric("dependency_ready_rate"),
            "mean_premature_extract_rate": _mean_metric("premature_extract_rate"),
            "mean_evidence_sufficient_rate": _mean_metric("evidence_sufficient_rate"),
            "answer_correct_given_sufficient": sufficient_correct,
            "questions_with_sufficient_evidence": sufficient_n,
            "accuracy_given_sufficient": (
                round(sufficient_correct / sufficient_n, 4) if sufficient_n else None
            ),
        },
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {out} correct={correct}/{len(rows)} "
        f"mean_tools={report['mean_tool_calls']} mean_llm={report['mean_llm_calls']}",
        flush=True,
    )


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


def compare3(args: argparse.Namespace) -> None:
    pure = json.loads(Path(args.pure).read_text(encoding="utf-8"))
    prove = json.loads(Path(args.prove).read_text(encoding="utf-8"))
    cghe = json.loads(Path(args.cghe).read_text(encoding="utf-8"))
    left = _index(pure)
    mid = _index(prove)
    right = _index(cghe)
    shared = sorted(set(left) & set(mid) & set(right))

    def mean_tools(index: Dict[int, Dict[str, Any]]) -> float:
        if not shared:
            return 0.0
        return sum(index[idx]["tool_calls"] for idx in shared) / len(shared)

    def mean_llm(index: Dict[int, Dict[str, Any]]) -> float:
        if not shared:
            return 0.0
        return sum(int(index[idx].get("llm_calls") or 0) for idx in shared) / len(shared)

    summary = {
        "pure": args.pure,
        "prove": args.prove,
        "cghe": args.cghe,
        "n": len(shared),
        "pure_correct": sum(1 for idx in shared if left[idx]["correct"]),
        "prove_correct": sum(1 for idx in shared if mid[idx]["correct"]),
        "cghe_correct": sum(1 for idx in shared if right[idx]["correct"]),
        "prove_vs_pure_gain": (
            sum(1 for idx in shared if mid[idx]["correct"])
            - sum(1 for idx in shared if left[idx]["correct"])
        ),
        "cghe_vs_pure_gain": (
            sum(1 for idx in shared if right[idx]["correct"])
            - sum(1 for idx in shared if left[idx]["correct"])
        ),
        "cghe_vs_prove_gain": (
            sum(1 for idx in shared if right[idx]["correct"])
            - sum(1 for idx in shared if mid[idx]["correct"])
        ),
        "pure_mean_tool_calls": round(mean_tools(left), 4),
        "prove_mean_tool_calls": round(mean_tools(mid), 4),
        "cghe_mean_tool_calls": round(mean_tools(right), 4),
        "pure_mean_llm_calls": round(mean_llm(left), 4),
        "prove_mean_llm_calls": round(mean_llm(mid), 4),
        "cghe_mean_llm_calls": round(mean_llm(right), 4),
        "prove_better_than_pure": [
            idx for idx in shared if mid[idx]["correct"] and not left[idx]["correct"]
        ],
        "cghe_better_than_prove": [
            idx for idx in shared if right[idx]["correct"] and not mid[idx]["correct"]
        ],
        "prove_selected": sum(1 for idx in shared if mid[idx].get("prove_selected")),
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
    rows = [
        row for row in rows
        if int(row.get("round", -1)) != int(entry["round"]) or row.get("ablation") != entry.get("ablation")
    ]
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
    run.add_argument("--prove-n", type=int, default=0)
    run.add_argument(
        "--prove-granularity",
        default="",
        choices=["", "plan", "plan_and_command", "iteration"],
    )
    run.add_argument("--prove-disabled", action="store_true")
    run.set_defaults(func=worker)

    diff = sub.add_parser("compare")
    diff.add_argument("--pure", required=True)
    diff.add_argument("--cghe", required=True)
    diff.add_argument("--summary", default="")
    diff.set_defaults(func=compare)

    three = sub.add_parser("compare3")
    three.add_argument("--pure", required=True)
    three.add_argument("--prove", required=True)
    three.add_argument("--cghe", required=True)
    three.add_argument("--summary", default="")
    three.set_defaults(func=compare3)

    book = sub.add_parser("ledger")
    book.add_argument("--entry", required=True)
    book.set_defaults(func=ledger_append)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
