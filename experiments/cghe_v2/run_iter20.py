"""Fixed-sample Hotpot20 + Bamboogle20 runner for CGHE iteration rounds.

Examples
--------
python experiments/cghe_v2/run_iter20.py --system cghe_pev --round R0 --smoke
python experiments/cghe_v2/run_iter20.py --system pure_pev --round R0
python experiments/cghe_v2/run_iter20.py --system cghe_pev --round R5 --ablation no_role_bind
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
EXP = ROOT / "experiments" / "cghe_v2" / "iter20"
SAMPLE_PATH = EXP / "sample_ids.json"
DATASETS = {
    "hotpot": ROOT / "data" / "hotpotqa.json",
    "bamboogle": ROOT / "data" / "bamboogle.json",
}
PROJECTS = {
    "pure_pev": ROOT / "user_space" / "projects" / "pure_pev",
    "cghe_pev": ROOT / "user_space" / "projects" / "cghe_pev",
}
_ANSWER_TAIL = re.compile(r"(?is)\s*when ready, output the final answer.*$")


def _clean_question(text: Any) -> str:
    return _ANSWER_TAIL.sub("", str(text or "")).strip()


def _load_sample() -> Dict[str, Any]:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def _items_for(dataset: str, ids: List[int]) -> List[Dict[str, Any]]:
    rows = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
    by_idx = {int(row["idx"]): row for row in rows}
    out = []
    for idx in ids:
        row = dict(by_idx[idx])
        row["question"] = _clean_question(row.get("question") or row.get("query") or "")
        row["dataset"] = dataset
        out.append(row)
    return out


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
        "cghe_frontier": trace.get("cghe_frontier") or [],
        "cghe_feedback": trace.get("cghe_feedback") or [],
        "cghe_metrics": trace.get("cghe_metrics") or {},
        "cghe_select_policy": trace.get("cghe_select_policy", ""),
        "conclusion": trace.get("conclusion", ""),
        "llm_calls": int(calls.get("count") or 0),
    }
    return {key: value for key, value in kept.items() if value not in ("", 0, None, [], {})}


def _prepare_imports(system: str) -> None:
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
    adapted = PROJECTS[system] / "adapted"
    for path in (str(ROOT), str(ROOT / "mas"), str(adapted)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["AGENTFLOW_TEMPERATURE"] = "0"
    os.environ["AGENTFLOW_MAX_STEPS"] = os.environ.get("AGENTFLOW_MAX_STEPS", "10")
    os.environ["AGENTFLOW_TOP_LOGPROBS"] = "5"
    os.environ.setdefault("AGENTFLOW_LLM_TIMEOUT", "120")
    from workflow.env_load import load_repo_dotenv

    load_repo_dotenv()


def _apply_ablation(ablation: str) -> None:
    os.environ.pop("CGHE_NO_ROLE_BIND", None)
    os.environ.pop("CGHE_NO_EXTRACT_PRUNE", None)
    os.environ["CGHE_SELECT_POLICY"] = "max_out_degree"
    if ablation == "no_role_bind":
        os.environ["CGHE_NO_ROLE_BIND"] = "1"
    elif ablation == "frontier":
        os.environ["CGHE_SELECT_POLICY"] = "frontier"
    elif ablation == "no_extract_prune":
        os.environ["CGHE_NO_EXTRACT_PRUNE"] = "1"
    elif ablation not in {"full", ""}:
        raise SystemExit(f"unknown ablation: {ablation}")


def _run_question(ctx: Any, item: Dict[str, Any], system: str, ablation: str, max_steps: int) -> Dict[str, Any]:
    from workflow.user_gateway.sandbox import bind_episode_id, reset_episode_id

    dataset = str(item.get("dataset") or "hotpot")
    token = bind_episode_id(f"iter20-{dataset}-{system}-{ablation}-{item['idx']}")
    payload = {"question": item["question"], "source": dataset}
    steps: List[Dict[str, Any]] = []
    answer = ""
    error = ""
    started = time.time()
    try:
        for _ in range(max_steps):
            print(f"{dataset} idx={item['idx']} plan", flush=True)
            plan = ctx.plan_window({"payload": payload})
            body = plan.get("payload") or {}
            args = body.get("args") if isinstance(body.get("args"), dict) else {}
            trace = body.get("trace") if isinstance(body.get("trace"), dict) else {}
            print(
                f"{dataset} idx={item['idx']} tool={args.get('tool_name','')} "
                f"mode={trace.get('cghe_mode','')}",
                flush=True,
            )
            steps.append({
                "phase": "plan",
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
    golds = item.get("answer") or []
    if isinstance(golds, str):
        golds = [golds]
    golds = [str(g) for g in golds]
    tool_calls = sum(1 for step in steps if step["phase"] == "exec")
    llm_calls = sum(int((step.get("trace") or {}).get("llm_calls") or 0) for step in steps)
    metrics: Dict[str, Any] = {}
    for step in reversed(steps):
        trace = step.get("trace") if isinstance(step.get("trace"), dict) else {}
        if isinstance(trace.get("cghe_metrics"), dict) and trace.get("cghe_metrics"):
            metrics = dict(trace["cghe_metrics"])
            break
    return {
        "dataset": dataset,
        "idx": int(item["idx"]),
        "question": item["question"],
        "gold": golds,
        "answer": answer,
        "correct": _local_correct(answer, golds),
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "seconds": round(time.time() - started, 1),
        "cghe_metrics": metrics,
        "error": error,
        "steps": steps,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "mean_tool_calls": round(sum(int(row.get("tool_calls") or 0) for row in rows) / n, 4) if n else 0.0,
        "mean_llm_calls": round(sum(int(row.get("llm_calls") or 0) for row in rows) / n, 4) if n else 0.0,
        "mean_seconds": round(sum(float(row.get("seconds") or 0) for row in rows) / n, 2) if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=sorted(PROJECTS))
    parser.add_argument("--round", required=True)
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--dataset", default="both", choices=["both", "hotpot", "bamboogle"])
    args = parser.parse_args()

    sample = _load_sample()
    if args.smoke:
        hot_ids = list(sample["smoke_hotpot_ids"])
        bam_ids = list(sample["smoke_bamboogle_ids"])
    else:
        hot_ids = list(sample["hotpot_ids"])
        bam_ids = list(sample["bamboogle_ids"])

    items: List[Dict[str, Any]] = []
    if args.dataset in {"both", "hotpot"}:
        items.extend(_items_for("hotpot", hot_ids))
    if args.dataset in {"both", "bamboogle"}:
        items.extend(_items_for("bamboogle", bam_ids))

    os.environ["CGHE_ABLATION"] = "full"
    _apply_ablation(args.ablation if args.system == "cghe_pev" else "full")
    _prepare_imports(args.system)
    import context as ctx_mod

    if args.system == "cghe_pev":
        os.environ["CGHE_SQLITE_PATH"] = str(
            EXP / args.round / f"{args.system}_{args.ablation}.sqlite"
        )
        if args.ablation != "no_memory":
            db = Path(ctx_mod.cghe_settings()["sqlite_path"])
            if db.exists():
                db.unlink()

    ctx = ctx_mod.get_episode_context()
    ctx.components()
    rows = []
    for item in items:
        row = _run_question(ctx, item, args.system, args.ablation, args.max_steps)
        rows.append(row)
        print(
            f"{args.system}:{args.ablation} {row['dataset']} idx={row['idx']} "
            f"correct={row['correct']} tools={row['tool_calls']} "
            f"llm={row['llm_calls']} seconds={row['seconds']}",
            flush=True,
        )

    by_ds = {
        "hotpot": [row for row in rows if row["dataset"] == "hotpot"],
        "bamboogle": [row for row in rows if row["dataset"] == "bamboogle"],
    }
    report = {
        "system": args.system,
        "round": args.round,
        "ablation": args.ablation,
        "smoke": bool(args.smoke),
        "max_steps": args.max_steps,
        "sample": sample,
        "overall": _summarize(rows),
        "by_dataset": {name: _summarize(items) for name, items in by_ds.items()},
        "rows": rows,
    }
    out_dir = EXP / args.round
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out = out_dir / f"{args.system}_{args.ablation}_{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {out} overall={report['overall']['correct']}/{report['overall']['n']} "
        f"hotpot={report['by_dataset']['hotpot']['accuracy']} "
        f"bamboogle={report['by_dataset']['bamboogle']['accuracy']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
