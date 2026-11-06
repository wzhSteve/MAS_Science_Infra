"""Full QA eval for pure_pev with incremental jsonl.

Example
-------
python adapted/eval_full_qa.py --dataset hotpotqa --out artifacts/full_qa_run
python adapted/eval_full_qa.py --dataset musique --out artifacts/full_qa_run
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
PROJECT = ROOT / "user_space" / "projects" / "pure_pev"
ADAPTED = PROJECT / "adapted"
DATASETS = {
    "hotpotqa": ROOT / "data" / "hotpotqa.json",
    "bamboogle": ROOT / "data" / "bamboogle.json",
    "musique": ROOT / "data" / "musique.json",
    "2wiki": ROOT / "data" / "2wiki.json",
}
# Episode payload source tags used by wrap_answer / scoring branches.
SOURCE_TAGS = {
    "hotpotqa": "hotpot",
    "bamboogle": "bamboogle",
    "musique": "musique",
    "2wiki": "2wiki",
}
_ANSWER_TAIL = re.compile(r"(?is)\s*when ready, output the final answer.*$")

for path in (str(ROOT), str(ROOT / "mas"), str(ADAPTED)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _clean_question(text: Any) -> str:
    return _ANSWER_TAIL.sub("", str(text or "")).strip()


def _parse_golds(raw: Any) -> List[str]:
    if isinstance(raw, list):
        return [str(x) for x in raw]
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        if parsed is not None:
            return [str(parsed)]
    except Exception:
        pass
    return [text]


def _local_correct(prediction: str, golds: List[str]) -> bool:
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


def _load_items(dataset: str) -> List[Dict[str, Any]]:
    rows = json.loads(DATASETS[dataset].read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["question"] = _clean_question(item.get("question") or item.get("query") or "")
        item["golds"] = _parse_golds(item.get("answer"))
        item["dataset"] = dataset
        raw_idx = item.get("idx")
        if raw_idx is None:
            raw_idx = item.get("pid")
        item["idx"] = int(raw_idx) if raw_idx is not None else len(out)
        out.append(item)
    return out


def _done_idxs(jsonl_path: Path) -> Set[int]:
    done: Set[int] = set()
    if not jsonl_path.is_file():
        return done
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if "idx" in row:
            done.add(int(row["idx"]))
    return done


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _configure_env(serp_account: str) -> None:
    from workflow.env_load import load_repo_dotenv

    load_repo_dotenv()
    os.environ["AGENTFLOW_SFT_PROMPTS"] = "1"
    os.environ["AGENTFLOW_TOP_LOGPROBS"] = "5"
    os.environ.setdefault("AGENTFLOW_MAX_STEPS", "10")
    os.environ.setdefault("AGENTFLOW_LLM_TIMEOUT", "180")
    os.environ.setdefault("enable_thinking", "false")
    # Pin LLM to .env OPENAI_* / MODEL_Name (dmxapi + qwen3.5-27b).
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_API_BASE_URL",
        "MODEL_Name",
        "OPENAI_MODEL",
        "SERVER_MODEL",
    ):
        val = os.environ.get(key, "").strip()
        if val:
            os.environ[key] = val
    if os.environ.get("MODEL_Name"):
        os.environ.setdefault("OPENAI_MODEL", os.environ["MODEL_Name"])
        os.environ.setdefault("SERVER_MODEL", os.environ["MODEL_Name"])
    # secondary (default for this eval): remap KEY_2 / ZONES_2 onto the
    # names mas.tools.search actually reads. primary keeps the main account.
    if serp_account == "secondary":
        key = os.environ.get("BRIGHT_DATA_API_KEY_2", "").strip()
        zones = os.environ.get("BRIGHT_DATA_SERP_ZONES_2", "").strip() or "serp_api1"
        if not key:
            raise SystemExit("BRIGHT_DATA_API_KEY_2 is empty; refuse to fall back to primary")
        os.environ["BRIGHT_DATA_API_KEY"] = key
        os.environ["BRIGHT_DATA_SERP_ZONES"] = zones
        os.environ["BRIGHT_DATA_SERP_ZONE"] = zones.split(",")[0].strip()



def _preflight_search() -> None:
    from mas.tools import search as S
    from tools_local import Bing_Search_Tool

    print(
        f"proxy={S._env_proxy()} alive={S.proxy_alive(ttl=0)} "
        f"key_set={bool(os.environ.get('BRIGHT_DATA_API_KEY'))} "
        f"zones={os.environ.get('BRIGHT_DATA_SERP_ZONES')}",
        flush=True,
    )
    out = str(Bing_Search_Tool().execute(query="HotpotQA wikipedia") or "")
    print(f"preflight_len={len(out)} head={out[:160]!r}", flush=True)
    if "search_unavailable" in out.lower() or "ProxyError" in out:
        raise SystemExit("Bing preflight failed")


def _run_one(ctx: Any, item: Dict[str, Any], max_steps: int) -> Dict[str, Any]:
    from workflow.user_gateway.sandbox import bind_episode_id, reset_episode_id
    from rl.rewards.outcome import extract_answer_text

    dataset = item["dataset"]
    source = SOURCE_TAGS.get(dataset, dataset)
    token = bind_episode_id(f"full-{dataset}-{item['idx']}")
    payload = {"question": item["question"], "source": source}
    answer = ""
    error = ""
    steps: List[Dict[str, Any]] = []
    obtained: List[Any] = []
    outline: List[Any] = []
    started = time.time()
    try:
        for _ in range(max_steps):
            plan = ctx.plan_window({"payload": payload})
            body = plan.get("payload") or {}
            if body.get("done") or body.get("ready_to_stop"):
                answer = str(body.get("answer") or "")
                break
            ctx.exec_window({"payload": payload})
            verify = ctx.verify_window({"payload": payload})
            vbody = verify.get("payload") or {}
            ep = ctx.episode(payload)
            actions = ep.memory.get_actions() or {}
            key = f"Action Step {ep.step_count}"
            act = actions.get(key) if isinstance(actions, dict) else None
            if isinstance(act, dict):
                steps.append({
                    "step": ep.step_count,
                    "sub_goal": str(act.get("sub_goal") or "")[:300],
                    "tool_name": str(act.get("tool_name") or ""),
                    "actual_satisfied": act.get("actual_satisfied"),
                    "conclusion": act.get("conclusion"),
                    "verifier_reason": str(act.get("verifier_reason") or "")[:240],
                })
            if vbody.get("ready_to_stop"):
                answer = str(vbody.get("answer") or "")
                break
        episode = ctx.episode(payload)
        obtained = list(getattr(episode.memory, "obtained_information", None) or [])
        outline = list(getattr(episode.memory, "outline", None) or [])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        reset_episode_id(token)

    extracted = extract_answer_text(answer) or ""
    correct = False if (error and not extracted) else _local_correct(answer, item["golds"])
    return {
        "dataset": dataset,
        "idx": item["idx"],
        "question": item["question"],
        "gold": item["golds"],
        "answer": extracted,
        "raw_answer": str(answer or "")[:500],
        "correct": bool(correct),
        "seconds": round(time.time() - started, 1),
        "n_steps": len(steps),
        "steps": steps,
        "obtained": obtained[:8],
        "outline": outline,
        "error": error,
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    errors = sum(1 for row in rows if row.get("error"))
    return {
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "errors": errors,
        "mean_seconds": round(sum(float(row.get("seconds") or 0) for row in rows) / n, 2) if n else 0.0,
        "mean_steps": round(sum(int(row.get("n_steps") or 0) for row in rows) / n, 2) if n else 0.0,
    }


def _read_all_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not jsonl_path.is_file():
        return rows
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    rows.sort(key=lambda row: int(row.get("idx") or 0))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--out", required=True, help="Output directory for jsonl + summary")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument(
        "--serp-account",
        choices=["primary", "secondary"],
        default="primary",
        help="primary=BRIGHT_DATA_API_KEY; secondary=KEY_2",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke tests")
    args = parser.parse_args()

    _configure_env(args.serp_account)
    os.environ["AGENTFLOW_MAX_STEPS"] = str(args.max_steps)

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{args.dataset}.jsonl"
    summary_path = out_dir / f"{args.dataset}_summary.json"

    items = _load_items(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]
    done = _done_idxs(jsonl_path)
    todo = [item for item in items if item["idx"] not in done]
    print(
        f"dataset={args.dataset} total={len(items)} done={len(done)} todo={len(todo)} "
        f"serp={args.serp_account} out={out_dir}",
        flush=True,
    )

    if not args.skip_preflight:
        _preflight_search()

    import context as ctx_mod

    ctx = ctx_mod.get_episode_context()
    ctx.components()

    for item in todo:
        row = _run_one(ctx, item, args.max_steps)
        _append_jsonl(jsonl_path, row)
        print(
            f"{args.dataset} idx={row['idx']} correct={row['correct']} "
            f"steps={row['n_steps']} sec={row['seconds']} answer={row['answer'][:80]!r} "
            f"err={str(row.get('error') or '')[:100]!r}",
            flush=True,
        )

    rows = _read_all_rows(jsonl_path)
    # Keep only idxs that belong to this run's item list (supports --limit).
    wanted = {item["idx"] for item in items}
    rows = [row for row in rows if int(row.get("idx") or -1) in wanted]
    summary = {
        "dataset": args.dataset,
        "max_steps": args.max_steps,
        "serp_account": args.serp_account,
        "model": os.environ.get("MODEL_Name") or os.environ.get("OPENAI_MODEL"),
        "base": os.environ.get("OPENAI_API_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        "bright_zones": os.environ.get("BRIGHT_DATA_SERP_ZONES"),
        "jsonl": str(jsonl_path),
        **_summarize(rows),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"DONE {args.dataset} accuracy={summary['accuracy']} "
        f"correct={summary['correct']}/{summary['n']} -> {summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
