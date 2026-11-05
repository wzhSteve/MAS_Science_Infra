"""Run 5 HotpotQA items through the same path as MAS Infra UI eval.

Logs go to stdout, artifacts/cghe_ui5_run.log, and (when possible) an inline
PROCS run so the MAS console can tail them.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PROJECT = Path(__file__).resolve().parents[1]
ADAPTED = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "mas"), str(ADAPTED)]

from workflow.env_load import load_repo_dotenv

load_repo_dotenv()
os.environ.setdefault("AGENTFLOW_TOP_LOGPROBS", "5")
os.environ.setdefault("AGENTFLOW_MAX_STEPS", "4")
os.environ.setdefault("AGENTFLOW_MAX_TIME", "240")
os.environ.setdefault("CGHE_N", "4")

LOG_PATH = PROJECT / "artifacts" / "cghe_ui5_run.log"
RESULT_PATH = PROJECT / "artifacts" / "cghe_ui5_results.json"
WORKFLOW_PATH = PROJECT / "contracts" / "workflow.yaml"
EXPERIMENT_ID = os.environ.get("CGHE_UI_EXPERIMENT") or "demo"


def _llm():
    from workflow.runtime import LLMConfig

    return LLMConfig(
        endpoint=(
            os.environ.get("OPENAI_API_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).rstrip("/"),
        model=os.environ.get("MODEL_Name") or os.environ.get("OPENAI_MODEL") or "qwen3.5-27b",
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
    )


def _tasks(limit: int = 5):
    from science_infra.control.task_files import load_task_rows

    rows = load_task_rows(ROOT / "data" / "hotpotqa.json")
    return rows[:limit]


def _workflow():
    import yaml

    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def main() -> int:
    from science_infra.control.process_manager import PROCS
    from science_infra.control.rollout_runs import _execute_question

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")

    procs_run = None
    try:
        procs_run = PROCS.start_inline(
            kind="eval",
            experiment_id=EXPERIMENT_ID,
            meta={"project": "cghe_pev", "tag": "hotpot_ui5", "n": 5},
        )
        PROCS.append_log(EXPERIMENT_ID, procs_run["run_id"], "==== CGHE Hotpot UI5 start ====")
        print(f"UI run_id={procs_run['run_id']} experiment={EXPERIMENT_ID}", flush=True)
        print(f"log_path={procs_run.get('log_path')}", flush=True)
    except Exception as exc:
        print(f"PROCS inline unavailable ({type(exc).__name__}: {exc}); stdout+file only", flush=True)

    workflow = _workflow()
    llm = _llm()
    tasks = _tasks(5)
    results = []

    def _sink(line: str, *, index: int = 0, total: int = 5) -> None:
        text = str(line or "")
        if not text.strip():
            return
        prefixed = f"[{index}/{total}] {text}" if not text.startswith("[") else text
        print(prefixed, flush=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(prefixed + "\n")
        if procs_run is not None:
            try:
                PROCS.append_log(EXPERIMENT_ID, procs_run["run_id"], prefixed)
            except Exception:
                pass

    for index, task in enumerate(tasks, 1):
        q = str(task.get("question") or "")[:160]
        _sink("", index=index, total=len(tasks))
        _sink(f"==== question {index}/{len(tasks)} ====", index=index, total=len(tasks))
        _sink(f"Q: {q}", index=index, total=len(tasks))
        gold = str(task.get("answer") or "")

        def on_window(line: str, _i: int = index, _n: int = len(tasks)) -> None:
            _sink(line, index=_i, total=_n)

        try:
            summary, traj = _execute_question(
                EXPERIMENT_ID,
                workflow,
                task,
                llm=llm,
                model_summary=None,
                model_public=None,
                on_window=on_window,
            )
            answer = str(summary.get("final_answer") or "")
            error = ""
            if isinstance(summary.get("error"), dict):
                error = str(summary["error"].get("message") or "")
            hit = bool(gold) and gold.casefold() in answer.casefold()
        except Exception as exc:
            summary, traj, answer, error, hit = {}, None, "", f"{type(exc).__name__}: {exc}", False
            _sink(f"ERROR {error}", index=index, total=len(tasks))

        _sink(f"gold={gold}", index=index, total=len(tasks))
        _sink(f"answer={answer[:240]}", index=index, total=len(tasks))
        _sink(f"hit={hit}", index=index, total=len(tasks))
        results.append({
            "index": index - 1,
            "id": task.get("id"),
            "question": task.get("question"),
            "gold": gold,
            "answer": answer,
            "hit": hit,
            "error": error,
            "status": (summary or {}).get("status"),
        })

    RESULT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = sum(1 for row in results if row.get("hit"))
    summary_line = f"DONE n={len(results)} hits={hits} em={hits / max(1, len(results)):.3f} log={LOG_PATH}"
    _sink(summary_line, index=len(tasks), total=len(tasks))
    if procs_run is not None:
        try:
            PROCS.finish_inline(
                EXPERIMENT_ID,
                procs_run["run_id"],
                state="succeeded",
                message=summary_line,
                returncode=0,
            )
        except Exception:
            pass
    print(json.dumps({"experiment": EXPERIMENT_ID, "run": procs_run, "hits": hits, "n": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
