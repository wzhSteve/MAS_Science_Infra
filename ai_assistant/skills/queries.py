"""Read-only fact queries backing the assistant skills.

Everything here reuses the same control-plane functions the UI calls, so the
assistant's advice is grounded in the real machine state and the experiment
YAMLs instead of model guesses. Never mutates anything.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[redacted]" if re.search(r"key|secret|password|token", str(key), re.I) else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"sk-[A-Za-z0-9]{8,}", "[redacted]", value)
    return value


def query_gpu_env(experiment_id: Optional[str] = None) -> str:
    """Real GPU inventory + the experiment's derived training plan (read-only).

    The plan call is wrapped so a broken experiment never hides the GPU facts.
    """
    from science_infra.control.services import list_gpus

    gpus = list_gpus()
    payload: Dict[str, Any] = {"gpus": gpus.get("gpus") or [], "count": gpus.get("count") or 0}
    if gpus.get("error"):
        payload["gpu_error"] = str(gpus["error"])
    try:
        from science_infra.control.experiments import recommend_rl

        payload["recommendation_by_gpu_count"] = recommend_rl(payload["count"] or 1)
    except Exception as exc:  # noqa: BLE001
        payload["recommendation_error"] = str(exc)
    if experiment_id:
        try:
            from science_infra.control.training import TrainingError, build_training_plan

            plan = build_training_plan(experiment_id)
            payload["experiment"] = {
                "id": experiment_id,
                "ready": plan.ready,
                "algorithm": plan.algorithm,
                "profile": plan.profile,
                "gpu_ids": plan.gpu_ids,
                "group_n": plan.group_n,
                "trainable_agents": plan.trainable_agents,
                "active_agents": plan.active_agents,
                "model_path": plan.source.model_path,
                "checks": plan.checks,
                "blocking_issues": plan.blocking_issues,
                "warnings": plan.warnings,
                "launch_preview": plan.launch_preview,
            }
        except TrainingError as exc:
            payload["experiment"] = {"id": experiment_id, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            payload["experiment"] = {"id": experiment_id, "error": str(exc)}
    return json.dumps(_redact(payload), ensure_ascii=False, indent=1)


def query_run(experiment_id: str, run_id: Optional[str] = None, tail_lines: int = 160) -> str:
    """Run status (+ stdout tail) for diagnosis. Defaults to the newest train/eval run."""
    from science_infra.control.process_manager import PROCS

    rows = [row for row in PROCS.list_runs(experiment_id) if row.get("kind") in ("train", "eval")]
    rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
    if run_id:
        row = PROCS.status(run_id, experiment_id)
        if row is None:
            return json.dumps({"error": f"run {run_id} 不存在", "runs": rows[:8]}, ensure_ascii=False)
    else:
        if not rows:
            return json.dumps({"runs": [], "hint": "该实验还没有训练/评测记录。"}, ensure_ascii=False)
        row = rows[0]
    run_id = str(row.get("run_id") or "")
    payload: Dict[str, Any] = {"run": row, "recent_runs": rows[:8]}
    log_dir = PROCS.run_dir(experiment_id, run_id)
    stdout = Path(log_dir) / "stdout.log"
    if stdout.is_file():
        try:
            lines = stdout.read_text(encoding="utf-8", errors="replace").splitlines()
            payload["stdout_tail"] = "\n".join(lines[-max(20, int(tail_lines)):])
        except OSError as exc:
            payload["stdout_tail_error"] = str(exc)
    else:
        payload["stdout_tail"] = "(stdout.log 尚未生成)"
    return json.dumps(_redact(payload), ensure_ascii=False, indent=1)


def query_experiment(experiment_id: str) -> str:
    """Summarize the experiment's five YAML sections (llm/workflow/rl/harness)."""
    from science_infra.control.experiments import load_bundle

    bundle = load_bundle(experiment_id)
    workflow = bundle.get("workflow") or {}
    agents = [
        {"id": a.get("id"), "kind": a.get("kind"), "model": a.get("model")}
        for a in (workflow.get("agents") or [])
        if isinstance(a, dict)
    ]
    sampling = workflow.get("sampling") if isinstance(workflow.get("sampling"), dict) else {}
    rl = bundle.get("rl") or {}
    llm = bundle.get("llm") or {}
    payload = {
        "experiment_id": experiment_id,
        "llm": {"kind": llm.get("kind"), "model": llm.get("model"), "base_url": llm.get("base_url")},
        "workflow": {
            "entry_agent": workflow.get("entry_agent"),
            "agents": agents,
            "routers": [r.get("id") for r in (workflow.get("routers") or []) if isinstance(r, dict)],
            "sampling": {
                "mode": sampling.get("mode"),
                "group_n": sampling.get("group_n"),
                "beam_size": sampling.get("beam_size"),
                "sites": sampling.get("sites") or [],
            },
        },
        "rl": {
            key: rl.get(key)
            for key in ("algo", "profile", "n_runners", "rollout_per_gpu", "devices", "model_path")
            if key in rl
        }
        | {"data": rl.get("data") or {}},
        "harness": bundle.get("harness") or {},
    }
    return json.dumps(_redact(payload), ensure_ascii=False, indent=1)


def skill_queries() -> List[Dict[str, Any]]:
    return [
        {"name": "query_gpu_env", "fn": query_gpu_env},
        {"name": "query_run", "fn": query_run},
        {"name": "query_experiment", "fn": query_experiment},
    ]
