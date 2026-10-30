#!/usr/bin/env python3
"""Offline CGHE recovery demo.

The default command replays a local planner → executor → verifier → recovery
trajectory and does not call a model or the network. ``--live`` is opt-in and
is not used by the default path.

The repo ``.env`` is loaded with the same ``load_repo_dotenv`` helper as
``entry.py``. Printed text is scrubbed of credential-shaped environment values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

_ADAPTED = Path(__file__).resolve().parent
_PROJECT = _ADAPTED.parent
_REPO = _ADAPTED.parents[3]
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PROXY", "CREDENTIAL")
_OFFLINE_COMMAND = "python user_space/projects/cghe_pev/adapted/run_cghe_example.py"
_LIVE_COMMAND = "python user_space/projects/cghe_pev/adapted/run_cghe_example.py --live"
_QUESTION = "Where is the capital of France?"
_VERIFIER_NO = (
    "Conclusion: CONTINUE\n"
    "Satisfied: NO\n"
    "Reason: the search tool is missing\n"
    "Obtained: NONE\n"
)


def load_env() -> bool:
    """Load the repo ``.env`` with ``entry.py``'s ``load_repo_dotenv``.

    The repo root stays on the path so ``workflow`` can import ``rl`` the same
    way a MAS process does. The loader module is executed directly when that
    package import is unavailable, and it is still ``workflow/env_load.py``.
    """
    mas = _REPO / "mas"
    for path in (str(_REPO), str(mas)):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        from workflow.env_load import _candidate_paths, load_repo_dotenv
    except Exception:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "cghe_env_load",
            mas / "workflow" / "env_load.py",
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("workflow.env_load is missing")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        load_repo_dotenv = module.load_repo_dotenv
        _candidate_paths = module._candidate_paths
    # workflow.collector calls this helper while the package is imported, so a
    # second call returns None even though the repo file was already applied.
    if load_repo_dotenv() is not None:
        return True
    return any(path.is_file() for path in _candidate_paths())


def redact(text: str) -> str:
    """Remove credential-shaped environment values from text that will be printed."""
    cleaned = str(text or "")
    for name, value in os.environ.items():
        upper = name.upper()
        if not any(marker in upper for marker in _SECRET_MARKERS):
            continue
        secret = str(value or "")
        if len(secret) < 6:
            continue
        cleaned = cleaned.replace(secret, "[redacted]")
    return cleaned


def _candidate_payload() -> Dict[str, Any]:
    def step(tool: str, subgoal: str, cost: float) -> Dict[str, Any]:
        return {
            "subgoal": subgoal,
            "action": subgoal,
            "tool": tool,
            "preconditions": [{"predicate": "ToolExists", "tool": tool}],
            "required_state": [],
            "estimated_cost": cost,
        }

    fetch = step("Web_Fetch_Tool", "open the page", 4.0)
    return {
        "plans": [
            {
                "plan_id": "bing",
                "steps": [step("Bing_Search_Tool", "search the capital", 1.0)],
            },
            {"plan_id": "fetch_a", "steps": [fetch]},
            {"plan_id": "fetch_b", "steps": [dict(fetch)]},
            {
                "plan_id": "code",
                "steps": [step("Python_Code_Generator_Tool", "compute locally", 0.2)],
            },
        ]
    }


def candidate_json() -> str:
    return json.dumps(_candidate_payload(), ensure_ascii=False)


class _Planner:
    def __init__(self, reply_box: Dict[str, str]) -> None:
        self.available_tools = [
            "Generalist_Solution_Generator_Tool",
            "Python_Code_Generator_Tool",
            "Bing_Search_Tool",
            "Web_Fetch_Tool",
        ]
        self.reply_box = reply_box
        self.generate_calls: List[str] = []
        self.engine_calls: List[str] = []
        self._last_plan: Dict[str, Any] = {}

    def analyze_query(self, question: str, image: Any) -> str:
        return "mock analysis"

    def generate_next_step(self, question: str, *args: Any, **kwargs: Any) -> str:
        self.generate_calls.append(str(question or ""))
        self._last_plan = {
            "outline": [],
            "subgoal": "find the capital",
            "tool_name": "Bing_Search_Tool",
            "tool_input": "capital of France",
            "prediction": "",
            "prediction_reason": "",
        }
        return (
            "Context: capital of France\n"
            "Sub-Goal: find the capital\n"
            "Tool Name: Bing_Search_Tool\n"
        )

    def extract_context_subgoal_and_tool(self, response: Any) -> tuple:
        context = subgoal = tool = ""
        for line in str(response or "").splitlines():
            if line.startswith("Context:"):
                context = line.split(":", 1)[1].strip()
            elif line.startswith("Sub-Goal:"):
                subgoal = line.split(":", 1)[1].strip()
            elif line.startswith("Tool Name:"):
                tool = line.split(":", 1)[1].strip()
        return context, subgoal, tool

    def llm_engine(self, prompt: str) -> str:
        self.engine_calls.append(str(prompt or "")[:80])
        return self.reply_box.get("text") or ""

    def generate_final_output(self, *args: Any, **kwargs: Any) -> str:
        return "Paris"

    def generate_direct_output(self, *args: Any, **kwargs: Any) -> str:
        return "Paris"


class _Executor:
    def __init__(self, root: str) -> None:
        self.root_cache_dir = root
        self.executed: List[tuple] = []
        self._cache = ""

    def set_query_cache_dir(self, path: str) -> None:
        self._cache = path

    def generate_tool_command(self, *args: Any, **kwargs: Any) -> str:
        return "execution = tool.execute(query='capital of France')"

    def extract_explanation_and_command(self, tool_command: Any) -> tuple:
        return "", "", str(tool_command or "")

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        self.executed.append((str(tool_name or ""), str(command or "")))
        if "bing" in str(tool_name or "").lower():
            return {"error_code": "TOOL_NOT_FOUND", "tool": "Bing_Search_Tool"}
        return "mock page text"


class _Verifier:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def verificate_context(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        return self.text

    def extract_conclusion(self, response: Any) -> tuple:
        return "the search tool is missing", "CONTINUE"


def _message(dst: str, question: str, kind: str = "") -> Dict[str, Any]:
    body: Dict[str, Any] = {"dst": dst, "payload": {"question": question}}
    if kind:
        body["kind"] = kind
    return body


def run_window_case(
    *,
    enabled: bool,
    candidate_text: str,
    question: str = _QUESTION,
    sqlite_path: Optional[str] = None,
    verifier_text: str = _VERIFIER_NO,
) -> Dict[str, Any]:
    """Drive one mocked episode through two planner turns. No network and no model."""
    import context as context_mod
    from unittest.mock import patch

    previous_sft = os.environ.get("AGENTFLOW_SFT_PROMPTS")
    os.environ["AGENTFLOW_SFT_PROMPTS"] = "0"
    owned = sqlite_path is None
    tmp = tempfile.TemporaryDirectory(prefix="cghe_case_") if owned else None
    db_path = sqlite_path or str(Path(tmp.name) / "evidence.sqlite")  # type: ignore[union-attr]
    cache = tempfile.mkdtemp(prefix="cghe_cache_")
    reply_box = {"text": candidate_text}
    planner = _Planner(reply_box)
    executor = _Executor(cache)
    verifier = _Verifier(verifier_text)
    seen: List[Any] = []
    original_steer = context_mod._steer
    settings = {
        "enabled": bool(enabled),
        "n": 4,
        "confidence_threshold": 1.0,
        "sqlite_path": db_path,
        "recovery_rounds": 2,
        "tool_calls": 12,
        "replans": 1,
    }

    def _settings() -> Dict[str, Any]:
        return dict(settings)

    def _spy(ep: Any, outcome: Any) -> None:
        seen.append(outcome)
        return original_steer(ep, outcome)

    ctx = None
    try:
        context_mod._ensure_import()
        with patch.object(context_mod, "cghe_settings", _settings), patch.object(context_mod, "_steer", _spy):
            ctx = context_mod.EpisodeContext()
            ctx.components = lambda: _components(planner, executor, verifier)  # type: ignore[method-assign]
            ctx.pin_engines = lambda: None  # type: ignore[method-assign]
            ctx.plan_window(_message("planner", question, "plan_step"))
            ctx.exec_window(_message("executor", question))
            ctx.verify_window(_message("verifier", question, "verify"))
            continued = ctx.plan_window(_message("planner", question, "plan_step"))
            decision = ""
            episodes = list(ctx._episodes.values())
            ep = episodes[-1] if episodes else None
            if ep is not None:
                decision = str((ep._cghe_directive or {}).get("decision") or "continue_pev")
            if decision == "continue_execution":
                ctx.exec_window(_message("executor", question))
            payload = (continued or {}).get("payload") or {}
            args = payload.get("args") or {}
            active: List[str] = []
            graph_open = False
            if ep is not None and getattr(ep, "_cghe", None) is not None and ep._cghe.graph is not None:
                graph_open = True
                active = list(ep._cghe.graph.active_plan_ids())
            trace = ((continued or {}).get("payload") or {}).get("trace") or {}
            return {
                "question": question,
                "enabled": bool(enabled),
                "generate_next_step_calls": len(planner.generate_calls),
                "candidate_engine_calls": len(planner.engine_calls),
                "decisions": [str(item.decision) for item in seen],
                "task_impossible_flags": [bool(item.task_impossible) for item in seen],
                "final_decision": str(trace.get("cghe_decision") or decision or "continue_pev"),
                "final_tool": str(args.get("tool_name") or ""),
                "final_subgoal": str(args.get("sub_goal") or ""),
                "active_plan_ids": active,
                "graph_open": graph_open,
                "executed_tools": [name for name, _command in executor.executed],
                "controller_present": bool(ep is not None and ep._cghe is not None),
            }
    finally:
        if ctx is not None:
            for ep in list(ctx._episodes.values()):
                controller = getattr(ep, "_cghe", None)
                if controller is not None:
                    controller.memory.close()
        if previous_sft is None:
            os.environ.pop("AGENTFLOW_SFT_PROMPTS", None)
        else:
            os.environ["AGENTFLOW_SFT_PROMPTS"] = previous_sft
        if tmp is not None:
            tmp.cleanup()


def _components(planner: _Planner, executor: _Executor, verifier: _Verifier) -> Any:
    import threading

    class _Comps:
        pass

    comps = _Comps()
    comps.planner = planner
    comps.executor = executor
    comps.verifier = verifier
    comps.available_tools = list(planner.available_tools)
    comps.toolbox_metadata = {name: {"name": name} for name in planner.available_tools}
    comps.tool_instances = {}
    comps.log_key = threading.local()
    return comps


def replay_audit() -> Dict[str, Any]:
    """Show which tools recovery is willing to run again. Uses a throwaway database."""
    from recovery_controller import BING_TOOL, FETCH_TOOL, GENERALIST_TOOL, PYTHON_TOOL, RecoveryController

    tmp = tempfile.TemporaryDirectory(prefix="cghe_replay_")
    controller = RecoveryController(
        db_path=str(Path(tmp.name) / "replay.sqlite"),
        tool_calls=4,
        environment="cghe_pev",
    )
    try:
        python_replay = controller.may_auto_replay(PYTHON_TOOL, command="print(1)")
        generalist_replay = controller.may_auto_replay(GENERALIST_TOOL, command="draft")
        bing = controller.authorize_tool_call(BING_TOOL, command="capital of France")
        controller.commit_tool_call(bing)
        bing_replay = controller.may_auto_replay(BING_TOOL, command="capital of France")
        fetch = controller.authorize_tool_call(FETCH_TOOL, command="https://example.test/paris")
        controller.commit_tool_call(fetch)
        fetch_replay = controller.may_auto_replay(FETCH_TOOL, command="https://example.test/paris")
        return {
            "python_replay_allowed": bool(python_replay.allowed),
            "python_reason": python_replay.reason,
            "generalist_replay_allowed": bool(generalist_replay.allowed),
            "generalist_reason": generalist_replay.reason,
            "bing_replay_allowed": bool(bing_replay.allowed),
            "fetch_replay_allowed": bool(fetch_replay.allowed),
        }
    finally:
        controller.memory.close()
        tmp.cleanup()


def run_offline_demo() -> Dict[str, Any]:
    """Default end-to-end path: load ``.env``, then replay the mock recovery."""
    dotenv_loaded = load_env()
    case = run_window_case(enabled=True, candidate_text=candidate_json(), question=_QUESTION)
    summary = {
        "mode": "mock",
        "live_default": False,
        "network": False,
        "model_calls": 0,
        "dotenv_loaded": bool(dotenv_loaded),
        "commands": {"offline": _OFFLINE_COMMAND, "live": _LIVE_COMMAND},
        "trajectory": {
            "question": case["question"],
            "first_tool_result": "TOOL_NOT_FOUND",
            "verifier": "Satisfied: NO",
            "decisions": case["decisions"],
            "task_impossible": any(case["task_impossible_flags"]),
            "selected_tool": case["final_tool"],
            "selected_subgoal": case["final_subgoal"],
            "active_plan_ids": case["active_plan_ids"],
            "executed_tools": case["executed_tools"],
            "generate_next_step_calls": case["generate_next_step_calls"],
            "candidate_engine_calls": case["candidate_engine_calls"],
        },
        "replay": replay_audit(),
    }
    return summary


def _live() -> int:
    """One real planner window. Not called unless the operator passes ``--live``."""
    load_env()
    print("live mode calls the configured chat endpoint. It is not the default.")
    try:
        import context as context_mod

        ctx = context_mod.get_episode_context()
        result = ctx.plan_window({
            "dst": "planner",
            "kind": "plan_step",
            "payload": {"question": _QUESTION},
        })
    except Exception as exc:  # noqa: BLE001
        print(redact(f"live planning failed: {type(exc).__name__}: {exc}"))
        return 1
    payload = result.get("payload") if isinstance(result, dict) else {}
    args = (payload or {}).get("args") or {}
    safe = {
        "mode": "live",
        "tool": str(args.get("tool_name") or ""),
        "sub_goal": str((payload or {}).get("sub_goal") or "")[:200],
        "done": bool((payload or {}).get("done")),
    }
    print(redact(json.dumps(safe, ensure_ascii=False, indent=2)))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CGHE recovery demo. Default is an offline mock trajectory.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call the configured model for one planner window. Omit this to stay offline.",
    )
    args = parser.parse_args(argv)
    if args.live:
        return _live()
    print(redact(json.dumps(run_offline_demo(), ensure_ascii=False, indent=2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
