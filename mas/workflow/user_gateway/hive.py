"""Scaffold HIVE PEV wrappers into user_space. Does not touch ref_Rep/HIVE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import ensure_project_layout, project_dir, user_agent_id
from .registry import load_manifest, save_manifest
from .tools import bind_project_tools

EXECUTOR_LOCAL = "executor"
ROUTER_ID = "route_exec"
ROLE_IDS = ("planner", "verifier")

CONTEXT_PY = '''"""Shared HIVE episode context. One SystemMemory for planner/executor/verifier.

This module lives in user_space. It may import MAS.hive after putting
ref_Rep/HIVE/MAS on sys.path. It must not import science_infra.
Collect / eval / sampling call run_window only — never Solver.solve.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

SOLVE_CALLS = 0


def _as_payload(message: Any) -> Dict[str, Any]:
    if hasattr(message, "payload") and isinstance(getattr(message, "payload"), dict):
        return dict(message.payload)
    if isinstance(message, dict):
        if isinstance(message.get("payload"), dict):
            return dict(message["payload"])
        return dict(message)
    return {}


def _text(*vals: Any) -> str:
    for val in vals:
        if val is None:
            continue
        text = str(val).strip()
        if text:
            return text
    return ""


def _use_mock() -> bool:
    flag = os.environ.get("SCIENCE_HIVE_MOCK", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _hive_mas_root() -> Optional[Path]:
    env = os.environ.get("SCIENCE_HIVE_MAS_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        return path if path.is_dir() else None
    search = [Path(__file__).resolve(), Path.cwd()]
    for start in search:
        for parent in [start, *start.parents]:
            cand = parent / "ref_Rep" / "HIVE" / "MAS"
            if cand.is_dir():
                return cand
    return None


def _map_hive_llm_env() -> None:
    """User-zone only. Do not write management config."""
    model = (
        os.environ.get("AI_ASSISTANT_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL")
        or ""
    ).strip()
    base = (
        os.environ.get("AI_ASSISTANT_API_BASE")
        or os.environ.get("AI_ASSISTANT_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip()
    key = (
        os.environ.get("AI_ASSISTANT_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    if model:
        os.environ.setdefault("MODEL_Name", model)
    if base:
        os.environ.setdefault("OPENAI_API_BASE_URL", base.rstrip("/"))
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)


class _MockMemory:
    def __init__(self) -> None:
        self.toolbox_metadata = {"Python_Coder_Tool": {"description": "mock python"}}
        self.outline: Dict[str, Any] = {"1": "solve the question"}
        self.obtained_information: List[str] = []
        self.evidence_records: List[Any] = []

    def set_outline(self, outline: Dict[str, Any]) -> None:
        self.outline = dict(outline or {})

    def get_outline(self) -> Dict[str, Any]:
        return dict(self.outline)

    def get_obtained_information(self) -> List[str]:
        return list(self.obtained_information)

    def get_obtained_information_for_prompt(self) -> str:
        return "\\n".join(f"- {item}" for item in self.obtained_information)

    def add_obtained_information(self, info: Any) -> None:
        text = str(info or "").strip()
        if text and text not in self.obtained_information:
            self.obtained_information.append(text)

    def get_task_profile(self) -> None:
        return None

    def set_phase(self, _phase: Any) -> None:
        return None


class _MockPlanner:
    available_tools = ["Python_Coder_Tool"]

    def generate_next_step(self, question: str, *args: Any, **kwargs: Any) -> Any:
        step = SimpleNamespace(
            context=str(question or "mock"),
            sub_goal=str(question or "mock"),
            tool_name="Python_Coder_Tool",
        )
        return step, ""

    def extract_context_subgoal_and_tool(self, plan: Any, **kwargs: Any) -> Any:
        return (
            str(getattr(plan, "context", "") or ""),
            str(getattr(plan, "sub_goal", "") or ""),
            str(getattr(plan, "tool_name", "") or "Python_Coder_Tool"),
        )


class _MockExecutor:
    def generate_tool_command(self, *args: Any, **kwargs: Any) -> Any:
        return "print(2)", "print(2)", "print(2)"

    def extract_explanation_and_command(self, tool_command: Any) -> Any:
        if isinstance(tool_command, tuple) and len(tool_command) == 3:
            return tool_command
        text = str(tool_command)
        return text, text, text

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        return f"hive-mock:{tool_name}:{command}"


class _MockSolver:
    def __init__(self, planner: Any, executor: Any, diagnoser: Any, system_memory: Any) -> None:
        self.planner = planner
        self.executor = executor
        self.diagnoser = diagnoser
        self.system_memory = system_memory

    def solve(self, question: str, *args: Any, **kwargs: Any) -> Any:
        global SOLVE_CALLS
        SOLVE_CALLS += 1
        raise RuntimeError("Solver.solve is not allowed on the collect/sampling path")

    def _run_verification(self, question: str, image_path: Any, step_ctx: Any, step_count: int, **kwargs: Any) -> Any:
        output = str(getattr(step_ctx, "result_executor", "") or "")
        ok = bool(output) and "error" not in output.lower()
        if ok:
            self.system_memory.add_obtained_information(output)
        return SimpleNamespace(
            subgoal_complete=ok,
            step_conclusion="COMPLETE" if ok else "INCOMPLETE",
            analysis="mock verify",
            slot_updates=[],
            evidence_type="DIRECT" if ok else "EMPTY",
        )


def _build_mock() -> Dict[str, Any]:
    memory = _MockMemory()
    planner = _MockPlanner()
    executor = _MockExecutor()
    diagnoser = SimpleNamespace(name="mock-diagnoser")
    solver = _MockSolver(planner, executor, diagnoser, memory)
    return {
        "planner": planner,
        "executor": executor,
        "diagnoser": diagnoser,
        "system_memory": memory,
        "solver": solver,
        "mocked": True,
    }


def _build_real() -> Dict[str, Any]:
    mas_root = _hive_mas_root()
    if mas_root is None:
        raise ImportError("ref_Rep/HIVE/MAS not found")
    mas_s = str(mas_root)
    if mas_s not in sys.path:
        sys.path.insert(0, mas_s)
    _map_hive_llm_env()
    from MAS.hive.solver import construct_solver  # type: ignore

    model = os.environ.get("MODEL_Name") or os.environ.get("AI_ASSISTANT_MODEL") or "gpt-4o"
    solver = construct_solver(llm_engine_name=model, verbose=False)
    orig = solver.solve

    def _blocked(question: str, *args: Any, **kwargs: Any) -> Any:
        global SOLVE_CALLS
        SOLVE_CALLS += 1
        raise RuntimeError("Solver.solve is not allowed on the collect/sampling path")

    solver.solve = _blocked  # type: ignore[method-assign]
    solver._contrast_solve = orig
    return {
        "planner": solver.planner,
        "executor": solver.executor,
        "diagnoser": solver.diagnoser,
        "system_memory": solver.system_memory,
        "solver": solver,
        "mocked": False,
    }


class HiveEpisodeContext:
    """One shared HIVE solver graph for the three canvas windows."""

    def __init__(self) -> None:
        self.question = ""
        self.step_count = 0
        self.last_feedback = ""
        self.last_context = ""
        self.last_sub_goal = ""
        self.last_tool_name = "Python_Coder_Tool"
        self.last_command = ""
        self.last_tool_output = ""
        self.last_step_ctx: Any = None
        bundle = _build_mock() if _use_mock() else self._try_real()
        self.planner = bundle["planner"]
        self.executor = bundle["executor"]
        self.diagnoser = bundle["diagnoser"]
        self.system_memory = bundle["system_memory"]
        self.solver = bundle["solver"]
        self.mocked = bool(bundle["mocked"])

    def _try_real(self) -> Dict[str, Any]:
        try:
            return _build_real()
        except Exception:
            if os.environ.get("SCIENCE_HIVE_REQUIRE_REAL", "").strip():
                raise
            return _build_mock()

    def hydrate_from_message(self, message: Any) -> None:
        payload = _as_payload(message)
        question = _text(payload.get("question"), payload.get("query"), payload.get("input"))
        if question and "[Verifier feedback" not in question:
            self.question = question.split("\\n\\nDecide the next step", 1)[0].strip() or self.question
        inbound_text = _text(payload.get("input"), payload.get("reason"))
        if "[Verifier feedback" in inbound_text:
            self.last_feedback = inbound_text
        args = payload.get("args") if isinstance(payload.get("args"), dict) else payload
        if isinstance(args, dict):
            self.last_context = _text(args.get("context"), self.last_context)
            self.last_sub_goal = _text(args.get("sub_goal"), payload.get("sub_goal"), self.last_sub_goal)
            self.last_tool_name = _text(args.get("tool_name"), self.last_tool_name) or "Python_Coder_Tool"
            q2 = _text(args.get("question"))
            if q2:
                self.question = q2
        output = payload.get("output")
        if output:
            self.last_tool_output = str(output)
        self._restore_memory_min()

    def hydrate_from_messages(self, messages: Optional[List[Any]]) -> None:
        for item in messages or []:
            self.hydrate_from_message(item)

    def _restore_memory_min(self) -> None:
        memory = self.system_memory
        outline = {}
        if hasattr(memory, "get_outline"):
            outline = memory.get_outline() or {}
        if not outline and self.question:
            outline = {"1": self.last_sub_goal or self.question}
            if hasattr(memory, "set_outline"):
                memory.set_outline(outline)
        if self.last_tool_output and hasattr(memory, "add_obtained_information"):
            memory.add_obtained_information(self.last_tool_output)

    def plan_window(self, message: Any) -> Dict[str, Any]:
        self.hydrate_from_message(message)
        payload = _as_payload(message)
        resume = payload.get("resume_messages")
        if isinstance(resume, list):
            self.hydrate_from_messages(resume)
        question = self.question or _text(payload.get("question"), payload.get("input"))
        self.question = question
        self.step_count += 1
        target = ""
        if hasattr(self.system_memory, "get_outline"):
            outline = self.system_memory.get_outline() or {}
            if isinstance(outline, dict) and outline:
                key = sorted(outline, key=lambda x: int(x) if str(x).isdigit() else str(x))[0]
                target = str(outline.get(key) or "")
        obtained = []
        if hasattr(self.system_memory, "get_obtained_information_for_prompt"):
            obtained = self.system_memory.get_obtained_information_for_prompt()
        signal = {"recommendation": "modify_state", "state_action": self.last_feedback} if self.last_feedback else None
        plan, _prompt = self.planner.generate_next_step(
            question, None, target or question, self.step_count, 20, obtained, {}, signal
        )
        context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(
            plan, target_information=target or question, question=question, obtained_information=obtained
        )
        self.last_context = str(context or question)
        self.last_sub_goal = str(sub_goal or question)
        self.last_tool_name = str(tool_name or "Python_Coder_Tool")
        done = True
        return {
            "kind": "plan_step",
            "payload": {
                "next": "u_hive__executor",
                "args": {
                    "question": question,
                    "context": self.last_context,
                    "sub_goal": self.last_sub_goal,
                    "tool_name": self.last_tool_name,
                    "query": self.last_sub_goal,
                },
                "sub_goal": self.last_sub_goal,
                "done": done,
            },
        }

    def exec_window(self, message: Any) -> Dict[str, Any]:
        self.hydrate_from_message(message)
        payload = _as_payload(message)
        question = _text(payload.get("question"), self.question)
        context = _text(payload.get("context"), self.last_context, question)
        sub_goal = _text(payload.get("sub_goal"), self.last_sub_goal, question)
        tool_name = _text(payload.get("tool_name"), self.last_tool_name) or "Python_Coder_Tool"
        metadata = {}
        if hasattr(self.system_memory, "toolbox_metadata"):
            metadata = (self.system_memory.toolbox_metadata or {}).get(tool_name) or {}
        command_blob = self.executor.generate_tool_command(
            question, None, context, sub_goal, tool_name, metadata, self.step_count, {}, None
        )
        if hasattr(self.executor, "extract_explanation_and_command"):
            _analysis, _expl, command = self.executor.extract_explanation_and_command(command_blob)
        elif isinstance(command_blob, tuple) and len(command_blob) == 3:
            command = command_blob[2]
        else:
            command = str(command_blob)
        self.last_command = str(command)
        result = self.executor.execute_tool_command(tool_name, str(command))
        output = result if isinstance(result, str) else str(result)
        self.last_tool_output = output
        self.last_step_ctx = SimpleNamespace(
            step_key=str(self.step_count or 1),
            target_information=sub_goal or question,
            context=context,
            sub_goal=sub_goal,
            tool_name=tool_name,
            command=str(command),
            result_executor=result,
            first_attempt_command=str(command),
        )
        if hasattr(self.system_memory, "add_obtained_information"):
            self.system_memory.add_obtained_information(output)
        return {
            "kind": "tool_result",
            "payload": {"output": output, "ok": True, "evidence_type": "DIRECT" if output else "EMPTY"},
        }

    def verify_window(self, message: Any) -> Dict[str, Any]:
        self.hydrate_from_message(message)
        payload = _as_payload(message)
        if not self.last_step_ctx:
            self.last_step_ctx = SimpleNamespace(
                step_key=str(self.step_count or 1),
                target_information=self.last_sub_goal or self.question,
                context=self.last_context or self.question,
                sub_goal=self.last_sub_goal or self.question,
                tool_name=self.last_tool_name,
                command=self.last_command,
                result_executor=self.last_tool_output or payload.get("reason") or "",
                first_attempt_command=self.last_command,
            )
        verification = self.solver._run_verification(
            self.question, None, self.last_step_ctx, self.step_count or 1
        )
        ok = bool(getattr(verification, "subgoal_complete", False))
        conclusion = str(getattr(verification, "step_conclusion", "") or ("COMPLETE" if ok else "INCOMPLETE"))
        if conclusion.upper() == "COMPLETE":
            ok = True
        reason = str(getattr(verification, "analysis", "") or conclusion)
        return {
            "kind": "verify",
            "payload": {
                "ok": ok,
                "reason": reason,
                "step_conclusion": "COMPLETE" if ok else "INCOMPLETE",
                "slot_updates": list(getattr(verification, "slot_updates", []) or []),
            },
        }


_CTX: Optional[HiveEpisodeContext] = None


def get_episode_context(*, reset: bool = False) -> HiveEpisodeContext:
    global _CTX
    if reset or _CTX is None:
        _CTX = HiveEpisodeContext()
    return _CTX


def solve_call_count() -> int:
    return SOLVE_CALLS
'''

ENTRY_PY = '''"""HIVE window dispatcher. Management runtime calls run_window only."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

from context import get_episode_context  # noqa: E402


def _dst(message: Any) -> str:
    return str(getattr(message, "dst", None) or (message.get("dst") if isinstance(message, dict) else "") or "")


def _kind(message: Any) -> str:
    return str(getattr(message, "kind", None) or (message.get("kind") if isinstance(message, dict) else "") or "")


class UserAgent:
    def run_window(self, message: Any) -> Any:
        ctx = get_episode_context()
        dst = _dst(message)
        kind = _kind(message)
        if kind == "plan_step" or dst == "planner" or "planner" in dst:
            return ctx.plan_window(message)
        if kind == "tool_invoke" or "executor" in dst:
            return ctx.exec_window(message)
        if kind == "verify" or "verifier" in dst:
            return ctx.verify_window(message)
        return ctx.exec_window(message)
'''

UPLOAD_README = """# HIVE 源码不复制到用户区

HIVE 本体留在仓库 `ref_Rep/HIVE`。本项目只写 thin 适配器：

- `adapted/context.py`：共享 `HiveEpisodeContext`（Planner / Executor / Diagnoser / SystemMemory）
- `adapted/entry.py`：按窗口分发 `run_window`

不要改 `ref_Rep/HIVE`。不要把 `Solver.solve` 当作 Collect / 训练 / 采样入口。
"""


def _dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def hive_sampling_sites(router_id: str = ROUTER_ID) -> List[Dict[str, Any]]:
    return [
        {
            "id": "after_planner",
            "enabled": True,
            "anchor": {"kind": "after_agent_turn", "agent_id": "planner"},
            "gate": {"type": "always"},
            "fork": {"beam_size": 2, "resume_mode": "messages"},
        },
        {
            "id": f"after_{router_id}",
            "enabled": True,
            "anchor": {"kind": "after_agent_turn", "agent_id": router_id},
            "gate": {"type": "always"},
            "fork": {"beam_size": 2, "resume_mode": "messages"},
        },
        {
            "id": "after_verifier",
            "enabled": True,
            "anchor": {"kind": "after_verifier", "agent_id": "verifier"},
            "gate": {"type": "always"},
            "fork": {"beam_size": 2, "resume_mode": "messages"},
        },
    ]


def hive_workflow(project_id: str) -> Dict[str, Any]:
    executor_id = user_agent_id(project_id, EXECUTOR_LOCAL)
    return {
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "hub": {
            "role": "planner",
            "skills": [],
            "max_feedback_hops": 3,
            "system_prompt": "HIVE planner window. Output JSON next/args/sub_goal/done.",
        },
        "agents": [
            {
                "id": "planner",
                "kind": "planner",
                "role": "planner",
                "trainable": True,
                "system_prompt": "Wrap HIVE Planner.generate_next_step. next must be the executor tool id.",
                "tools": [executor_id],
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "HIVE.Planner"},
            },
            {
                "id": executor_id,
                "kind": "tool",
                "role": "tool",
                "trainable": False,
                "system_prompt": "Wrap HIVE Executor tool pool (Wikipedia/Google/Web/Python/BaseGenerator).",
                "profile": {"backend": "user_space", "user_project": project_id, "llm_required": False},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "HIVE.Executor"},
            },
            {
                "id": "verifier",
                "kind": "verifier",
                "role": "verifier",
                "trainable": False,
                "system_prompt": "Wrap HIVE Solver._run_verification (Diagnoser + after_verify).",
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "HIVE.Diagnoser"},
            },
        ],
        "routers": [
            {
                "id": ROUTER_ID,
                "candidates": [executor_id],
                "strategy": "from_plan",
                "output_contract": "json",
            }
        ],
        "edges": [
            {"from": "planner", "to": ROUTER_ID, "kind": "route"},
            {"from": ROUTER_ID, "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
        "sampling": {
            "mode": "arpo",
            "group_n": 4,
            "beam_size": 2,
            "sites": hive_sampling_sites(),
        },
    }


def ensure_hive_project(project_id: str = "hive", *, title: str = "HIVE PEV wrap") -> Dict[str, Any]:
    from .registry import create_project

    try:
        return load_manifest(project_id)
    except FileNotFoundError:
        return create_project(project_id, title=title, mode="native_mas")


def scaffold_hive_pev(project_id: str = "hive", *, title: Optional[str] = None) -> Dict[str, Any]:
    ensure_project_layout(project_id)
    try:
        manifest = load_manifest(project_id)
    except FileNotFoundError:
        manifest = ensure_hive_project(project_id, title=title or "HIVE PEV wrap")
    title = title or str(manifest.get("title") or "HIVE PEV wrap")
    root = project_dir(project_id)
    adapted = root / "adapted"
    adapted.mkdir(parents=True, exist_ok=True)
    (adapted / "__init__.py").write_text("", encoding="utf-8")
    (adapted / "context.py").write_text(CONTEXT_PY, encoding="utf-8")
    (adapted / "entry.py").write_text(ENTRY_PY, encoding="utf-8")
    (root / "upload" / "README.md").write_text(UPLOAD_README, encoding="utf-8")
    workflow = hive_workflow(project_id)
    _dump(root / "contracts" / "workflow.yaml", workflow)
    _dump(root / "contracts" / "sampling.yaml", workflow["sampling"])
    executor_id = user_agent_id(project_id, EXECUTOR_LOCAL)
    manifest.update(
        {
            "mode": "native_mas",
            "status": "ready",
            "title": title,
            "agent_ids": ["planner", "verifier"],
            "tool_ids": [executor_id],
            "entry": "adapted/entry.py:UserAgent",
            "wraps": "ref_Rep/HIVE",
        }
    )
    save_manifest(project_id, manifest)
    bind_project_tools(project_id)
    return {
        "project_id": project_id,
        "mode": "native_mas",
        "agent_ids": ["planner", "verifier"],
        "tool_ids": [executor_id],
        "workflow": workflow,
    }
