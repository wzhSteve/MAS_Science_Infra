"""Scaffold EPC-AW Planner / Executor / Diagnoser as three canvas windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .detect import find_epc_aw_import_root
from .hive import ROUTER_ID, hive_sampling_sites
from .paths import ensure_project_layout, project_dir, user_agent_id
from .registry import create_project, load_manifest, save_manifest
from .tools import bind_project_tools

EXECUTOR_LOCAL = "executor"

CONTEXT_PY = '''"""Shared EPC-AW episode context. One SystemMemory for planner/executor/diagnoser.

Lives in user_space. May import MAS.epc_aw after putting the uploaded or
ref_Rep/EPC-AW tree on sys.path. Collect / eval / sampling call run_window only.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

SOLVE_CALLS = 0
EXECUTOR_ID = "__EXECUTOR_ID__"


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
    flag = os.environ.get("SCIENCE_EPC_AW_MOCK") or os.environ.get("SCIENCE_HIVE_MOCK") or ""
    return flag.strip().lower() in {"1", "true", "yes", "on"}


def _plan_n() -> int:
    raw = os.environ.get("SCIENCE_EPC_AW_N") or "1"
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _is_llm_error(val: Any) -> bool:
    return isinstance(val, dict) and "error" in val


def _llm_error_text(val: Any) -> str:
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False, default=str)
    return str(val)


def _import_root() -> Optional[Path]:
    env = os.environ.get("SCIENCE_EPC_AW_MAS_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        if (path / "MAS" / "epc_aw").is_dir():
            return path
        if path.name == "MAS" and (path / "epc_aw").is_dir():
            return path.parent
    here = Path(__file__).resolve().parent
    upload = here.parent / "upload"
    if upload.is_dir():
        for solver in upload.rglob("epc_aw/solver.py"):
            mas = solver.parent.parent
            if mas.name == "MAS":
                return mas.parent
    for start in (here, Path.cwd()):
        for parent in [start, *start.parents]:
            cand = parent / "ref_Rep" / "EPC-AW"
            if (cand / "MAS" / "epc_aw" / "solver.py").is_file():
                return cand
    return None


def _ensure_import_paths(root: Path) -> None:
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    tools_parent = root / "MAS" / "epc_aw"
    if tools_parent.is_dir() and str(tools_parent) not in sys.path:
        sys.path.insert(0, str(tools_parent))


@contextmanager
def _epc_tools_scope(root: Optional[Path] = None) -> Iterator[None]:
    """Temporarily prefer EPC-AW tools.* over management mas/tools."""
    root = root or _import_root()
    parent = str((root / "MAS" / "epc_aw").resolve()) if root is not None else ""
    saved = {key: sys.modules[key] for key in list(sys.modules) if key == "tools" or key.startswith("tools.")}
    for key in saved:
        sys.modules.pop(key, None)
    inserted = False
    if parent and parent not in sys.path:
        sys.path.insert(0, parent)
        inserted = True
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "tools" or key.startswith("tools."):
                sys.modules.pop(key, None)
        sys.modules.update(saved)
        if inserted:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass


def _map_llm_env() -> None:
    candidates: List[str] = []
    for key in ("AI_ASSISTANT_MODEL", "OPENAI_MODEL", "MODEL", "MODEL_Name", "MODEL_NAME"):
        val = (os.environ.get(key) or "").strip()
        if val and val not in candidates:
            candidates.append(val)
    model = candidates[0] if candidates else ""
    base = (
        os.environ.get("AI_ASSISTANT_API_BASE")
        or os.environ.get("AI_ASSISTANT_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip()
    key = (os.environ.get("AI_ASSISTANT_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if candidates:
        os.environ.setdefault("MODEL_NAME", model)
        os.environ.setdefault("MODEL_Name", model)
        existing = [x.strip() for x in (os.environ.get("SERVER_MODEL") or "").split(",") if x.strip()]
        for item in candidates:
            if item not in existing:
                existing.append(item)
        os.environ["SERVER_MODEL"] = ",".join(existing)
    if base:
        os.environ.setdefault("OPENAI_API_BASE_URL", base.rstrip("/"))
    if key:
        os.environ.setdefault("OPENAI_API_KEY", key)


def _merge_no_think(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    extra = dict(kwargs.get("extra_body") or {})
    extra["enable_thinking"] = False
    ck = dict(extra.get("chat_template_kwargs") or {})
    ck["enable_thinking"] = False
    extra["chat_template_kwargs"] = ck
    reason = extra.get("reasoning")
    extra["reasoning"] = dict(reason) if isinstance(reason, dict) else {}
    extra["reasoning"]["enabled"] = False
    out = dict(kwargs)
    out["extra_body"] = extra
    return out


def _wrap_no_think(fn: Any) -> Any:
    if fn is None or getattr(fn, "_science_no_think", False):
        return fn

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **_merge_no_think(kwargs))

    wrapped._science_no_think = True  # type: ignore[attr-defined]
    return wrapped


def _disable_engine_thinking(eng: Any) -> None:
    client = getattr(eng, "client", None)
    if client is None:
        return
    chat = getattr(client, "chat", None)
    completions = getattr(chat, "completions", None) if chat is not None else None
    if completions is not None and hasattr(completions, "create"):
        completions.create = _wrap_no_think(completions.create)
    beta = getattr(client, "beta", None)
    bchat = getattr(beta, "chat", None) if beta is not None else None
    bcomp = getattr(bchat, "completions", None) if bchat is not None else None
    if bcomp is not None:
        if hasattr(bcomp, "parse"):
            bcomp.parse = _wrap_no_think(bcomp.parse)
        if hasattr(bcomp, "create"):
            bcomp.create = _wrap_no_think(bcomp.create)


def _patch_one_engine(eng: Any) -> None:
    try:
        if not bool(getattr(eng, "is_chat_model", False)):
            eng.is_chat_model = True
    except Exception:
        pass
    model = str(getattr(eng, "model_string", "") or os.environ.get("MODEL_Name") or "")
    if model and hasattr(eng, "SERVER_MODEL"):
        sm = list(getattr(eng, "SERVER_MODEL") or [])
        if model not in sm:
            sm.append(model)
            try:
                eng.SERVER_MODEL = sm
            except Exception:
                pass
    _disable_engine_thinking(eng)
    _cap = 4096
    orig = getattr(eng, "generate", None)
    if callable(orig) and not getattr(eng, "_science_token_capped", False):
        def _capped(*args: Any, _orig=orig, **kwargs: Any) -> Any:
            raw = kwargs.get("max_tokens")
            try:
                cur = int(raw) if raw is not None else _cap
            except (TypeError, ValueError):
                cur = _cap
            kwargs["max_tokens"] = min(cur, _cap)
            return _orig(*args, **kwargs)

        try:
            eng.generate = _capped
            eng._science_token_capped = True
        except Exception:
            pass


def _install_engine_patches() -> None:
    os.environ.setdefault("ENABLE_THINKING", "0")
    try:
        from MAS.epc_aw.engine.openai import ChatOpenAI  # type: ignore
    except Exception:
        return
    if getattr(ChatOpenAI.__init__, "_science_patched", False):
        return
    orig_init = ChatOpenAI.__init__

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        _patch_one_engine(self)

    _init._science_patched = True  # type: ignore[attr-defined]
    ChatOpenAI.__init__ = _init  # type: ignore[method-assign]


def _patch_engines(solver: Any) -> None:
    seen: List[int] = []
    roles = (
        getattr(solver, "planner", None),
        getattr(solver, "executor", None),
        getattr(solver, "diagnoser", None),
    )
    for role in roles:
        if role is None:
            continue
        for attr in dir(role):
            if "llm" not in attr.lower():
                continue
            try:
                eng = getattr(role, attr)
            except Exception:
                continue
            if eng is None or not hasattr(eng, "is_chat_model"):
                continue
            marker = id(eng)
            if marker in seen:
                continue
            seen.append(marker)
            _patch_one_engine(eng)


def _feasibility() -> str:
    try:
        from MAS.epc_aw.solver import feasibility_criteria  # type: ignore

        return str(feasibility_criteria)
    except Exception:
        return "Score the plan from 1 to 5 based on intrinsic feasibility and reliability."


def _first_outline_target(memory: Any) -> str:
    outline: Any = {}
    if hasattr(memory, "get_outline"):
        try:
            outline = memory.get_outline() or {}
        except Exception:
            outline = {}
    if isinstance(outline, dict) and outline:
        key = sorted(outline, key=lambda x: int(x) if str(x).isdigit() else str(x))[0]
        text = str(outline.get(key) or "").strip()
        if text:
            return text
    return "solve the current outline step"


def _parse_verify_result(result: Any) -> Tuple[str, bool, Any, str]:
    if _is_llm_error(result):
        return f"llm error: {_llm_error_text(result)}", False, "", "CONTINUE"
    if isinstance(result, tuple) and len(result) >= 4:
        return str(result[0]), bool(result[1]), result[2], str(result[3] or "CONTINUE")
    if hasattr(result, "model_dump"):
        try:
            result = result.model_dump()
        except Exception:
            pass
    if isinstance(result, str):
        text = result.strip()
        if text.startswith("```"):
            start = text.find("{")
            end = text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            result = json.loads(text)
        except Exception:
            return text, False, "", "CONTINUE"
    if isinstance(result, dict):
        if _is_llm_error(result):
            return f"llm error: {_llm_error_text(result)}", False, "", "CONTINUE"
        analysis = result.get("Analysis") or result.get("analysis") or ""
        flag = result.get("New_Obtained_Information_Flag", result.get("add_obtained_information_flag"))
        info = result.get("New_Obtained_Information", result.get("obtained_information", ""))
        conclusion = result.get("Conclusion") or result.get("conclusion") or "CONTINUE"
        flag_b = flag is True or "true" in str(flag).lower()
        return str(analysis), flag_b, info, str(conclusion)
    return str(result or ""), False, "", "CONTINUE"


class _MockRoleMemory:
    def __init__(self) -> None:
        self.actions: List[Any] = []
        self.constraints: List[Any] = []

    def add_action(self, *args: Any, **kwargs: Any) -> None:
        self.actions.append((args, kwargs))

    def add_epistemic_constraint(self, item: Any) -> None:
        self.constraints.append(item)

    def get_actions(self) -> List[Any]:
        return list(self.actions)

    def get_epistemic_constraint(self) -> str:
        return "\\n".join(str(x) for x in self.constraints)


class _MockMemory:
    def __init__(self) -> None:
        self.toolbox_metadata = {
            "Python_Coder_Tool": {"description": "mock python"},
            "Base_Generator_Tool": {"description": "mock generator"},
        }
        self.outline: Dict[str, Any] = {}
        self.obtained_information: List[str] = []

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

    def get_agent_profile(self) -> Dict[str, Any]:
        return {}

    def get_last_step_plan_scores(self) -> Dict[str, Any]:
        return {}

    def get_toolbox_metadata(self) -> Dict[str, Any]:
        return dict(self.toolbox_metadata)


class _MockPlanner:
    def __init__(self) -> None:
        self.available_tools = ["Python_Coder_Tool", "Base_Generator_Tool"]
        self.toolbox_metadata = {
            "Python_Coder_Tool": {"description": "mock python"},
            "Base_Generator_Tool": {"description": "mock generator"},
        }
        self.n = _plan_n()
        self.memory = _MockRoleMemory()

    def analyze_query(self, question: str, image: Any = None) -> Any:
        del image
        return "mock analysis", {"1": "solve the current outline step"}

    def generate_next_step(self, question: str, *args: Any, **kwargs: Any) -> Any:
        n = max(int(getattr(self, "n", 1) or 1), _plan_n())
        step = SimpleNamespace(
            context=str(question or "mock"),
            sub_goal="compute the answer",
            tool_name="Python_Coder_Tool",
        )
        if n >= 2:
            alt = SimpleNamespace(
                context=str(question or "mock"),
                sub_goal="alternative plan",
                tool_name="Base_Generator_Tool",
            )
            return [step, alt], ""
        return step, ""

    def extract_context_subgoal_and_tool(self, plan: Any, **kwargs: Any) -> Any:
        del kwargs
        return (
            str(getattr(plan, "context", "") or ""),
            str(getattr(plan, "sub_goal", "") or ""),
            str(getattr(plan, "tool_name", "") or "Python_Coder_Tool"),
        )

    def report_generation(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return [], {"0": 5.0, "1": 3.0}

    def belief_prediction(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return {"0": 4.0, "1": 2.0}, {"0": 4.0, "1": 2.0}

    def compute_bts_for_plans(self, planner_scores: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        scores = planner_scores if isinstance(planner_scores, dict) else {"0": 1.0}
        return {str(k): {"bts_score": float(v), "A_bar": float(v)} for k, v in scores.items()}


class _MockExecutor:
    def __init__(self) -> None:
        self.memory = _MockRoleMemory()
        self.executed: List[str] = []

    def generate_tool_command(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return "print(2)", "print(2)", "print(2)"

    def extract_explanation_and_command(self, tool_command: Any) -> Any:
        if isinstance(tool_command, tuple) and len(tool_command) == 3:
            return tool_command
        text = str(tool_command)
        return text, text, text

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        self.executed.append(str(tool_name))
        if "no matched" in str(tool_name).lower() or " " in str(tool_name):
            raise ImportError(f"No module named \\'tools.{tool_name}\\'")
        return f"epc-aw-mock:{tool_name}:{command}"

    def generate_direct_output(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "2"

    def report_generation(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return [], {"0": 5.0, "1": 3.0}

    def belief_prediction(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return {"0": 4.0, "1": 2.0}, {"0": 4.0, "1": 2.0}


class _MockDiagnoser:
    def __init__(self) -> None:
        self.memory = _MockRoleMemory()
        self.verify_calls = 0

    def verificate_context(self, *args: Any, **kwargs: Any) -> Any:
        self.verify_calls += 1
        output = str(kwargs.get("result_executor") or (args[4] if len(args) > 4 else ""))
        if self.verify_calls <= 1:
            return ("need another step", True, f"partial:{output}", "CONTINUE")
        return ("enough information", True, f"final:{output}", "STOP")

    def extract_conclusion(self, response: Any) -> Any:
        if isinstance(response, tuple) and len(response) >= 4:
            return str(response[0]), str(response[3])
        return "mock verify", "CONTINUE"

    def update_outline(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return {"1": "updated outline after verify"}

    def epc_aw_diagnosis(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return "mock diagnosis", "mock constraint"

    def report_generation(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return [], {"0": 5.0, "1": 3.0}

    def belief_prediction(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return {"0": 4.0, "1": 2.0}, {"0": 4.0, "1": 2.0}


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

    def process_next_step(self, next_step_list: list) -> Any:
        cleaned = {str(i): item for i, item in enumerate(next_step_list or [])}
        return cleaned, []


def _build_mock() -> Dict[str, Any]:
    memory = _MockMemory()
    planner = _MockPlanner()
    executor = _MockExecutor()
    diagnoser = _MockDiagnoser()
    return {
        "planner": planner,
        "executor": executor,
        "diagnoser": diagnoser,
        "system_memory": memory,
        "solver": _MockSolver(planner, executor, diagnoser, memory),
        "mocked": True,
    }


def _build_real() -> Dict[str, Any]:
    root = _import_root()
    if root is None:
        raise ImportError("EPC-AW MAS not found in upload/ or ref_Rep/EPC-AW")
    _ensure_import_paths(root)
    _map_llm_env()
    os.environ.setdefault("ENABLE_THINKING", "0")
    from MAS.epc_aw.solver import construct_solver  # type: ignore

    _install_engine_patches()

    model = (
        os.environ.get("MODEL_NAME")
        or os.environ.get("MODEL_Name")
        or os.environ.get("AI_ASSISTANT_MODEL")
        or "gpt-4o"
    )
    n = _plan_n()
    enabled = [
        "Base_Generator_Tool",
        "Python_Coder_Tool",
        "Wikipedia_Search_Tool",
        "Web_Search_Tool",
        "Google_Search_Tool",
    ]
    with _epc_tools_scope(root):
        solver = construct_solver(
            llm_engine_name=model,
            verbose=False,
            n=n,
            enabled_tools=enabled,
        )
    if hasattr(solver, "planner") and hasattr(solver.planner, "n"):
        solver.planner.n = n
    _patch_engines(solver)
    if hasattr(solver.executor, "set_query_cache_dir"):
        cache_dir = os.environ.get("SCIENCE_EPC_AW_CACHE") or str(Path.cwd() / "artifacts" / "epc_aw_cache")
        try:
            solver.executor.set_query_cache_dir(cache_dir)
        except Exception:
            pass
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


class EpcAwEpisodeContext:
    def __init__(self) -> None:
        self.question = ""
        self.step_count = 0
        self.last_feedback = ""
        self.last_context = ""
        self.last_sub_goal = ""
        self.last_tool_name = "Python_Coder_Tool"
        self.last_command = ""
        self.last_tool_output = ""
        self.last_analysis = ""
        self.last_explanation = ""
        self.last_plan: Any = None
        self.last_verify_analysis = ""
        self.query_analysis = ""
        self.planner_selected_index = "0"
        self.bts_selected_index = "0"
        self.tool_available = True
        self._analyzed = False
        self.json_data: Dict[str, Any] = {}
        self.last_trace: Dict[str, Any] = {}
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
            if os.environ.get("SCIENCE_EPC_AW_REQUIRE_REAL", "").strip():
                raise
            return _build_mock()

    def hydrate_from_message(self, message: Any) -> None:
        payload = _as_payload(message)
        question = _text(payload.get("question"), payload.get("query"), payload.get("input"))
        if question and "[Verifier feedback" not in question:
            self.question = question.split("\\n\\nDecide the next step", 1)[0].strip() or self.question
        inbound = _text(payload.get("input"), payload.get("reason"))
        if "[Verifier feedback" in inbound:
            self.last_feedback = inbound
        args = payload.get("args") if isinstance(payload.get("args"), dict) else payload
        if isinstance(args, dict):
            self.last_context = _text(args.get("context"), self.last_context)
            self.last_sub_goal = _text(args.get("sub_goal"), payload.get("sub_goal"), self.last_sub_goal)
            self.last_tool_name = _text(args.get("tool_name"), self.last_tool_name) or "Python_Coder_Tool"
            if "tool_available" in args:
                self.tool_available = bool(args.get("tool_available"))
            q2 = _text(args.get("question"))
            if q2:
                self.question = q2
        if payload.get("output"):
            self.last_tool_output = str(payload.get("output"))

    def hydrate_from_messages(self, messages: Optional[List[Any]]) -> None:
        for item in messages or []:
            self.hydrate_from_message(item)

    def _obtained(self) -> Any:
        memory = self.system_memory
        if hasattr(memory, "get_obtained_information"):
            return memory.get_obtained_information()
        if hasattr(memory, "get_obtained_information_for_prompt"):
            return memory.get_obtained_information_for_prompt()
        return []

    def _available_tools(self) -> List[str]:
        tools = getattr(self.planner, "available_tools", None)
        if isinstance(tools, list) and tools:
            return [str(x) for x in tools]
        return ["Python_Coder_Tool", "Base_Generator_Tool"]

    def _analyze_once(self, question: str) -> None:
        if self._analyzed:
            return
        self._analyzed = True
        if not hasattr(self.planner, "analyze_query"):
            if hasattr(self.system_memory, "set_outline") and not (hasattr(self.system_memory, "get_outline") and self.system_memory.get_outline()):
                self.system_memory.set_outline({"1": "solve the current outline step"})
            return
        try:
            analysis, outline = self.planner.analyze_query(question, None)
        except Exception as exc:
            self.last_trace["llm_error"] = str(exc)
            if hasattr(self.system_memory, "set_outline") and not (
                hasattr(self.system_memory, "get_outline") and self.system_memory.get_outline()
            ):
                self.system_memory.set_outline({"1": "solve the current outline step"})
            return
        if _is_llm_error(analysis) or _is_llm_error(outline):
            self.last_trace["llm_error"] = analysis if _is_llm_error(analysis) else outline
            outline = {"1": "solve the current outline step"}
        self.query_analysis = analysis if not _is_llm_error(analysis) else ""
        if isinstance(outline, dict) and outline:
            self.system_memory.set_outline(outline)
        elif hasattr(self.system_memory, "set_outline"):
            self.system_memory.set_outline({"1": "solve the current outline step"})

    def _normalize_tool(self, tool_name: str) -> Tuple[str, bool]:
        available = self._available_tools()
        name = str(tool_name or "").strip()
        if not name or "none" in name.lower():
            name = "Base_Generator_Tool"
        if name in available:
            return name, True
        if name.lower().startswith("no matched") or name not in available:
            if "Base_Generator_Tool" in available:
                return "Base_Generator_Tool", True
            return name, False
        return name, name in available

    def _select_plan(self, plan: Any, question: str, target: str) -> Any:
        if _is_llm_error(plan):
            self.last_trace["llm_error"] = plan
            return SimpleNamespace(context=target, sub_goal=target, tool_name="Base_Generator_Tool")
        if not (isinstance(plan, list) and len(plan) > 1):
            if isinstance(plan, list) and plan:
                return plan[0]
            return plan
        cleaned: Any = plan
        if hasattr(self.solver, "process_next_step"):
            try:
                cleaned, _ = self.solver.process_next_step(plan)
            except Exception:
                cleaned = {str(i): item for i, item in enumerate(plan)}
        if isinstance(cleaned, list):
            cleaned = {str(i): item for i, item in enumerate(cleaned)}
        if not isinstance(cleaned, dict) or not cleaned:
            return plan[0]
        self.last_trace["bts"] = {"n": len(cleaned), "selected": "0"}
        try:
            meta = getattr(self.system_memory, "toolbox_metadata", {}) or {}
            obtained = self._obtained()
            criteria = _feasibility()
            profile = self.system_memory.get_agent_profile() if hasattr(self.system_memory, "get_agent_profile") else {}
            last_scores = (
                self.system_memory.get_last_step_plan_scores()
                if hasattr(self.system_memory, "get_last_step_plan_scores")
                else {}
            )
            _ranked_p, planner_scores = self.planner.report_generation(
                question, target, cleaned, meta, criteria, obtained
            )
            executor_scores_by_planner, diagnoser_scores_by_planner = self.planner.belief_prediction(
                question, target, cleaned, meta, criteria,
                agent_profile=profile, last_step_plan_scores=last_scores, obtained_informtion=obtained,
            )
            _ranked_e, executor_scores = self.executor.report_generation(
                question, target, cleaned, meta, criteria, obtained
            )
            planner_scores_by_executor, diagnoser_scores_by_executor = self.executor.belief_prediction(
                question, target, cleaned, meta, criteria,
                agent_profile=profile, last_step_plan_scores=last_scores, obtained_informtion=obtained,
            )
            _ranked_d, diagnoser_scores = self.diagnoser.report_generation(
                question, target, cleaned, meta, criteria, obtained
            )
            planner_scores_by_diagnoser, executor_scores_by_diagnoser = self.diagnoser.belief_prediction(
                question, target, cleaned, meta, criteria,
                agent_profile=profile, last_step_plan_scores=last_scores, obtained_informtion=obtained,
            )
            if any(_is_llm_error(x) for x in (planner_scores, executor_scores, diagnoser_scores)):
                err = next(x for x in (planner_scores, executor_scores, diagnoser_scores) if _is_llm_error(x))
                self.last_trace["llm_error"] = err
                self.planner_selected_index = "0"
                self.bts_selected_index = "0"
                return cleaned.get("0") or next(iter(cleaned.values()))
            plan_results = self.planner.compute_bts_for_plans(
                planner_scores,
                executor_scores,
                diagnoser_scores,
                executor_scores_by_planner,
                diagnoser_scores_by_planner,
                planner_scores_by_executor,
                diagnoser_scores_by_executor,
                planner_scores_by_diagnoser,
                executor_scores_by_diagnoser,
            )
            if isinstance(planner_scores, dict) and planner_scores:
                self.planner_selected_index = str(max(planner_scores.keys(), key=lambda k: planner_scores[k]))
            else:
                self.planner_selected_index = "0"
            if isinstance(plan_results, dict) and plan_results:
                self.bts_selected_index = str(
                    max(plan_results.keys(), key=lambda k: (plan_results[k].get("bts_score", 0), plan_results[k].get("A_bar", 0)))
                )
            else:
                self.bts_selected_index = self.planner_selected_index
            self.last_trace["bts"] = {
                "n": len(cleaned),
                "selected": self.bts_selected_index,
                "planner_selected": self.planner_selected_index,
                "scores": {k: (v.get("bts_score") if isinstance(v, dict) else v) for k, v in (plan_results or {}).items()},
            }
            return cleaned.get(self.bts_selected_index) or cleaned.get("0") or next(iter(cleaned.values()))
        except Exception as exc:
            self.last_trace["llm_error"] = str(exc)
            self.planner_selected_index = "0"
            self.bts_selected_index = "0"
            return cleaned.get("0") or next(iter(cleaned.values()))

    def _safe_add_action(self, memory: Any, *args: Any) -> None:
        if memory is None or not hasattr(memory, "add_action"):
            return
        try:
            memory.add_action(*args)
        except Exception:
            pass

    def plan_window(self, message: Any) -> Dict[str, Any]:
        self.last_trace = {}
        self.hydrate_from_message(message)
        payload = _as_payload(message)
        resume = payload.get("resume_messages")
        if isinstance(resume, list):
            self.hydrate_from_messages(resume)
        question = self.question or _text(payload.get("question"), payload.get("input"))
        self.question = question
        self.step_count += 1
        self._analyze_once(question)
        target = _first_outline_target(self.system_memory)
        obtained = self._obtained()
        try:
            raw = self.planner.generate_next_step(
                question, None, target, _feasibility(), self.step_count, 20, obtained, self.json_data
            )
        except Exception as exc:
            self.last_trace["llm_error"] = str(exc)
            raw = SimpleNamespace(context=target, sub_goal=target, tool_name="Base_Generator_Tool"), ""
        plan = raw[0] if isinstance(raw, tuple) and raw else raw
        if _is_llm_error(plan):
            self.last_trace["llm_error"] = plan
            plan = SimpleNamespace(context=target, sub_goal=target, tool_name="Base_Generator_Tool")
        selected = self._select_plan(plan, question, target)
        self.last_plan = selected
        if hasattr(self.planner, "extract_context_subgoal_and_tool"):
            try:
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(selected)
            except TypeError:
                context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(
                    selected, target_information=target, question=question, obtained_information=obtained
                )
            except Exception as exc:
                self.last_trace["llm_error"] = str(exc)
                context, sub_goal, tool_name = target, target, "Base_Generator_Tool"
        else:
            context, sub_goal, tool_name = target, target, "Python_Coder_Tool"
        tool_name, tool_ok = self._normalize_tool(str(tool_name or ""))
        self.tool_available = tool_ok
        self.last_context = str(context or target)
        self.last_sub_goal = str(sub_goal or target)
        self.last_tool_name = tool_name
        outline = self.system_memory.get_outline() if hasattr(self.system_memory, "get_outline") else {}
        self.last_trace.update(
            {
                "outline": outline,
                "tool": tool_name,
                "context": self.last_context,
                "sub_goal": self.last_sub_goal,
                "tool_available": tool_ok,
            }
        )
        if "bts" not in self.last_trace:
            self.last_trace["bts"] = {"n": _plan_n() if not isinstance(plan, list) else max(1, len(plan)), "selected": "0"}
        return {
            "kind": "plan_step",
            "payload": {
                "next": EXECUTOR_ID,
                "args": {
                    "question": question,
                    "context": self.last_context,
                    "sub_goal": self.last_sub_goal,
                    "tool_name": self.last_tool_name,
                    "tool_available": tool_ok,
                    "query": self.last_sub_goal,
                },
                "sub_goal": self.last_sub_goal,
                "done": False,
                "trace": dict(self.last_trace),
            },
        }

    def exec_window(self, message: Any) -> Dict[str, Any]:
        self.last_trace = {}
        self.hydrate_from_message(message)
        payload = _as_payload(message)
        question = _text(payload.get("question"), self.question)
        context = _text(payload.get("context"), self.last_context, question)
        sub_goal = _text(payload.get("sub_goal"), self.last_sub_goal)
        tool_name = _text(payload.get("tool_name"), self.last_tool_name) or "Python_Coder_Tool"
        available = self._available_tools()
        illegal = (
            tool_name not in available
            or tool_name.lower().startswith("no matched")
            or payload.get("tool_available") is False
        )
        if illegal:
            output = f"Tool '{tool_name}' is not available or not found."
            self.last_command = "No command was generated because the tool was not found."
            self.last_tool_output = output
            self.last_trace = {
                "tool": tool_name,
                "command": self.last_command,
                "output": output,
                "analysis": "skipped execute_tool_command",
            }
            return {
                "kind": "tool_result",
                "payload": {
                    "output": output,
                    "ok": False,
                    "evidence_type": "ERROR",
                    "command": self.last_command,
                    "analysis": "skipped execute_tool_command",
                    "trace": dict(self.last_trace),
                },
            }
        metadata = {}
        if hasattr(self.system_memory, "toolbox_metadata"):
            metadata = (self.system_memory.toolbox_metadata or {}).get(tool_name) or {}
        try:
            command_blob = self.executor.generate_tool_command(
                question, None, context, sub_goal, tool_name, metadata, self.step_count, self.json_data
            )
        except Exception as exc:
            self.last_trace["llm_error"] = str(exc)
            command_blob = None
        if _is_llm_error(command_blob):
            self.last_trace["llm_error"] = command_blob
            output = f"llm error: {_llm_error_text(command_blob)}"
            return {
                "kind": "tool_result",
                "payload": {
                    "output": output,
                    "ok": False,
                    "evidence_type": "ERROR",
                    "trace": dict(self.last_trace),
                },
            }
        analysis, explanation, command = "", "", ""
        if command_blob is not None and hasattr(self.executor, "extract_explanation_and_command"):
            try:
                analysis, explanation, command = self.executor.extract_explanation_and_command(command_blob)
            except Exception as exc:
                self.last_trace["llm_error"] = str(exc)
                command = str(command_blob)
        elif isinstance(command_blob, tuple) and len(command_blob) == 3:
            analysis, explanation, command = command_blob
        elif command_blob is not None:
            command = str(command_blob)
        self.last_analysis = str(analysis or "")
        self.last_explanation = str(explanation or "")
        self.last_command = str(command or "")
        try:
            with _epc_tools_scope():
                result = self.executor.execute_tool_command(tool_name, self.last_command)
        except Exception as exc:
            result = f"tool error: {exc}"
        output = result if isinstance(result, str) else str(result)
        self.last_tool_output = output
        self.last_trace = {
            "tool": tool_name,
            "command": self.last_command,
            "output": output,
            "analysis": self.last_analysis,
            "explanation": self.last_explanation,
        }
        return {
            "kind": "tool_result",
            "payload": {
                "output": output,
                "ok": True,
                "evidence_type": "DIRECT" if output else "EMPTY",
                "command": self.last_command,
                "analysis": self.last_analysis,
                "explanation": self.last_explanation,
                "trace": dict(self.last_trace),
            },
        }

    def verify_window(self, message: Any) -> Dict[str, Any]:
        self.last_trace = {}
        self.hydrate_from_message(message)
        question = self.question or "question"
        target = self.last_sub_goal or _first_outline_target(self.system_memory)
        outline = self.system_memory.get_outline() if hasattr(self.system_memory, "get_outline") else {}
        try:
            result = self.diagnoser.verificate_context(
                question,
                None,
                target,
                outline,
                self.last_tool_output,
                self.step_count or 1,
                self._obtained(),
            )
        except Exception as exc:
            self.last_trace["llm_error"] = str(exc)
            result = {"error": type(exc).__name__, "message": str(exc)}
        if _is_llm_error(result):
            self.last_trace["llm_error"] = result
        analysis, new_flag, new_info, conclusion = _parse_verify_result(result)
        self.last_verify_analysis = analysis
        if new_flag and new_info and hasattr(self.system_memory, "add_obtained_information"):
            self.system_memory.add_obtained_information(new_info)
        ready = str(conclusion).strip().upper() == "STOP" and bool(new_flag) and self.step_count > 1
        updated_outline = outline
        if not ready:
            epc_aw_analysis = ""
            if str(self.planner_selected_index) != str(self.bts_selected_index) and hasattr(self.diagnoser, "epc_aw_diagnosis"):
                try:
                    epc_aw_analysis, constraint = self.diagnoser.epc_aw_diagnosis(
                        self.last_plan,
                        self.last_plan,
                        target,
                        outline,
                        self.last_tool_output,
                        self.step_count,
                        self.json_data,
                    )
                    planner_mem = getattr(self.planner, "memory", None)
                    if planner_mem is not None and hasattr(planner_mem, "add_epistemic_constraint"):
                        planner_mem.add_epistemic_constraint(constraint)
                except Exception as exc:
                    self.last_trace["llm_error"] = str(exc)
                    epc_aw_analysis = ""
            if hasattr(self.diagnoser, "update_outline"):
                try:
                    updated_outline = self.diagnoser.update_outline(
                        question,
                        self.last_tool_output,
                        target,
                        outline,
                        self.last_tool_output,
                        self.step_count,
                        self._obtained(),
                        epc_aw_analysis,
                        getattr(self.system_memory, "toolbox_metadata", {}),
                    )
                    if isinstance(updated_outline, dict) and hasattr(self.system_memory, "set_outline"):
                        self.system_memory.set_outline(updated_outline)
                except Exception as exc:
                    self.last_trace["llm_error"] = str(exc)
                    updated_outline = outline
            action_args = (
                self.step_count,
                target,
                self.last_plan,
                self.last_tool_name,
                self.last_command,
                self.last_tool_output,
                analysis,
            )
            if new_flag:
                self._safe_add_action(getattr(self.diagnoser, "memory", None), *action_args)
            else:
                self._safe_add_action(getattr(self.planner, "memory", None), *action_args)
            self._safe_add_action(
                getattr(self.executor, "memory", None),
                self.step_count,
                target,
                self.last_plan,
                self.last_tool_name,
                self.last_command,
                self.last_tool_output,
            )
        answer = ""
        if ready and hasattr(self.executor, "generate_direct_output"):
            try:
                answer = str(self.executor.generate_direct_output(question, analysis, self.system_memory) or "")
            except Exception as exc:
                self.last_trace["llm_error"] = str(exc)
                answer = ""
        self.last_trace.update(
            {
                "analysis": analysis,
                "new_info": new_info,
                "conclusion": conclusion,
                "outline": outline,
                "outline_updated": updated_outline,
                "ready_to_stop": ready,
            }
        )
        return {
            "kind": "verify",
            "payload": {
                "ok": bool(ready),
                "reason": str(analysis or conclusion),
                "step_conclusion": "COMPLETE" if ready else "INCOMPLETE",
                "slot_updates": [],
                "ready_to_stop": ready,
                "answer": answer,
                "direct_output": answer,
                "trace": dict(self.last_trace),
            },
        }


_CTX: Optional[EpcAwEpisodeContext] = None


def get_episode_context(*, reset: bool = False) -> EpcAwEpisodeContext:
    global _CTX
    if reset or _CTX is None:
        _CTX = EpcAwEpisodeContext()
    return _CTX


def solve_call_count() -> int:
    return SOLVE_CALLS
'''

ENTRY_PY = '''"""EPC-AW window dispatcher. Management runtime calls run_window only."""

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
        if kind == "verify" or "verifier" in dst or "diagnoser" in dst:
            return ctx.verify_window(message)
        return ctx.exec_window(message)
'''

UPLOAD_README = """# EPC-AW 按 PEV 窗口封装

上传的源码留在 `upload/`。本项目只写 thin 适配器：

- `adapted/context.py`：共享 Planner / Executor / Diagnoser / SystemMemory
- `adapted/entry.py`：按窗口分发 `run_window`

不要改 `ref_Rep/EPC-AW`。不要把 `Solver.solve` 当作 Collect / 训练 / 采样入口。
"""


def _dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def epc_aw_workflow(project_id: str) -> Dict[str, Any]:
    executor_id = user_agent_id(project_id, EXECUTOR_LOCAL)
    return {
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "hub": {
            "role": "planner",
            "skills": [],
            "max_feedback_hops": 6,
            "system_prompt": "EPC-AW planner window. Output JSON next/args/sub_goal/done.",
        },
        "agents": [
            {
                "id": "planner",
                "kind": "planner",
                "role": "planner",
                "trainable": True,
                "system_prompt": "Wrap EPC-AW Planner.generate_next_step. next must be the executor tool id.",
                "tools": [executor_id],
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "EPC-AW.Planner"},
            },
            {
                "id": executor_id,
                "kind": "tool",
                "role": "tool",
                "trainable": False,
                "system_prompt": "Wrap EPC-AW Executor tool pool (Wikipedia/Google/Web/Python/BaseGenerator).",
                "profile": {"backend": "user_space", "user_project": project_id, "llm_required": False},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "EPC-AW.Executor"},
            },
            {
                "id": "verifier",
                "kind": "verifier",
                "role": "verifier",
                "trainable": False,
                "system_prompt": "Wrap EPC-AW Diagnoser.verificate_context.",
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "EPC-AW.Diagnoser"},
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


def ensure_epc_aw_project(project_id: str, *, title: str = "EPC-AW PEV wrap") -> Dict[str, Any]:
    try:
        return load_manifest(project_id)
    except FileNotFoundError:
        return create_project(project_id, title=title, mode="native_mas")


def scaffold_epc_aw_pev(project_id: str, *, title: Optional[str] = None) -> Dict[str, Any]:
    ensure_project_layout(project_id)
    try:
        manifest = load_manifest(project_id)
    except FileNotFoundError:
        manifest = ensure_epc_aw_project(project_id, title=title or "EPC-AW PEV wrap")
    title = title or str(manifest.get("title") or "EPC-AW PEV wrap")
    root = project_dir(project_id)
    adapted = root / "adapted"
    adapted.mkdir(parents=True, exist_ok=True)
    executor_id = user_agent_id(project_id, EXECUTOR_LOCAL)
    (adapted / "__init__.py").write_text("", encoding="utf-8")
    (adapted / "context.py").write_text(CONTEXT_PY.replace("__EXECUTOR_ID__", executor_id), encoding="utf-8")
    (adapted / "entry.py").write_text(ENTRY_PY, encoding="utf-8")
    readme = root / "upload" / "README.md"
    if not any((root / "upload").rglob("epc_aw/solver.py")):
        readme.write_text(UPLOAD_README, encoding="utf-8")
    workflow = epc_aw_workflow(project_id)
    _dump(root / "contracts" / "workflow.yaml", workflow)
    _dump(root / "contracts" / "sampling.yaml", workflow["sampling"])
    source = find_epc_aw_import_root(project_id)
    manifest.update(
        {
            "mode": "native_mas",
            "status": "ready",
            "title": title,
            "agent_ids": ["planner", "verifier"],
            "tool_ids": [executor_id],
            "entry": "adapted/entry.py:UserAgent",
            "wraps": "EPC-AW.PEV",
            "source": str(source) if source else "",
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
