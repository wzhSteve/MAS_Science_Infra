"""Scaffold EPC-AW Planner / Executor / Diagnoser as three canvas windows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .detect import find_epc_aw_import_root
from .paths import ensure_project_layout, project_dir, user_agent_id
from .registry import create_project, load_manifest, save_manifest
from .tools import bind_project_tools

EXECUTOR_LOCAL = "executor"

CONTEXT_PY = r'''"""Shared EPC-AW episode context. One SystemMemory for planner/executor/diagnoser.

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


DEFAULT_RUNTIME: Dict[str, Any] = {
    "n": 1,
    "max_steps": 20,
    "max_time": 3000,
    "max_tokens": 4000,
    "temperature": 0.0,
    "enabled_tools": [
        "Base_Generator_Tool",
        "Python_Coder_Tool",
        "Wikipedia_Search_Tool",
        "Bing_Search_Tool",
        "Web_Fetch_Tool",
    ],
}

_INFRA_TOOL_ERRORS = (
    "Binary Location Must be a String",
    "Binary Location",
    "signal only works in main thread",
    "signal only works",
    "session not created",
    "only supports Chrome version",
)

_TRANSPORT_MARKERS = (
    "HTTPSConnectionPool",
    "Max retries exceeded",
    "NameResolutionError",
    "ConnectTimeout",
    "Read timed out",
    "Connection refused",
    "Network is unreachable",
)

_SEARCH_FALLBACK = (
    "Wikipedia_Search_Tool",
    "Bing_Search_Tool",
    "Web_Search_Tool",
    "Google_Search_Tool",
)


def _is_transport_failure(output: Any) -> bool:
    """Search host did not answer. Distinct from a Python snippet timeout."""
    text = str(output or "")
    if text.startswith("Wikipedia unreachable:") or text.startswith("Bing unreachable:"):
        return True
    return any(marker in text for marker in _TRANSPORT_MARKERS)


def _next_search_tool(available: List[str], failed: List[str]) -> str:
    blocked = set(failed)
    for name in _SEARCH_FALLBACK:
        if name in available and name not in blocked:
            return name
    return ""


def _runtime_path() -> Path:
    return Path(__file__).resolve().parent.parent / "contracts" / "runtime.yaml"


def _load_runtime() -> Dict[str, Any]:
    cfg = dict(DEFAULT_RUNTIME)
    cfg["enabled_tools"] = list(DEFAULT_RUNTIME["enabled_tools"])
    path = _runtime_path()
    if path.is_file():
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(data, dict):
                for key in DEFAULT_RUNTIME:
                    if key not in data:
                        continue
                    cfg[key] = data[key]
        except Exception:
            pass
    env_n = os.environ.get("SCIENCE_EPC_AW_N")
    if env_n not in (None, ""):
        try:
            cfg["n"] = int(env_n)
        except ValueError:
            pass
    else:
        os.environ["SCIENCE_EPC_AW_N"] = str(int(cfg.get("n") or 1))
    try:
        cfg["n"] = max(1, min(16, int(cfg.get("n") or 1)))
    except (TypeError, ValueError):
        cfg["n"] = 1
    try:
        cfg["max_steps"] = max(1, min(40, int(cfg.get("max_steps") or 20)))
    except (TypeError, ValueError):
        cfg["max_steps"] = 20
    try:
        cfg["max_time"] = max(30, min(10000, int(cfg.get("max_time") or 3000)))
    except (TypeError, ValueError):
        cfg["max_time"] = 3000
    try:
        cfg["max_tokens"] = max(256, min(16384, int(cfg.get("max_tokens") or 4000)))
    except (TypeError, ValueError):
        cfg["max_tokens"] = 4000
    try:
        cfg["temperature"] = float(cfg.get("temperature") if cfg.get("temperature") is not None else 0.0)
    except (TypeError, ValueError):
        cfg["temperature"] = 0.0
    tools = cfg.get("enabled_tools")
    if not isinstance(tools, list) or not tools:
        cfg["enabled_tools"] = list(DEFAULT_RUNTIME["enabled_tools"])
    else:
        cfg["enabled_tools"] = [str(item) for item in tools if str(item).strip()]
    return cfg


def _plan_n() -> int:
    return int(_load_runtime().get("n") or 1)


def _max_steps() -> int:
    return int(_load_runtime().get("max_steps") or 20)


def _is_llm_error(val: Any) -> bool:
    return isinstance(val, dict) and "error" in val


def _llm_error_text(val: Any) -> str:
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False, default=str)
    return str(val)

def _progress(text: str) -> None:
    try:
        import science_user

        report = getattr(science_user, "report_progress", None)
        if report:
            report(text)
    except Exception:
        return


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
    """Point EPC-AW at the experiment endpoint. Overwrites; ignores Assistant vars."""
    candidates: List[str] = []
    for key in ("OPENAI_MODEL", "MODEL", "MODEL_NAME", "MODEL_Name"):
        val = (os.environ.get(key) or "").strip()
        if val and val not in candidates:
            candidates.append(val)
    model = candidates[0] if candidates else ""
    base = (
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip().rstrip("/")
    key = (os.environ.get("OPENAI_API_KEY") or "").strip() or "EMPTY"
    if model:
        os.environ["MODEL_NAME"] = model
        os.environ["MODEL_Name"] = model
        os.environ["MODEL"] = model
        os.environ["OPENAI_MODEL"] = model
        existing = [x.strip() for x in (os.environ.get("SERVER_MODEL") or "").split(",") if x.strip()]
        for item in candidates:
            if item not in existing:
                existing.append(item)
        os.environ["SERVER_MODEL"] = ",".join(existing)
    if base:
        os.environ["OPENAI_API_BASE"] = base
        os.environ["OPENAI_BASE_URL"] = base
        os.environ["OPENAI_API_BASE_URL"] = base
    os.environ["OPENAI_API_KEY"] = key


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
    _wrap_engine_completion_budget(eng)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _estimate_prompt_tokens(prompt: Any) -> int:
    if prompt is None:
        return 0
    if isinstance(prompt, list):
        text = " ".join(str(x) for x in prompt if not isinstance(x, (bytes, bytearray)))
    else:
        text = str(prompt)
    return max(1, (len(text) + 3) // 3)


def _clamp_max_tokens(requested: Any, prompt: Any) -> int:
    try:
        want = int(requested)
    except (TypeError, ValueError):
        want = 2048
    cap = _env_int("SCIENCE_EPC_AW_MAX_COMPLETION", 0) or _env_int("MAS_MAX_COMPLETION_TOKENS", 2048)
    model_len = _env_int("SCIENCE_EPC_AW_MAX_MODEL_LEN", 8192)
    prompt_tokens = _estimate_prompt_tokens(prompt)
    remaining = max(256, model_len - prompt_tokens - 64)
    return max(256, min(want, cap, remaining))


def _budget_wrapper(orig: Any) -> Any:
    import inspect

    def wrapped(self: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            ba = inspect.signature(orig).bind(self, prompt, *args, **kwargs)
            ba.apply_defaults()
            if "max_tokens" in ba.arguments:
                ba.arguments["max_tokens"] = _clamp_max_tokens(ba.arguments.get("max_tokens"), ba.arguments.get("prompt", prompt))
            return orig(*ba.args, **ba.kwargs)
        except Exception:
            kwargs["max_tokens"] = _clamp_max_tokens(kwargs.get("max_tokens", 2048), prompt)
            return orig(self, prompt, *args, **kwargs)

    wrapped._science_budget = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_engine_completion_budget(eng: Any) -> None:
    fn = getattr(eng, "_generate_text", None)
    if fn is None or getattr(fn, "_science_budget", False):
        return
    orig = getattr(fn, "__func__", fn)
    eng._generate_text = _budget_wrapper(orig).__get__(eng, type(eng))  # type: ignore[method-assign]
    eng._generate_text._science_budget = True  # type: ignore[attr-defined]


def _install_completion_budget() -> None:
    """Class-level wrap so ChatOpenAI / ChatVLLM clamp completion under max_model_len."""
    for mod_name, cls_name in (
        ("MAS.epc_aw.engine.openai", "ChatOpenAI"),
        ("MAS.epc_aw.engine.vllm", "ChatVLLM"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        fn = getattr(cls, "_generate_text", None)
        if fn is None or getattr(fn, "_science_budget", False):
            continue
        setattr(cls, "_generate_text", _budget_wrapper(fn))


def _install_client_timeout() -> None:
    try:
        import openai
    except Exception:
        return
    orig = openai.OpenAI.__init__
    if getattr(orig, "_science_timeout", False):
        return

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        raw = os.environ.get("SCIENCE_LLM_TIMEOUT") or "30"
        try:
            seconds = max(5.0, float(raw))
        except ValueError:
            seconds = 30.0
        kwargs.setdefault("timeout", seconds)
        kwargs.setdefault("max_retries", 0)
        orig(self, *args, **kwargs)

    _init._science_timeout = True  # type: ignore[attr-defined]
    openai.OpenAI.__init__ = _init  # type: ignore[method-assign]



def _patch_module_timeout(mod: Any) -> None:
    """SIGALRM only works on the main thread. Off-thread, the outer future is the timeout."""
    import threading
    from contextlib import contextmanager

    timeout_fn = getattr(mod, "timeout", None)
    if timeout_fn is None or getattr(timeout_fn, "_science_threadsafe", False):
        return
    orig = timeout_fn

    @contextmanager
    def timeout(seconds: Any):  # type: ignore[no-redef]
        if threading.current_thread() is threading.main_thread():
            with orig(seconds):
                yield
        else:
            yield

    timeout._science_threadsafe = True  # type: ignore[attr-defined]
    mod.timeout = timeout


def _install_threadsafe_timeouts() -> None:
    """Replace SIGALRM timeouts so tools work inside eval worker threads."""
    import importlib
    import re as _re
    import threading
    from concurrent.futures import ThreadPoolExecutor

    try:
        from MAS.epc_aw.models.executor import (  # type: ignore
            Executor,
            TOOL_NAME_MAPPING_LONG,
            TOOL_NAME_MAPPING_SHORT,
        )
    except Exception:
        return

    if not getattr(Executor.execute_tool_command, "_science_threadsafe", False):
        _orig_exec = Executor.execute_tool_command

        def execute_tool_command(self: Any, tool_name: str, command: str) -> Any:
            if threading.current_thread() is threading.main_thread():
                return _orig_exec(self, tool_name, command)

            def split_commands(cmd: str) -> List[str]:
                pattern = r".*?execution\s*=\s*tool\.execute\([^\n]*\)\s*(?:\n|$)"
                blocks = _re.findall(pattern, cmd, _re.DOTALL)
                return [block.strip() for block in blocks if block.strip()]

            def execute_with_timeout(block: str, local_context: dict) -> Any:
                def run() -> Any:
                    exec(block, globals(), local_context)
                    return local_context.get("execution")

                pool = ThreadPoolExecutor(max_workers=1)
                try:
                    fut = pool.submit(run)
                    return fut.result()
                except Exception as exc:
                    return f"Error in execute_tool_command: {exc}"
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)

            if tool_name in TOOL_NAME_MAPPING_LONG:
                dir_name = TOOL_NAME_MAPPING_LONG[tool_name]["dir_name"]
                class_name = TOOL_NAME_MAPPING_LONG[tool_name]["class_name"]
            elif tool_name in TOOL_NAME_MAPPING_SHORT:
                long_name = TOOL_NAME_MAPPING_SHORT[tool_name]
                if long_name in TOOL_NAME_MAPPING_LONG:
                    dir_name = TOOL_NAME_MAPPING_LONG[long_name]["dir_name"]
                    class_name = TOOL_NAME_MAPPING_LONG[long_name]["class_name"]
                else:
                    dir_name = tool_name.lower().replace("_tool", "")
                    class_name = tool_name
            else:
                dir_name = tool_name.lower().replace("_tool", "")
                class_name = tool_name
            module_name = f"tools.{dir_name}.tool"
            try:
                module = importlib.import_module(module_name)
                _patch_module_timeout(module)
                tool_class = getattr(module, class_name)
                tool = tool_class()
                tool.set_custom_output_dir(self.query_cache_dir)
                command_blocks = split_commands(command)
                executions = []
                for block in command_blocks:
                    local_context = {"tool": tool}
                    result = execute_with_timeout(block, local_context)
                    if result is not None:
                        executions.append(result)
                    else:
                        executions.append(f"No execution captured from block: {block}")
                return executions
            except Exception as e:
                return f"Error in execute_tool_command: {str(e)}"

        execute_tool_command._science_threadsafe = True  # type: ignore[attr-defined]
        Executor.execute_tool_command = execute_tool_command  # type: ignore[method-assign]

    for mod_name in ("tools.python_coder.tool", "MAS.epc_aw.tools.python_coder.tool"):
        try:
            _patch_module_timeout(importlib.import_module(mod_name))
        except Exception:
            continue


def _install_engine_patches() -> None:
    os.environ.setdefault("ENABLE_THINKING", "0")
    os.environ.setdefault("SCIENCE_EPC_AW_MAX_MODEL_LEN", "8192")
    os.environ.setdefault("SCIENCE_EPC_AW_MAX_COMPLETION", "2048")
    _install_client_timeout()
    _install_threadsafe_timeouts()
    _install_completion_budget()
    try:
        from engine_factory_patch import patch_factory_for_vllm

        patch_factory_for_vllm()
    except Exception:
        pass
    for mod_name, cls_name in (
        ("MAS.epc_aw.engine.openai", "ChatOpenAI"),
        ("MAS.epc_aw.engine.vllm", "ChatVLLM"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        if getattr(cls.__init__, "_science_patched", False):
            continue
        orig_init = cls.__init__

        def _make_init(orig: Any) -> Any:
            def _init(self: Any, *args: Any, **kwargs: Any) -> None:
                orig(self, *args, **kwargs)
                _patch_one_engine(self)

            _init._science_patched = True  # type: ignore[attr-defined]
            return _init

        cls.__init__ = _make_init(orig_init)  # type: ignore[method-assign]


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
            if eng is None:
                continue
            if not (hasattr(eng, "is_chat_model") or hasattr(eng, "_generate_text") or hasattr(eng, "model_string")):
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
    """Match Solver.solve: preceding_step stays 1 → outline['1']."""
    outline: Any = {}
    if hasattr(memory, "get_outline"):
        try:
            outline = memory.get_outline() or {}
        except Exception:
            outline = {}
    if isinstance(outline, dict) and outline:
        for key in ("1", 1):
            text = str(outline.get(key) or "").strip()
            if text:
                return text
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
        return "\n".join(str(x) for x in self.constraints)


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
        return "\n".join(f"- {item}" for item in self.obtained_information)

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
            raise ImportError(f"No module named \'tools.{tool_name}\'")
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
        "available_tools": list(planner.available_tools),
        "skipped_tools": [],
    }



def _find_chrome_binary() -> Optional[str]:
    env = (os.environ.get("CHROME_BINARY") or os.environ.get("GOOGLE_CHROME_BIN") or "").strip()
    candidates = [env] if env else []
    candidates.extend(
        [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
            "/usr/bin/chromium",
            "/usr/bin/chromium-browser",
            "/snap/bin/chromium",
        ]
    )
    for cand in candidates:
        path = str(cand or "").strip()
        if path and Path(path).is_file():
            return path
    try:
        import shutil

        for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            found = shutil.which(name)
            if found and Path(found).is_file():
                return found
    except Exception:
        pass
    return _playwright_chrome()


def _playwright_chrome() -> Optional[str]:
    """Playwright's Chrome for Testing is a real browser when system Chrome is absent."""
    roots: List[Path] = []
    env = (os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or "").strip()
    if env and env != "0":
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "ms-playwright")
    found: List[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        found.extend(p for p in root.glob("chromium-*/chrome-linux*/chrome") if p.is_file())
    found = [p for p in found if os.access(p, os.X_OK)]
    if not found:
        return None
    found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(found[0])


def _chrome_major(binary: str) -> Optional[int]:
    """Major version so ChromeDriver matches Playwright's Chrome for Testing."""
    import re
    import subprocess

    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        text = f"{proc.stdout or ''} {proc.stderr or ''}"
    except Exception:
        return None
    match = re.search(r"(\d+)\.", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _kill_google_driver(driver: Any, profile_path: str = "") -> None:
    """SIGKILL the browser. driver.quit() blocks when the navigation itself is stuck."""
    import signal

    pids = []
    service = getattr(driver, "service", None)
    proc = getattr(service, "process", None) if service is not None else None
    if getattr(proc, "pid", None):
        pids.append(int(proc.pid))
    if getattr(driver, "browser_pid", None):
        pids.append(int(driver.browser_pid))
    marker = str(profile_path or "")
    if marker:
        proc_root = Path("/proc")
        if proc_root.is_dir():
            for entry in proc_root.iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    raw = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
                except OSError:
                    continue
                if marker in raw:
                    pids.append(int(entry.name))
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def _google_navigate(driver: Any, raw_get: Any, url: str, timeout_s: float, args: tuple, kwargs: dict, profile_path: str = "") -> Any:
    """Bound driver.get. www.google.com is unreachable from this host and Chrome will not time out by itself."""
    import threading

    done = threading.Event()
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = raw_get(url, *args, **kwargs)
        except Exception as exc:
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    if not done.wait(timeout_s):
        _kill_google_driver(driver, profile_path)
        raise TimeoutError(f"Google navigation timed out after {timeout_s}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def _patch_google_chrome_or_demote(enabled_tools: List[str]) -> List[str]:
    """Inject Chrome binary into scraper, or drop Google_Search_Tool when missing."""
    tools = [str(item) for item in (enabled_tools or []) if str(item).strip()]
    chrome = _find_chrome_binary()
    if chrome:
        os.environ.setdefault("CHROME_BINARY", chrome)
        try:
            from tools.google_search import local_google_scraper as scraper_mod  # type: ignore
        except Exception:
            scraper_mod = None
        if scraper_mod is not None and not getattr(scraper_mod.get_stealth_driver, "_science_chrome_bin", False):
            major = _chrome_major(chrome)

            def get_stealth_driver(*args: Any, **kwargs: Any) -> Any:
                import undetected_chromedriver as uc  # type: ignore

                options = uc.ChromeOptions()
                options.page_load_strategy = "eager"
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1366,768")
                options.add_argument("--headless=new")
                options.add_argument("--disable-blink-features=AutomationControlled")
                profile_path = os.path.abspath(f"./chrome_profile/{int(__import__('time').time())}")
                os.makedirs(profile_path, exist_ok=True)
                options.add_argument(f"--user-data-dir={profile_path}")
                launch: Dict[str, Any] = {
                    "options": options,
                    "browser_executable_path": chrome,
                    "headless": True,
                }
                if major:
                    launch["version_main"] = major
                driver = uc.Chrome(**launch)
                try:
                    driver.set_page_load_timeout(20)
                except Exception:
                    pass
                raw_get = driver.get

                def get(url: str = "", *args: Any, **kwargs: Any) -> Any:
                    return _google_navigate(driver, raw_get, url, 20.0, args, kwargs, profile_path)

                driver.get = get  # type: ignore[method-assign]
                try:
                    driver.execute_cdp_cmd(
                        "Page.addScriptToEvaluateOnNewDocument",
                        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
                    )
                except Exception:
                    pass
                return driver, profile_path

            get_stealth_driver._science_chrome_bin = True  # type: ignore[attr-defined]
            scraper_mod.get_stealth_driver = get_stealth_driver  # type: ignore[method-assign]
        return tools
    if "Google_Search_Tool" in tools:
        tools = [name for name in tools if name != "Google_Search_Tool"]
        _progress("skip Google_Search_Tool: Chrome binary not found")
    return tools


def _patch_google_search_model_string() -> None:
    """ref_Rep Google tool uses self.model_string in USE_LOCAL path but never sets it."""
    try:
        from tools.google_search.tool import Google_Search_Tool  # type: ignore
    except Exception:
        return
    if getattr(Google_Search_Tool.__init__, "_science_model_fix", False):
        return
    orig = Google_Search_Tool.__init__

    def _init(self: Any, model_string: Any = None, *args: Any, **kwargs: Any) -> None:
        model = model_string or os.environ.get("MODEL_Name") or os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL")
        orig(self, model, *args, **kwargs)
        if getattr(self, "model_string", None) is None:
            try:
                self.model_string = model
            except Exception:
                pass
        if getattr(self, "search_model", None) is None:
            try:
                self.search_model = model
            except Exception:
                pass

    _init._science_model_fix = True  # type: ignore[attr-defined]
    Google_Search_Tool.__init__ = _init  # type: ignore[method-assign]


_FAST_TIMEOUT = (3, 8)
_WEB_TEXT_LIMIT = 1200


def _proxy_candidates() -> List[Optional[Dict[str, str]]]:
    """SSH RemoteForward / local Clash first, then a direct connection."""
    raws: List[str] = []
    for key in ("WEB_SEARCH_PROXY", "GOOGLE_CHROME_PROXY", "CHROME_PROXY"):
        value = (os.environ.get(key) or "").strip()
        if value and "://" not in value:
            value = "http://" + value
        if value and value not in raws:
            raws.append(value)
    default = "http://127.0.0.1:7890"
    if default not in raws:
        raws.append(default)
    ordered: List[Optional[Dict[str, str]]] = [{"http": raw, "https": raw} for raw in raws]
    ordered.append(None)
    return ordered


def _fast_get(url: str, params: Optional[Dict[str, str]] = None) -> Any:
    import requests

    errors: List[str] = []
    for proxies in _proxy_candidates():
        label = (proxies or {}).get("http") or "direct"
        try:
            return requests.get(
                url,
                params=params,
                timeout=_FAST_TIMEOUT,
                headers={"User-Agent": "MAS-Science-Infra/1.0"},
                proxies=proxies,
            )
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    raise RuntimeError("; ".join(errors) or "no route")


def _html_to_text(html: str) -> str:
    import re
    from html.parser import HTMLParser

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.parts: List[str] = []
            self.skip = 0

        def handle_starttag(self, tag: str, attrs: Any) -> None:
            if tag in {"script", "style", "noscript"}:
                self.skip += 1

        def handle_endtag(self, tag: str) -> None:
            if tag in {"script", "style", "noscript"} and self.skip:
                self.skip -= 1

        def handle_data(self, data: str) -> None:
            if not self.skip and data:
                self.parts.append(data)

    parser = _Text()
    try:
        parser.feed(html or "")
    except Exception:
        return " ".join((html or "").split())
    return re.sub(r"\s+", " ", " ".join(parser.parts)).strip()


_BING_SKIP = ("bing.com", "microsoft.com", "msn.com", "live.com", "microsoftonline")
_DISPATCH_TOOLS = {
    "Bing_Search_Tool": "bing",
    "Google_Search_Tool": "bing",
    "Wikipedia_Search_Tool": "wiki",
    "Web_Fetch_Tool": "fetch",
    "Web_Search_Tool": "fetch",
}
_DISPATCH_META: Dict[str, Dict[str, Any]] = {
    "Bing_Search_Tool": {
        "tool_name": "Bing_Search_Tool",
        "tool_description": (
            "Search Bing and return up to 5 hits with title, url, and snippet. "
            "Does not open those pages. If a snippet is not enough, call Web_Fetch_Tool "
            "with one of the urls on the next step."
        ),
        "tool_version": "1.0.0",
        "input_types": {"query": "str - the search query."},
        "output_type": "str - numbered hits, each with title, url, and snippet.",
        "demo_commands": [{"command": 'execution = tool.execute(query="Citibank founded")', "description": "Search Bing."}],
        "require_llm_engine": False,
    },
    "Web_Fetch_Tool": {
        "tool_name": "Web_Fetch_Tool",
        "tool_description": (
            "Open exactly one url copied from an earlier Bing or Wikipedia hit. "
            "HTML returns text plus image urls. An image response returns type and size only. "
            "PDF returns its text layer. Requires url. This tool does not search."
        ),
        "tool_version": "1.0.0",
        "input_types": {
            "query": "str - what to look for on the page.",
            "url": "str - the page, image, or pdf url to open.",
        },
        "output_type": "str - page text, image metadata, or pdf text.",
        "demo_commands": [{
            "command": 'execution = tool.execute(query="founding year", url="https://example.com")',
            "description": "Read one page.",
        }],
        "require_llm_engine": False,
    },
    "Wikipedia_Search_Tool": {
        "tool_name": "Wikipedia_Search_Tool",
        "tool_description": (
            "Search Wikipedia titles only. Returns title, url, and snippet. "
            "Does not open the page. Call Web_Fetch_Tool with one url on a later step if the snippet is not enough."
        ),
        "tool_version": "1.0.0",
        "input_types": {"query": "str - the search query."},
        "output_type": "str - numbered hits, each with title, url, and snippet.",
        "demo_commands": [{"command": 'execution = tool.execute(query="Citibank")', "description": "Search Wikipedia."}],
        "require_llm_engine": False,
    },
}


def _format_hits(hits: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for index, hit in enumerate(hits, start=1):
        lines.append(f"{index}. title: {hit.get('title') or ''}")
        lines.append(f"   url: {hit.get('url') or ''}")
        lines.append(f"   snippet: {hit.get('snippet') or ''}")
    return "\n".join(lines)


def _strip_tags(text: str) -> str:
    import html as html_lib
    import re

    plain = re.sub(r"<[^>]+>", "", text or "")
    return " ".join(html_lib.unescape(plain).split())


def _parse_bing_html(html: str, limit: int = 5) -> List[Dict[str, str]]:
    import re

    pattern = re.compile(
        r'<h2[^>]*>\s*<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>\s*</h2>\s*<div class="b_caption"[^>]*>\s*<p[^>]*>(.*?)</p>',
        re.S | re.I,
    )
    hits: List[Dict[str, str]] = []
    for url, title, snippet in pattern.findall(html or ""):
        if any(host in url for host in _BING_SKIP):
            continue
        title_text = _strip_tags(title)
        snippet_text = _strip_tags(snippet)
        if not title_text:
            continue
        hits.append({"title": title_text, "url": url, "snippet": snippet_text})
        if len(hits) >= limit:
            break
    return hits


def _bing_hits(query: str) -> str:
    """Bing result cards only. Does not open the result URLs."""
    q = str(query or "").strip()
    if not q:
        return "Bing_Search_Tool requires a query"
    try:
        resp = _fast_get("https://cn.bing.com/search", {"q": q, "count": "5"})
        resp.raise_for_status()
        hits = _parse_bing_html(getattr(resp, "text", "") or "")
        return _format_hits(hits) or f"No Bing results for {q}"
    except Exception as exc:
        return f"Bing unreachable: {exc}"


def _wikipedia_hits(query: str) -> str:
    """One MediaWiki search. Returns titles, urls, and snippets. Does not open pages."""
    q = str(query or "").strip()
    if not q:
        return "Wikipedia_Search_Tool requires a query"
    try:
        search = _fast_get(
            "https://en.wikipedia.org/w/api.php",
            {"action": "query", "list": "search", "srsearch": q, "srlimit": "5", "format": "json"},
        )
        search.raise_for_status()
        raw = ((search.json().get("query") or {}).get("search") or [])[:5]
        hits: List[Dict[str, str]] = []
        for item in raw:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            hits.append({
                "title": title,
                "url": "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                "snippet": _strip_tags(str(item.get("snippet") or "")),
            })
        return _format_hits(hits) or f"No Wikipedia results for {q}"
    except Exception as exc:
        return f"Wikipedia unreachable: {exc}"


def _html_images(html: str, page_url: str, limit: int = 8) -> List[str]:
    import re
    from urllib.parse import urljoin

    lines: List[str] = []
    for tag in re.findall(r"<img\b[^>]*>", html or "", re.I)[:limit]:
        src_match = re.search(r'\b(?:src|data-src)\s*=\s*["\']([^"\']+)["\']', tag, re.I)
        if not src_match:
            continue
        src = src_match.group(1).strip()
        if not src or src.startswith("data:"):
            continue
        alt_match = re.search(r'\balt\s*=\s*["\']([^"\']*)["\']', tag, re.I)
        alt = " ".join((alt_match.group(1) if alt_match else "").split())
        full = urljoin(page_url, src)
        lines.append(f"- {full}" + (f" alt: {alt}" if alt else ""))
    return lines


def _pdf_text(data: bytes) -> str:
    if not data or not data.startswith(b"%PDF"):
        return ""
    try:
        import io

        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(data))
        parts: List[str] = []
        for page in list(reader.pages)[:3]:
            parts.append(page.extract_text() or "")
        return " ".join(" ".join(parts).split())
    except Exception:
        return ""


def _response_bytes(resp: Any, limit: int = 200000) -> bytes:
    raw = getattr(resp, "raw", None)
    data = b""
    if raw is not None and hasattr(raw, "read"):
        try:
            data = raw.read(limit, decode_content=True) or b""
        except TypeError:
            try:
                data = raw.read(limit) or b""
            except Exception:
                data = b""
        except Exception:
            data = b""
    if not data:
        data = getattr(resp, "content", b"") or b""
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    data = data[:limit]
    if data.startswith(b"\x1f\x8b"):
        import gzip
        try:
            data = gzip.decompress(data)
        except Exception:
            pass
    return data


def _web_fetch(query: str, url: str) -> str:
    """Open one url and branch on the response type. Does not search."""
    target = str(url or "").strip()
    if not target:
        return "Web_Fetch_Tool requires a url"
    try:
        import requests

        resp = requests.get(
            target,
            timeout=_FAST_TIMEOUT,
            headers={"User-Agent": "MAS-Science-Infra/1.0"},
            stream=True,
        )
    except Exception as exc:
        return f"Web fetch failed: {exc}"
    try:
        status = int(getattr(resp, "status_code", 0) or 0)
        if status >= 400:
            return f"Web fetch failed: HTTP {status} {target}"
        headers = getattr(resp, "headers", None) or {}
        try:
            ctype = str(headers.get("Content-Type") or "").split(";")[0].strip().lower()
        except Exception:
            ctype = ""
        path = target.lower().split("?", 1)[0]
        if ctype.startswith("image/") or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            size = ""
            try:
                size = str(headers.get("Content-Length") or "")
            except Exception:
                size = ""
            return (
                f"type: image\nurl: {target}\ncontent_type: {ctype or 'image'}\n"
                f"bytes: {size or 'unknown'}"
            )
        data = _response_bytes(resp)
        if "pdf" in ctype or path.endswith(".pdf"):
            text = _pdf_text(data)
            if not text:
                return f"type: pdf\nurl: {target}\nPDF has no text layer; it looks like a scan."
            return f"type: pdf\nurl: {target}\n{text[:_WEB_TEXT_LIMIT]}"
        try:
            body = data.decode("utf-8", "replace")
        except Exception:
            body = ""
        if "html" in ctype or body.lstrip().startswith("<"):
            text = _html_to_text(body)[:_WEB_TEXT_LIMIT]
            images = _html_images(body, target)
        else:
            text = " ".join(body.split())[:_WEB_TEXT_LIMIT]
            images = []
        q = str(query or "").strip()
        lines = [f"type: html\nurl: {target}"]
        if q:
            lines.append(f"query: {q}")
        lines.append(text or "(empty page)")
        if images:
            lines.append("images:")
            lines.extend(images)
        return "\n".join(lines)
    except Exception as exc:
        return f"Web fetch failed: {exc}"
    finally:
        close = getattr(resp, "close", None)
        if close:
            try:
                close()
            except Exception:
                pass


def _first_url(text: str) -> str:
    import re

    match = re.search(r"https?://[^\s'\"<>)]+", text or "")
    return match.group(0).rstrip(".,)") if match else ""


def _call_args(command: str, fallback_query: str, context: str) -> Tuple[str, str]:
    import re

    query = ""
    url = ""
    query_match = re.search(r"""query\s*=\s*(['"])(.*?)\1""", command or "", re.S)
    url_match = re.search(r"""url\s*=\s*(['"])(.*?)\1""", command or "", re.S)
    if query_match:
        query = query_match.group(2).strip()
    if url_match:
        url = url_match.group(2).strip()
    if not url:
        url = _first_url(command) or _first_url(fallback_query) or _first_url(context)
    if not query:
        query = str(fallback_query or "").strip()
    return query, url


def _run_dispatch(tool_name: str, query: str, url: str) -> str:
    kind = _DISPATCH_TOOLS.get(tool_name)
    if kind == "bing":
        return _bing_hits(query)
    if kind == "wiki":
        return _wikipedia_hits(query)
    if kind == "fetch":
        return _web_fetch(query, url)
    return f"No result was generated because the tool was not found: {tool_name}"


def _register_dispatch_tools(solver: Any, enabled: Optional[List[str]] = None) -> None:
    """Expose Bing and Web_Fetch without a ref_Rep package.

    Only names listed in the runtime enabled_tools are added. An imported
    Web_Search_Tool is not promoted just because Wikipedia was loaded.
    """
    planner = getattr(solver, "planner", None)
    if planner is None:
        return
    tools = getattr(planner, "available_tools", None)
    if not isinstance(tools, list):
        tools = []
        planner.available_tools = tools
    meta = getattr(planner, "toolbox_metadata", None)
    if not isinstance(meta, dict):
        meta = {}
        planner.toolbox_metadata = meta
    allowed = {str(name) for name in (enabled or [])}
    for name, item in _DISPATCH_META.items():
        if allowed and name not in allowed:
            continue
        if name not in tools:
            tools.append(name)
        meta[name] = dict(item)
    wiki = meta.get("Wikipedia_Search_Tool")
    if isinstance(wiki, dict):
        wiki = dict(wiki)
        wiki["tool_description"] = (
            "Search Wikipedia titles only. Returns title, url, and snippet. "
            "Does not open the page. Call Web_Fetch_Tool with one url on a later step if the snippet is not enough."
        )
        wiki["demo_commands"] = [{"command": 'execution = tool.execute(query="Citibank")', "description": "Search Wikipedia."}]
        meta["Wikipedia_Search_Tool"] = wiki
    memory = getattr(solver, "system_memory", None)
    mem_meta = getattr(memory, "toolbox_metadata", None) if memory is not None else None
    if isinstance(mem_meta, dict) and mem_meta is not meta:
        mem_meta.update({name: dict(item) for name, item in meta.items() if name in _DISPATCH_META or name == "Wikipedia_Search_Tool"})


def _install_fast_search() -> None:
    """Wikipedia's own execute must not open the article or call Web_Search."""
    try:
        import importlib

        mod = importlib.import_module("tools.wikipedia_search.tool")
    except Exception:
        return
    cls = getattr(mod, "Wikipedia_Search_Tool", None)
    if cls is None or getattr(getattr(cls, "execute", None), "_science_fast", False):
        return

    def execute(self: Any, query: str = "", *args: Any, **kwargs: Any) -> str:
        return _wikipedia_hits(str(query or kwargs.get("query") or ""))

    execute._science_fast = True  # type: ignore[attr-defined]
    cls.execute = execute  # type: ignore[method-assign]


def _guard_wikipedia_sys_exit() -> None:
    """Wikipedia_Search_Tool calls sys.exit when OPENAI_API_KEY missing — convert to error string."""
    try:
        from tools.wikipedia_search import tool as wiki_mod  # type: ignore
    except Exception:
        return
    cls = getattr(wiki_mod, "Wikipedia_Search_Tool", None)
    if cls is None or getattr(cls.execute, "_science_no_exit", False):
        return
    orig = cls.execute

    def execute(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return orig(self, *args, **kwargs)
        except SystemExit as exc:
            return f"Wikipedia_Search_Tool aborted: missing API key or sys.exit({exc.code})"

    execute._science_no_exit = True  # type: ignore[attr-defined]
    cls.execute = execute  # type: ignore[method-assign]


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
    runtime = _load_runtime()
    n = int(runtime.get("n") or 1)
    enabled = list(runtime.get("enabled_tools") or list(DEFAULT_RUNTIME["enabled_tools"]))
    with _epc_tools_scope(root):
        enabled = _patch_google_chrome_or_demote(enabled)
        solver = construct_solver(
            llm_engine_name=model,
            verbose=False,
            n=n,
            enabled_tools=enabled,
            max_steps=int(runtime.get("max_steps") or 20),
            max_time=int(runtime.get("max_time") or 3000),
            max_tokens=int(runtime.get("max_tokens") or 4000),
            temperature=float(runtime.get("temperature") or 0.0),
        )
        _install_threadsafe_timeouts()
        _patch_google_search_model_string()
        _guard_wikipedia_sys_exit()
        _install_fast_search()
    _register_dispatch_tools(solver, enabled)
    if hasattr(solver, "planner") and hasattr(solver.planner, "n"):
        solver.planner.n = n
    _patch_engines(solver)
    _install_threadsafe_timeouts()
    _install_completion_budget()
    available = list(getattr(getattr(solver, "planner", None), "available_tools", None) or [])
    missing = [name for name in enabled if name not in available]
    _progress(f"available_tools={available}")
    if missing:
        _progress(f"tools skipped (missing dep or init failed): {missing}")
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
        "available_tools": available,
        "skipped_tools": missing,
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
        self.planner_selected_plan: Any = None
        self.bts_selected_plan: Any = None
        self.tool_available = True
        self.failed_tools: List[str] = []
        self.max_steps = _max_steps()
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
        self._active_question = ""

    def reset_for_question(self) -> None:
        """Clear per-question state so eval questions do not contaminate each other."""
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
        self.last_plan = None
        self.last_verify_analysis = ""
        self.query_analysis = ""
        self.planner_selected_index = "0"
        self.bts_selected_index = "0"
        self.planner_selected_plan = None
        self.bts_selected_plan = None
        self.tool_available = True
        self.failed_tools = []
        self.max_steps = _max_steps()
        self._analyzed = False
        self.json_data = {}
        self.last_trace = {}
        self._active_question = ""
        memory = self.system_memory
        if hasattr(memory, "obtained_information") and isinstance(getattr(memory, "obtained_information", None), list):
            memory.obtained_information.clear()
        if hasattr(memory, "set_outline"):
            try:
                memory.set_outline({})
            except Exception:
                pass
        if hasattr(memory, "last_step_plan_scores") and isinstance(getattr(memory, "last_step_plan_scores", None), dict):
            memory.last_step_plan_scores = {
                "plan_list": [],
                "planner": {},
                "executor": {},
                "diagnoser": {},
            }
        if hasattr(memory, "actions") and isinstance(getattr(memory, "actions", None), dict):
            memory.actions.clear()
        for role in (self.planner, self.executor, self.diagnoser):
            role_mem = getattr(role, "memory", None)
            if role_mem is None:
                continue
            if hasattr(role_mem, "actions"):
                actions = getattr(role_mem, "actions")
                if isinstance(actions, list):
                    actions.clear()
                elif isinstance(actions, dict):
                    actions.clear()
            if hasattr(role_mem, "epistemic_constraint") and isinstance(getattr(role_mem, "epistemic_constraint", None), list):
                role_mem.epistemic_constraint.clear()
            if hasattr(role_mem, "constraints") and isinstance(getattr(role_mem, "constraints", None), list):
                role_mem.constraints.clear()
        if hasattr(self.executor, "executed") and isinstance(getattr(self.executor, "executed", None), list):
            self.executor.executed.clear()
        if hasattr(self.diagnoser, "verify_calls"):
            try:
                self.diagnoser.verify_calls = 0
            except Exception:
                pass

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
            self.question = question.split("\n\nDecide the next step", 1)[0].strip() or self.question
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
        """Keep original tool name. Illegal tools are skipped in exec_window (Solver.solve style)."""
        available = self._available_tools()
        name = str(tool_name or "").strip()
        if not name or "none" in name.lower():
            return "Base_Generator_Tool", "Base_Generator_Tool" in available
        if name.lower().startswith("no matched"):
            return name, False
        return name, name in available

    def _select_plan(self, plan: Any, question: str, target: str) -> Any:
        if _is_llm_error(plan):
            self.last_trace["llm_error"] = plan
            self.planner_selected_plan = SimpleNamespace(context=target, sub_goal=target, tool_name="Base_Generator_Tool")
            self.bts_selected_plan = self.planner_selected_plan
            return self.bts_selected_plan
        # Solver.solve: BTS whenever plan_list is a list (including len==1).
        if not isinstance(plan, list):
            self.planner_selected_plan = plan
            self.bts_selected_plan = plan
            self.planner_selected_index = "0"
            self.bts_selected_index = "0"
            return plan
        if not plan:
            fallback = SimpleNamespace(context=target, sub_goal=target, tool_name="Base_Generator_Tool")
            self.planner_selected_plan = fallback
            self.bts_selected_plan = fallback
            return fallback
        cleaned: Any = plan
        if hasattr(self.solver, "process_next_step"):
            try:
                cleaned, _ = self.solver.process_next_step(plan)
            except Exception:
                cleaned = {str(i): item for i, item in enumerate(plan)}
        if isinstance(cleaned, list):
            cleaned = {str(i): item for i, item in enumerate(cleaned)}
        if not isinstance(cleaned, dict) or not cleaned:
            fallback = plan[0]
            self.planner_selected_plan = fallback
            self.bts_selected_plan = fallback
            self.planner_selected_index = "0"
            self.bts_selected_index = "0"
            return fallback
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
                fallback = cleaned.get("0") or next(iter(cleaned.values()))
                self.planner_selected_plan = fallback
                self.bts_selected_plan = fallback
                return fallback
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
            self.planner_selected_plan = cleaned.get(self.planner_selected_index) or cleaned.get("0") or next(iter(cleaned.values()))
            self.bts_selected_plan = cleaned.get(self.bts_selected_index) or cleaned.get("0") or next(iter(cleaned.values()))
            return self.bts_selected_plan
        except Exception as exc:
            self.last_trace["llm_error"] = str(exc)
            self.planner_selected_index = "0"
            self.bts_selected_index = "0"
            fallback = cleaned.get("0") or next(iter(cleaned.values()))
            self.planner_selected_plan = fallback
            self.bts_selected_plan = fallback
            return fallback

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
        prev_q = getattr(self, "_active_question", "") or ""
        if question and question != prev_q:
            if prev_q or self._analyzed or self.step_count > 0:
                self.reset_for_question()
            self._active_question = question
        self.question = question
        self.step_count += 1
        _progress("正在生成下一步")
        self._analyze_once(question)
        target = _first_outline_target(self.system_memory)
        obtained = self._obtained()
        try:
            raw = self.planner.generate_next_step(
                question, None, target, _feasibility(), self.step_count, self.max_steps or _max_steps(), obtained, self.json_data
            )
        except Exception as exc:
            self.last_trace["llm_error"] = str(exc)
            raw = SimpleNamespace(context=target, sub_goal=target, tool_name="Base_Generator_Tool"), ""
        plan = raw[0] if isinstance(raw, tuple) and raw else raw
        if _is_llm_error(plan):
            self.last_trace["llm_error"] = plan
            plan = SimpleNamespace(context=target, sub_goal=target, tool_name="Base_Generator_Tool")
        _progress("正在选择方案")
        selected = self._select_plan(plan, question, target)
        self.last_plan = selected
        if self.bts_selected_plan is None:
            self.bts_selected_plan = selected
        if self.planner_selected_plan is None:
            self.planner_selected_plan = selected
        # Prefer structured attrs when present. Avoid extract() on error/SimpleNamespace
        # fallbacks — that path yields "No matched tool given:" and skips execution.
        if self.last_trace.get("llm_error") or _is_llm_error(selected):
            context, sub_goal, tool_name = target, target, "Base_Generator_Tool"
        elif hasattr(selected, "tool_name") and getattr(selected, "tool_name", None) is not None:
            context = str(getattr(selected, "context", "") or target)
            sub_goal = str(getattr(selected, "sub_goal", "") or target)
            tool_name = str(getattr(selected, "tool_name", "") or "Base_Generator_Tool")
        elif hasattr(self.planner, "extract_context_subgoal_and_tool"):
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
        tool_name = str(tool_name or "").strip()
        if not tool_name or tool_name.lower().startswith("no matched"):
            self.last_trace["tool_fallback"] = tool_name or "empty"
            tool_name = "Base_Generator_Tool"
        tool_name, tool_ok = self._normalize_tool(tool_name)
        prev_output = str(self.last_tool_output or "")
        prev_goal = str(self.last_sub_goal or "")
        if _is_transport_failure(prev_output):
            if prev_goal and sub_goal != prev_goal:
                self.last_trace["subgoal_held"] = {"from": sub_goal, "to": prev_goal}
                sub_goal = prev_goal
            if tool_name == "Base_Generator_Tool" or tool_name in self.failed_tools:
                replacement = _next_search_tool(self._available_tools(), self.failed_tools)
                if replacement and replacement != tool_name:
                    self.last_trace["search_fallback"] = {"from": tool_name, "to": replacement}
                    tool_name = replacement
                    tool_ok = True
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
            output = "No result was generated because the tool was not found."
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
        _progress("正在生成工具命令")
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
        _progress(f"正在执行 {tool_name}")
        if tool_name in _DISPATCH_TOOLS:
            query, url = _call_args(self.last_command, sub_goal or question, context)
            result = _run_dispatch(tool_name, query, url)
        else:
            try:
                with _epc_tools_scope():
                    # Each scope reimports tools.*, so the Chrome binary patch must be reapplied.
                    _patch_google_chrome_or_demote(self._available_tools() or ["Google_Search_Tool"])
                    _guard_wikipedia_sys_exit()
                    _install_fast_search()
                    result = self.executor.execute_tool_command(tool_name, self.last_command)
            except Exception as exc:
                result = f"tool error: {exc}"
        output = result if isinstance(result, str) else str(result)
        self.last_tool_output = output
        transport_fail = _is_transport_failure(output)
        infra_fail = transport_fail or any(token in output for token in _INFRA_TOOL_ERRORS)
        if infra_fail and tool_name and tool_name not in self.failed_tools:
            self.failed_tools.append(tool_name)
            _progress(f"mark failed_tools += {tool_name}")
        self.last_trace = {
            "tool": tool_name,
            "command": self.last_command,
            "output": output,
            "analysis": self.last_analysis,
            "explanation": self.last_explanation,
            "failed_tools": list(self.failed_tools),
        }
        return {
            "kind": "tool_result",
            "payload": {
                "output": output,
                "ok": not infra_fail,
                "evidence_type": "ERROR" if infra_fail else ("DIRECT" if output else "EMPTY"),
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
        _progress("正在校验结果")
        if _is_transport_failure(self.last_tool_output):
            self.last_trace["transport_failure"] = True
            analysis = "检索没有连上，这条结果不能当作证据。"
            new_flag = False
            new_info = ""
            conclusion = "CONTINUE"
        else:
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
            if (
                not self.last_trace.get("transport_failure")
                and str(self.planner_selected_index) != str(self.bts_selected_index)
                and hasattr(self.diagnoser, "epc_aw_diagnosis")
            ):
                try:
                    epc_aw_analysis, constraint = self.diagnoser.epc_aw_diagnosis(
                        self.planner_selected_plan if self.planner_selected_plan is not None else self.last_plan,
                        self.bts_selected_plan if self.bts_selected_plan is not None else self.last_plan,
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
            if not self.last_trace.get("transport_failure") and hasattr(self.diagnoser, "update_outline"):
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
        hit_limit = self.step_count >= (self.max_steps or _max_steps())
        should_answer = ready or str(conclusion).strip().upper() == "STOP" or hit_limit
        if should_answer and hasattr(self.executor, "generate_direct_output"):
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
        stop_now = bool(ready) or hit_limit
        return {
            "kind": "verify",
            "payload": {
                "ok": stop_now,
                "reason": str(analysis or conclusion),
                "step_conclusion": "COMPLETE" if stop_now else "INCOMPLETE",
                "slot_updates": [],
                "ready_to_stop": stop_now,
                "answer": answer,
                "direct_output": answer,
                "trace": dict(self.last_trace),
            },
        }


_CTX: Optional[EpcAwEpisodeContext] = None


def _endpoint_fingerprint() -> str:
    return "|".join(
        (
            os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "",
            os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or "",
        )
    )


def get_episode_context(*, reset: bool = False) -> EpcAwEpisodeContext:
    global _CTX
    fingerprint = _endpoint_fingerprint()
    if _CTX is None or getattr(_CTX, "_endpoint_fp", None) != fingerprint:
        _CTX = EpcAwEpisodeContext()
        _CTX._endpoint_fp = fingerprint
    elif reset:
        _CTX.reset_for_question()
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


def epc_aw_sampling_sites(executor_id: str) -> List[Dict[str, Any]]:
    """Branch after each EPC-AW window. No router site."""
    return [
        {
            "id": "after_planner",
            "enabled": True,
            "anchor": {"kind": "after_agent_turn", "agent_id": "planner"},
            "gate": {"type": "always"},
            "fork": {"beam_size": 2, "resume_mode": "messages"},
        },
        {
            "id": f"after_{executor_id}",
            "enabled": True,
            "anchor": {"kind": "after_agent_turn", "agent_id": executor_id},
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
                "label": "epc_aw_planner",
                "trainable": True,
                "system_prompt": "Wrap EPC-AW Planner.generate_next_step. next is the executor window.",
                "tools": [],
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "EPC-AW.Planner"},
            },
            {
                "id": executor_id,
                "kind": "blank",
                "role": "executor",
                "label": "epc_aw_executor",
                "trainable": False,
                "system_prompt": "Wrap EPC-AW Executor window (tool command + execute).",
                "profile": {"backend": "user_space", "user_project": project_id, "llm_required": False},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "EPC-AW.Executor"},
            },
            {
                "id": "verifier",
                "kind": "verifier",
                "role": "verifier",
                "label": "epc_aw_verifier",
                "trainable": False,
                "system_prompt": "Wrap EPC-AW Diagnoser.verificate_context.",
                "profile": {"backend": "user_space", "user_project": project_id},
                "meta": {"origin": "user", "user_project": project_id, "wraps": "EPC-AW.Diagnoser"},
            },
        ],
        "routers": [],
        "edges": [
            {"from": "planner", "to": executor_id, "kind": "message"},
            {"from": executor_id, "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
        "sampling": {
            "mode": "arpo",
            "group_n": 4,
            "beam_size": 2,
            "sites": epc_aw_sampling_sites(executor_id),
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
    runtime_path = root / "contracts" / "runtime.yaml"
    if not runtime_path.is_file():
        _dump(
            runtime_path,
            {
                "n": 1,
                "max_steps": 20,
                "max_time": 3000,
                "max_tokens": 4000,
                "temperature": 0.0,
                "enabled_tools": [
                    "Base_Generator_Tool",
                    "Python_Coder_Tool",
                    "Wikipedia_Search_Tool",
                    "Bing_Search_Tool",
                    "Web_Fetch_Tool",
                ],
            },
        )
    source = find_epc_aw_import_root(project_id)
    manifest.update(
        {
            "mode": "native_mas",
            "status": "ready",
            "title": title,
            "agent_ids": ["planner", executor_id, "verifier"],
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
        "agent_ids": ["planner", executor_id, "verifier"],
        "tool_ids": [executor_id],
        "workflow": workflow,
    }
