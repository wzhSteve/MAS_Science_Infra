"""Shared EPC-AW episode context. One SystemMemory for planner/executor/diagnoser.

Lives in user_space. May import MAS.epc_aw after putting the uploaded or
upload/ EPC-AW tree on sys.path. Collect / eval / sampling call run_window only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple

SOLVE_CALLS = 0
EXECUTOR_ID = "u_epc-aw-main__executor"


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
    flag = os.environ.get("SCIENCE_EPC_AW_MOCK") or ""
    return flag.strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_RUNTIME: Dict[str, Any] = {
    "n": 1,
    "max_steps": 20,
    "max_time": 3000,
    "max_tokens": 4000,
    "temperature": 0.7,
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
)


def _is_transport_failure(output: Any) -> bool:
    """Search host did not answer. Distinct from a Python snippet timeout."""
    text = str(output or "")
    if text.startswith("Wikipedia unreachable:"):
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


# cheap: n=1 + lite prompts — BTS/belief/epc_aw_diagnosis never fire.
# paper: n>=2 + official prompts — Solver.solve path with BTS consensus and diagnosis.
PROFILES: Dict[str, Dict[str, Any]] = {
    "cheap": {"n": 1, "prompts_lite": True},
    "paper": {"n": 3, "prompts_lite": False},
}


def _runtime_yaml() -> Dict[str, Any]:
    path = _runtime_path()
    if not path.is_file():
        return {}
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _profile_name() -> str:
    raw = os.environ.get("SCIENCE_EPC_AW_PROFILE") or _runtime_yaml().get("profile") or ""
    name = str(raw).strip().lower()
    return name if name in PROFILES else ""


def _profile_overrides() -> Dict[str, Any]:
    name = _profile_name()
    if not name:
        return {}
    over = dict(PROFILES[name])
    if os.environ.get("SCIENCE_EPC_AW_PROFILE"):
        return over
    data = _runtime_yaml()
    for key in over:
        if key in data:
            over[key] = data[key]
    return over


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
    over = _profile_overrides()
    if "n" in over:
        cfg["n"] = over["n"]
    cfg["profile"] = _profile_name() or "custom"
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


def _upstream_tool_modules(sub: str) -> List[Any]:
    """EPC-AW tool modules under both import names (``MAS.epc_aw.tools.<sub>`` and
    the Initializer's ``tools.<sub>``). Only modules whose file lives in the EPC-AW
    tree are returned, so management ``mas/tools`` can never be patched by mistake.
    """
    import importlib

    root = _import_root()
    if root is None:
        return []
    epc_tools = (root / "MAS" / "epc_aw" / "tools").resolve()
    found: List[Any] = []
    for name in (f"MAS.epc_aw.tools.{sub}", "tools." + sub):
        try:
            mod = importlib.import_module(name)
        except Exception:
            continue
        path = Path(str(getattr(mod, "__file__", "") or "")).resolve()
        if epc_tools in path.parents and mod not in found:
            found.append(mod)
    return found


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
    # mas/ is often already on sys.path ahead of this project. If the EPC
    # tools parent is present but not first, ``import tools`` binds management
    # mas/tools and every official tool then fails to load.
    prior_index: Optional[int] = None
    if parent:
        if parent in sys.path:
            prior_index = sys.path.index(parent)
            sys.path.remove(parent)
        sys.path.insert(0, parent)
    try:
        yield
    finally:
        for key in list(sys.modules):
            if key == "tools" or key.startswith("tools."):
                sys.modules.pop(key, None)
        sys.modules.update(saved)
        if parent and parent in sys.path:
            sys.path.remove(parent)
        if prior_index is not None:
            sys.path.insert(min(prior_index, len(sys.path)), parent)


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


_NO_THINK_SUFFIX = "\n/no_think"


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
    # Belt and braces: vLLM's OpenAI layer drops unknown fields like
    # ``enable_thinking``, and the LiteLLM proxy may strip ``extra_body``
    # entirely. Qwen3's chat template also honors the soft ``/no_think``
    # switch inside the message content, so inject it into the last user
    # message as well. Idempotent: never append twice.
    messages = out.get("messages")
    if isinstance(messages, list) and messages:
        for item in reversed(messages):
            if not isinstance(item, dict) or item.get("role") != "user":
                continue
            content = item.get("content")
            if isinstance(content, str):
                if not content.rstrip().endswith("/no_think"):
                    item["content"] = content.rstrip() + _NO_THINK_SUFFIX
            elif isinstance(content, list):
                for part in reversed(content):
                    if isinstance(part, dict) and part.get("type") == "text":
                        text = str(part.get("text") or "")
                        if not text.rstrip().endswith("/no_think"):
                            part["text"] = text.rstrip() + _NO_THINK_SUFFIX
                        break
            break
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
    _wrap_engine_llm_recorder(eng)


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


# --- Fix E: LLM call recording (prompt head / model / n / response head) ----
_LLM_CALL_LOG: List[Dict[str, Any]] = []
_LLM_CALL_LOG_CAP = 64


def _prompt_head(prompt: Any, limit: int = 500) -> str:
    if isinstance(prompt, list):
        text = " ".join(str(x) for x in prompt if not isinstance(x, (bytes, bytearray)))
    else:
        text = str(prompt if prompt is not None else "")
    text = " ".join(text.split())
    return text[:limit]


def _response_head(response: Any, limit: int = 300) -> str:
    if isinstance(response, list):
        text = " ".join(str(x) for x in response if not isinstance(x, (bytes, bytearray)))
    else:
        text = str(response if response is not None else "")
    text = " ".join(text.split())
    return text[:limit]


def record_llm_call(entry: Dict[str, Any]) -> None:
    """Append one _generate_text record to the episode-level ring buffer."""
    _LLM_CALL_LOG.append(dict(entry))
    if len(_LLM_CALL_LOG) > _LLM_CALL_LOG_CAP:
        del _LLM_CALL_LOG[: len(_LLM_CALL_LOG) - _LLM_CALL_LOG_CAP]


def drain_llm_calls() -> List[Dict[str, Any]]:
    """Return and clear records logged since the last drain."""
    out = list(_LLM_CALL_LOG)
    _LLM_CALL_LOG.clear()
    return out


def _wrap_engine_llm_recorder(eng: Any) -> None:
    """Wrap eng._generate_text (already budget-wrapped) so each call is logged."""
    fn = getattr(eng, "_generate_text", None)
    if fn is None or getattr(fn, "_science_llm_log", False):
        return
    orig = getattr(fn, "__func__", fn)

    def wrapped(self: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        model = str(getattr(self, "model_string", "") or "")
        n = kwargs.get("n") if isinstance(kwargs.get("n"), int) else 1
        try:
            resp = orig(self, prompt, *args, **kwargs)
        except Exception as exc:
            record_llm_call(
                {
                    "model": model,
                    "n": n,
                    "prompt_head": _prompt_head(prompt),
                    "response_head": f"{type(exc).__name__}: {exc}"[:300],
                }
            )
            raise
        response_head = resp
        if isinstance(resp, list) and resp:
            response_head = resp[0]
        record_llm_call(
            {
                "model": model,
                "n": n,
                "prompt_head": _prompt_head(prompt),
                "response_head": _response_head(response_head),
            }
        )
        return resp

    wrapped._science_llm_log = True  # type: ignore[attr-defined]
    wrapped._science_budget = True  # type: ignore[attr-defined]
    eng._generate_text = wrapped.__get__(eng, type(eng))  # type: ignore[method-assign]


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


# --- Prompt lite injection (mirrors agentflow-side isomorphic task) -------
# ref models do open(prompt_file).read() then .format(**placeholders) on every
# window call. Redirect those four main-chain templates to adapted/prompts/
# *.lite.txt via per-module open shims (NOT builtins.open — a global builtins
# patch previously broke MasExecutionRunner isolation).
_LITE_DIR = Path(__file__).resolve().parent / "prompts"
_LITE_REDIRECTS = {
    "prompts/planner/generate_next_step.txt": "planner/generate_next_step.lite.txt",
    "prompts/diagnoser/verificate_context.txt": "diagnoser/verificate_context.lite.txt",
    "prompts/executor/generate_tool_command.txt": "executor/generate_tool_command.lite.txt",
    "prompts/executor/direct_output.txt": "executor/direct_output.lite.txt",
}
_LITE_MARKER = "_science_prompt_lite"
_LITE_ANNOUNCED = False


def _prompts_lite_enabled() -> bool:
    """contracts/runtime.yaml: prompts_lite (default true); profile overrides it."""
    raw = os.environ.get("SCIENCE_EPC_AW_PROMPTS_LITE")
    if raw is None or not raw.strip():
        over = _profile_overrides()
        data = _runtime_yaml()
        if "prompts_lite" in over:
            raw = "1" if bool(over["prompts_lite"]) else "0"
        elif "prompts_lite" in data:
            raw = "1" if bool(data["prompts_lite"]) else "0"
    if raw is None or not raw.strip():
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _is_lite_candidate(path: str) -> Optional[Path]:
    """Map an EPC-AW prompts/<role>/<name>.txt read to adapted/*.lite.txt."""
    try:
        text = str(path)
    except Exception:
        return None
    # Normalize separators; avoid Path.resolve() (slow / noisy under isolation).
    norm = text.replace("\\", "/")
    for suffix, rel in _LITE_REDIRECTS.items():
        if norm.endswith(suffix):
            lite = _LITE_DIR / rel
            return lite if lite.is_file() else None
    return None


def _restore_builtins_open_if_hijacked() -> None:
    """Undo legacy builtins.open patch left by earlier opt-epc revisions."""
    import builtins

    current = builtins.open
    if getattr(current, _LITE_MARKER, False):
        orig = getattr(current, "_science_orig_open", None)
        if orig is not None:
            builtins.open = orig  # type: ignore[assignment]


def _install_lite_prompts() -> None:
    """Install per-model-module open() shims redirecting to adapted lite templates.

    Idempotent via _science_prompt_lite marker on each module.open. Never
    patches builtins.open (evaluator isolation previously crashed on that).
    """
    _restore_builtins_open_if_hijacked()
    if not _prompts_lite_enabled():
        return
    redirects = {
        suffix: str(_LITE_DIR / rel)
        for suffix, rel in _LITE_REDIRECTS.items()
        if (_LITE_DIR / rel).is_file()
    }
    if not redirects:
        return
    import builtins

    modules = []
    for mod_name in (
        "MAS.epc_aw.models.planner",
        "MAS.epc_aw.models.diagnoser",
        "MAS.epc_aw.models.executor",
    ):
        try:
            modules.append(__import__(mod_name, fromlist=["_"]))
        except Exception:
            continue
    if not modules:
        return

    for module in modules:
        existing = getattr(module, "open", None)
        if getattr(existing, _LITE_MARKER, False):
            continue
        orig_open = existing if existing is not None else builtins.open

        def _make_open(orig_open, redirects):
            def _lite_open(file, *args, **kwargs):
                try:
                    path = str(file).replace("\\", "/")
                except Exception:
                    path = ""
                for suffix, lite_path in redirects.items():
                    if path.endswith(suffix) and path != lite_path:
                        return orig_open(lite_path, *args, **kwargs)
                return orig_open(file, *args, **kwargs)

            return _lite_open

        shim = _make_open(orig_open, redirects)
        setattr(shim, _LITE_MARKER, True)
        shim._science_orig_open = orig_open  # type: ignore[attr-defined]
        module.open = shim  # type: ignore[attr-defined]

    global _LITE_ANNOUNCED
    if not _LITE_ANNOUNCED:
        _LITE_ANNOUNCED = True
        print(
            f"EPC-AW prompts_lite: {len(redirects)} templates -> adapted/prompts/**/*.lite.txt",
            flush=True,
        )


def _install_json_tolerant_parsing() -> None:
    """Coerce fenced/partial LLM JSON output before json.loads raises ValueError.

    The EPC-AW engine classes funnel every call through _generate_text, and
    the model layer consumes the result with json.loads. Wrap
    ChatOpenAI/ChatVLLM._generate_text so its return value is pre-cleaned:
    strip code fences, extract the first balanced JSON object/array, try the
    ```json block, or degrade to a structured error payload instead of
    letting ValueError propagate.
    """
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
        if fn is None or getattr(fn, "_science_json_tolerant", False):
            continue
        setattr(cls, "_generate_text", _json_tolerant_wrapper(fn))


def _install_memory_dedup(limit: int = 12) -> None:
    """Dedup role-memory get_actions() by (tool_name, tool_input), newest first.

    EPC-AW's Memory.get_actions() returns {'Planner Step N': {...}} dicts that
    grow every step and then get str()-formatted into subsequent prompts.
    Repeated (tool, command) turns collapse to one entry, capped at `limit`.
    """
    for spec in (
        ("MAS.epc_aw.models.memory", "Memory"),
    ):
        try:
            mod = __import__(spec[0], fromlist=[spec[1]])
            cls = getattr(mod, spec[1])
        except Exception:
            continue
        fn = getattr(cls, "get_actions", None)
        if fn is None or getattr(fn, "_science_dedup", False):
            continue

        def make_dedup(orig: Any) -> Any:
            def get_actions(self: Any) -> Any:
                actions = orig(self)
                if not isinstance(actions, dict):
                    return actions
                try:
                    deduped: Dict[str, Any] = {}
                    seen: set = set()
                    for key in sorted(actions, key=_action_step_key, reverse=True):
                        item = actions[key]
                        token = _action_dedup_token(item)
                        if token is not None and token in seen:
                            continue
                        if token is not None:
                            seen.add(token)
                        deduped[key] = item
                        if len(deduped) >= limit:
                            break
                    if len(deduped) == len(actions):
                        return actions
                    # keep chronological order (oldest first) for prompt coherence
                    return {k: deduped[k] for k in sorted(deduped, key=_action_step_key)}
                except Exception:
                    return actions

            get_actions._science_dedup = True  # type: ignore[attr-defined]
            return get_actions

        cls.get_actions = make_dedup(fn)  # type: ignore[method-assign]


def _action_step_key(key: Any) -> Tuple[int, str]:
    text = str(key)
    digits = re.findall(r"\d+", text)
    return (int(digits[-1]) if digits else 0), text


def _action_dedup_token(item: Any) -> Optional[Tuple[str, str]]:
    if not isinstance(item, dict):
        return None
    tool = str(item.get("tool_name") or item.get("tool") or "")
    command = str(item.get("tool_input") or item.get("command") or "")
    if not tool and not command:
        return None
    return tool, command


def _json_strip_code_fence(text: str) -> str:
    t = text.strip()
    m = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)```", t)
    if m:
        return m.group(1).strip()
    return t


def _json_first_balanced(text: str) -> str:
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        in_str = False
        esc = False
        for i, ch in enumerate(text):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                if depth > 0:
                    in_str = True
                elif start < 0:
                    pass
            elif ch == opener:
                if depth == 0 and start < 0:
                    start = i
                depth += 1
            elif ch == closer and depth > 0:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return ""


def _json_tolerant_clean(text: str) -> str:
    """fence strip → balanced substring → ```json block → raw as-is."""
    if not isinstance(text, str):
        return text
    out = _json_strip_code_fence(text)
    if not out:
        return text
    # The global tolerant_loads hook calls back into this function.
    raw_loads = getattr(json.loads, "_science_orig_loads", json.loads)
    try:
        raw_loads(out)
        return out
    except Exception:
        pass
    balanced = _json_first_balanced(out)
    if balanced:
        try:
            raw_loads(balanced)
            return balanced
        except Exception:
            pass
    return out


_TEXT_FALLBACK_FORMATS = frozenset({"ToolCommand", "NextStep"})


def _json_tolerant_wrapper(orig: Any) -> Any:
    def wrapped(self: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        resp = orig(self, prompt, *args, **kwargs)
        response_format = kwargs.get("response_format")
        if response_format is None:
            for a in args:
                if isinstance(a, type) and getattr(a, "model_json_schema", None):
                    response_format = a
                    break
        if response_format is None:
            return resp
        if isinstance(resp, list):
            # n>1 candidates (ChatOpenAI server path): clean each, keep the list shape.
            return [wrapped_one(r, response_format) for r in resp]
        return wrapped_one(resp, response_format)

    def wrapped_one(resp: Any, response_format: Any) -> Any:
        try:
            cleaned = _json_tolerant_clean(resp)
            try:
                json.loads(cleaned)  # verifiable payload
            except Exception:
                # Executor.extract_explanation_and_command / parse_response_to_fields
                # parse plain text themselves; an error dict would skip that path.
                if getattr(response_format, "__name__", "") in _TEXT_FALLBACK_FORMATS and isinstance(resp, str):
                    return resp
                return {
                    "error": "json_parse_failed",
                    "message": _response_head(resp, 400),
                }
            return cleaned
        except Exception as exc:
            return {
                "error": "json_parse_failed",
                "message": f"{type(exc).__name__}: {exc}"[:400],
            }

    wrapped._science_json_tolerant = True  # type: ignore[attr-defined]
    wrapped._science_budget = True  # type: ignore[attr-defined]
    return wrapped


def _install_json_tolerant_parse_hook() -> None:
    """Belt-and-braces: json.loads of LLM text is also fenced/balanced-cleaned.

    Some EPC-AW call sites parse strings that never passed a response_format,
    so globally guard json.loads only for str inputs — non-str payloads
    pass through untouched.
    """
    if getattr(json.loads, "_science_json_tolerant", False):
        return
    orig_loads = json.loads

    def tolerant_loads(s: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        if isinstance(s, str):
            try:
                return orig_loads(s, *args, **kwargs)
            except ValueError:
                cleaned = _json_tolerant_clean(s)
                if cleaned != s:
                    return orig_loads(cleaned, *args, **kwargs)
                raise
        return orig_loads(s, *args, **kwargs)

    tolerant_loads._science_json_tolerant = True  # type: ignore[attr-defined]
    tolerant_loads._science_orig_loads = orig_loads  # type: ignore[attr-defined]
    json.loads = tolerant_loads  # type: ignore[assignment]


def _install_engine_patches() -> None:
    os.environ.setdefault("ENABLE_THINKING", "0")
    os.environ.setdefault("SCIENCE_EPC_AW_MAX_MODEL_LEN", "8192")
    os.environ.setdefault("SCIENCE_EPC_AW_MAX_COMPLETION", "2048")
    _install_client_timeout()
    _install_threadsafe_timeouts()
    _install_completion_budget()
    _install_lite_prompts()
    _install_json_tolerant_parse_hook()
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
    """Google search is removed. Drop the name if an old list still contains it."""
    return [
        str(item) for item in (enabled_tools or [])
        if str(item).strip() and str(item) != "Google_Search_Tool"
    ]


def _patch_google_search_model_string() -> None:
    """Google search is removed. Kept as a no-op for existing call sites."""
    return None


def _fix_google_model_string(_cls: Any) -> None:
    return None




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


_DISPATCH_TOOLS = {
    "Bing_Search_Tool": "bing",
    "Wikipedia_Search_Tool": "wiki",
    "Web_Fetch_Tool": "fetch",
    "Web_Search_Tool": "fetch",
}
_DISPATCH_META: Dict[str, Dict[str, Any]] = {
    "Web_Fetch_Tool": {
        "tool_name": "Web_Fetch_Tool",
        "tool_description": (
            "Input is a URL. Opens exactly that page and extracts its readable text content "
            "(HTML text, image metadata, or PDF text layer). This tool does not search; "
            "get a url from Bing_Search_Tool or Wikipedia_Search_Tool first."
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
            "Wikipedia search. Returns up to 10 entries, each with a title, its URL, "
            "and a short truncated description. Does not open pages. "
            "Call Web_Fetch_Tool with one url on a later step to read a page."
        ),
        "tool_version": "1.0.0",
        "input_types": {"query": "str - the search query."},
        "output_type": "str - numbered entries with title, URL, and truncated description.",
        "demo_commands": [{"command": 'execution = tool.execute(query="Citibank")', "description": "Search Wikipedia."}],
        "require_llm_engine": False,
    },
    "Bing_Search_Tool": {
        "tool_name": "Bing_Search_Tool",
        "tool_description": (
            "Bing web search. Returns up to 10 entries, each with a title, its URL, "
            "and a short truncated description. Does not open pages. "
            "If a title is promising, call Web_Fetch_Tool with that url on the next step."
        ),
        "tool_version": "1.0.0",
        "input_types": {"query": "str - the search query."},
        "output_type": "str - numbered entries with title, URL, and truncated description.",
        "demo_commands": [{"command": 'execution = tool.execute(query="Citibank founded")', "description": "Search the web with Bing."}],
        "require_llm_engine": False,
    },
}


def _strip_tags(text: str) -> str:
    import html as html_lib
    import re

    plain = re.sub(r"<[^>]+>", "", text or "")
    return " ".join(html_lib.unescape(plain).split())


def _bing_hits(query: str) -> str:
    """Bing web search: top-10 (title, url) pairs. Does not open the URLs.

    Delegates to the mas.tools.search Bing kernel (same 7890 proxy chain as
    _fast_get's environment) so both projects share one SERP implementation.
    No Wikipedia fallback: a Bing failure is an explicit failure.
    """
    q = str(query or "").strip()
    if not q:
        return "Bing_Search_Tool requires a query"
    try:
        from mas.tools.search import bing_search

        return str(bing_search(q, 10) or "")
    except Exception as exc:
        return f"Bing unreachable: {exc}"


def _wikipedia_hits(query: str) -> str:
    """One MediaWiki search. Title, URL, and truncated snippet. Does not open pages."""
    q = str(query or "").strip()
    if not q:
        return "Wikipedia_Search_Tool requires a query"
    try:
        from mas.tools.search import truncate_description

        search = _fast_get(
            "https://en.wikipedia.org/w/api.php",
            {"action": "query", "list": "search", "srsearch": q, "srlimit": "10", "format": "json"},
        )
        search.raise_for_status()
        raw = ((search.json().get("query") or {}).get("search") or [])[:10]
        lines = [f"wikipedia results for: {q}"]
        for i, item in enumerate(raw, 1):
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            lines.append(f"{i}. {title}")
            lines.append("   URL: https://en.wikipedia.org/wiki/" + title.replace(" ", "_"))
            desc = truncate_description(str(item.get("snippet") or ""))
            if desc:
                lines.append(f"   {desc}")
        return "\n".join(lines) or f"No Wikipedia results for {q}"
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
    resp: Any = None
    try:
        import requests
    except Exception as exc:
        return f"Web fetch failed: {exc}"
    # Same proxy-chain fallback as _fast_get (SSH RemoteForward / local
    # Clash first, direct last): bare direct egress is blocked on this
    # host, so try every candidate before giving up.
    errors: List[str] = []
    for proxies in _proxy_candidates():
        label = (proxies or {}).get("http") or "direct"
        try:
            resp = requests.get(
                target,
                timeout=_FAST_TIMEOUT,
                headers={"User-Agent": "MAS-Science-Infra/1.0"},
                stream=True,
                proxies=proxies,
            )
            break
        except Exception as exc:
            errors.append(f"{label}: {exc}")
    if resp is None:
        return f"Web fetch failed: {'; '.join(errors[-4:]) if errors else 'no route'}"
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


# --- Query sanitizing (mirrors agentflow/adapted/context.py) ---------------
# Dataset prompts append instruction tails ("?When ready, output the final
# answer...") and planners prepend imperatives ("Search the web for: ...").
# Both pollute SERP queries; strip them before any search.
_QUERY_TAIL_RE = re.compile(
    r"(?i)\??\s*when\s+ready\s*,?\s*output\s+the\s+final\s+answer.*$"
    r"|\?\s*when\s+ready\b.*$"
    r"|do\s+not\s+generate\s+any\s+content\b.*$",
    re.S,
)
_QUERY_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:please\s+)?"
    r"(?:search\s+(?:the\s+web|wikipedia|the\s+internet|online)?\s*for"
    r"|look\s*up"
    r"|find\s+(?:information\s+)?(?:about|on)"
    r"|fetch\s+(?:the\s+)?(?:url|page|website)"
    r"|compute"
    r"|calculate"
    r")\s*[:：]?\s*"
)
_QUERY_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "was", "are", "were", "be", "been", "his", "her", "its", "their",
    "before", "after", "answering", "please", "prefer",
    "search", "web", "internet", "online", "look", "up", "find", "about",
    "when", "ready", "output", "final", "answer", "generate", "any",
    "content", "tag", "tags", "enclosed", "do", "not", "did",
    "what", "which", "who", "where", "whom", "whose", "how",
}


def _condense_query(query: str) -> str:
    """Strip instruction tails / imperative prefixes, cut at the first "?",
    and keep an entity-first keyword window for SERP queries."""
    q = str(query or "").strip()
    if not q:
        return q
    text = _QUERY_TAIL_RE.sub("", q)
    text = _QUERY_PREFIX_RE.sub("", text).strip()
    if len(text) <= 48 and "?" not in text:
        return text
    text = text.split("?", 1)[0]
    text = re.sub(r"(?i)^(what|which|who|where|when|whom|whose|how)\s+(is|was|are|were|did|does|do|has|have|had)\b", " ", text)
    text = re.sub(r"(?i)\b(the actor who|the actress who|the person who|who played|that appeared in|which appeared in|use (wikipedia|web|google)[^.]*\.?)", " ", text)
    text = re.sub(r"[?.!]+$", " ", text)
    text = re.sub(r"^(?:What|Which|Who|Where|When|Whom|Whose|How)\s+", "", text, flags=re.I)
    words = text.split()
    keep: List[str] = []
    for w in words:
        lowered = w.lower().strip(".,;:!?\"'()")
        if lowered in _QUERY_STOPWORDS:
            continue
        is_entity = bool(re.match(r"^[A-Z0-9]", w)) or w.isdigit()
        is_year = bool(re.match(r"^\d{4}$", w))
        if is_entity and not is_year:
            keep.append(w)
        elif is_year and keep:
            keep.append(w)
        elif keep and len(keep) <= 5 and w not in ("played", "role", "title"):
            continue
    keep2 = [w for w in (keep or words) if w.lower().strip(".,;:!?\"'()") not in _QUERY_STOPWORDS]
    if len(keep2) > 8:
        keep2 = keep2[:8]
    return " ".join(keep2).strip() or q


# SERP-list detector shared by the verify stop-gate below.
_SERP_LIST_PREFIXES = (
    "bing results for:",
    "wikipedia results for:",
)


def _looks_like_serp_list(text: str) -> bool:
    head = str(text or "").lstrip()[:120].lower()
    return any(head.startswith(p.lower()) for p in _SERP_LIST_PREFIXES)


def _serp_first_url(text: str) -> str:
    """First http(s) URL from a SERP list body (title+URL lines). Empty if none."""
    import re

    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # Only lines that carry a URL payload (SERP entries or "   URL: ...").
        if "http://" not in stripped and "https://" not in stripped:
            continue
        match = re.search(r"https?://[^\s'\"<>)]+", stripped)
        if match:
            return match.group(0).rstrip(".,)")
    return ""


def _run_dispatch(tool_name: str, query: str, url: str) -> str:
    if tool_name == "Google_Search_Tool":
        return "No result was generated because the tool was not found: Google_Search_Tool"
    kind = _DISPATCH_TOOLS.get(tool_name)
    if kind == "bing":
        return _bing_hits(_condense_query(query))
    if kind == "wiki":
        return _wikipedia_hits(_condense_query(query))
    if kind == "fetch":
        return _web_fetch(query, url)
    return f"No result was generated because the tool was not found: {tool_name}"


def _register_dispatch_tools(solver: Any, enabled: Optional[List[str]] = None) -> None:
    """Expose Google/Wikipedia search and Web_Fetch without the upstream tool package.

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
            "Wikipedia search. Returns up to 10 entries, each with a title, its URL, "
            "and a short truncated description. Does not open pages. "
            "Call Web_Fetch_Tool with one url on a later step to read a page."
        )
        wiki["demo_commands"] = [{"command": 'execution = tool.execute(query="Citibank")', "description": "Search Wikipedia."}]
        meta["Wikipedia_Search_Tool"] = wiki
    memory = getattr(solver, "system_memory", None)
    mem_meta = getattr(memory, "toolbox_metadata", None) if memory is not None else None
    if isinstance(mem_meta, dict) and mem_meta is not meta:
        mem_meta.update({name: dict(item) for name, item in meta.items() if name in _DISPATCH_META or name == "Wikipedia_Search_Tool"})


def _install_fast_search() -> None:
    """Wikipedia's own execute must not open the article or call Web_Search."""
    def execute(self: Any, query: str = "", *args: Any, **kwargs: Any) -> str:
        return _wikipedia_hits(str(query or kwargs.get("query") or ""))

    execute._science_fast = True  # type: ignore[attr-defined]
    for mod in _upstream_tool_modules("wikipedia_search.tool"):
        cls = getattr(mod, "Wikipedia_Search_Tool", None)
        if cls is not None and not getattr(getattr(cls, "execute", None), "_science_fast", False):
            cls.execute = execute  # type: ignore[method-assign]


def _guard_wikipedia_sys_exit() -> None:
    """Wikipedia_Search_Tool calls sys.exit when OPENAI_API_KEY missing — convert to error string."""
    for mod in _upstream_tool_modules("wikipedia_search.tool"):
        cls = getattr(mod, "Wikipedia_Search_Tool", None)
        if cls is not None and not getattr(cls.execute, "_science_no_exit", False):
            _wrap_wikipedia_no_exit(cls)


def _wrap_wikipedia_no_exit(cls: Any) -> None:
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
        raise ImportError("EPC-AW MAS not found in upload/ (set SCIENCE_EPC_AW_DIR or upload EPC-AW-main)")
    _ensure_import_paths(root)
    _map_llm_env()
    os.environ.setdefault("ENABLE_THINKING", "0")
    from MAS.epc_aw.solver import construct_solver  # type: ignore

    _install_engine_patches()
    # JSON-tolerant wrap must land AFTER construct_solver imports the engine
    # classes but BEFORE any instance is created. The engine __init__ patch
    # above runs first; the class-level wrap below is applied once and also
    # via instances created later (idempotent _science_json_tolerant marker).
    _install_json_tolerant_parsing()
    _install_memory_dedup()

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
        drain_llm_calls()
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

    def _drain_llm_calls(self) -> Dict[str, Any]:
        """Fix E: collect _generate_text records for this window's trace."""
        calls = drain_llm_calls()
        summaries = [
            {
                "model": str(c.get("model") or ""),
                "n": int(c.get("n") or 1),
                "prompt_head": str(c.get("prompt_head") or "")[:500],
                "response_head": str(c.get("response_head") or "")[:300],
            }
            for c in calls
        ]
        return {"count": len(summaries), "calls": summaries}

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
        # --- Fix D: search→fetch hard constraint state machine -----------------
        # If the previous round ran a search tool and its output is still a raw
        # SERP list (titles+URLs) that no fetch/generator has digested yet, this
        # round is forced to Web_Fetch_Tool on the first SERP URL — the LLM does
        # not get to pick (mirrors agentflow/adapted/context.py).
        forced_fetch_url = ""
        prev_output = str(self.last_tool_output or "")
        if (
            self.step_count > 1
            and str(self.last_tool_name or "") in ("Wikipedia_Search_Tool", "Web_Search_Tool", "Bing_Search_Tool")
            and _looks_like_serp_list(prev_output)
            and "Web_Fetch_Tool" in self._available_tools()
            and "Web_Fetch_Tool" not in self.failed_tools
            and not _is_transport_failure(prev_output)
        ):
            forced_fetch_url = _serp_first_url(prev_output)
        if forced_fetch_url:
            prev_search_tool = str(self.last_tool_name or "")
            self.last_trace["forced_fetch"] = {"url": forced_fetch_url, "from": prev_search_tool}
            self.last_context = str(self.last_context or question)
            self.last_sub_goal = f"Fetch {forced_fetch_url} and extract the facts needed to answer the current step"
            self.last_tool_name = "Web_Fetch_Tool"
            self.tool_available = True
            self.last_trace.update(
                {
                    "outline": self.system_memory.get_outline() if hasattr(self.system_memory, "get_outline") else {},
                    "tool": "Web_Fetch_Tool",
                    "context": self.last_context,
                    "sub_goal": self.last_sub_goal,
                    "tool_available": True,
                    "forced_fetch": {"url": forced_fetch_url, "from": prev_search_tool},
                }
            )
            self.last_trace["llm_calls"] = self._drain_llm_calls()
            return {
                "kind": "plan_step",
                "payload": {
                    "next": EXECUTOR_ID,
                    "args": {
                        "question": question,
                        "context": self.last_context,
                        "sub_goal": self.last_sub_goal,
                        "tool_name": "Web_Fetch_Tool",
                        "tool_available": True,
                        "query": "",
                        "url": forced_fetch_url,
                    },
                    "sub_goal": self.last_sub_goal,
                    "done": False,
                    "trace": dict(self.last_trace),
                },
            }
        # --- End Fix D state machine -------------------------------------------
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
        self.last_trace["llm_calls"] = self._drain_llm_calls()
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
                "llm_calls": self._drain_llm_calls(),
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
            self.last_trace["llm_calls"] = self._drain_llm_calls()
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
            "llm_calls": self._drain_llm_calls(),
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
        # Stop-gate: a bare SERP list is titles+URLs, not a digested answer.
        # Reject STOP while steps remain and no generator/digested evidence
        # exists (mirrors agentflow/adapted/context.py).
        if ready and _looks_like_serp_list(self.last_tool_output or "") and self.step_count < int(self.max_steps or _max_steps()):
            got_answer = False
            for item in (self._obtained() or []):
                chunk = str(item or "")
                if chunk.strip() and not _looks_like_serp_list(chunk):
                    got_answer = True
                    break
            if not got_answer:
                ready = False
                self.last_trace["stop_gate"] = "serp_only_evidence"
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
            # The direct_output prompt mandates "1. Process Summary ... 2. Answer: X".
            # Keep only the Answer section so the grader sees the answer itself,
            # not a process narrative that buries (and truncates) it.
            m = re.search(
                r"(?is)(?:^|\n)[ \t]*(?:2?[.\-:]?\s*)?(?:\*\*)?\s*Answer\s*(?:\*\*)?\s*[.:：]\s*(.*?)(?:\n\s*\n|\Z)",
                answer,
            )
            if m and m.group(1).strip():
                answer = m.group(1).strip().strip("*").strip()
            elif re.search(r"(?i)process summary", answer):
                # Malformed two-section output with no explicit Answer line:
                # fall back to the last non-empty line, usually the answer.
                tail = [ln.strip(" -*") for ln in answer.strip().splitlines() if ln.strip()]
                if tail:
                    answer = tail[-1]
        self.last_trace.update(
            {
                "analysis": analysis,
                "new_info": new_info,
                "conclusion": conclusion,
                "outline": outline,
                "outline_updated": updated_outline,
                "ready_to_stop": ready,
                "llm_calls": self._drain_llm_calls(),
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
