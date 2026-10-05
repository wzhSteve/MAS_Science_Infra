"""AgentFlow episode context. Planner, executor, and diagnoser share one solver.

Collect / eval / training call run_window only. Solver.solve is not an entry.
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ENABLED_TOOLS = [
    "Base_Generator_Tool",
    "Python_Coder_Tool",
    "Wikipedia_Search_Tool",
    "Bing_Search_Tool",
    "Web_Fetch_Tool",
]
EXECUTOR_ID = "u_agentflow__executor"
_NATIVE = {
    "Base_Generator_Tool": ("base_generator", "Base_Generator_Tool"),
    "Python_Coder_Tool": ("python_coder", "Python_Coder_Tool"),
}
_FORBIDDEN = ("Google_Search_Tool", "Web_Search_Tool", "think")
_PROJECT = Path(__file__).resolve().parents[1]
_AGENTFLOW_ROOT = Path(__file__).resolve().parents[4] / "ref_Rep" / "agentflow"
_CTX: Optional["EpisodeContext"] = None


def enabled_tool_names() -> List[str]:
    cfg = _load_runtime()
    names = [str(item) for item in (cfg.get("enabled_tools") or [])]
    return names or list(ENABLED_TOOLS)


def _load_runtime() -> Dict[str, Any]:
    path = _PROJECT / "contracts" / "runtime.yaml"
    cfg: Dict[str, Any] = {
        "n": 1,
        "max_steps": 3,
        "max_time": 180,
        "max_tokens": 512,
        "temperature": 0.7,
        "enabled_tools": list(ENABLED_TOOLS),
    }
    if not path.is_file():
        return cfg
    try:
        import yaml
    except ImportError:
        return cfg
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if isinstance(raw, dict):
        cfg.update(raw)
    tools = cfg.get("enabled_tools")
    if not isinstance(tools, list) or not tools:
        cfg["enabled_tools"] = list(ENABLED_TOOLS)
    else:
        cfg["enabled_tools"] = [str(item) for item in tools]
    return cfg


def _map_llm_env() -> str:
    base = (
        os.environ.get("OPENAI_API_BASE_URL")
        or os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).strip().rstrip("/")
    if base:
        os.environ["OPENAI_API_BASE_URL"] = base
        os.environ["OPENAI_API_BASE"] = base
        os.environ["OPENAI_BASE_URL"] = base
    # Prefer training-bound MODEL_NAME over .env OPENAI_MODEL (e.g. qwen3.5-27b).
    model = (
        os.environ.get("MODEL_NAME")
        or os.environ.get("MODEL_Name")
        or os.environ.get("OPENAI_MODEL")
        or os.environ.get("MODEL")
        or "gpt-4o"
    ).strip()
    os.environ["MODEL_NAME"] = model
    os.environ["MODEL_Name"] = model
    os.environ["OPENAI_MODEL"] = model
    os.environ["MODEL"] = model
    existing = [x.strip() for x in (os.environ.get("SERVER_MODEL") or "").split(",") if x.strip()]
    if model and model not in existing:
        existing.append(model)
    os.environ["SERVER_MODEL"] = ",".join(existing)
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "EMPTY"
    return model


def _ensure_import() -> None:
    # Append, do not prepend: ref_Rep/agentflow/logging.py shadows stdlib logging.
    root = str(_AGENTFLOW_ROOT)
    if root not in sys.path:
        sys.path.append(root)


def _payload(message: Any) -> Dict[str, Any]:
    if hasattr(message, "payload") and isinstance(getattr(message, "payload"), dict):
        return dict(message.payload)
    if isinstance(message, dict):
        inner = message.get("payload")
        if isinstance(inner, dict):
            return dict(inner)
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


def _canonical(name: str) -> str:
    raw = str(name or "").strip()
    lowered = raw.lower().replace(" ", "_")
    if raw in ENABLED_TOOLS:
        return raw
    if "wiki" in lowered:
        return "Wikipedia_Search_Tool"
    if "python" in lowered or "coder" in lowered:
        return "Python_Coder_Tool"
    if "fetch" in lowered or "web_search" in lowered or lowered.startswith("web"):
        return "Web_Fetch_Tool"
    if "bing" in lowered or "google" in lowered or "search" in lowered:
        return "Bing_Search_Tool"
    if "base" in lowered or "generator" in lowered:
        return "Base_Generator_Tool"
    return "Base_Generator_Tool"


def _parse_execute_args(command: str) -> Dict[str, Any]:
    match = re.search(r"tool\.execute\((.*)\)", command or "", re.S)
    if not match:
        return {}
    inner = match.group(1).strip()
    try:
        call = ast.parse(f"f({inner})", mode="eval").body
    except SyntaxError:
        return {"query": inner}
    out: Dict[str, Any] = {}
    if isinstance(call, ast.Call):
        for kw in call.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                out[kw.arg] = kw.value.value
    return out


def _bing(query: str) -> str:
    q = str(query or "").strip()
    if not q:
        return "Bing_Search_Tool requires a query"
    try:
        import requests

        resp = requests.get(
            "https://cn.bing.com/search",
            params={"q": q, "count": "5"},
            timeout=20,
            headers={"User-Agent": "MAS-Science-Infra/1.0"},
        )
        resp.raise_for_status()
        titles = re.findall(r"<h2[^>]*>\s*<a[^>]+href=\"(https?://[^\"]+)\"[^>]*>(.*?)</a>", resp.text or "", re.S | re.I)
        lines = [f"Bing results for {q}"]
        kept = 0
        for url, title in titles:
            if any(host in url for host in ("bing.com", "microsoft.com", "msn.com")):
                continue
            plain = re.sub(r"<[^>]+>", "", title)
            plain = " ".join(plain.split())
            if not plain:
                continue
            kept += 1
            lines.append(f"{kept}. {plain}\n   url: {url}")
            if kept >= 5:
                break
        return "\n".join(lines) if kept else f"No Bing results for {q}"
    except Exception as exc:
        return f"Bing unreachable: {exc}"


def _wiki(query: str) -> str:
    q = str(query or "").strip()
    if not q:
        return "Wikipedia_Search_Tool requires a query"
    try:
        import requests

        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": q,
                "srlimit": "5",
                "format": "json",
            },
            timeout=20,
            headers={"User-Agent": "MAS-Science-Infra/1.0"},
        )
        resp.raise_for_status()
        rows = ((resp.json().get("query") or {}).get("search") or [])[:5]
        lines = [f"Wikipedia results for {q}"]
        for index, item in enumerate(rows, start=1):
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            snippet = re.sub(r"<[^>]+>", "", str(item.get("snippet") or ""))
            lines.append(f"{index}. {title}\n   url: {url}\n   snippet: {' '.join(snippet.split())}")
        return "\n".join(lines) if len(lines) > 1 else f"No Wikipedia results for {q}"
    except Exception as exc:
        return f"Wikipedia unreachable: {exc}"


def _fetch(query: str, url: str) -> str:
    target = str(url or "").strip()
    if not target:
        return "Web_Fetch_Tool requires a url"
    try:
        import requests

        resp = requests.get(target, timeout=20, headers={"User-Agent": "MAS-Science-Infra/1.0"})
        status = int(getattr(resp, "status_code", 0) or 0)
        if status >= 400:
            return f"Web fetch failed: HTTP {status} {target}"
        text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", resp.text or "", flags=re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = " ".join(text.split())
        note = f"query: {query}\n" if query else ""
        return f"{note}url: {target}\n{text[:4000]}"
    except Exception as exc:
        return f"Web fetch failed: {exc}"


def _native_execute(tool_name: str, args: Dict[str, Any], command: str) -> str:
    dir_name, class_name = _NATIVE[tool_name]
    _ensure_import()
    import importlib

    module = importlib.import_module(f"agentflow.tools.{dir_name}.tool")
    tool = getattr(module, class_name)()
    call_args = dict(args)
    if "query" not in call_args:
        call_args["query"] = command or tool_name
    try:
        return str(tool.execute(**call_args))
    except Exception as exc:
        return f"{tool_name} failed: {exc}"


def _execute(tool_name: str, command: str) -> str:
    name = _canonical(tool_name)
    args = _parse_execute_args(command)
    if name == "Bing_Search_Tool":
        return _bing(str(args.get("query") or command))
    if name == "Web_Fetch_Tool":
        return _fetch(str(args.get("query") or ""), str(args.get("url") or ""))
    if name == "Wikipedia_Search_Tool":
        return _wiki(str(args.get("query") or command))
    if name in _NATIVE:
        return _native_execute(name, args, command)
    return f"tool not enabled: {tool_name}"


def _tool_meta(name: str) -> Dict[str, Any]:
    descriptions = {
        "Base_Generator_Tool": "Answer a query step by step when no specialized tool fits.",
        "Python_Coder_Tool": "Write and run Python for a computation.",
        "Wikipedia_Search_Tool": "Search Wikipedia titles. Returns title, url, and snippet.",
        "Bing_Search_Tool": "Search Bing. Returns title and url. Does not open the page.",
        "Web_Fetch_Tool": "Open one url copied from an earlier search hit.",
    }
    return {
        "tool_name": name,
        "tool_description": descriptions.get(name, name),
        "tool_version": "1.0.0",
        "input_types": {"query": "str"},
        "output_type": "str",
        "demo_commands": [{"command": f'execution = tool.execute(query="example")', "description": name}],
        "require_llm_engine": name in _NATIVE,
    }


def _install_five(solver: Any, enabled: List[str]) -> None:
    meta = {name: _tool_meta(name) for name in enabled}
    for holder_name in ("planner", "diagnoser"):
        holder = getattr(solver, holder_name, None)
        if holder is None:
            continue
        existing = getattr(holder, "toolbox_metadata", None)
        if isinstance(existing, dict):
            for name in enabled:
                if name in existing and isinstance(existing[name], dict):
                    meta[name] = dict(existing[name])
                    meta[name]["tool_name"] = name
        holder.available_tools = list(enabled)
        holder.toolbox_metadata = dict(meta)
    memory = getattr(solver, "system_memory", None)
    if memory is not None:
        memory.toolbox_metadata = dict(meta)
    planner = getattr(solver, "planner", None)
    if planner is None or not hasattr(planner, "extract_context_subgoal_and_tool"):
        return
    original = planner.extract_context_subgoal_and_tool

    def _extract(response: Any) -> Any:
        context, sub_goal, tool_name = original(response)
        tool_name = _canonical(str(tool_name or ""))
        if tool_name not in enabled:
            tool_name = "Base_Generator_Tool"
        return context, sub_goal, tool_name

    planner.extract_context_subgoal_and_tool = _extract


def _clamp_max_tokens(requested: Any, prompt: Any = None) -> int:
    try:
        want = int(requested)
    except (TypeError, ValueError):
        want = 512
    runtime = _load_runtime()
    cap = int(runtime.get("max_tokens") or 512)
    model_len = int(os.environ.get("TIR_MAX_MODEL_LEN") or 0) or 2560
    text = "" if prompt is None else (prompt if isinstance(prompt, str) else str(prompt))
    prompt_tokens = max(1, (len(text) + 3) // 3)
    remaining = max(64, model_len - prompt_tokens - 64)
    return max(64, min(want, cap, remaining))


def _install_completion_budget() -> None:
    """Keep AgentFlow OpenAI/vLLM completions under the training context window."""
    import inspect

    def _budget_wrapper(orig: Any) -> Any:
        def wrapped(self: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                ba = inspect.signature(orig).bind(self, prompt, *args, **kwargs)
                ba.apply_defaults()
                if "max_tokens" in ba.arguments:
                    ba.arguments["max_tokens"] = _clamp_max_tokens(
                        ba.arguments.get("max_tokens"), ba.arguments.get("prompt", prompt)
                    )
                return orig(*ba.args, **ba.kwargs)
            except TypeError:
                kwargs["max_tokens"] = _clamp_max_tokens(kwargs.get("max_tokens", 512), prompt)
                return orig(self, prompt, *args, **kwargs)

        wrapped._agentflow_budget = True  # type: ignore[attr-defined]
        return wrapped

    for mod_name, cls_name in (
        ("agentflow.engine.openai", "ChatOpenAI"),
        ("agentflow.engine.vllm", "ChatVLLM"),
    ):
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except Exception:
            continue
        if not getattr(getattr(cls, "__init__", None), "_agentflow_model_force", False):
            orig_init = cls.__init__

            def _init_factory(orig: Any) -> Any:
                def _init(self: Any, *args: Any, **kwargs: Any) -> Any:
                    forced = (
                        os.environ.get("MODEL_Name")
                        or os.environ.get("MODEL_NAME")
                        or kwargs.get("model_string")
                        or (args[0] if args else None)
                        or "gpt-4o"
                    )
                    os.environ["MODEL_Name"] = str(forced)
                    os.environ["MODEL_NAME"] = str(forced)
                    os.environ["OPENAI_MODEL"] = str(forced)
                    if "model_string" in kwargs:
                        kwargs["model_string"] = forced
                    elif args:
                        args = (forced,) + tuple(args[1:])
                    return orig(self, *args, **kwargs)

                _init._agentflow_model_force = True  # type: ignore[attr-defined]
                return _init

            cls.__init__ = _init_factory(orig_init)  # type: ignore[method-assign]
        fn = getattr(cls, "_generate_text", None)
        if fn is None or getattr(fn, "_agentflow_budget", False):
            continue
        setattr(cls, "_generate_text", _budget_wrapper(fn))


def _build_solver() -> Any:
    _ensure_import()
    _map_llm_env()
    _install_completion_budget()
    from agentflow.solver import construct_solver

    runtime = _load_runtime()
    enabled = [name for name in enabled_tool_names() if name not in _FORBIDDEN]
    # Search tools are dispatched here. Loading AgentFlow's Wikipedia class
    # also imports Web_Search_Tool, which must stay off the planner's list.
    native = [name for name in enabled if name in _NATIVE]
    model = os.environ.get("MODEL_NAME") or "gpt-4o"
    solver = construct_solver(
        llm_engine_name=model,
        enabled_tools=native or ["Base_Generator_Tool"],
        tool_engine=["Default"] * max(1, len(native)),
        verbose=False,
        n=int(runtime.get("n") or 1),
        max_steps=int(runtime.get("max_steps") or 3),
        max_time=int(runtime.get("max_time") or 180),
        max_tokens=int(runtime.get("max_tokens") or 512),
        temperature=float(runtime.get("temperature") or 0.7),
        output_types="direct",
    )
    _install_five(solver, enabled)
    _patch_engines_for_local(solver, model)
    print("AgentFlow enabled_tools:", ",".join(enabled), flush=True)
    return solver


def _patch_engines_for_local(solver: Any, model: str) -> None:
    """Force MAS training-bound OpenAI client; drop AgentFlow cache / thinking."""
    model = str(model or "").strip()

    def _fix(eng: Any) -> None:
        if eng is None or not hasattr(eng, "_generate_text"):
            return
        if model:
            try:
                eng.model_string = model
            except Exception:
                pass
        try:
            eng.is_chat_model = True
        except Exception:
            pass
        # MAS RL needs a real Proxy completion (token_ids); never hit disk cache.
        if hasattr(eng, "use_cache"):
            eng.use_cache = False
        if hasattr(eng, "SERVER_MODEL"):
            sm = [str(x).strip() for x in (getattr(eng, "SERVER_MODEL") or []) if str(x).strip()]
            if model and model not in sm:
                sm.append(model)
            eng.SERVER_MODEL = sm
        if hasattr(eng, "support_structured_output"):
            eng.support_structured_output = False
        # Rebuild client against the bound training endpoint.
        base = (
            os.environ.get("OPENAI_API_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or ""
        ).strip()
        key = (os.environ.get("OPENAI_API_KEY") or "EMPTY").strip()
        if base and hasattr(eng, "client"):
            try:
                from openai import OpenAI

                eng.client = OpenAI(api_key=key, base_url=base)
            except Exception:
                pass

    seen: set[int] = set()
    holders = [solver]
    for name in ("planner", "executor", "diagnoser"):
        holders.append(getattr(solver, name, None))
    for holder in holders:
        if holder is None:
            continue
        for attr in dir(holder):
            try:
                eng = getattr(holder, attr)
            except Exception:
                continue
            if eng is None or id(eng) in seen:
                continue
            if hasattr(eng, "_generate_text"):
                seen.add(id(eng))
                _fix(eng)
    print(
        f"AgentFlow llm_model={model} endpoint={os.environ.get('OPENAI_API_BASE_URL')} engines={len(seen)}",
        flush=True,
    )


class EpisodeContext:
    def __init__(self) -> None:
        self.solver: Any = None
        self.build_error = ""
        self.question = ""
        self.step_count = 0
        self.obtained: List[str] = []
        self.last_output = ""
        self.stop = False
        self.json_data: Dict[str, Any] = {}

    def _solver(self) -> Any:
        if self.solver is None and not self.build_error:
            try:
                self.solver = _build_solver()
            except Exception as exc:
                self.build_error = str(exc)
        if self.build_error:
            raise RuntimeError(self.build_error)
        return self.solver

    def _reset_if_new(self, question: str) -> None:
        if question and question != self.question:
            self.question = question
            self.step_count = 0
            self.obtained = []
            self.last_output = ""
            self.stop = False
            self.json_data = {"query": question}

    def plan_window(self, message: Any) -> Dict[str, Any]:
        payload = _payload(message)
        question = _text(payload.get("question"), payload.get("input"), self.question)
        self._reset_if_new(question)
        runtime = _load_runtime()
        max_steps = int(runtime.get("max_steps") or 3)
        self.step_count += 1
        if self.stop or self.step_count > max_steps:
            return {
                "kind": "plan_step",
                "payload": {
                    "next": EXECUTOR_ID,
                    "args": {"question": question},
                    "sub_goal": "stop",
                    "done": True,
                    "trace": {"enabled_tools": enabled_tool_names()},
                },
            }
        try:
            solver = self._solver()
            if self.step_count == 1 and hasattr(solver.planner, "analyze_query"):
                analyzed = solver.planner.analyze_query(question, None)
                outline = analyzed[1] if isinstance(analyzed, tuple) and len(analyzed) > 1 else {"1": question}
                if not isinstance(outline, dict) or not outline:
                    outline = {"1": question}
                if hasattr(solver.system_memory, "set_outline"):
                    solver.system_memory.set_outline(outline)
            outline = {}
            if hasattr(solver.system_memory, "get_outline"):
                outline = solver.system_memory.get_outline() or {}
            target = str(outline.get(str(self.step_count)) or outline.get("1") or question)
            raw = solver.planner.generate_next_step(
                question, None, target, "", self.step_count, max_steps, self.obtained, self.json_data
            )
            context, sub_goal, tool_name = solver.planner.extract_context_subgoal_and_tool(raw)
        except Exception as exc:
            context, sub_goal, tool_name = question, question, "Base_Generator_Tool"
            return {
                "kind": "plan_step",
                "payload": {
                    "next": EXECUTOR_ID,
                    "args": {
                        "question": question,
                        "context": context,
                        "sub_goal": sub_goal,
                        "tool_name": tool_name,
                        "query": sub_goal,
                    },
                    "sub_goal": sub_goal,
                    "done": False,
                    "trace": {"llm_error": str(exc), "enabled_tools": enabled_tool_names(), "tool": tool_name},
                },
            }
        tool_name = _canonical(tool_name)
        return {
            "kind": "plan_step",
            "payload": {
                "next": EXECUTOR_ID,
                "args": {
                    "question": question,
                    "context": str(context or question),
                    "sub_goal": str(sub_goal or question),
                    "tool_name": tool_name,
                    "query": str(sub_goal or question),
                },
                "sub_goal": str(sub_goal or question),
                "done": False,
                "trace": {"tool": tool_name, "enabled_tools": enabled_tool_names()},
            },
        }

    def exec_window(self, message: Any) -> Dict[str, Any]:
        payload = _payload(message)
        question = _text(payload.get("question"), self.question)
        context = _text(payload.get("context"), question)
        sub_goal = _text(payload.get("sub_goal"), question)
        tool_name = _canonical(_text(payload.get("tool_name"), "Base_Generator_Tool"))
        if tool_name not in enabled_tool_names():
            output = f"tool not enabled: {tool_name}"
            self.last_output = output
            return {
                "kind": "tool_result",
                "payload": {"output": output, "ok": False, "evidence_type": "ERROR", "tool": tool_name},
            }
        command = ""
        try:
            solver = self._solver()
            meta = {}
            metadata = getattr(solver.planner, "toolbox_metadata", {}) or {}
            if isinstance(metadata, dict):
                meta = metadata.get(tool_name) or {}
            blob = solver.executor.generate_tool_command(
                question, None, context, sub_goal, tool_name, meta, self.step_count, self.json_data
            )
            if hasattr(solver.executor, "extract_explanation_and_command"):
                _analysis, _explanation, command = solver.executor.extract_explanation_and_command(blob)
            else:
                command = str(blob or "")
            output = _execute(tool_name, str(command or ""))
            ok = not str(output).lower().startswith(("tool not", "bing unreachable", "wikipedia unreachable", "web fetch failed"))
        except Exception as exc:
            output = f"{tool_name} failed: {exc}"
            ok = False
        self.last_output = output
        self.obtained.append(f"{tool_name}: {output[:500]}")
        return {
            "kind": "tool_result",
            "payload": {
                "output": str(output),
                "ok": bool(ok),
                "evidence_type": "DIRECT" if ok else "ERROR",
                "tool": tool_name,
                "command": str(command),
            },
        }

    def verify_window(self, message: Any) -> Dict[str, Any]:
        payload = _payload(message)
        question = _text(payload.get("question"), self.question)
        output = _text(payload.get("output"), self.last_output)
        runtime = _load_runtime()
        max_steps = int(runtime.get("max_steps") or 3)
        conclusion = ""
        reason = output[:500]
        try:
            solver = self._solver()
            outline = {}
            if hasattr(solver.system_memory, "get_outline"):
                outline = solver.system_memory.get_outline() or {}
            target = str(outline.get(str(self.step_count)) or question)
            verified = solver.diagnoser.verificate_context(
                question, None, target, outline, output, self.step_count, self.obtained
            )
            if isinstance(verified, tuple) and len(verified) >= 4:
                reason = str(verified[0] or reason)
                conclusion = str(verified[3] or "")
        except Exception as exc:
            reason = f"{reason}\nverifier error: {exc}"
        self.stop = "STOP" in conclusion.upper() or self.step_count >= max_steps
        return {
            "kind": "verify",
            "payload": {
                "ok": not self.stop,
                "reason": reason[:1000],
                "step_conclusion": "COMPLETE" if self.stop else "INCOMPLETE",
                "slot_updates": [],
                "conclusion": conclusion,
            },
        }


def get_episode_context() -> EpisodeContext:
    global _CTX
    fingerprint = "|".join(
        [
            os.environ.get("OPENAI_API_BASE_URL") or "",
            os.environ.get("MODEL_Name") or os.environ.get("MODEL_NAME") or "",
        ]
    )
    if _CTX is None or getattr(_CTX, "_endpoint_fp", None) != fingerprint:
        _CTX = EpisodeContext()
        _CTX._endpoint_fp = fingerprint  # type: ignore[attr-defined]
    return _CTX
