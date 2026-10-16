"""AgentFlow (vendored in upload/AgentFlow-main) as three MAS windows.

``Solver.solve`` from ``agentflow/agentflow/solver.py`` is split at its window
boundaries; each window runs the official model method verbatim:

==============  ==========================================================
planner         step 1: ``Planner.analyze_query``; every step:
                ``Planner.generate_next_step`` +
                ``extract_context_subgoal_and_tool``
executor        ``Executor.generate_tool_command`` +
                ``extract_explanation_and_command`` +
                ``execute_tool_command``; then ``Memory.add_action``
verifier        ``Verifier.verificate_context`` + ``extract_conclusion``;
                on STOP / max_steps / max_time, official order:
                ``Planner.generate_final_output`` then
                ``Planner.generate_direct_output``. The graded answer is
                ``direct_output`` only; ``final_output`` stays in json/trace.
==============  ==========================================================

One shared ``Memory`` per episode, no outline, no BTS. Engine roles follow
``model_engine=[trainable, fixed, fixed, fixed]``: only
``generate_next_step`` hits the trainable engine (see ``engine.py``).
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

from engine import LocalChatEngine, drain_calls  # noqa: E402
from tools_local import DEFAULT_TOOLS, build_toolbox  # noqa: E402

EXECUTOR_ID = "u_agentflow__executor"
_PROJECT = _ADAPTED.parent
# Vendored copy of the official ``agentflow`` package (models/engine/tools).
# Appended, never prepended, so it never shadows MAS modules.
_AGENTFLOW_ROOT = _PROJECT / "upload" / "AgentFlow-main" / "agentflow"

# ``solver.py`` construct_solver / main() defaults.
RUNTIME_DEFAULTS: Dict[str, Any] = {
    "n": 1,
    "max_steps": 10,
    "max_time": 300,
    "max_tokens": 2048,
    "temperature": 0.7,
    "enabled_tools": list(DEFAULT_TOOLS),
    # 0 keeps official make_json_serializable_truncated (100000 chars).
    "memory_result_chars": 0,
}
_MAX_EPISODES = 128

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
_ANSWER_LINE_RE = re.compile(
    r"(?is)(?:^|\n)[ \t]*(?:\d[.)]\s*)?(?:\*\*)?\s*Answer\s*(?:\*\*)?\s*[:：][ \t]*(?:\*\*)?[ \t]*(.*?)(?:\n\s*\n|\Z)"
)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def load_runtime() -> Dict[str, Any]:
    cfg = dict(RUNTIME_DEFAULTS)
    path = _PROJECT / "contracts" / "runtime.yaml"
    if path.is_file():
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(raw, dict):
            cfg.update(raw)
    for key in ("max_steps", "max_time", "max_tokens"):
        env = os.environ.get(f"AGENTFLOW_{key.upper()}", "").strip()
        if env.isdigit():
            cfg[key] = int(env)
    tools = cfg.get("enabled_tools")
    cfg["enabled_tools"] = [str(t) for t in tools] if isinstance(tools, list) and tools else list(DEFAULT_TOOLS)
    return cfg


def _ensure_import() -> None:
    root = str(_AGENTFLOW_ROOT)
    if root not in sys.path:
        sys.path.append(root)


def _payload(message: Any) -> Dict[str, Any]:
    if hasattr(message, "payload") and isinstance(getattr(message, "payload"), dict):
        return dict(message.payload)
    if isinstance(message, dict):
        inner = message.get("payload")
        return dict(inner) if isinstance(inner, dict) else dict(message)
    return {}


def _episode_key(payload: Dict[str, Any]) -> str:
    try:
        from workflow.user_gateway.public import current_episode_id

        eid = current_episode_id()
    except Exception:  # noqa: BLE001
        eid = ""
    return eid or f"q:{payload.get('question') or ''}"


def wrap_answer(direct_output: str, *, source: str = "") -> str:
    """``<answer>`` for the reward scorer; official output ends with ``**Answer:** X``."""
    text = str(direct_output or "").strip()
    if not text:
        return ""
    tagged = _ANSWER_TAG_RE.search(text)
    if tagged:
        body = tagged.group(1).strip()
    else:
        line = _ANSWER_LINE_RE.search(text)
        body = line.group(1).strip() if line and line.group(1).strip() else text
    if str(source or "").lower() == "gsm8k":
        nums = _NUM_RE.findall(body)
        body = nums[-1] if nums else body
    return f"<answer>{body}</answer>"


class Components:
    """Planner / Verifier / Executor built once; engines read the endpoint per call."""

    def __init__(self, runtime: Dict[str, Any]) -> None:
        _ensure_import()
        from agentflow.models.executor import Executor
        from agentflow.models.planner import Planner
        from agentflow.models.verifier import Verifier

        temp = float(runtime.get("temperature", 0.7))
        cap = int(runtime.get("max_tokens") or 2048)
        self.log_key = threading.local()

        def engine(role: str, trainable: bool = False, temperature: float = temp) -> LocalChatEngine:
            return LocalChatEngine(
                role, trainable=trainable, temperature=temperature, max_tokens_cap=cap,
                log_key=lambda: getattr(self.log_key, "value", ""),
            )

        self.available_tools, self.toolbox_metadata, instances = build_toolbox(
            list(runtime["enabled_tools"]), lambda: engine("tool", temperature=0.0)
        )
        from tools_local import _placeholder_keys

        with _placeholder_keys():
            self.planner = Planner(
                llm_engine_name="gpt-4o", llm_engine_fixed_name="gpt-4o",
                toolbox_metadata=self.toolbox_metadata, available_tools=self.available_tools,
                verbose=False, temperature=temp,
            )
            self.verifier = Verifier(
                llm_engine_name="gpt-4o", llm_engine_fixed_name="gpt-4o",
                toolbox_metadata=self.toolbox_metadata, available_tools=self.available_tools,
                verbose=False, temperature=temp,
            )
            self.executor = Executor(
                llm_engine_name="gpt-4o", root_cache_dir=tempfile.mkdtemp(prefix="agentflow_cache_"),
                verbose=False, temperature=temp, tool_instances_cache=instances,
            )
        self.planner.llm_engine = engine("planner", trainable=True)
        self.planner.llm_engine_fixed = engine("planner_fixed")
        self.verifier.llm_engine = engine("verifier")
        self.verifier.llm_engine_fixed = engine("verifier")
        self.executor.llm_generate_tool_command = engine("executor")


class Episode:
    def __init__(self, comps: Components, question: str, source: str) -> None:
        from agentflow.models.memory import Memory

        self.question = question
        self.source = source
        self.memory = Memory()
        # Planner keeps query_analysis on self; Executor keeps its cache dir.
        self.planner = copy.copy(comps.planner)
        self.executor = copy.copy(comps.executor)
        self.executor.set_query_cache_dir(tempfile.mkdtemp(dir=comps.executor.root_cache_dir))
        self.verifier = comps.verifier
        self.toolbox_metadata = comps.toolbox_metadata
        self.json_data: Dict[str, Any] = {"query": question, "image": None}
        self.query_analysis = ""
        self.step_count = 0
        self.start = time.time()
        self.context: Optional[str] = None
        self.sub_goal: Optional[str] = None
        self.tool_name: Optional[str] = None
        self.done = False
        self.direct_output = ""
        self.final_output = ""

    def export_state(self, *, replay_plan: bool = False) -> Dict[str, Any]:
        """Serializable Memory + step so a branch can continue this episode.

        ``replay_plan`` drops the current planner step (the tool has not run
        yet) so the child calls ``generate_next_step`` again for that step.
        A post-tool export keeps the action and the child plans the next step.
        """
        step = int(self.step_count)
        actions = {
            str(key): {
                "tool_name": action.get("tool_name"),
                "sub_goal": action.get("sub_goal"),
                "command": action.get("command"),
                "result": action.get("result"),
            }
            for key, action in self.memory.get_actions().items()
            if isinstance(action, dict)
        }
        if replay_plan:
            step = max(0, step - 1)
            actions.pop(f"Action Step {self.step_count}", None)
        return {
            "question": self.question,
            "source": self.source,
            "query_analysis": self.query_analysis,
            "step_count": step,
            "actions": actions,
            "query": self.memory.get_query(),
            "files": list(self.memory.get_files()),
            "analyzed": bool(self.query_analysis),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Replace Memory and the step counter. Does not call ``analyze_query``."""
        self.query_analysis = str(state.get("query_analysis") or "")
        self.step_count = int(state.get("step_count") or 0)
        self.memory.actions = {}
        for key, action in (state.get("actions") or {}).items():
            if not isinstance(action, dict):
                continue
            self.memory.actions[str(key)] = {
                "tool_name": action.get("tool_name"),
                "sub_goal": action.get("sub_goal"),
                "command": action.get("command"),
                "result": action.get("result"),
            }
        query = state.get("query")
        if query:
            self.memory.set_query(str(query))
        self.memory.files = [dict(item) for item in (state.get("files") or []) if isinstance(item, dict)]


class EpisodeContext:
    def __init__(self) -> None:
        self._comps: Optional[Components] = None
        self._build_error = ""
        self._lock = threading.Lock()
        self._episodes: "OrderedDict[str, Episode]" = OrderedDict()

    def components(self) -> Components:
        with self._lock:
            if self._comps is None and not self._build_error:
                try:
                    self._comps = Components(load_runtime())
                    print("AgentFlow tools:", ",".join(self._comps.available_tools), flush=True)
                except Exception as exc:  # noqa: BLE001
                    self._build_error = f"{type(exc).__name__}: {exc}"
            if self._build_error:
                raise RuntimeError(f"AgentFlow build failed: {self._build_error}")
            return self._comps  # type: ignore[return-value]

    def episode(self, payload: Dict[str, Any], *, create: bool = False) -> Episode:
        key = _episode_key(payload)
        comps = self.components()
        with self._lock:
            if key == "q:" and self._episodes:
                key = next(reversed(self._episodes))
            comps.log_key.value = key
            ep = self._episodes.get(key)
            question = str(payload.get("question") or (ep.question if ep else "") or payload.get("input") or "")
            if ep is None or (create and ep.done):
                ep = Episode(comps, question, str(payload.get("source") or ""))
                self._episodes[key] = ep
                while len(self._episodes) > _MAX_EPISODES:
                    self._episodes.popitem(last=False)
            self._episodes.move_to_end(key)
            return ep

    def _trace(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        key = getattr(self.components().log_key, "value", "")
        calls = drain_calls(key)
        return {**extra, "llm_calls": {"count": len(calls), "calls": calls}}

    # -- planner window -------------------------------------------------
    def plan_window(self, message: Any) -> Dict[str, Any]:
        payload = _payload(message)
        ep = self.episode(payload, create=True)
        state = payload.get("agentflow_state")
        if isinstance(state, dict) and state:
            ep.restore_state(state)
        if ep.done:
            return self._finish_plan(ep)
        ep.step_count += 1
        # A restored episode already has query_analysis; do not re-run step 1.
        if not ep.query_analysis:
            ep.query_analysis = ep.planner.analyze_query(ep.question, None)
            ep.json_data["query_analysis"] = ep.query_analysis
        runtime = load_runtime()
        next_step = ep.planner.generate_next_step(
            ep.question, None, ep.query_analysis, ep.memory,
            ep.step_count, int(runtime["max_steps"]), ep.json_data,
        )
        ep.context, ep.sub_goal, ep.tool_name = ep.planner.extract_context_subgoal_and_tool(next_step)
        return {
            "kind": "plan_step",
            "payload": {
                "next": EXECUTOR_ID,
                "args": {"context": ep.context or "", "sub_goal": ep.sub_goal or "", "tool_name": ep.tool_name or ""},
                "sub_goal": ep.sub_goal or "",
                "done": False,
                "agentflow_state": ep.export_state(replay_plan=True),
                "trace": self._trace({"step": ep.step_count, "tool": ep.tool_name}),
            },
        }

    def _finish_plan(self, ep: Episode) -> Dict[str, Any]:
        answer = wrap_answer(ep.direct_output, source=ep.source)
        return {
            "kind": "plan_step",
            "payload": {
                "next": "", "args": {}, "sub_goal": "stop", "done": True,
                "answer": answer, "ready_to_stop": True, "trace": self._trace({"stop": True}),
            },
        }

    # -- executor window ------------------------------------------------
    def exec_window(self, message: Any) -> Dict[str, Any]:
        from agentflow.models.utils import make_json_serializable_truncated

        ep = self.episode(_payload(message))
        tool_name = ep.tool_name
        if tool_name is None or tool_name not in ep.planner.available_tools:
            command = "No command was generated because the tool was not found."
            result: Any = "No result was generated because the tool was not found."
            ok = False
        else:
            tool_command = ep.executor.generate_tool_command(
                ep.question, None, ep.context, ep.sub_goal, tool_name,
                ep.toolbox_metadata[tool_name], ep.step_count, ep.json_data,
            )
            _analysis, _explanation, command = ep.executor.extract_explanation_and_command(tool_command)
            result = ep.executor.execute_tool_command(tool_name, command)
            limit = int(load_runtime().get("memory_result_chars") or 0)
            result = make_json_serializable_truncated(result, limit) if limit > 0 else make_json_serializable_truncated(result)
            ep.json_data[f"tool_result_{ep.step_count}"] = result
            ok = not (isinstance(result, str) and result.startswith(("Error", "Execution timed out")))
        ep.memory.add_action(ep.step_count, tool_name, ep.sub_goal, command, result)
        output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
        short = _short_tool_name(tool_name)
        return {
            "kind": "tool_result",
            "payload": {
                "output": output,
                "ok": bool(ok),
                "evidence_type": "DIRECT" if ok else "ERROR",
                "tool": short,
                "tool_name": tool_name or "",
                "command": str(command),
                "agentflow_state": ep.export_state(replay_plan=False),
                "trace": self._trace({"tool": short, "step": ep.step_count}),
            },
        }

    # -- verifier window ------------------------------------------------
    def verify_window(self, message: Any) -> Dict[str, Any]:
        ep = self.episode(_payload(message))
        runtime = load_runtime()
        stop_verification = ep.verifier.verificate_context(
            ep.question, None, ep.query_analysis, ep.memory, ep.step_count, ep.json_data
        )
        analysis, conclusion = ep.verifier.extract_conclusion(stop_verification)
        out_of_budget = ep.step_count >= int(runtime["max_steps"]) or (
            time.time() - ep.start >= float(runtime["max_time"])
        )
        stop = conclusion == "STOP" or out_of_budget
        answer = ""
        if stop:
            # Official default output_types=final,direct. final is the write-up;
            # direct is the answer the scorer wraps.
            ep.planner.query_analysis = ep.query_analysis
            ep.final_output = str(ep.planner.generate_final_output(ep.question, None, ep.memory) or "")
            ep.direct_output = str(ep.planner.generate_direct_output(ep.question, None, ep.memory) or "")
            ep.json_data["final_output"] = ep.final_output
            ep.json_data["direct_output"] = ep.direct_output
            ep.done = True
            answer = wrap_answer(ep.direct_output, source=ep.source)
        return {
            "kind": "verify",
            "payload": {
                "ok": stop,
                "reason": str(analysis or "")[:1000],
                "step_conclusion": "COMPLETE" if stop else "INCOMPLETE",
                "slot_updates": [],
                "conclusion": conclusion,
                "answer": answer,
                "direct_output": answer,
                "ready_to_stop": stop,
                "agentflow_state": ep.export_state(replay_plan=False),
                "trace": self._trace({
                    "conclusion": conclusion,
                    "step": ep.step_count,
                    "budget_stop": out_of_budget,
                    "final_output": ep.final_output,
                }),
            },
        }


def _short_tool_name(long_name: Optional[str]) -> str:
    """Class name for the runtime's n_search / n_python counters."""
    from tools_local import OFFICIAL_TOOLS

    mapping = {
        "Generalist_Solution_Generator_Tool": "Base_Generator_Tool",
        "Python_Code_Generator_Tool": "Python_Coder_Tool",
        "Bing_Search_Tool": "Bing_Search_Tool",
        "Wikipedia_RAG_Search_Tool": "Wikipedia_Search_Tool",
        "Web_Fetch_Tool": "Web_Fetch_Tool",
    }
    name = mapping.get(str(long_name or ""), str(long_name or ""))
    return name if name in OFFICIAL_TOOLS else ""


_CTX: Optional[EpisodeContext] = None
_CTX_LOCK = threading.Lock()


def get_episode_context() -> EpisodeContext:
    global _CTX
    with _CTX_LOCK:
        if _CTX is None:
            _CTX = EpisodeContext()
        return _CTX


def enabled_tool_names() -> List[str]:
    return list(load_runtime()["enabled_tools"])


def reset_for_tests() -> None:
    global _CTX
    with _CTX_LOCK:
        _CTX = None
