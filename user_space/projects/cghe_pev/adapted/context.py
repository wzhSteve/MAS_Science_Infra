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

One shared ``Memory`` per episode. ``actions`` stay the history. ``outline`` and
``obtained_information`` hang on that same object. Engine roles follow
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
from uuid import uuid4
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

_ADAPTED = Path(__file__).resolve().parent
if str(_ADAPTED) not in sys.path:
    sys.path.insert(0, str(_ADAPTED))

from engine import LocalChatEngine, drain_calls  # noqa: E402
from tools_local import DEFAULT_TOOLS, build_toolbox  # noqa: E402

EXECUTOR_ID = "u_cghe_pev__executor"
_PROJECT = _ADAPTED.parent
# Vendored copy of the official ``agentflow`` package (models/engine/tools).
# Appended, never prepended, so it never shadows MAS modules.
_AGENTFLOW_ROOT = _PROJECT / "upload" / "AgentFlow-main" / "agentflow"

# ``solver.py`` construct_solver / main() defaults.
RUNTIME_DEFAULTS: Dict[str, Any] = {
    "n": 1,
    "max_steps": 10,
    "max_time": 300,
    "max_tokens": 4096,
    "temperature": 0.7,
    "enabled_tools": list(DEFAULT_TOOLS),
    # 0 keeps official make_json_serializable_truncated (100000 chars).
    "memory_result_chars": 0,
    # Used only when contracts/runtime.yaml has no cghe block. enabled stays off.
    "cghe": {
        "enabled": False,
        "n": 10,
        "confidence_threshold": 1.0,
        "sqlite_path": "artifacts/cghe_evidence.sqlite",
        "recovery_rounds": 2,
        "tool_calls": 12,
        "replans": 1,
        "pruning": True,
        "shared_priority": True,
        "isolate_memory": False,
        "algorithm": "v1",
        "lambda_cost": 0.0,
        "mu_risk": 0.0,
        "epv_threshold": 0.0,
        "max_step1_attempts": 3,
        "max_regenerations": 5,
        "max_total_step1_attempts": 12,
        "max_verified_snippet_prunes_per_episode": 2,
        "max_inherited_ce": 32,
        "round_regen": True,
        "selection": {"strategy": "max_step1_path_coverage"},
        "pruning_gates": {
            "require_verified_counterexample": True,
            "require_direct_observation": True,
            "require_complete_scope": True,
            "prune_only_explicitly_dependent_plans": True,
        },
    },
}
# One environment for tool observations and RecoveryController scope checks.
CGHE_ENVIRONMENT = "cghe_pev"
_PEV_DIRECTIVE: Dict[str, Any] = {
    "decision": "continue_pev",
    "reason": "",
    "intervene": False,
    "message": "",
    "node_id": "",
    "plan_id": "",
    "subgoal": "",
    "action": "",
    "tool": "",
    "round_id": "",
}
_ACTION_EXTRA_KEYS = (
    "prediction", "prediction_reason", "tool_input", "actual", "conclusion",
    "prediction_event_id", "planner_step", "predicted_at", "p_satisfy",
    "predicted_before_execution_at",
    "executed_at",
    "p_unsatisfy", "s_satisfy", "s_unsatisfy", "prediction_mismatch",
    "satisfy_token_ids", "unsatisfy_token_ids",
    "prediction_token_start",
    "actual_satisfied", "verifier_reason", "recovery_chain_id",
    "parent_prediction_event_id", "failure_confirmed",
    "raw_tool_result", "execution_log", "cghe_node_id", "cghe_plan_id",
    "cghe_authorized", "cghe_replay", "cghe_auth_reason",
    "cghe_feedback", "cghe_input_url",
)
_EXECUTION_LOG_CHARS = 20000
_MAX_EPISODES = 128

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
_ANSWER_LINE_RE = re.compile(
    r"(?is)(?:^|\n)[ \t]*(?:\d[.)]\s*)?(?:\*\*)?\s*Answer\s*(?:\*\*)?\s*[:：][ \t]*(?:\*\*)?[ \t]*(.*?)(?:\n\s*\n|\Z)"
)
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def sft_prompts_enabled() -> bool:
    """Adapted prompts and tag→execute live in ``sft_prompts.py``.

    ``AGENTFLOW_SFT_PROMPTS`` still forces the bench baseline off or the short
    prompts on. When it is unset, ``contracts/runtime.yaml`` ``sft_prompts`` decides.
    """
    raw = os.environ.get("AGENTFLOW_SFT_PROMPTS", "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return bool(load_runtime().get("sft_prompts"))


def apply_sft_tags(tool_name: Optional[str], raw: Any, command: str) -> str:
    """After the official extractor: turn ``<search>`` / ``<python>`` into ``tool.execute``."""
    if not sft_prompts_enabled():
        return command
    from sft_prompts import tags_to_execute

    return tags_to_execute(str(tool_name or ""), str(raw or ""), str(command or ""))


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
    temp_env = os.environ.get("AGENTFLOW_TEMPERATURE", "").strip()
    if temp_env:
        cfg["temperature"] = float(temp_env)
    # Paper trains at 3 planner turns. AGENTFLOW_MAX_STEPS still wins.
    # Eval shells do not set TIR_ALGO / AGENTFLOW_TRAIN, so they keep max_steps.
    training = bool(os.environ.get("TIR_ALGO", "").strip() or os.environ.get("AGENTFLOW_TRAIN", "").strip())
    if training and not os.environ.get("AGENTFLOW_MAX_STEPS", "").strip():
        cap = cfg.get("train_max_steps")
        if str(cap).isdigit():
            cfg["max_steps"] = int(cap)
    tools = cfg.get("enabled_tools")
    cfg["enabled_tools"] = [str(t) for t in tools] if isinstance(tools, list) and tools else list(DEFAULT_TOOLS)
    return cfg


def cghe_settings() -> Dict[str, Any]:
    """CGHE block only. Tool list and PEV step limits stay on the runtime root."""
    raw = load_runtime().get("cghe")
    merged = dict(RUNTIME_DEFAULTS["cghe"])
    if isinstance(raw, dict):
        merged.update(raw)
    enabled = merged.get("enabled")
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() in ("1", "true", "yes", "on")
    path = Path(str(merged.get("sqlite_path") or RUNTIME_DEFAULTS["cghe"]["sqlite_path"]))
    sqlite_env = os.environ.get("CGHE_SQLITE_PATH", "").strip()
    if sqlite_env:
        path = Path(sqlite_env)
    if not path.is_absolute():
        path = _PROJECT / path
    try:
        count = int(merged.get("n") or 10)
    except (TypeError, ValueError):
        count = 10
    try:
        threshold = float(merged.get("confidence_threshold"))
    except (TypeError, ValueError):
        threshold = 1.0

    def positive(key: str, default: int) -> int:
        try:
            return max(0, int(merged.get(key)))
        except (TypeError, ValueError):
            return default

    def flag(key: str, default: bool = True) -> bool:
        value = merged.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    ablation = os.environ.get("CGHE_ABLATION", "").strip().lower() or "full"
    pruning = flag("pruning")
    shared_priority = flag("shared_priority")
    isolate_memory = flag("isolate_memory", False)
    if ablation == "graph_only":
        pruning = False
    elif ablation == "no_shared":
        shared_priority = False
    elif ablation == "no_memory":
        isolate_memory = True
    elif ablation not in {
        "full",
        "shared_only",
        "constraint_inference_only",
        "epv_only",
        "counterfactual_only",
        "epv_plus_counterfactual",
        "full_cghe_v2",
    }:
        raise ValueError(f"unknown CGHE_ABLATION: {ablation}")
    n_env = os.environ.get("CGHE_N", "").strip()
    if n_env.isdigit():
        count = int(n_env)

    env_rr = os.environ.get("CGHE_ROUND_REGEN")
    if env_rr is not None and str(env_rr).strip() != "":
        round_regen = str(env_rr).strip().lower() not in {"0", "false", "no"}
    else:
        round_regen = bool(merged.get("round_regen", True))

    return {
        "enabled": bool(enabled),
        "n": max(1, count),
        "confidence_threshold": threshold,
        "sqlite_path": str(path),
        "recovery_rounds": positive("recovery_rounds", 2),
        "tool_calls": positive("tool_calls", 12),
        "replans": positive("replans", 1),
        "pruning": pruning,
        "shared_priority": shared_priority,
        "isolate_memory": isolate_memory,
        "ablation": ablation,
        "algorithm": _algorithm(merged.get("algorithm")),
        "lambda_cost": _float_setting(merged.get("lambda_cost"), 0.0),
        "mu_risk": _float_setting(merged.get("mu_risk"), 0.0),
        "epv_threshold": _float_setting(merged.get("epv_threshold"), 0.0),
        "max_step1_attempts": positive("max_step1_attempts", 3),
        "max_total_step1_attempts": positive("max_total_step1_attempts", 12),
        "max_verified_snippet_prunes_per_episode": positive("max_verified_snippet_prunes_per_episode", 2),
        "max_inherited_ce": positive("max_inherited_ce", 32),
        "round_regen": round_regen,
        "max_regenerations": positive("max_regenerations", 5 if round_regen else 2),
        "selection": merged.get("selection") if isinstance(merged.get("selection"), dict) else {
            "strategy": "max_step1_path_coverage",
        },
        "pruning_gates": merged.get("pruning_gates") if isinstance(merged.get("pruning_gates"), dict) else {
            "require_verified_counterexample": True,
            "require_direct_observation": True,
            "require_complete_scope": True,
            "prune_only_explicitly_dependent_plans": True,
        },
    }


def _algorithm(value: Any) -> str:
    name = str(os.environ.get("CGHE_ALGORITHM") or value or "v1").strip().lower()
    if name not in {"v1", "v2"}:
        raise ValueError(f"unknown CGHE algorithm: {name}")
    return name


def _float_setting(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def cghe_enabled() -> bool:
    return bool(cghe_settings()["enabled"])


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
    # Strip leaked <think> and peel \\boxed{...} so LLM judges see the short answer.
    try:
        from rl.rewards.outcome import extract_answer_text, strip_thinking, unwrap_boxed

        cleaned = strip_thinking(body) or body
        extracted = extract_answer_text(f"<answer>{cleaned}</answer>")
        body = extracted or unwrap_boxed(cleaned) or cleaned
    except Exception:  # noqa: BLE001
        body = re.sub(r"<think>.*?</think>", "", body, flags=re.I | re.S).strip() or body
    if str(source or "").lower() == "gsm8k":
        nums = _NUM_RE.findall(body)
        body = nums[-1] if nums else body
    body = _prefer_short_answer_span(str(body or "").strip())
    if not body:
        return ""
    return f"<answer>{body}</answer>"


def _prefer_short_answer_span(body: str) -> str:
    """Prefer a short span matching Hotpot/Bamboogle gold granularity.

    Drops long explanatory sentences while keeping full dates / proper names.
    """
    text = str(body or "").strip()
    if not text:
        return ""
    # Prefer \\boxed{...} if still nested.
    boxed = re.search(r"\\boxed\{([^{}]{1,160})\}", text)
    if boxed:
        text = boxed.group(1).strip()
    # Drop "X is located in Y, and ..." multi-clause dumps: keep first clause if short.
    if len(text) > 80 and ("," in text or " and " in text.lower() or "。" in text):
        for sep in (". ", "。", "; ", "；", "\n"):
            if sep in text:
                first = text.split(sep, 1)[0].strip()
                if 1 <= len(first) <= 80:
                    text = first
                    break
    # Strip leading labels like "Answer:" / "The answer is".
    text = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s*(?:is|:)\s*", "", text, flags=re.I).strip()
    # Cap runaway prose; keep up to ~12 tokens for entity answers.
    words = text.split()
    if len(words) > 16 and len(text) > 100:
        text = " ".join(words[:12]).strip(" ,;:")
    return text.strip()


class Components:
    """Planner / Verifier / Executor built once; engines read the endpoint per call."""

    def __init__(self, runtime: Dict[str, Any]) -> None:
        _ensure_import()
        from agentflow.models.executor import Executor
        from agentflow.models.planner import Planner
        from agentflow.models.verifier import Verifier

        temp = float(runtime.get("temperature", 0.7))
        cap = int(runtime.get("max_tokens") or 4096)
        self.log_key = threading.local()

        def engine(role: str, trainable: bool = False, temperature: float = temp) -> LocalChatEngine:
            return LocalChatEngine(
                role, trainable=trainable, temperature=temperature, max_tokens_cap=cap,
                log_key=lambda: getattr(self.log_key, "value", ""),
            )

        self.available_tools, self.toolbox_metadata, instances = build_toolbox(
            list(runtime["enabled_tools"]), lambda: engine("tool", temperature=0.0)
        )
        self.tool_instances = instances
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
        # Only the planner samples. Frozen roles stay at temperature 0 so tool
        # text is part of the environment, not another policy.
        self.planner.llm_engine = engine("planner", trainable=True, temperature=temp)
        self.planner.llm_engine_fixed = engine("planner_fixed", temperature=0.0)
        self.verifier.llm_engine = engine("verifier", temperature=0.0)
        self.verifier.llm_engine_fixed = engine("verifier", temperature=0.0)
        self.executor.llm_generate_tool_command = engine("executor", temperature=0.0)


class Episode:
    def __init__(self, comps: Components, question: str, source: str) -> None:
        from agentflow.models.memory import Memory

        self.question = question
        self.source = source
        self.memory = Memory()
        self.memory.outline = []
        self.memory.obtained_information = []
        self.memory.planning_graph = {}
        self._pending_plan: Dict[str, Any] = {}
        self.recovery_chain_id = uuid4().hex
        self._resume_parent_prediction_event_id = ""
        self._cghe = None
        self._cghe_directive = dict(_PEV_DIRECTIVE)
        self._cghe_metrics: Dict[str, Any] = {
            "steps": 0,
            "deps_ready_selections": 0,
            "premature_extract": 0,
            "evidence_sufficient_steps": 0,
            "answer_from_sufficient": 0,
            "feedback": [],
        }
        self._cghe_snapshot: Dict[str, Any] = {}
        self._cghe_graph_mode = ""
        self._cghe_pages: List[str] = []
        # Planner keeps query_analysis on self; Executor keeps its cache dir.
        self.planner = copy.copy(comps.planner)
        self.executor = copy.copy(comps.executor)
        self.executor.set_query_cache_dir(tempfile.mkdtemp(dir=comps.executor.root_cache_dir))
        self.verifier = comps.verifier
        if sft_prompts_enabled():
            from sft_prompts import bind_episode

            bind_episode(self.planner, self.executor, self.verifier)
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
        from sft_prompts import normalize_outline

        step = int(self.step_count)
        actions = {}
        for key, action in self.memory.get_actions().items():
            if not isinstance(action, dict):
                continue
            saved = {
                "tool_name": action.get("tool_name"),
                "sub_goal": action.get("sub_goal"),
                "command": action.get("command"),
                "result": action.get("result"),
            }
            for extra in _ACTION_EXTRA_KEYS:
                if action.get(extra) not in (None, ""):
                    saved[extra] = action.get(extra)
            actions[str(key)] = saved
        if replay_plan:
            step = max(0, step - 1)
            actions.pop(f"Action Step {self.step_count}", None)
        state = {
            "question": self.question,
            "source": self.source,
            "query_analysis": self.query_analysis,
            "step_count": step,
            "actions": actions,
            "outline": normalize_outline(getattr(self.memory, "outline", None) or []),
            "obtained_information": [str(item) for item in (getattr(self.memory, "obtained_information", None) or [])],
            "planning_graph": dict(getattr(self.memory, "planning_graph", None) or {}),
            "query": self.memory.get_query(),
            "files": list(self.memory.get_files()),
            "analyzed": bool(self.query_analysis),
            "recovery_chain_id": self.recovery_chain_id,
            "parent_prediction_event_id": self._latest_prediction_event_id(),
        }
        if cghe_enabled():
            state["cghe_directive"] = dict(self._cghe_directive)
            if self._cghe is not None:
                state["cghe"] = _plain_cghe_state(self._cghe)
        return state

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Replace Memory and the step counter. Does not call ``analyze_query``."""
        from sft_prompts import normalize_outline

        self.query_analysis = str(state.get("query_analysis") or "")
        self.step_count = int(state.get("step_count") or 0)
        self.recovery_chain_id = str(state.get("recovery_chain_id") or self.recovery_chain_id)
        self._resume_parent_prediction_event_id = str(state.get("parent_prediction_event_id") or "")
        self.memory.actions = {}
        for key, action in (state.get("actions") or {}).items():
            if not isinstance(action, dict):
                continue
            saved = {
                "tool_name": action.get("tool_name"),
                "sub_goal": action.get("sub_goal"),
                "command": action.get("command"),
                "result": action.get("result"),
            }
            for extra in _ACTION_EXTRA_KEYS:
                if action.get(extra) not in (None, ""):
                    saved[extra] = action.get(extra)
            self.memory.actions[str(key)] = saved
        query = state.get("query")
        if query:
            self.memory.set_query(str(query))
        self.memory.files = [dict(item) for item in (state.get("files") or []) if isinstance(item, dict)]
        self.memory.outline = normalize_outline(state.get("outline"))
        self.memory.obtained_information = [
            str(item) for item in (state.get("obtained_information") or []) if str(item or "").strip()
        ]
        graph_state = state.get("planning_graph")
        self.memory.planning_graph = dict(graph_state) if isinstance(graph_state, dict) else {}
        self._restore_cghe(state)

    def cghe_controller(self):
        """Lazily open recovery for this episode. ``None`` when CGHE is off."""
        if not cghe_enabled():
            return None
        if self._cghe is None:
            from recovery_controller import RecoveryController

            cfg = cghe_settings()
            db_path = cfg["sqlite_path"]
            if cfg.get("isolate_memory"):
                db_path = str(Path(db_path).with_name(f"{Path(db_path).stem}_{self.recovery_chain_id}.sqlite"))
            self._cghe = RecoveryController(
                tool_calls=cfg["tool_calls"],
                replans=cfg["replans"],
                recovery_rounds=cfg["recovery_rounds"],
                confidence_threshold=cfg["confidence_threshold"],
                environment=CGHE_ENVIRONMENT,
                task_id=self.recovery_chain_id,
                task_type="",
                task_goal=self.question,
                db_path=db_path,
                pruning=bool(cfg.get("pruning", True)),
                shared_priority=bool(cfg.get("shared_priority", True)),
                algorithm=str(cfg.get("algorithm") or "v1"),
                v2_mode=str(cfg.get("ablation") or "full"),
                lambda_cost=float(cfg.get("lambda_cost") or 0.0),
                mu_risk=float(cfg.get("mu_risk") or 0.0),
            )
        return self._cghe

    def _restore_cghe(self, state: Dict[str, Any]) -> None:
        """Rebuild the controller from plain fields. A disabled switch drops it."""
        if not cghe_enabled():
            self._cghe_directive = dict(_PEV_DIRECTIVE)
            if self._cghe is not None:
                self._cghe.memory.close()
                self._cghe = None
            return
        directive = dict(_PEV_DIRECTIVE)
        raw_directive = state.get("cghe_directive")
        if isinstance(raw_directive, dict):
            for key in _PEV_DIRECTIVE:
                if raw_directive.get(key) not in (None,):
                    directive[key] = raw_directive.get(key)
        self._cghe_directive = directive
        raw = state.get("cghe")
        if isinstance(raw, dict):
            controller = self.cghe_controller()
            controller.restore_state(raw)
            if raw.get("confidence_threshold") is not None:
                threshold = float(raw["confidence_threshold"])
                controller.confidence_threshold = threshold
                controller.analyzer.confidence_threshold = threshold
                controller.pruner.confidence_threshold = threshold
            return
        if self._cghe is not None:
            self._cghe.memory.close()
            self._cghe = None

    def _latest_prediction_event_id(self) -> str:
        for action in reversed(list(self.memory.get_actions().values())):
            if isinstance(action, dict) and action.get("prediction_event_id"):
                return str(action["prediction_event_id"])
        return ""


def tool_result_failed(result: Any) -> bool:
    """True when a tool result is an error, a timeout, or an unavailable search/fetch.

    ``make_json_serializable_truncated`` wraps strings in a list, so a check
    that only looks at ``str.startswith`` misses ``["Error: ..."]``. An empty
    list is not a failure.
    """
    if result is None or result == [] or result == "":
        return False
    if isinstance(result, (list, tuple, dict)):
        text = json.dumps(result, ensure_ascii=False)
    else:
        text = str(result)
    blob = text.lstrip()
    low = blob.lower()
    if blob.startswith(("Error", "Execution timed out", '["Error', '["Execution timed out')):
        return True
    return "search_unavailable" in low or "fetch_unavailable" in low


def _plain_cghe_state(controller: Any) -> Dict[str, Any]:
    """JSON-safe controller state. A SQLite connection cannot pass through."""
    state = controller.export_state()
    _assert_plain_cghe(state)
    return json.loads(json.dumps(state, ensure_ascii=False))


def _assert_plain_cghe(value: Any) -> None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"_conn", "connection", "memory"}:
                raise RuntimeError("CGHE state must not include a SQLite connection")
            _assert_plain_cghe(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_plain_cghe(item)
        return
    raise RuntimeError(f"CGHE state is not plain data: {type(value).__name__}")


def _registered_tool(name: Any) -> str:
    """Map a normalized graph tool back to a registered tool name when possible."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    from sft_prompts import resolve_tool_name

    return resolve_tool_name(raw) or raw


def _directive_from_outcome(outcome: Any) -> Dict[str, Any]:
    """Planner steering only. continue_pev and stop_budget stay on the original step."""
    decision = str(getattr(outcome, "decision", "") or "")
    reason = str(getattr(outcome, "reason", "") or "")
    if decision not in ("need_universe", "continue_execution") or not bool(getattr(outcome, "intervene", False)):
        cleared = dict(_PEV_DIRECTIVE)
        cleared["reason"] = reason
        cleared["message"] = str(getattr(outcome, "message", "") or "")
        return cleared
    directive = dict(_PEV_DIRECTIVE)
    directive.update({
        "decision": decision,
        "reason": reason,
        "intervene": True,
        "message": str(getattr(outcome, "message", "") or ""),
        "round_id": str(getattr(outcome, "round_id", "") or ""),
    })
    node = getattr(outcome, "selected_node", None)
    if node is not None:
        plan_ids = tuple(getattr(node, "candidate_plan_ids", ()) or ())
        directive.update({
            "node_id": str(getattr(node, "node_id", "") or ""),
            "plan_id": str(plan_ids[0]) if plan_ids else "",
            "subgoal": str(getattr(node, "subgoal", "") or ""),
            "action": str(getattr(node, "action", "") or ""),
            "tool": _registered_tool(getattr(node, "tool", "") or getattr(node, "action", "")),
        })
    if decision == "continue_execution" and not directive["tool"] and not directive["subgoal"]:
        cleared = dict(_PEV_DIRECTIVE)
        cleared["reason"] = reason or "selected node has no executable subgoal or tool"
        return cleared
    return directive


def _steer(ep: "Episode", outcome: Any) -> None:
    ep._cghe_directive = _directive_from_outcome(outcome)


def _directive_continue(node: Any, reason: str, round_id: str) -> Dict[str, Any]:
    plan_ids = tuple(getattr(node, "candidate_plan_ids", ()) or ())
    directive = dict(_PEV_DIRECTIVE)
    directive.update({
        "decision": "continue_execution",
        "reason": reason,
        "intervene": True,
        "node_id": str(getattr(node, "node_id", "") or ""),
        "plan_id": str(plan_ids[0]) if plan_ids else "",
        "subgoal": str(getattr(node, "subgoal", "") or ""),
        "action": str(getattr(node, "action", "") or ""),
        "tool": _registered_tool(getattr(node, "tool", "") or getattr(node, "action", "")),
        "task_id": str(getattr(node, "task_id", "") or ""),
        "round_id": str(round_id or ""),
    })
    return directive


def _recovery_round_open(ep: "Episode") -> bool:
    ctrl = getattr(ep, "_cghe", None)
    if ctrl is None or ctrl.graph is None:
        return False
    snap = ctrl.export_state()
    return not bool(snap.get("round_exhausted")) and not bool(snap.get("stopped"))


def _bind_plan_environment(plans: Any) -> None:
    """Fill a blank plan environment so scope matching can see this episode.

    A plan that already names an environment is left unchanged. This does not
    add premises and does not mark any model text as verified evidence.
    """
    from cghe_types import Scope

    for plan in plans:
        current = plan.context
        if str(getattr(current, "environment", "") or "").strip():
            continue
        plan.context = Scope(
            environment=CGHE_ENVIRONMENT,
            tool=getattr(current, "tool", "") or "",
            resource=getattr(current, "resource", "") or "",
            identity=getattr(current, "identity", "") or "",
            parameter=getattr(current, "parameter", "") or "",
            argument=getattr(current, "argument", "") or "",
            schema_signature=getattr(current, "schema_signature", "") or "",
            task_type=getattr(current, "task_type", "") or "",
            call_instance=getattr(current, "call_instance", "") or "",
        )


def _lock_recovery_step(ep: "Episode", directive: Dict[str, Any]) -> bool:
    """Use the selected node as this step. Returns false when it is not executable."""
    tool = _registered_tool(directive.get("tool") or directive.get("action") or "")
    subgoal = str(directive.get("subgoal") or "").strip()
    action = str(directive.get("action") or "").strip()
    if not tool or not subgoal:
        return False
    context = action if action and action != tool else subgoal
    node_id = str(directive.get("node_id") or "")
    controller = getattr(ep, "_cghe", None)
    graph = getattr(controller, "graph", None) if controller is not None else None
    if graph is not None and node_id:
        try:
            bound = str(graph.tool_input_for(node_id) or "").strip()
        except Exception:
            bound = ""
        if bound:
            context = bound
        elif "fetch" in tool.lower():
            return False
        elif "generalist" in tool.lower() or "generator" in tool.lower():
            # Extract must wait for page body; snippets alone are not enough.
            return False
    ep.context = context
    ep.sub_goal = subgoal
    ep.tool_name = tool
    ep._tool_parse_note = ""
    plan = {
        "outline": [],
        "subgoal": subgoal,
        "tool_name": tool,
        "tool_input": context,
        "prediction": "",
        "prediction_reason": "",
        "cghe_node_id": node_id,
        "cghe_task_id": str(directive.get("task_id") or ""),
    }
    ep._pending_plan = plan
    ep.planner._last_plan = dict(plan)
    return True


def _graph_for(ep: "Episode") -> Any:
    controller = getattr(ep, "_cghe", None)
    return getattr(controller, "graph", None) if controller is not None else None


def _graph_select_policy() -> str:
    """Default max_out_degree; set CGHE_SELECT_POLICY=frontier for ablation."""
    raw = str(os.environ.get("CGHE_SELECT_POLICY") or "max_out_degree").strip().lower()
    if raw in {"frontier", "shared", "first_plan", "max_out_degree"}:
        return raw
    return "max_out_degree"


def _store_graph_in_memory(ep: "Episode", graph: Any, *, selected_id: str = "") -> None:
    from sft_prompts import ensure_memory_fields

    ensure_memory_fields(ep.memory)
    snap = _graph_snapshot(graph, selected_id=selected_id)
    ep.memory.planning_graph = dict(snap)
    ep._cghe_snapshot = dict(snap)


def _metrics(ep: "Episode") -> Dict[str, Any]:
    data = getattr(ep, "_cghe_metrics", None)
    if not isinstance(data, dict):
        data = {
            "steps": 0,
            "deps_ready_selections": 0,
            "premature_extract": 0,
            "evidence_sufficient_steps": 0,
            "answer_from_sufficient": 0,
            "feedback": [],
        }
        ep._cghe_metrics = data
    return data


def _ingest_graph_result(
    ep: "Episode",
    *,
    tool_name: Any,
    raw_result: Any,
    ok: bool,
) -> Dict[str, Any]:
    """Store artifacts and three-way feedback for the locked graph node."""
    from sft_prompts import assess_step_evidence, _extract_search_results, _extract_urls, page_has_answer_clues

    directive = getattr(ep, "_cghe_directive", None) or {}
    node_id = str(directive.get("node_id") or "")
    graph = _graph_for(ep)
    metrics = _metrics(ep)
    metrics["steps"] = int(metrics.get("steps") or 0) + 1
    execution = "success" if ok else "error"
    text = raw_result if isinstance(raw_result, str) else json.dumps(raw_result, ensure_ascii=False, default=str)
    low = text.lower()
    if ok and any(tok in low for tok in ("timeout", "timed out")):
        execution = "timeout"
    if any(tok in low for tok in ("fetch_blocked", "blocked:", "access denied", "403 forbidden")):
        execution = "error"
        ok = False
    evidence = assess_step_evidence(tool_name, text, ok=ok and execution == "success")
    name = str(tool_name or "").lower()
    if (
        execution == "success"
        and "fetch" in name
        and evidence == "insufficient"
        and page_has_answer_clues(str(getattr(ep, "question", "") or ""), text)
    ):
        evidence = "sufficient"
    goal = "unmet"
    if execution != "success":
        goal = "unmet"
    elif "bing" in name or "wiki" in name:
        goal = "partial" if evidence in ("weak", "insufficient", "sufficient") else "unmet"
    elif "fetch" in name:
        goal = "partial" if evidence in ("insufficient", "sufficient") else "unmet"
    elif "generalist" in name or "generator" in name:
        goal = "met" if evidence == "sufficient" else "partial"
    feedback = {
        "node_id": node_id,
        "task_id": str(directive.get("task_id") or ""),
        "execution": execution,
        "evidence": evidence,
        "goal": goal,
    }
    if graph is not None and node_id:
        if execution == "success":
            if "bing" in name or "wiki" in name:
                results = _extract_search_results(text, limit=8)
                urls = [row["url"] for row in results] or _extract_urls(text, limit=8)
                graph.store_search_results(node_id, urls, results=results)
            elif "fetch" in name:
                graph.store_page_content(node_id, text)
                body = str(text or "").strip()
                if len(body) >= 80:
                    pages = list(getattr(ep, "_cghe_pages", None) or [])
                    pages.append(body)
                    ep._cghe_pages = pages[-8:]
        feedback = graph.record_feedback(
            node_id, execution=execution, evidence=evidence, goal=goal,
        )
    if evidence == "sufficient":
        metrics["evidence_sufficient_steps"] = int(metrics.get("evidence_sufficient_steps") or 0) + 1
    if ("generalist" in name or "generator" in name) and evidence != "sufficient":
        # Extract ran without sufficient page-backed evidence.
        metrics["premature_extract"] = int(metrics.get("premature_extract") or 0) + 1
    bucket = list(metrics.get("feedback") or [])
    bucket.append(feedback)
    metrics["feedback"] = bucket[-20:]
    action = ep.memory.actions.get(f"Action Step {ep.step_count}")
    if isinstance(action, dict):
        action["cghe_feedback"] = dict(feedback)
        if node_id:
            action["cghe_node_id"] = node_id
        if feedback.get("input_url"):
            action["cghe_input_url"] = feedback.get("input_url")
    return feedback


def _json_tool_result(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False)
        safe: Any = value
    except TypeError:
        safe = json.loads(json.dumps(value, ensure_ascii=False, default=str))
    if isinstance(safe, str) and len(safe) > _EXECUTION_LOG_CHARS:
        return safe[:_EXECUTION_LOG_CHARS]
    return safe


def _format_execution_log(tool_name: Any, command: Any, result: Any) -> str:
    if isinstance(result, str):
        rendered = result
    else:
        rendered = json.dumps(result, ensure_ascii=False, default=str)
    return f"tool={tool_name}\ncommand={command}\nresult={rendered}"[:_EXECUTION_LOG_CHARS]


def _authorize_cghe_call(controller: Any, tool_name: str, command: str):
    """First call is a normal authorization. A repeat uses may_auto_replay only when safe."""
    from recovery_controller import call_fingerprint, retry_policy_for

    environment = controller.environment
    fingerprint = call_fingerprint(tool_name, command=command, environment=environment)
    prior = int(dict(controller.export_state().get("fingerprints") or {}).get(fingerprint, 0))
    if prior > 0 and retry_policy_for(tool_name).safe_to_retry:
        return controller.may_auto_replay(tool_name, command=command, environment=environment)
    return controller.authorize_tool_call(
        tool_name, command=command, environment=environment, replay=False,
    )


def _preserved_tool_result(ep: "Episode", tool_name: str, command: str, reason: str) -> Any:
    """Keep the original tool result when recovery refuses to run the call again."""
    actions = ep.memory.get_actions() if ep.memory is not None else {}
    for action in reversed(list((actions or {}).values())):
        if not isinstance(action, dict):
            continue
        if str(action.get("tool_name") or "") != str(tool_name or ""):
            continue
        if str(action.get("command") or "") != str(command or ""):
            continue
        if "raw_tool_result" in action:
            return action.get("raw_tool_result")
        if action.get("result") not in (None, ""):
            return action.get("result")
    return f"recovery refused to replay {tool_name}: {reason}"


def _mark_directive_node_failed(ep: "Episode", controller: Any) -> None:
    node_id = str((getattr(ep, "_cghe_directive", None) or {}).get("node_id") or "")
    graph = getattr(controller, "graph", None)
    if not node_id or graph is None:
        return
    if any(node.node_id == node_id for node in graph.nodes()):
        graph.mark_failed(node_id)


def _execute_authorized(ep: "Episode", tool_name: str, command: str) -> Any:
    if not cghe_enabled():
        return ep.executor.execute_tool_command(tool_name, command)
    controller = ep.cghe_controller()
    if controller is None:
        return ep.executor.execute_tool_command(tool_name, command)
    auth = _authorize_cghe_call(controller, tool_name, command)
    ep._cghe_last_auth = {
        "allowed": bool(auth.allowed),
        "replay": bool(auth.replay),
        "reason": str(auth.reason or "")[:200],
    }
    if not auth.allowed:
        _mark_directive_node_failed(ep, controller)
        return _preserved_tool_result(ep, tool_name, command, auth.reason)
    result = ep.executor.execute_tool_command(tool_name, command)
    controller.commit_tool_call(auth)
    if not bool(getattr(auth, "replay", False)):
        _note_step1_execution(ep, controller)
    return result


def _stamp_execution_observation(
    ep: "Episode",
    action: Dict[str, Any],
    tool_name: Any,
    command: Any,
    raw_result: Any,
) -> None:
    if not cghe_enabled():
        return
    action["execution_log"] = _format_execution_log(tool_name, command, raw_result)
    action["raw_tool_result"] = _json_tool_result(raw_result)
    directive = getattr(ep, "_cghe_directive", None) or {}
    if str(directive.get("decision") or "") == "continue_execution":
        if directive.get("node_id"):
            action["cghe_node_id"] = str(directive.get("node_id") or "")
        if directive.get("plan_id"):
            action["cghe_plan_id"] = str(directive.get("plan_id") or "")
    auth = getattr(ep, "_cghe_last_auth", None)
    if isinstance(auth, dict):
        action["cghe_authorized"] = bool(auth.get("allowed"))
        action["cghe_replay"] = bool(auth.get("replay"))
        if auth.get("reason"):
            action["cghe_auth_reason"] = str(auth.get("reason") or "")[:200]


def _plan_argument_for_action(ep: "Episode", action: Dict[str, Any]) -> str:
    """Planner-side argument for this step: graph node argument, else tool input / subgoal."""
    node_id = str(action.get("cghe_node_id") or "")
    controller = getattr(ep, "_cghe", None)
    graph = getattr(controller, "graph", None) if controller is not None else None
    if graph is not None and node_id:
        try:
            node = graph.node(node_id)
            argument = str(getattr(node, "argument", "") or getattr(node, "action", "") or "").strip()
            if argument:
                return argument
        except Exception:
            pass
    for key in ("tool_input", "command", "sub_goal"):
        value = str(action.get(key) or "").strip()
        if value:
            return value
    return str(getattr(ep, "context", "") or getattr(ep, "sub_goal", "") or "").strip()


def _observation_from_action(
    action: Dict[str, Any],
    *,
    verifier_label: str,
    hypothesis: str,
    ep: Optional["Episode"] = None,
) -> Any:
    from cghe_types import ExecutionObservation

    tool_name = str(action.get("tool_name") or "")
    command = str(action.get("command") or "")
    argument = _plan_argument_for_action(ep, action) if ep is not None else str(
        action.get("tool_input") or action.get("sub_goal") or command or ""
    ).strip()
    parameters: Dict[str, Any] = {}
    if argument:
        if "bing" in tool_name.lower() or "search" in tool_name.lower():
            parameters = {"query": argument}
            if "query=" not in command.casefold():
                command = f"execution = tool.execute(query={argument!r})"
        elif "fetch" in tool_name.lower():
            parameters = {"resource": argument}
    return ExecutionObservation(
        tool_name=tool_name,
        environment=CGHE_ENVIRONMENT,
        tool_result=action.get("raw_tool_result", action.get("result")),
        execution_log=str(action.get("execution_log") or ""),
        llm_hypothesis=hypothesis,
        verifier_label=verifier_label,
        plan_id=str(action.get("cghe_plan_id") or ""),
        node_id=str(action.get("cghe_node_id") or ""),
        command=command,
        parameters=parameters,
        resource=str(parameters.get("resource") or ""),
    )


def _update_v2_hypotheses(controller: Any, *, observation: str, verified: bool) -> None:
    if getattr(controller, "algorithm", "v1") != "v2":
        return
    from v2.counterfactual import update_hypothesis

    for item in list(getattr(controller, "v2_hypotheses", []) or []):
        update_hypothesis(item, observation=observation, verified_counterexample=verified)


def _verifier_signal(verdict: Dict[str, Any]) -> tuple:
    satisfied = str(verdict.get("satisfied") or "").upper()
    if satisfied == "YES":
        return "满足", True
    if satisfied == "NO":
        return "不满足", False
    return "uncertain", None


def _note_step1_execution(ep: "Episode", controller: Any) -> None:
    """Count one real Step-1 Executor call. Later nodes and replays are not Step-1 attempts."""
    from cghe_loop import session_for

    graph = getattr(controller, "graph", None)
    directive = getattr(ep, "_cghe_directive", None) or {}
    node_id = str(directive.get("node_id") or "")
    if graph is None or not node_id or not graph.is_step1_node(node_id):
        return
    attempt_id = f"{graph.dag_id}:{node_id}:{ep.step_count}"
    ep._cghe_open_attempt_id = attempt_id
    session = session_for(controller, cghe_settings())
    session.note_step1_execution(
        graph,
        node_id,
        attempt_id=attempt_id,
        selected_plan_ids=graph.plans_starting_with(node_id),
        coverage=graph.compute_step1_coverage().get(node_id, 0),
    )


def _handoff_active(ep: "Episode") -> bool:
    data = getattr(ep, "_cghe_handoff", None)
    return isinstance(data, dict) and bool(data.get("active"))


def _complete_cghe_plans(ep: "Episode") -> None:
    """Mark the handed-off plans completed only when the task itself stops."""
    graph = _graph_for(ep)
    if graph is None:
        return
    handoff = getattr(ep, "_cghe_handoff", None) or {}
    plan_ids = list(handoff.get("selected_plan_ids") or [])
    if not plan_ids:
        plan_ids = list(graph.active_plan_ids())
    graph.complete_plans(plan_ids)


def _plan_successors(graph: Any, node_id: str) -> list[str]:
    """Pending downstream nodes on ACTIVE plans that contain ``node_id``."""
    successors: list[str] = []
    for plan_id in graph.active_plan_ids():
        sequence = list(graph.plan_node_ids(plan_id))
        if node_id not in sequence:
            continue
        index = sequence.index(node_id)
        if index + 1 >= len(sequence):
            continue
        nxt = sequence[index + 1]
        status = str(getattr(graph.node(nxt).status, "value", graph.node(nxt).status) or "")
        if status == "pending" and nxt not in successors:
            successors.append(nxt)
    return successors


def _lock_successor(
    ep: "Episode",
    controller: Any,
    graph: Any,
    node_id: str,
    successors: list[str],
    *,
    reason: str,
) -> bool:
    """Lock the highest-coverage pending successor (usually Fetch after Bing)."""
    from cghe_loop import session_for
    from cghe_dag_log import log_advance

    if not successors:
        return False
    session = session_for(controller, cghe_settings())
    coverage = graph.compute_step1_coverage()
    nxt = max(successors, key=lambda item: (coverage.get(item, 0), item))
    session.log_selection(graph, nxt, coverage.get(nxt, 0), {item: coverage.get(item, 0) for item in successors})
    log_advance(node_id, nxt, int(coverage.get(nxt, 0)), graph)
    ep._cghe_graph_mode = "switch"
    _store_graph_in_memory(ep, graph, selected_id=nxt)
    ep._cghe_directive = _directive_continue(graph.node(nxt), reason, graph.round_id)
    ep._cghe_directive["coverage"] = int(coverage.get(nxt, 0))
    graph.protect_children_of = node_id
    return True


def _advance_to_plan_successor(
    ep: "Episode",
    controller: Any,
    node_id: str,
    *,
    soft: bool = False,
    verifier_status: str = "SUCCESS",
) -> bool:
    """Advance along the current plan. Soft Bing failures do not mark the node satisfied."""
    from cghe_loop import session_for

    graph = getattr(controller, "graph", None)
    session = session_for(controller, cghe_settings())
    if not node_id or graph is None or not any(node.node_id == node_id for node in graph.nodes()):
        return False
    successors = _plan_successors(graph, node_id)
    if not soft:
        graph.mark_satisfied(node_id)
    else:
        # Keep the Bing node reusable for sibling OR plans; just remember soft miss.
        try:
            if hasattr(graph, "mark_attempted"):
                graph.mark_attempted(node_id)
            elif hasattr(graph, "note_attempt"):
                graph.note_attempt(node_id)
        except Exception:
            pass
    attempt_id = str(getattr(ep, "_cghe_open_attempt_id", "") or "")
    if attempt_id:
        graph.finish_attempt(
            attempt_id,
            execution_status="success" if not soft else "soft_continue",
            verifier_status=verifier_status,
        )
    session.events.append({
        "event": "verification_completed",
        "verifier_status": verifier_status if not soft else (verifier_status or "SOFT_CONTINUE"),
        "node_id": node_id,
        "dag_id": graph.dag_id,
        "task_completed": False,
        "soft_advance": bool(soft),
    })
    reason = (
        "soft-advance to Fetch after Bing snippet gap"
        if soft
        else f"advanced along path after success"
    )
    if _lock_successor(ep, controller, graph, node_id, successors, reason=reason):
        return True
    return False


def _advance_after_success(ep: "Episode", controller: Any, node_id: str) -> None:
    """A successful step stays on this DAG. The next ready step 1 is selected by coverage."""
    from cghe_loop import session_for

    graph = getattr(controller, "graph", None)
    session = session_for(controller, cghe_settings())
    if not node_id or graph is None or not any(node.node_id == node_id for node in graph.nodes()):
        ep._cghe_directive = dict(_PEV_DIRECTIVE)
        ep._cghe_directive["reason"] = "verified step had no graph node"
        return
    if _advance_to_plan_successor(ep, controller, node_id, soft=False, verifier_status="SUCCESS"):
        return
    # No downstream on this plan: pick another Step-1 or regenerate.
    selection = graph.select_step1()
    if selection is None or graph.coverage_saturated():
        ep._cghe_pending_reasons = list(graph.regeneration_reasons() or ["NO_EXECUTABLE_CANDIDATE"])
        ep._cghe_directive = dict(_PEV_DIRECTIVE)
        ep._cghe_directive["reason"] = "step succeeded; DAG needs another decision"
        ep._cghe_graph_mode = "rebuild"
        return
    nxt, coverage, eligible = selection
    session.log_selection(graph, nxt, coverage, eligible)
    ep._cghe_graph_mode = "switch"
    _store_graph_in_memory(ep, graph, selected_id=nxt)
    ep._cghe_directive = _directive_continue(
        graph.node(nxt), f"advanced to step1 coverage {coverage}", graph.round_id,
    )
    ep._cghe_directive["coverage"] = int(coverage)


def _cghe_on_verdict(ep: "Episode", verdict: Dict[str, Any]) -> None:
    """Record the verifier judgment. Recovery steers only an explicit, admitted failure."""
    if not cghe_enabled():
        return
    action = ep.memory.actions.get(f"Action Step {ep.step_count}")
    if not isinstance(action, dict):
        return
    from recovery_controller import verifier_explicitly_failed

    label, flag = _verifier_signal(verdict)
    controller = ep.cghe_controller()
    if controller is None:
        return
    # Judge the planner subgoal via the verifier, not whether the tool call
    # mechanically returned links or a long page body.
    if flag is True:
        observation = _observation_from_action(action, verifier_label=label, hypothesis="", ep=ep)
        controller.record_verified_success(
            observation,
            action=str(action.get("sub_goal") or ""),
            subgoal=str(action.get("sub_goal") or ""),
            tool=str(action.get("tool_name") or ""),
            source="verifier",
            note="verifier satisfied",
        )
        try:
            from hypothesis_round_memory import build_verifier_feedback

            graph = getattr(controller, "graph", None)
            fb = build_verifier_feedback(
                satisfied=True,
                label=label or "满足",
                analysis=None,
                action=action,
                evidence_text="verifier satisfied",
                dag_version=str(getattr(graph, "version", "") or getattr(graph, "dag_id", "") or ""),
                round_id=int(getattr(getattr(controller, "cghe_session", None), "planning_round", 0) or 0),
            )
            _hypothesis_memory(controller).update_from_feedback(
                fb, environment_fingerprint=_env_fingerprint(ep),
            )
        except Exception:
            pass
        _update_v2_hypotheses(controller, observation="verifier_satisfied", verified=False)
        _advance_after_success(ep, controller, str(action.get("cghe_node_id") or ""))
        return
    if _switch_after_graph_failure(ep, controller, action, label, flag):
        return
    if not verifier_explicitly_failed(label, flag):
        if str((ep._cghe_directive or {}).get("decision") or "") == "continue_execution":
            ep._cghe_directive = dict(_PEV_DIRECTIVE)
            ep._cghe_directive["reason"] = "verifier did not explicitly mark the step unsatisfied"
        return
    observation = _observation_from_action(
        action,
        verifier_label=label,
        hypothesis=str(verdict.get("reason") or ""),
        ep=ep,
    )
    outcome = controller.handle_failure(
        observation,
        verifier_label=label,
        verifier_satisfied=flag,
        action=str(action.get("sub_goal") or action.get("command") or ""),
    )
    _steer(ep, outcome)


def _graph_snapshot(graph: Any, *, selected_id: str = "") -> Dict[str, Any]:
    if graph is None:
        return {}
    from cghe_types import PlanStatus

    outlines = []
    for plan in graph.plans():
        if plan.status != PlanStatus.ACTIVE:
            continue
        outlines.append({
            "plan_id": plan.plan_id,
            "steps": [
                {
                    "subgoal": step.subgoal,
                    "tool": step.tool,
                    "task_id": step.task_id,
                    "consumes": list(step.consumes),
                    "produces": list(step.produces),
                    "instance_key": step.instance_key,
                }
                for step in plan.steps
            ],
        })
    frontier = []
    seen = []
    for plan_id in graph.active_plan_ids():
        node_id = graph.frontier_node_id(plan_id)
        if not node_id or node_id in seen:
            continue
        seen.append(node_id)
        node = graph.node(node_id)
        frontier.append({
            "node_id": node_id,
            "subgoal": node.subgoal,
            "tool": node.tool,
            "task_id": node.task_id,
            "vote": graph.frontier_vote(node_id),
            "out_degree": graph.out_degree(node_id),
            "selected": node_id == selected_id,
            "input_url": graph.bound_url(node_id) if "search_url" in node.consumes else "",
        })
    return {
        "cghe_outlines": outlines,
        "cghe_frontier": frontier,
        "cghe_round_id": str(getattr(graph, "round_id", "") or ""),
        "cghe_select_policy": _graph_select_policy(),
    }


def _is_search_action(action: Dict[str, Any]) -> bool:
    tool = str(action.get("tool_name") or action.get("tool") or "").casefold()
    return "bing" in tool or "search" in tool


def _is_fetch_or_extract_action(action: Dict[str, Any]) -> bool:
    tool = str(action.get("tool_name") or action.get("tool") or "").casefold()
    return "fetch" in tool or "extract" in tool or "generalist" in tool or "generator" in tool


def _snippet_prune_budget_ok(ep: "Episode") -> bool:
    used = int(getattr(ep, "_cghe_snippet_prunes", 0) or 0)
    limit = int(cghe_settings().get("max_verified_snippet_prunes_per_episode") or 2)
    return used < max(0, limit)


def _note_snippet_prune(ep: "Episode") -> None:
    ep._cghe_snippet_prunes = int(getattr(ep, "_cghe_snippet_prunes", 0) or 0) + 1


def _upstream_bing_query(graph: Any, node_id: str, plan_id: str = "") -> str:
    """Recover the Step-1 Bing query for a Fetch/Extract node on the same plan."""
    plan_ids = [plan_id] if plan_id else list(graph.active_plan_ids())
    for pid in plan_ids:
        sequence = list(graph.plan_node_ids(pid))
        if node_id not in sequence:
            continue
        for prior_id in sequence:
            if prior_id == node_id:
                break
            node = graph.node(prior_id)
            tool = str(getattr(node, "tool", "") or getattr(node, "action", "") or "").casefold()
            if "bing" in tool or "search" in tool:
                argument = str(getattr(node, "argument", "") or getattr(node, "action", "") or "").strip()
                if argument:
                    return argument
    # Fallback: any Bing ancestor among plan ids that still list this node.
    for pid in graph.active_plan_ids():
        sequence = list(graph.plan_node_ids(pid))
        if node_id not in sequence:
            continue
        for prior_id in sequence:
            if prior_id == node_id:
                break
            node = graph.node(prior_id)
            tool = str(getattr(node, "tool", "") or "").casefold()
            if "bing" in tool or "search" in tool:
                argument = str(getattr(node, "argument", "") or "").strip()
                if argument:
                    return argument
    return ""


def _fetch_result_is_empty_or_error(action: Dict[str, Any], observation: Any) -> bool:
    """True only for empty/blocked Fetch bodies — not 'page OK but verifier wants more'."""
    text = str(
        getattr(observation, "tool_result", None)
        or action.get("raw_tool_result")
        or action.get("result")
        or ""
    ).strip()
    if not text:
        return True
    low = text.casefold()
    if low in {"null", "none", "empty", "no result", "no results", "[]", "{}"}:
        return True
    error_markers = (
        "http error", "status code 4", "status code 5", "403", "404", "429", "503",
        "blocked", "access denied", "timeout", "timed out", "proxyerror",
        "fetch_unavailable", "connection error", "traceback",
    )
    if any(m in low for m in error_markers) and len(text) < 400:
        return True
    # Very short body with no article signal → treat as empty page.
    if len(text) < 80 and "http" not in low and "<p" not in low:
        return True
    return False


def _deferred_snippet_ce_from_fetch(
    ep: "Episode",
    controller: Any,
    action: Dict[str, Any],
    observation: Any,
    label: str,
) -> Optional[Any]:
    """Only after Fetch/Extract returns empty/error, retire the upstream Bing query.

    Verifier-unsatisfied on a non-empty page must NOT become a hard snippet CE —
    that recreates Soft-Bing's failure mode one hop later.
    """
    from failure_analyzer import FailureAnalyzer
    from cghe_types import ExecutionObservation

    graph = getattr(controller, "graph", None)
    node_id = str(action.get("cghe_node_id") or "")
    if graph is None or not node_id:
        return None
    if not _is_fetch_or_extract_action(action):
        return None
    if not _fetch_result_is_empty_or_error(action, observation):
        return None
    query = _upstream_bing_query(graph, node_id, str(action.get("cghe_plan_id") or ""))
    if not query:
        return None
    if not _snippet_prune_budget_ok(ep):
        return None
    synthetic = ExecutionObservation(
        tool_name="Bing_Search_Tool",
        environment=str(getattr(observation, "environment", "") or CGHE_ENVIRONMENT),
        command=f"execution = tool.execute(query={query!r})",
        tool_result="No results",
        verifier_label=label or "不满足",
        plan_id=str(action.get("cghe_plan_id") or getattr(observation, "plan_id", "") or ""),
        parameters={"query": query},
    )
    analysis = FailureAnalyzer().analyze(synthetic)
    if not analysis.verified_counterexample or analysis.constraint is None:
        return None
    try:
        analysis.constraint.source = "deferred_fetch_empty_or_error"
        analysis.constraint.note = "Fetch empty/error; retire upstream search query"
    except Exception:
        pass
    return analysis


def _truncate_records(items: list, limit: int) -> list:
    if limit <= 0:
        return []
    if len(items) <= limit:
        return list(items)
    return list(items)[-limit:]


def _round_regen_enabled() -> bool:
    return bool(cghe_settings().get("round_regen", True))


def _hypothesis_memory(controller: Any) -> Any:
    mem = getattr(controller, "hypothesis_memory", None)
    if mem is not None:
        return mem
    from hypothesis_round_memory import HypothesisMemory

    mem = HypothesisMemory()
    controller.hypothesis_memory = mem
    return mem


def _env_fingerprint(ep: "Episode") -> str:
    # Lightweight environment fingerprint; extend if richer MAS state is available.
    return f"q:{norm_key(str(ep.question or '')[:120])}|step:{int(ep.step_count)}"


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())


def _end_round_for_regen(
    ep: "Episode",
    controller: Any,
    *,
    reasons: list[str],
    feedback: Any = None,
) -> bool:
    """Seal current DAG and request a NEW planning DAG (round-based regeneration)."""
    from cghe_loop import session_for

    session = session_for(controller, cghe_settings())
    memory = _hypothesis_memory(controller)
    if feedback is not None:
        memory.update_from_feedback(feedback, environment_fingerprint=_env_fingerprint(ep))
    graph = getattr(controller, "graph", None)
    version = str(getattr(graph, "version", "") or getattr(graph, "round_id", "") or "")
    if graph is not None and not getattr(graph, "sealed", False):
        session.archive_active_dag(reason=",".join(reasons) or "round_end")
    print(
        f"[CGHE][ROUND] End version={version} reasons={reasons} "
        f"hypotheses={len(memory)} last={getattr(memory.last_feedback, 'planning_implication', None)}",
        flush=True,
    )
    ep._cghe_graph_mode = "rebuild"
    ep._cghe_pending_reasons = list(reasons or ["ROUND_FEEDBACK_REGEN"])
    ep._cghe_directive = dict(_PEV_DIRECTIVE)
    ep._cghe_directive["reason"] = "round ended; generate a new Planning DAG from HypothesisMemory"
    # Clear active graph pointer so the next open builds/regenerates cleanly.
    # Historical snapshot remains in session.archived_dags / seal_snapshot.
    return True


def _switch_after_graph_failure(
    ep: "Episode",
    controller: Any,
    action: Dict[str, Any],
    label: str,
    flag: Optional[bool],
) -> bool:
    """Ordinary failures stay on the graph. Only a verified counterexample prunes plans.

    Soft Bing failures (URLs present, verifier unsatisfied) advance to Fetch on the
    same plan instead of pruning or regenerating.
    """
    from failure_analyzer import FailureAnalyzer
    from cghe_loop import prune_by_counterexample, session_for
    from cghe_types import CausalRelation, EvidenceStatus

    node_id = str(action.get("cghe_node_id") or "")
    graph = getattr(controller, "graph", None)
    if not node_id or graph is None or not any(node.node_id == node_id for node in graph.nodes()):
        return False
    observation = _observation_from_action(action, verifier_label=label, hypothesis="", ep=ep)
    analysis = FailureAnalyzer().analyze(observation)
    session = session_for(controller, cghe_settings())
    attempt_id = str(getattr(ep, "_cghe_open_attempt_id", "") or "")
    from cghe_dag_log import constraint_brief, log_prune, plans_text

    # Soft Bing (or non-empty Fetch) miss with a successor → same-plan advance.
    soft_codes = {"SUBGOAL_NOT_SATISFIED", "VERIFIER_UNCERTAIN", ""}
    obs_code_soft = str(getattr(analysis.evidence, "observation_code", "") or "")
    soft_search = (
        not analysis.verified_counterexample
        and _is_search_action(action)
        and obs_code_soft in soft_codes
    )
    soft_fetch = (
        not analysis.verified_counterexample
        and _is_fetch_or_extract_action(action)
        and obs_code_soft in soft_codes
        and not _fetch_result_is_empty_or_error(action, observation)
    )
    if (soft_search or soft_fetch) and _plan_successors(graph, node_id):
        try:
            controller._remember_failure(
                analysis, observation, action=str(action.get("sub_goal") or ""), promote=False,
            )
        except Exception:
            pass
        session.events.append({
            "event": "counterexample_validated",
            "predicate": "",
            "validation_status": EvidenceStatus.UNVERIFIED_HYPOTHESIS.value,
            "evidence_id": str(getattr(analysis.evidence, "evidence_id", "") or ""),
            "hard_constraint": False,
            "soft_advance": True,
            "soft_fetch": bool(soft_fetch),
            "constraint_brief": constraint_brief(analysis.constraint),
        })
        reason_status = label or ("SOFT_FETCH_CONTINUE" if soft_fetch else "SOFT_CONTINUE")
        if _advance_to_plan_successor(
            ep, controller, node_id, soft=True, verifier_status=reason_status,
        ):
            # Within-round soft continue: record UNVERIFIED, keep the same DAG.
            try:
                from hypothesis_round_memory import build_verifier_feedback

                fb = build_verifier_feedback(
                    satisfied=False,
                    label=label or "不满足",
                    analysis=analysis,
                    action=action,
                    evidence_text=str(getattr(observation, "tool_result", "") or "")[:400],
                    dag_version=str(getattr(graph, "version", "") or graph.dag_id),
                    round_id=int(getattr(session, "planning_round", 0) or 0),
                    soft_continue=True,
                )
                _hypothesis_memory(controller).update_from_feedback(
                    fb, environment_fingerprint=_env_fingerprint(ep),
                )
            except Exception:
                pass
            return True

    # Fetch empty/error only → deferred query-level CE against upstream Bing.
    deferred = None
    if (
        not analysis.verified_counterexample
        and _is_fetch_or_extract_action(action)
        and str(os.environ.get("CGHE_NO_PRUNE") or "").strip().lower() not in {"1", "true", "yes"}
    ):
        deferred = _deferred_snippet_ce_from_fetch(ep, controller, action, observation, label)
        if deferred is not None and getattr(deferred, "verified_counterexample", False):
            analysis = deferred

    pruned: list[str] = []
    evidence_id = ""
    before_ids = list(graph.active_plan_ids())
    no_prune = str(os.environ.get("CGHE_NO_PRUNE") or "").strip().lower() in {"1", "true", "yes"}
    allow_snippet_prune = True
    obs_code = str(getattr(getattr(analysis, "constraint", None), "observation_code", "") or "")
    if obs_code == "SNIPPETS_INSUFFICIENT" and not _snippet_prune_budget_ok(ep):
        allow_snippet_prune = False
        session.events.append({
            "event": "snippet_prune_budget_exhausted",
            "used": int(getattr(ep, "_cghe_snippet_prunes", 0) or 0),
            "limit": int(cghe_settings().get("max_verified_snippet_prunes_per_episode") or 2),
        })

    if (
        analysis.verified_counterexample
        and analysis.constraint is not None
        and not no_prune
        and allow_snippet_prune
    ):
        pruned = prune_by_counterexample(
            graph,
            analysis.constraint,
            evidence=analysis.evidence,
            executed_plan_id=str(action.get("cghe_plan_id") or observation.plan_id or ""),
        )
        if pruned and obs_code == "SNIPPETS_INSUFFICIENT":
            _note_snippet_prune(ep)
        evidence_id = str(analysis.constraint.evidence_id or analysis.constraint.constraint_id or "")
        try:
            controller.memory.record_counterexample(
                analysis.constraint,
                observation=observation,
                environment=str(getattr(observation, "environment", "") or ""),
                action=str(action.get("sub_goal") or ""),
                tool=str(action.get("tool_name") or ""),
                round_id=graph.dag_id,
                plan_id=str(action.get("cghe_plan_id") or ""),
            )
        except Exception:
            pass
        memory = controller.memory
        if hasattr(memory, "add_causal_edge"):
            memory.add_causal_edge(
                node_id,
                evidence_id or node_id,
                CausalRelation.OBSERVED_TO_CAUSE,
                evidence_id=evidence_id,
                direct=True,
            )
            memory.add_causal_edge(
                evidence_id or node_id,
                analysis.constraint.condition.predicate,
                CausalRelation.INVALIDATES,
                evidence_id=evidence_id,
                scope=analysis.constraint.scope,
                direct=True,
            )
            for plan_id in pruned:
                memory.add_causal_edge(
                    evidence_id or node_id,
                    plan_id,
                    CausalRelation.ELIMINATES,
                    evidence_id=evidence_id,
                    direct=True,
                )
        after_ids = list(graph.active_plan_ids())
        brief = constraint_brief(analysis.constraint)
        session.events.append({
            "event": "counterexample_validated",
            "predicate": analysis.constraint.condition.predicate,
            "scope": str(analysis.constraint.scope),
            "source": analysis.constraint.source,
            "validation_status": EvidenceStatus.VERIFIED_COUNTEREXAMPLE.value,
            "evidence_id": evidence_id,
            "constraint_brief": brief,
            "deferred_from_fetch": bool(deferred is not None),
        })
        session.events.append({
            "event": "plans_pruned",
            "evidence_id": evidence_id,
            "pruned_plan_ids": list(pruned),
            "remaining_active_plan_ids": after_ids,
            "plans_text": plans_text(graph),
            "constraint_brief": brief,
        })
        log_prune(before_ids, pruned, after_ids, analysis.constraint, verified=True)
        if (
            pruned
            and obs_code == "SNIPPETS_INSUFFICIENT"
            and hasattr(graph, "mark_failed")
        ):
            try:
                bing_id = node_id
                query = str(getattr(getattr(analysis.constraint, "scope", None), "argument", "") or "")
                if deferred is not None and query:
                    for pid in list(graph.active_plan_ids()) + (
                        [str(action.get("cghe_plan_id") or "")] if action.get("cghe_plan_id") else []
                    ):
                        if not pid:
                            continue
                        for nid in graph.plan_node_ids(pid):
                            node = graph.node(nid)
                            tool = str(getattr(node, "tool", "") or "").casefold()
                            arg = str(getattr(node, "argument", "") or "")
                            if ("bing" in tool or "search" in tool) and arg == query:
                                bing_id = nid
                                break
                graph.mark_failed(bing_id)
            except Exception:
                try:
                    graph.mark_failed(node_id)
                except Exception:
                    pass
    else:
        try:
            controller._remember_failure(analysis, observation, action=str(action.get("sub_goal") or ""), promote=False)
        except Exception:
            try:
                controller.memory.record_hypothesis(
                    observation=observation,
                    constraint=analysis.constraint,
                    environment=str(getattr(observation, "environment", "") or ""),
                    action=str(action.get("sub_goal") or ""),
                    tool=str(action.get("tool_name") or ""),
                    note="unverified failure is not a hard constraint",
                    round_id=graph.dag_id,
                )
            except Exception:
                pass
        session.events.append({
            "event": "counterexample_validated",
            "predicate": "",
            "validation_status": EvidenceStatus.UNVERIFIED_HYPOTHESIS.value,
            "evidence_id": str(getattr(analysis.evidence, "evidence_id", "") or ""),
            "hard_constraint": False,
            "constraint_brief": constraint_brief(analysis.constraint),
        })
        log_prune(before_ids, [], before_ids, analysis.constraint, verified=False)
    if attempt_id:
        try:
            graph.finish_attempt(
                attempt_id,
                execution_status="failure",
                verifier_status=label,
                evidence_id=evidence_id or None,
                pruning_applied=bool(pruned),
                pruned_plan_ids=pruned,
            )
        except Exception:
            pass
    session.events.append({
        "event": "verification_completed",
        "verifier_status": label or "FAILURE",
        "node_id": node_id,
        "dag_id": graph.dag_id,
        "pruning_applied": bool(pruned),
    })

    # Round-based regeneration (default): do NOT continue pruning/SELECT on the
    # same active DAG. Seal it, update HypothesisMemory, and request a NEW DAG.
    from hypothesis_round_memory import build_verifier_feedback

    hyp_mem = _hypothesis_memory(controller)
    feedback = build_verifier_feedback(
        satisfied=False,
        label=label or "不满足",
        analysis=analysis,
        action=action,
        evidence_text=str(
            getattr(observation, "tool_result", "") or action.get("result") or label or ""
        )[:500],
        dag_version=str(getattr(graph, "version", "") or graph.dag_id),
        round_id=int(getattr(session, "planning_round", 0) or 0),
        soft_continue=False,
    )
    if _round_regen_enabled() or (
        str(os.environ.get("CGHE_FORCE_REGEN_AFTER_HARD_PRUNE") or "").strip().lower() in {"1", "true", "yes"}
        and pruned
    ):
        reasons = ["ROUND_FEEDBACK_REGEN"]
        if pruned:
            reasons.append("VERIFIED_COUNTEREXAMPLE")
        if analysis.verified_counterexample:
            reasons.append(str(getattr(analysis.constraint, "observation_code", "") or "HARD_CE"))
        else:
            reasons.append("SUBGOAL_NOT_SATISFIED")
        return _end_round_for_regen(ep, controller, reasons=reasons, feedback=feedback)

    # Legacy same-DAG SELECT path (CGHE_ROUND_REGEN=0).
    hyp_mem.update_from_feedback(feedback, environment_fingerprint=_env_fingerprint(ep))
    decision, payload = session.next_decision()
    if decision == "REGENERATE":
        ep._cghe_graph_mode = "rebuild"
        ep._cghe_pending_reasons = list(payload or [])
        ep._cghe_directive = dict(_PEV_DIRECTIVE)
        ep._cghe_directive["reason"] = "regeneration required after persisted step1 result"
        return True
    if decision != "SELECT":
        ep._cghe_directive = dict(_PEV_DIRECTIVE)
        ep._cghe_directive["reason"] = str(decision)
        return True
    selected, coverage, eligible = payload
    session.log_selection(graph, selected, coverage, eligible)
    node = graph.node(selected)
    _store_graph_in_memory(ep, graph, selected_id=selected)
    ep._cghe_graph_mode = "switch"
    ep._cghe_directive = _directive_continue(
        node,
        f"step1 coverage {coverage} after unverified failure" if not pruned else f"step1 coverage {coverage} after verified prune",
        graph.round_id,
    )
    return True


def _lock_step1(ep: "Episode", graph: Any, node_id: str, coverage: int, reason: str) -> bool:
    node = graph.node(node_id)
    ep._cghe_graph_mode = "switch"
    _store_graph_in_memory(ep, graph, selected_id=node_id)
    ep._cghe_directive = _directive_continue(node, reason, graph.round_id)
    ep._cghe_directive["coverage"] = int(coverage)
    if _lock_recovery_step(ep, ep._cghe_directive):
        _metrics(ep)["deps_ready_selections"] = int(_metrics(ep).get("deps_ready_selections") or 0) + 1
        return True
    return False


def _continue_existing_graph(ep: "Episode") -> bool:
    """Lock the highest-coverage eligible Step 1. Do not walk downstream nodes."""
    from cghe_loop import session_for

    if _handoff_active(ep):
        return False
    if list(getattr(ep, "_cghe_pending_reasons", None) or []):
        # Round ended; must regenerate a NEW DAG rather than SELECT on the old one.
        return False
    controller = ep.cghe_controller()
    graph = getattr(controller, "graph", None) if controller is not None else None
    if graph is None:
        return False
    if getattr(graph, "sealed", False):
        ep._cghe_pending_reasons = list(getattr(ep, "_cghe_pending_reasons", None) or ["SEALED_DAG"])
        return False
    session = session_for(controller, cghe_settings())
    decision, payload = session.next_decision()
    if decision == "REGENERATE":
        ep._cghe_pending_reasons = list(payload or [])
        return False
    if decision != "SELECT":
        ep._cghe_directive = dict(_PEV_DIRECTIVE)
        ep._cghe_directive["reason"] = str(decision)
        return False
    node_id, coverage, eligible = payload
    session.log_selection(graph, node_id, coverage, eligible)
    return _lock_step1(ep, graph, node_id, coverage, f"step1 coverage {coverage}")


def _regenerate_layered_plan(ep: "Episode", controller: Any, reasons: list) -> bool:
    """One regeneration. A planner or schema failure does not spend a Step-1 attempt."""
    from cghe_loop import context_guidance, make_regeneration_context, session_for
    from sft_prompts import build_outlines_oneshot

    session = session_for(controller, cghe_settings())
    graph = getattr(controller, "graph", None)
    records = controller.planning_records()
    remaining = max(0, session.max_total - session.global_attempts)
    ce_limit = int(cghe_settings().get("max_inherited_ce") or 32)
    ctx = make_regeneration_context(
        query=str(ep.question or ""),
        mas_state=getattr(ep, "json_data", None),
        active_plans=list(graph.active_plans()) if graph is not None else [],
        verified_counterexamples=_truncate_records(list(records.verified_counterexamples), ce_limit),
        verified_successes=_truncate_records(list(records.verified_successes), ce_limit),
        unverified_failures=_truncate_records(list(records.diagnostic_hypotheses), ce_limit),
        observations=[],
        previous_attempts=list(getattr(graph, "attempt_history", []) or []) + list(session.archived_attempts),
        trigger_reasons=reasons,
        global_attempts_remaining=remaining,
    )
    engine = getattr(getattr(ep, "planner", None), "llm_engine", None)
    if not callable(engine):
        return False
    runtime = load_runtime()
    depth = min(3, max(1, int(runtime["max_steps"]) - int(ep.step_count) + 1))
    n = int(cghe_settings().get("n") or 8)
    hyp_mem = _hypothesis_memory(controller)
    try:
        guidance = ""
        if str(os.environ.get("CGHE_NO_MEMORY") or "").strip().lower() not in {"1", "true", "yes"}:
            guidance = context_guidance(ctx)
            hyp_guide = hyp_mem.planner_guidance(environment_fingerprint=_env_fingerprint(ep))
            guidance = (guidance + "\n" + hyp_guide).strip()
        plans = build_outlines_oneshot(
            engine, ep.question, n=n, depth=depth, guidance=guidance,
        )
    except Exception:
        plans = []
    from cghe_loop import drop_contradicted_plans

    if str(os.environ.get("CGHE_NO_MEMORY") or "").strip().lower() not in {"1", "true", "yes"}:
        plans = drop_contradicted_plans(plans, records.verified_counterexamples)
        plans = hyp_mem.filter_plans(plans, environment_fingerprint=_env_fingerprint(ep))
    installed = session.regenerate(plans, ctx)
    if installed is None:
        return False
    from cghe_loop import record_plan_dependencies

    record_plan_dependencies(controller.memory, installed)
    controller._round_exhausted = False
    ep._cghe_pending_reasons = []
    selection = installed.select_step1()
    if selection is None:
        return False
    node_id, coverage, eligible = selection
    session.log_selection(installed, node_id, coverage, eligible)
    ep._cghe_graph_mode = "rebuild"
    return _lock_step1(ep, installed, node_id, coverage, "regenerated step1 coverage " + str(coverage))


def _open_layered_plan(ep: "Episode") -> bool:
    """Build or continue a DAG and lock the Step-1 with the most ACTIVE paths."""
    from planning_graph import PlanningGraph
    from cghe_loop import record_plan_dependencies, session_for, validate_dag
    from sft_prompts import build_outlines_oneshot

    if _handoff_active(ep):
        return False
    if _continue_existing_graph(ep):
        return True
    controller = ep.cghe_controller()
    if controller is None:
        return False
    reasons = list(getattr(ep, "_cghe_pending_reasons", None) or [])
    graph = getattr(controller, "graph", None)
    if graph is not None and not reasons:
        reasons = list(graph.regeneration_reasons())
    if reasons:
        return _regenerate_layered_plan(ep, controller, reasons)
    engine = getattr(getattr(ep, "planner", None), "llm_engine", None)
    if not callable(engine):
        return False
    runtime = load_runtime()
    depth = min(3, max(1, int(runtime["max_steps"]) - int(ep.step_count) + 1))
    n = int(cghe_settings().get("n") or 8)
    hyp_mem = _hypothesis_memory(controller)
    try:
        guidance = ""
        if str(os.environ.get("CGHE_NO_MEMORY") or "").strip().lower() not in {"1", "true", "yes"}:
            guidance = hyp_mem.planner_guidance(environment_fingerprint=_env_fingerprint(ep))
        plans = build_outlines_oneshot(engine, ep.question, n=n, depth=depth, guidance=guidance or None)
    except Exception:
        return False
    from cghe_loop import drop_contradicted_plans

    plans = drop_contradicted_plans(plans, controller.planning_records().verified_counterexamples)
    plans = hyp_mem.filter_plans(plans, environment_fingerprint=_env_fingerprint(ep))
    if not plans:
        return False
    try:
        session = session_for(controller, cghe_settings())
        if int(getattr(session, "planning_round", 0) or 0) <= 0:
            session.planning_round = 1
        graph = PlanningGraph(list(plans), version=session.planning_round)
    except (TypeError, ValueError):
        return False
    if validate_dag(graph):
        return False
    from cghe_dag_log import log_dag_build, plans_text

    session.bind_graph(graph)
    hyp_mem.round_index = session.planning_round
    hyp_mem.register_dag_hypotheses(graph)
    record_plan_dependencies(controller.memory, graph)
    print(
        f"[CGHE][ROUND {session.planning_round}] Generated DAG version={graph.version} "
        f"plans={len(list(graph.active_plan_ids()))}",
        flush=True,
    )
    controller._round_exhausted = False
    session.events.append({
        "event": "dag_built",
        "dag_id": graph.dag_id,
        "plans_text": plans_text(graph),
        "coverage_map": dict(graph.compute_step1_coverage()),
        "active_plan_ids": list(graph.active_plan_ids()),
    })
    log_dag_build(graph)
    selection = graph.select_step1()
    if selection is None:
        return False
    node_id, coverage, eligible = selection
    session.log_selection(graph, node_id, coverage, eligible)
    ep._cghe_graph_mode = "rebuild"
    locked = _lock_step1(ep, graph, node_id, coverage, f"step1 coverage {coverage}")
    if locked:
        return True
    ep._cghe_directive = dict(_PEV_DIRECTIVE)
    ep._cghe_directive["reason"] = "selected node has no executable subgoal or tool"
    return False


def _cghe_trace(ep: "Episode") -> Dict[str, Any]:
    if not cghe_enabled():
        return {}
    from cghe_dag_log import human_lines, plans_text

    directive = getattr(ep, "_cghe_directive", None) or {}
    snapshot = getattr(ep, "_cghe_snapshot", None) or {}
    graph = getattr(getattr(ep, "_cghe", None), "graph", None)
    if graph is not None:
        snapshot = _graph_snapshot(graph, selected_id=str(directive.get("node_id") or ""))
    metrics = _metrics(ep)
    steps = max(1, int(metrics.get("steps") or 0))
    session = getattr(getattr(ep, "_cghe", None), "cghe_session", None)
    events = list(getattr(session, "events", None) or []) if session is not None else []
    return {
        "cghe_decision": str(directive.get("decision") or "continue_pev"),
        "cghe_reason": str(directive.get("reason") or "")[:300],
        "cghe_mode": str(getattr(ep, "_cghe_graph_mode", "") or ""),
        "cghe_node_id": str(directive.get("node_id") or ""),
        "cghe_task_id": str(directive.get("task_id") or ""),
        "cghe_outlines": snapshot.get("cghe_outlines") or [],
        "cghe_frontier": snapshot.get("cghe_frontier") or [],
        "cghe_select_policy": "max_step1_path_coverage",
        "cghe_events": events,
        "cghe_log_lines": human_lines(limit=120),
        "cghe_plans_text": plans_text(graph) if graph is not None else [],
        "cghe_round_id": snapshot.get("cghe_round_id") or "",
        "cghe_feedback": list(metrics.get("feedback") or [])[-5:],
        "cghe_metrics": {
            "dependency_ready_rate": round(
                float(metrics.get("deps_ready_selections") or 0) / steps, 4
            ),
            "premature_extract_rate": round(
                float(metrics.get("premature_extract") or 0) / steps, 4
            ),
            "evidence_sufficient_rate": round(
                float(metrics.get("evidence_sufficient_steps") or 0) / steps, 4
            ),
            "answer_correct_given_sufficient": int(
                metrics.get("answer_from_sufficient") or 0
            ),
            "steps": steps,
            "deps_ready_selections": int(metrics.get("deps_ready_selections") or 0),
            "premature_extract": int(metrics.get("premature_extract") or 0),
            "evidence_sufficient_steps": int(metrics.get("evidence_sufficient_steps") or 0),
        },
    }


def _close_cghe(ep: "Episode") -> None:
    controller = getattr(ep, "_cghe", None)
    if controller is not None:
        controller.memory.close()
        evidence = getattr(controller, "_v2_evidence", None)
        if evidence is not None:
            evidence.close()
        ep._cghe = None


class EpisodeContext:
    def __init__(self) -> None:
        self._comps: Optional[Components] = None
        self._build_error = ""
        self._lock = threading.Lock()
        self._episodes: "OrderedDict[str, Episode]" = OrderedDict()
        self._anon_key = ""

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
            # An empty id plus an empty question used to reuse the previous
            # episode. Keep one fresh key for this anonymous streak instead.
            if key == "q:":
                if not self._anon_key:
                    self._anon_key = f"q:anon:{uuid4().hex}"
                key = self._anon_key
            else:
                self._anon_key = ""
            comps.log_key.value = key
            ep = self._episodes.get(key)
            question = str(payload.get("question") or (ep.question if ep else "") or payload.get("input") or "")
            if ep is None or (create and ep.done):
                ep = Episode(comps, question, str(payload.get("source") or ""))
                self._episodes[key] = ep
                while len(self._episodes) > _MAX_EPISODES:
                    _old_key, old_ep = self._episodes.popitem(last=False)
                    _close_cghe(old_ep)
            self._episodes.move_to_end(key)
            return ep

    def pin_engines(self) -> None:
        """Copy the rollout endpoint onto engine instances before tool threads start."""
        try:
            from workflow.llm_scope import current_rollout_llm
        except Exception:
            return
        bound = current_rollout_llm()
        if not bound.get("endpoint"):
            return
        comps = self.components()
        engines = [
            comps.planner.llm_engine,
            comps.planner.llm_engine_fixed,
            comps.verifier.llm_engine,
            comps.verifier.llm_engine_fixed,
            getattr(comps.executor, "llm_generate_tool_command", None),
        ]
        for inst in (getattr(comps, "tool_instances", None) or {}).values():
            engines.append(getattr(inst, "llm_engine", None))
        for eng in engines:
            pin = getattr(eng, "pin_endpoint", None)
            if callable(pin):
                pin(bound)

    def _trace(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        key = getattr(self.components().log_key, "value", "")
        calls = drain_calls(key)
        return {**extra, "llm_calls": {"count": len(calls), "calls": calls}}

    # -- planner window -------------------------------------------------
    def plan_window(self, message: Any) -> Dict[str, Any]:
        payload = _payload(message)
        ep = self.episode(payload, create=True)
        self.pin_engines()
        state = payload.get("agentflow_state")
        if isinstance(state, dict) and state:
            ep.restore_state(state)
        if ep.done:
            return self._finish_plan(ep)
        ep.step_count += 1
        # A restored episode already has query_analysis; do not re-run step 1.
        from sft_prompts import (
            PYTHON_FALLBACK,
            PYTHON_OUTLINE_STEP,
            apply_outline,
            bind_current_subgoal,
            fallback_unparsed_tool,
            normalize_outline,
            question_needs_python,
        )

        if not ep.query_analysis:
            ep.query_analysis = ep.planner.analyze_query(ep.question, None)
            ep.json_data["query_analysis"] = ep.query_analysis
            apply_outline(ep.memory, getattr(ep.planner, "_initial_outline", None))
        locked = False
        if cghe_enabled() and _handoff_active(ep):
            handoff = ep._cghe_handoff
            if not handoff.get("seeded"):
                remaining = [
                    str(step.get("subgoal") or "")
                    for step in (handoff.get("remaining") or [])
                    if step.get("subgoal")
                ]
                if remaining:
                    apply_outline(ep.memory, remaining)
                handoff["seeded"] = True
        elif cghe_enabled():
            directive = dict(getattr(ep, "_cghe_directive", None) or _PEV_DIRECTIVE)
            if str(directive.get("decision") or "") == "continue_execution":
                ep._cghe_graph_mode = "switch"
                locked = _lock_recovery_step(ep, directive)
                if locked:
                    _metrics(ep)["deps_ready_selections"] = int(
                        _metrics(ep).get("deps_ready_selections") or 0
                    ) + 1
                if not locked:
                    ep._cghe_directive = dict(_PEV_DIRECTIVE)
                    ep._cghe_directive["reason"] = "selected node has no executable subgoal or tool"
            if not locked:
                locked = _open_layered_plan(ep)
            if not locked:
                locked = self._plan_cghe_step(ep)
        if not locked:
            runtime = load_runtime()
            next_step = ep.planner.generate_next_step(
                ep.question, None, ep.query_analysis, ep.memory,
                ep.step_count, int(runtime["max_steps"]), ep.json_data,
            )
            plan = getattr(ep.planner, "_last_plan", None) or {}
            ep._pending_plan = plan if isinstance(plan, dict) else {}
        else:
            plan = ep._pending_plan if isinstance(ep._pending_plan, dict) else {}
            ep._pending_plan = plan
        predicted_at = time.time()
        ep._pending_plan.update({
            "prediction_event_id": uuid4().hex,
            "planner_step": int(ep.step_count),
            "predicted_at": predicted_at,
            "predicted_before_execution_at": predicted_at,
            "recovery_chain_id": ep.recovery_chain_id,
            "parent_prediction_event_id": ep._resume_parent_prediction_event_id,
        })
        if ep._pending_plan.get("outline"):
            apply_outline(ep.memory, ep._pending_plan["outline"])
        if not locked:
            ep.context, ep.sub_goal, ep.tool_name = ep.planner.extract_context_subgoal_and_tool(next_step)
            needs_code = question_needs_python(ep.question)
            available = getattr(ep.planner, "available_tools", None)
            ep.tool_name, ep.context, ep.sub_goal, ep._tool_parse_note = fallback_unparsed_tool(
                ep.tool_name, ep.context, ep.sub_goal, available,
                PYTHON_FALLBACK if needs_code else None,
            )
            names = {str(item) for item in (available or [])}
            if needs_code and PYTHON_FALLBACK in names:
                ep.tool_name = PYTHON_FALLBACK
            if not normalize_outline(getattr(ep.memory, "outline", None)):
                step = PYTHON_OUTLINE_STEP if needs_code else "查找题目要的事实"
                apply_outline(ep.memory, [step])
            ep.sub_goal = bind_current_subgoal(ep.sub_goal, ep.memory)
        return {
            "kind": "plan_step",
            "payload": {
                "next": EXECUTOR_ID,
                "args": {"context": ep.context or "", "sub_goal": ep.sub_goal or "", "tool_name": ep.tool_name or ""},
                "sub_goal": ep.sub_goal or "",
                "done": False,
                "agentflow_state": ep.export_state(replay_plan=True),
                "trace": self._trace({"step": ep.step_count, "tool": ep.tool_name, **_cghe_trace(ep)}),
            },
        }

    def _plan_cghe_step(self, ep: Episode) -> bool:
        """Consume a recovery directive. False means the original single-step planner runs."""
        directive = dict(getattr(ep, "_cghe_directive", None) or _PEV_DIRECTIVE)
        decision = str(directive.get("decision") or "")
        if decision == "continue_execution":
            if _lock_recovery_step(ep, directive):
                return True
            ep._cghe_directive = dict(_PEV_DIRECTIVE)
            ep._cghe_directive["reason"] = "selected node has no executable subgoal or tool"
            return False
        if decision != "need_universe":
            return False
        controller = ep.cghe_controller()
        if controller is None:
            return False
        from planning_graph import UniverseFrozenError, plans_for_snippet_gap
        from sft_prompts import solicit_candidate_plans

        try:
            records = controller.planning_records()
            actions = list((ep.memory.get_actions() or {}).values())
            latest = actions[-1] if actions and isinstance(actions[-1], dict) else {}
            plans = list(plans_for_snippet_gap(latest, environment=CGHE_ENVIRONMENT))
            if not plans:
                plans = solicit_candidate_plans(
                    ep.planner,
                    ep.question,
                    ep.query_analysis,
                    ep.memory,
                    records,
                    int(cghe_settings()["n"]),
                )
        except Exception:
            return False
        if not plans:
            return False
        _bind_plan_environment(plans)
        try:
            outcome = controller.begin_round(plans)
        except (ValueError, UniverseFrozenError):
            return False
        _steer(ep, outcome)
        if str(ep._cghe_directive.get("decision") or "") != "continue_execution":
            return False
        if _lock_recovery_step(ep, ep._cghe_directive):
            return True
        ep._cghe_directive = dict(_PEV_DIRECTIVE)
        ep._cghe_directive["reason"] = "selected node has no executable subgoal or tool"
        return False

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
        executed_at = time.time()
        ep._cghe_last_auth = None
        self.pin_engines()
        from sft_prompts import last_python_error, repeated_python_program

        tool_name = ep.tool_name
        raw_result: Any = None
        if tool_name is None or tool_name not in ep.planner.available_tools:
            command = "No command was generated because the tool was not found."
            result: Any = "No result was generated because the tool was not found."
            raw_result = result
            ok = False
        else:
            ep.executor._python_repair = last_python_error(ep.memory) if "python" in tool_name.lower() else ""
            from sft_prompts import listed_fetch_command

            direct = listed_fetch_command(tool_name, ep.context, ep.memory)
            if not direct and "fetch" in str(tool_name or "").lower():
                graph = _graph_for(ep)
                node_id = str((getattr(ep, "_cghe_directive", None) or {}).get("node_id") or "")
                bound = ""
                if graph is not None and node_id:
                    try:
                        bound = str(graph.bound_url(node_id) or "").strip()
                    except Exception:
                        bound = ""
                if not bound:
                    bound = str(ep.context or "").strip()
                if bound.startswith("http://") or bound.startswith("https://"):
                    direct = f"execution = tool.execute(url={repr(bound)})"
            if direct:
                tool_command = direct
                command = direct
            else:
                tool_command = ep.executor.generate_tool_command(
                    ep.question, None, ep.context, ep.sub_goal, tool_name,
                    ep.toolbox_metadata[tool_name], ep.step_count, ep.json_data,
                )
                _analysis, _explanation, command = ep.executor.extract_explanation_and_command(tool_command)
                command = apply_sft_tags(tool_name, tool_command, command)
            if "python" in tool_name.lower() and repeated_python_program(command, ep.memory):
                result = "Error: this program already ran"
                raw_result = result
            else:
                result = _execute_authorized(ep, tool_name, command)
                raw_result = result
            limit = int(load_runtime().get("memory_result_chars") or 0)
            result = make_json_serializable_truncated(result, limit) if limit > 0 else make_json_serializable_truncated(result)
            ep.json_data[f"tool_result_{ep.step_count}"] = result
            ok = not tool_result_failed(result)
        note = str(getattr(ep, "_tool_parse_note", "") or "")
        if note:
            command = f"{note}\n{command}"
        ep.memory.add_action(ep.step_count, tool_name, ep.sub_goal, command, result)
        from sft_prompts import actual_label, normalize_prediction

        action = ep.memory.actions.get(f"Action Step {ep.step_count}")
        plan = ep._pending_plan if isinstance(ep._pending_plan, dict) else {}
        if isinstance(action, dict):
            prediction = normalize_prediction(plan.get("prediction"))
            reason = str(plan.get("prediction_reason") or "")
            tool_input = str(plan.get("tool_input") or ep.context or "")
            if prediction:
                action["prediction"] = prediction
            for key in (
                "p_satisfy", "p_unsatisfy", "s_satisfy", "s_unsatisfy", "prediction_mismatch",
                "satisfy_token_ids", "unsatisfy_token_ids",
                "prediction_token_start",
                "prediction_event_id", "planner_step", "predicted_at", "recovery_chain_id",
                "predicted_before_execution_at",
                "parent_prediction_event_id",
            ):
                if key in plan:
                    action[key] = plan[key]
            if reason:
                action["prediction_reason"] = reason
            if tool_input:
                action["tool_input"] = tool_input
            action["actual"] = actual_label(result)
            action["executed_at"] = executed_at
            _stamp_execution_observation(ep, action, tool_name, command, raw_result)
            if cghe_enabled():
                _ingest_graph_result(ep, tool_name=tool_name, raw_result=raw_result, ok=bool(ok))
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
        self.pin_engines()
        runtime = load_runtime()
        stop_verification = ep.verifier.verificate_context(
            ep.question, None, ep.query_analysis, ep.memory, ep.step_count, ep.json_data
        )
        from sft_prompts import (
            add_obtained_information,
            budget_unresolved_reason,
            parse_verifier_text,
            recent_has_fetch_body,
            recent_has_search_urls,
            last_python_print,
            recent_steps_look_empty,
        )

        verdict = parse_verifier_text(stop_verification)
        if recent_has_search_urls(ep.memory) and not recent_has_fetch_body(ep.memory):
            verdict["conclusion"] = "CONTINUE"
        feedback = {}
        action = ep.memory.actions.get(f"Action Step {ep.step_count}")
        if isinstance(action, dict) and isinstance(action.get("cghe_feedback"), dict):
            feedback = dict(action.get("cghe_feedback") or {})
        # Tool text alone is not enough to stop; need sufficient evidence.
        if str(feedback.get("evidence") or "") not in ("sufficient",) and verdict.get("conclusion") == "STOP":
            if not recent_has_fetch_body(ep.memory):
                verdict["conclusion"] = "CONTINUE"
            elif str(feedback.get("evidence") or "") in ("none", "weak", "insufficient", ""):
                verdict["conclusion"] = "CONTINUE"
        add_obtained_information(ep.memory, verdict.get("obtained"))
        if isinstance(action, dict):
            action["conclusion"] = verdict["conclusion"]
            satisfied = str(verdict.get("satisfied") or "").upper()
            if satisfied in ("YES", "NO"):
                action["actual_satisfied"] = satisfied == "YES"
                action["failure_confirmed"] = bool(
                    action.get("p_satisfy") is not None
                    and float(action["p_satisfy"]) < 0.5
                    and satisfied == "NO"
                    and not tool_result_failed(action.get("result"))
                )
            action["verifier_reason"] = str(verdict.get("reason") or "")
        _cghe_on_verdict(ep, verdict)
        _analysis, _token = ep.verifier.extract_conclusion(stop_verification)
        conclusion = verdict["conclusion"]
        analysis = str(verdict.get("reason") or _analysis or "")
        out_of_budget = ep.step_count >= int(runtime["max_steps"]) or (
            time.time() - ep.start >= float(runtime["max_time"])
        )
        if out_of_budget:
            analysis = budget_unresolved_reason(analysis, ep.memory)

        printed = last_python_print(ep.memory)
        obtained = list(getattr(ep.memory, "obtained_information", None) or [])
        obtained_text = "\n".join(str(item) for item in obtained if str(item or "").strip())
        no_evidence = (
            not printed
            and recent_steps_look_empty(ep.memory)
            and not recent_has_fetch_body(ep.memory)
            and not obtained_text.strip()
        )
        graph = _graph_for(ep)
        pages = list(getattr(ep, "_cghe_pages", None) or [])
        if graph is not None:
            for page in graph.collected_page_texts():
                if page and page not in pages:
                    pages.append(page)
        from sft_prompts import page_has_answer_clues

        evidence_ok = str(feedback.get("evidence") or "") == "sufficient"
        if graph is not None and graph.evidence_sufficient(str(ep.question or "")):
            evidence_ok = True
        if pages and any(page_has_answer_clues(str(ep.question or ""), page) for page in pages):
            evidence_ok = True
        has_any_observation = bool(
            printed
            or obtained_text.strip()
            or pages
            or recent_has_search_urls(ep.memory)
            or recent_has_fetch_body(ep.memory)
        )
        # Early stop: extract already produced a short boxed answer with evidence.
        boxed_early = ""
        if isinstance(action, dict):
            tool_name = str(action.get("tool_name") or action.get("tool") or "").lower()
            result_text = str(action.get("result") or action.get("raw_tool_result") or "")
            if ("generalist" in tool_name or "generator" in tool_name) and evidence_ok:
                m = re.search(r"\\boxed\{([^{}]{1,120})\}", result_text)
                if m:
                    cand = m.group(1).strip()
                    if cand and "unknown" not in cand.lower():
                        boxed_early = cand
        # Only force an empty stop when the episode truly has zero observations.
        # Weak search/fetch traces should still attempt final generation.
        force_empty = bool(
            out_of_budget
            and no_evidence
            and not has_any_observation
            and conclusion != "STOP"
        )
        stop = conclusion == "STOP" or out_of_budget or bool(boxed_early)
        if stop and conclusion == "STOP" and not evidence_ok and not out_of_budget and not boxed_early:
            # Keep searching/opening until evidence is sufficient.
            stop = False
            conclusion = "CONTINUE"
            analysis = (str(analysis or "") + " | blocked_stop_without_sufficient_evidence").strip(" |")
        answer = ""
        if stop and force_empty:
            ep.final_output = ""
            ep.direct_output = ""
            ep.json_data["final_output"] = ""
            ep.json_data["direct_output"] = ""
            ep.json_data["budget_empty_stop"] = True
            ep.done = True
            _complete_cghe_plans(ep)
            answer = ""
            analysis = (str(analysis or "") + " | budget_stop_without_evidence").strip(" |")
        elif stop:
            from sft_prompts import structured_answer_from_pages

            ep.planner.query_analysis = ep.query_analysis
            structured = ""
            if boxed_early:
                structured = f"<answer>\n\\boxed{{{boxed_early}}}\n</answer>"
                analysis = (str(analysis or "") + " | early_stop_boxed_extract").strip(" |")
            elif evidence_ok and pages and not cghe_enabled():
                engine = getattr(ep.planner, "llm_engine_fixed", None) or getattr(ep.planner, "llm_engine", None)
                if callable(engine):
                    structured = structured_answer_from_pages(engine, ep.question, pages)
            use_structured = bool(structured) and "unknown" not in structured.lower()
            if use_structured:
                ep.direct_output = structured
                ep.final_output = structured
                _metrics(ep)["answer_from_sufficient"] = 1
            elif evidence_ok:
                ep.final_output = str(ep.planner.generate_final_output(ep.question, None, ep.memory) or "")
                ep.direct_output = str(ep.planner.generate_direct_output(ep.question, None, ep.memory) or "")
                _metrics(ep)["answer_from_sufficient"] = 1
            else:
                ep.final_output = str(ep.planner.generate_final_output(ep.question, None, ep.memory) or "")
                ep.direct_output = str(ep.planner.generate_direct_output(ep.question, None, ep.memory) or "")
            ep.json_data["final_output"] = ep.final_output
            ep.json_data["direct_output"] = ep.direct_output
            ep.json_data["cghe_evidence_sufficient"] = bool(evidence_ok)
            ep.done = True
            _complete_cghe_plans(ep)
            answer = wrap_answer(ep.direct_output, source=ep.source)
            if not answer and printed:
                ep.direct_output = printed
                ep.json_data["direct_output"] = printed
                answer = wrap_answer(printed, source=ep.source)
        return {
            "kind": "verify",
            "payload": {
                "ok": stop and not force_empty,
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
                    "budget_empty_stop": force_empty,
                    "final_output": ep.final_output,
                    **_cghe_trace(ep),
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
        "Wikipedia_RAG_Search_Tool": "Bing_Search_Tool",
        "Wikipedia_Search_Tool": "Bing_Search_Tool",
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
