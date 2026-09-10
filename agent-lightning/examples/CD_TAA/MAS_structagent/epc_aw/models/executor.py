from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import (
    FinalAnswerDraft,
    PlannerDecision,
    ToolAction,
    parse_structured_response,
)
from MAS.epc_aw.models.memory import ExecutorMemory
from MAS.epc_aw.models.state import AgentState, ExecutionTrace, Milestone


_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts" / "executor"
_URL_RE = re.compile(r"https?://[^\s\])>'\"]+")


def _load_prompt(name: str) -> str:
    return (_PROMPT_ROOT / name).read_text(encoding="utf-8")


def _extract_urls(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"url", "link", "source"} and isinstance(item, str):
                found.extend(_URL_RE.findall(item))
            found.extend(_extract_urls(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_extract_urls(item))
    elif isinstance(value, str):
        found.extend(_URL_RE.findall(value))
    return list(dict.fromkeys(found))


class Executor:
    """Bounded tool actor that emits auditable ExecutionTrace objects."""

    def __init__(
        self,
        llm_engine_name: str,
        root_cache_dir: str = "solver_cache",
        num_threads: int = 1,
        max_time: int = 120,
        max_output_length: int = 100000,
        verbose: bool = False,
        base_url: Optional[str] = None,
        check_model: bool = True,
        temperature: float = 0.0,
        toolbox_metadata: Optional[Dict[str, Any]] = None,
        tool_instances: Optional[Dict[str, Any]] = None,
        llm_engine: Any = None,
    ):
        del num_threads, base_url, check_model
        self.llm_engine_name = llm_engine_name
        self.root_cache_dir = root_cache_dir
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.verbose = verbose
        self.temperature = temperature
        self.toolbox_metadata = toolbox_metadata or {}
        self.tool_instances = tool_instances or {}
        self.memory = ExecutorMemory()
        self.query_cache_dir = root_cache_dir
        self.llm_engine = llm_engine or create_llm_engine(
            model_string=llm_engine_name, is_multimodal=False, temperature=0.0
        )

    def set_query_cache_dir(self, query_cache_dir: Optional[str]) -> None:
        self.query_cache_dir = query_cache_dir or self.root_cache_dir
        os.makedirs(self.query_cache_dir, exist_ok=True)
        for tool in self.tool_instances.values():
            if hasattr(tool, "set_custom_output_dir"):
                tool.set_custom_output_dir(self.query_cache_dir)

    def generate_action(
        self,
        state: AgentState,
        milestone: Milestone,
        plan: PlannerDecision,
    ) -> ToolAction:
        metadata = self.toolbox_metadata.get(plan.tool_name) or {}
        prompt = _load_prompt("tool_action.txt").format(
            Question=state.question,
            Milestone=json.dumps(
                {"id": milestone.id, "description": milestone.description},
                ensure_ascii=False,
            ),
            Sub_Goal=plan.subgoal,
            Tool_Name=plan.tool_name,
            Tool_Metadata=json.dumps(metadata, ensure_ascii=False, default=str),
            Facts=json.dumps(
                state.compact_view()["facts"], ensure_ascii=False, default=str
            ),
        )
        response = self.llm_engine(
            prompt, response_format=ToolAction, temperature=0, n=1
        )
        action = parse_structured_response(response, ToolAction)
        if action.tool_name != plan.tool_name:
            raise ValueError(
                f"Actor changed selected tool from {plan.tool_name!r} "
                f"to {action.tool_name!r}"
            )
        self._validate_arguments(action.tool_name, action.arguments)
        return action

    def _validate_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        if tool_name not in self.tool_instances:
            raise ValueError(f"Tool instance not available: {tool_name}")
        expected = (self.toolbox_metadata.get(tool_name) or {}).get("input_types") or {}
        unknown = set(arguments) - set(expected)
        if unknown:
            raise ValueError(f"Unexpected arguments for {tool_name}: {sorted(unknown)}")
        missing = [
            name
            for name, description in expected.items()
            if name not in arguments
            and "optional" not in str(description).lower()
            and "default" not in str(description).lower()
        ]
        if missing:
            raise ValueError(f"Missing arguments for {tool_name}: {missing}")

    def execute_action(
        self,
        action: ToolAction,
        *,
        step: int,
        milestone_id: str,
        subgoal: str,
    ) -> ExecutionTrace:
        started = time.perf_counter()
        result: Any = None
        error: Optional[str] = None
        try:
            result = self.tool_instances[action.tool_name].execute(**action.arguments)
        except BaseException as exc:  # tools historically raise SystemExit too
            error = f"{type(exc).__name__}: {exc}"
        duration = time.perf_counter() - started
        if isinstance(result, str) and len(result) > self.max_output_length:
            result = result[: self.max_output_length] + " ...[truncated]"
        return ExecutionTrace(
            step=step,
            milestone_id=milestone_id,
            subgoal=subgoal,
            tool_name=action.tool_name,
            arguments=dict(action.arguments),
            result=result,
            error=error,
            duration_seconds=duration,
            source_urls=_extract_urls(result),
        )

    def generate_direct_output(self, question: str, state: AgentState) -> str:
        prompt = _load_prompt("final_answer.txt").format(
            Question=question,
            State=json.dumps(state.to_dict(), ensure_ascii=False, default=str),
        )
        response = self.llm_engine(
            prompt, response_format=FinalAnswerDraft, temperature=0, n=1
        )
        try:
            return parse_structured_response(response, FinalAnswerDraft).answer.strip()
        except (TypeError, ValueError, json.JSONDecodeError):
            # User-level formatting constraints (for example <answer> tags)
            # may legitimately outrank our JSON transport contract.
            text = str(response or "").strip()
            if not text:
                raise ValueError("Final answer model returned an empty response")
            return text

    # Legacy arbitrary-code execution is deliberately disabled.
    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        raise RuntimeError(
            "String command execution is disabled. Use generate_action() and "
            "execute_action() with structured arguments."
        )
