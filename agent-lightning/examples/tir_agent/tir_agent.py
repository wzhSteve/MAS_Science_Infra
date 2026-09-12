"""Tool-integrated reasoning agent: search + wikipedia + python, trained via Agent-lightning.

ReAct graph: agent ⇄ tools, with finalize nudge and react resume when no <answer>.
Extra state fields feed IGPO / GiGPO / ARPO annotations.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict, cast
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from algos.arpo_rollout import (
    deserialize_messages,
    extract_token_logprobs,
    obs_hash,
    serialize_messages,
)
from algos.rewards import compute_outcome_reward, extract_answer_text, has_answer_format, normalize_qa, parse_alias_field
from tools.langchain_tools import TOOL_MAP, TOOLS
from workflow.archive import ArchiveWriteError
from workflow.contracts import EventKind
from workflow.recorder import BoundaryRecorder, Redactor, message_records

try:
    from workflow.env_load import load_repo_dotenv

    load_repo_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)



SYSTEM_PROMPT = """You are a careful tool-using assistant following a ReAct loop:
Thought (brief) → Action (tool call if needed) → Observation (tool result) → repeat → Final Answer.

You may call these tools:
- `web_search`: open-web snippets for facts that are not in the question.
- `wikipedia_search`: short Wikipedia extracts for well-known entities.
- `execute_python`: run a short Python snippet. No imports. Prefer assigning the final value to `result`.

Rules:
- Arithmetic / GSM8K-style math: you MUST call `execute_python` to compute the number. Do not do long mental math.
- Factoid or multi-hop questions: you MUST call `wikipedia_search` or `web_search` BEFORE giving the final answer. Do not guess from memory.
- You may use more than one tool when needed (search then python, or two searches).
- After each observation, decide the next Thought/Action; do not stop mid-loop without an answer.
- When you know the final answer, stop calling tools and reply with EXACTLY:
<answer> YOUR_ANSWER </answer>
Do not wrap the tags in extra commentary.
""".strip()

FINALIZE_PROMPT = (
    "Stop using tools. Based on the conversation above, output ONLY the final answer "
    "in this exact format (no other text):\n<answer> YOUR_ANSWER </answer>"
)

RESUME_REACT_PROMPT = (
    "Your last reply did not contain a valid <answer> YOUR_ANSWER </answer>. "
    "Resume the ReAct loop: think briefly, call tools if you still need information, "
    "then output the final answer in exactly:\n<answer> YOUR_ANSWER </answer>"
)

SEARCH_TOOL_NAMES = {"web_search", "wikipedia_search"}
PYTHON_TOOL_NAMES = {"execute_python"}
# Tool-call schemas are sent on every OpenAI request but are not in `messages`.
_TOOL_SCHEMA_TOKENS = 512
_TOOL_OBS_CHARS = 700


def default_system_prompt(enabled_tools: Sequence[str]) -> str:
    """Describe only tools that the current Agent can actually call."""
    if set(enabled_tools) == set(TOOL_MAP):
        return SYSTEM_PROMPT
    return (
        "You are a careful assistant. "
        + ("Available tools: " + ", ".join(enabled_tools) + ". Use them when helpful. "
           if enabled_tools else "No tools are available. ")
        + "Use the task and provided context; do not invent observations. "
        "When finished, reply with <answer> YOUR_ANSWER </answer>."
    )


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (CJK-heavy tool dumps)."""
    if not text:
        return 0
    return max(1, (len(text) + 2) // 3)


def clip_text(text: str, max_chars: int) -> str:
    s = str(text or "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    keep = max(32, max_chars - 24)
    return s[:keep] + "\n...[truncated]"


def estimate_messages_tokens(messages: Sequence[Any]) -> int:
    n = _TOOL_SCHEMA_TOKENS
    for msg in messages:
        n += 8
        n += estimate_tokens(str(getattr(msg, "content", "") or ""))
        tcs = getattr(msg, "tool_calls", None) or []
        if tcs:
            try:
                n += estimate_tokens(json.dumps(tcs, ensure_ascii=False, default=str))
            except Exception:
                n += 80 * len(list(tcs))
    return n


def _clip_tool_messages(messages: List[AnyMessage], max_chars: int) -> List[AnyMessage]:
    out: List[AnyMessage] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            out.append(
                ToolMessage(
                    content=clip_text(str(getattr(msg, "content", "") or ""), max_chars),
                    tool_call_id=getattr(msg, "tool_call_id", "") or "tool",
                    name=getattr(msg, "name", None),
                )
            )
        else:
            out.append(msg)
    return out


def fit_messages_for_context(
    messages: List[AnyMessage],
    *,
    max_prompt_tokens: int,
    tool_max_chars: int = _TOOL_OBS_CHARS,
) -> List[AnyMessage]:
    """Keep system+first user, drop oldest ReAct turns, clip tool observations."""
    if max_prompt_tokens <= 0:
        return list(messages)
    chars = tool_max_chars
    fitted = _clip_tool_messages(list(messages), chars)
    while estimate_messages_tokens(fitted) > max_prompt_tokens and len(fitted) > 3:
        # Drop oldest turn after the opening system+user, skip a dangling tool msg.
        if len(fitted) > 3:
            fitted.pop(2)
        while len(fitted) > 3 and isinstance(fitted[2], ToolMessage):
            fitted.pop(2)
        chars = max(160, chars // 2)
        fitted = _clip_tool_messages(fitted, chars)
    return fitted


def remaining_completion_tokens(messages: List[AnyMessage], *, max_model_len: int, max_tokens: int) -> int:
    prompt = estimate_messages_tokens(messages)
    room = int(max_model_len) - prompt - 8
    return max(64, min(int(max_tokens), room))


def _is_context_error(exc: BaseException) -> bool:
    blob = str(exc).lower()
    return any(
        s in blob
        for s in (
            "contextwindow",
            "context length",
            "maximum context",
            "context_window",
            "please reduce the length",
        )
    )


class TirProblem(TypedDict, total=False):
    id: str
    question: str
    answer: str
    answers: Any
    source: str
    split: str
    resume_messages: Any
    resume_parent_id: str


class TurnRecord(TypedDict, total=False):
    turn: int
    tool_name: str
    obs_hash: str
    prefix_message_count: int


class AgentState(TypedDict, total=False):
    messages: List[AnyMessage]
    question: str
    num_turns: int
    asked_finalize: bool
    turn_records: List[TurnRecord]
    n_search: int
    n_python: int
    h_root: float
    h_tool: float
    consecutive_high: int
    last_entropy: float
    branch_messages: List[Dict[str, Any]]
    termination_reason: str
    error: str
    uncertainty_evidence: str
    branch_event_id: str


def last_assistant_text(messages: List[AnyMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return str(getattr(msg, "content", "") or "")
    return None


def extract_answer_from_messages(messages: List[AnyMessage]) -> Optional[str]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            ans = extract_answer_text(str(getattr(msg, "content", "") or ""))
            if ans is not None:
                return ans
    return None


def _gold_hit(text: str, golds: List[str]) -> float:
    blob = normalize_qa(text)
    if not blob:
        return 0.0
    for gold in golds:
        ng = normalize_qa(str(gold))
        if ng and ng in blob:
            return 1.0
    return 0.0


def ig_deltas_from_messages(messages: List[AnyMessage], golds: List[str]) -> List[float]:
    """One IG proxy per assistant turn: 1 if latest tool obs newly contains GT."""
    ig: List[float] = []
    hit = 0.0
    pending = 0.0
    for msg in messages:
        if isinstance(msg, ToolMessage):
            new_hit = _gold_hit(str(getattr(msg, "content", "") or ""), golds)
            pending = new_hit - hit
            hit = new_hit
        if isinstance(msg, AIMessage):
            ig.append(float(pending))
            pending = 0.0
    return ig


def _parse_resume_messages(raw: Any) -> List[AnyMessage]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        try:
            raw = json.loads(raw)
        except Exception:
            return []
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, list):
        return []
    return deserialize_messages([x for x in raw if isinstance(x, dict)])


class TirAgent:
    """LangGraph ReAct TIR agent: search / wikipedia / python, then <answer> close-out."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        temperature: float = 0.7,
        max_turns: int = 8,
        max_tokens: int = 1024,
        entropy_tokens: int = 8,
        entropy_threshold: float = 0.15,
        request_logprobs: bool = False,
        enabled_tools: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
        max_model_len: Optional[int] = None,
        api_key: Optional[str] = None,
        recorder: Optional[BoundaryRecorder] = None,
        model_identity: Optional[Dict[str, Any]] = None,
        sampling_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_turns = max_turns
        self.model_name = model_name
        self.entropy_tokens = entropy_tokens
        self.entropy_threshold = entropy_threshold
        self.max_tokens = int(max_tokens)
        self.recorder = recorder
        self.model_identity = model_identity or {"model": model_name}
        self.redact = Redactor([api_key] if api_key else [])
        self.last_state: AgentState = {}
        self.last_model_call_id: Optional[str] = None
        env_len = os.environ.get("TIR_MAX_MODEL_LEN", "").strip()
        self.max_model_len = int(max_model_len or (env_len or 0) or 0) or None
        names = list(enabled_tools) if enabled_tools is not None else [t.name for t in TOOLS]
        self.system_prompt = (system_prompt or "").strip() or default_system_prompt(names)
        self.tool_map = {n: TOOL_MAP[n] for n in names if n in TOOL_MAP}
        unknown = set(names) - set(TOOL_MAP)
        if unknown:
            raise ValueError(f"Unknown tools: {sorted(unknown)}")
        bound_tools = [self.tool_map[n] for n in names if n in self.tool_map]
        self.tool_definitions = [convert_to_openai_tool(t) for t in bound_tools]
        extra: Dict[str, Any] = {}
        if request_logprobs:
            extra = {"logprobs": True, "top_logprobs": 1}
        llm_kwargs: Dict[str, Any] = dict(
            model_provider="openai",
            openai_api_base=endpoint,
            openai_api_key=(api_key or "dummy") if api_key is not None else os.environ.get("OPENAI_API_KEY", "dummy"),
            temperature=temperature,
            max_retries=0,
            max_tokens=max_tokens,
        )
        if extra:
            llm_kwargs["model_kwargs"] = extra
        sampling = dict(sampling_parameters or {})
        forbidden = {
            "api_key", "openai_api_key", "base_url", "openai_api_base", "model", "model_provider",
            "tools", "functions", "callbacks", "max_retries",
        }
        if forbidden.intersection(sampling):
            raise ValueError("Sampling parameters cannot override model binding")
        llm_kwargs.update(sampling)
        # These are the resolved per-run limits, also used by context fitting.
        llm_kwargs["temperature"] = temperature
        llm_kwargs["max_tokens"] = max_tokens
        self.sampling = {**sampling, "temperature": temperature, "max_tokens": max_tokens, **extra}
        self.llm = init_chat_model(model_name, **llm_kwargs).bind_tools(bound_tools)
        fin_kwargs = dict(llm_kwargs)
        fin_kwargs["temperature"] = min(temperature, 0.3)
        fin_kwargs["max_tokens"] = min(max_tokens, 256)
        fin_kwargs.pop("model_kwargs", None)
        fin_kwargs.pop("logprobs", None)
        fin_kwargs.pop("top_logprobs", None)
        fin_kwargs.pop("tool_choice", None)
        fin_kwargs.pop("parallel_tool_calls", None)
        self.llm_finalize = init_chat_model(model_name, **fin_kwargs)
        self.finalize_sampling = {
            k: v for k, v in self.sampling.items()
            if k not in {"logprobs", "top_logprobs", "tool_choice", "parallel_tool_calls"}
        }
        self.finalize_sampling.update(temperature=fin_kwargs["temperature"], max_tokens=fin_kwargs["max_tokens"])

    @classmethod
    def from_spec(
        cls,
        spec: Any,
        *,
        endpoint: str,
        model_name: str,
        enabled_tools: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> "TirAgent":
        """Build from MASSpec-like object. Does not import workflow."""
        tools = list(enabled_tools) if enabled_tools is not None else list(getattr(spec, "tools", None) or [])
        prompt = system_prompt
        if not prompt:
            hub = getattr(spec, "hub", None)
            prompt = getattr(hub, "system_prompt", None) if hub is not None else None
        return cls(
            endpoint=endpoint,
            model_name=model_name,
            enabled_tools=tools,
            system_prompt=prompt or None,
            **kwargs,
        )

    def call_model(self, state: AgentState) -> AgentState:
        self.last_state = state
        messages = list(state["messages"])
        asked_finalize = bool(state.get("asked_finalize", False))
        num_turns = int(state.get("num_turns", 0))
        llm = self.llm_finalize if asked_finalize else self.llm
        max_out = self.max_tokens
        if asked_finalize:
            max_out = min(max_out, 256)
        if self.max_model_len:
            budget = max(256, int(self.max_model_len) - max_out - 8)
            messages = fit_messages_for_context(messages, max_prompt_tokens=budget)
            max_out = remaining_completion_tokens(
                messages, max_model_len=int(self.max_model_len), max_tokens=max_out
            )
        def invoke(effective: List[AnyMessage], limit: int, retry_of: Optional[str] = None) -> Any:
            call_id = uuid4().hex
            self.last_model_call_id = call_id
            sampling = dict(self.finalize_sampling if asked_finalize else self.sampling)
            sampling["max_tokens"] = limit
            bound = llm.bind(max_tokens=limit)
            provider = getattr(bound, "bound", None)
            build_payload = getattr(provider, "_get_request_payload", None)
            request = build_payload(effective, **bound.kwargs) if callable(build_payload) else None
            if request is not None:
                sampling = {
                    key: value for key, value in request.items()
                    if key not in {"messages", "tools", "model", "extra_headers"}
                }
            if self.recorder:
                self.recorder.emit(EventKind.MODEL_CALL, {
                    "messages": request.get("messages", message_records(effective)) if request else message_records(effective),
                    "tool_definitions": request.get("tools", []) if request else ([] if asked_finalize else self.tool_definitions),
                    "sampling_parameters": sampling, "model": self.model_identity,
                    "request_format": "provider_payload" if request else "langchain_messages",
                    "retry_of": retry_of, "finalize": asked_finalize,
                    "context_processing": {
                        "method": "estimated_tokens_clip_tools_drop_oldest" if self.max_model_len else "none",
                        "changed": message_records(effective) != message_records(state["messages"]),
                        "original_message_count": len(state["messages"]),
                        "effective_message_count": len(effective),
                        "max_model_len": self.max_model_len,
                        "token_count_is_estimate": True,
                    },
                }, model_call_id=call_id)
            try:
                response = bound.invoke(effective)
            except ArchiveWriteError:
                raise
            except Exception as exc:
                if self.recorder:
                    self.recorder.emit(EventKind.MODEL_RESULT, {
                        "status": "failed", "error": self.redact(str(exc)),
                        "error_code": type(exc).__name__,
                    }, model_call_id=call_id)
                raise
            if self.recorder:
                metadata = getattr(response, "response_metadata", {}) or {}
                self.recorder.emit(EventKind.MODEL_RESULT, {
                    "status": "succeeded", "message": message_records([response])[0],
                    "finish_reason": metadata.get("finish_reason") or metadata.get("stop_reason"),
                    "model": metadata.get("model_name") or metadata.get("model"),
                    "requested_model": self.model_name,
                    "usage": getattr(response, "usage_metadata", None) or metadata.get("token_usage"),
                    "logprobs": metadata.get("logprobs") or (getattr(response, "additional_kwargs", {}) or {}).get("logprobs"),
                }, model_call_id=call_id)
            return response

        try:
            response = invoke(messages, max_out)
        except ArchiveWriteError:
            raise
        except Exception as e:
            if self.max_model_len and _is_context_error(e):
                tight = max(256, int(self.max_model_len) // 2)
                messages = fit_messages_for_context(messages, max_prompt_tokens=tight, tool_max_chars=200)
                response = invoke(messages, 64, self.last_model_call_id)
            else:
                raise

        # Legacy h_* fields contain chosen-token surprisal only when observed.
        # Missing logprobs use a neutral compatibility value, never fabricated entropy.
        logprobs = extract_token_logprobs(response, max_tokens=self.entropy_tokens)
        h = -sum(logprobs) / len(logprobs) if logprobs else 0.0
        h_root = float(state.get("h_root") or 0.0)
        h_tool = float(state.get("h_tool") or 0.0)
        last_entropy = float(state.get("last_entropy") or 0.0)
        consecutive_high = int(state.get("consecutive_high") or 0)
        if num_turns == 0:
            h_root = h
        elif state.get("turn_records") and logprobs:
            h_tool = h if h_tool == 0.0 else 0.5 * h_tool + 0.5 * h
            if (h - last_entropy) > self.entropy_threshold:
                consecutive_high += 1
            else:
                consecutive_high = 0
        result = {
            **state,
            "messages": list(state["messages"]) + [response],
            "num_turns": num_turns + 1,
            "asked_finalize": asked_finalize,
            "h_root": h_root,
            "h_tool": h_tool,
            "last_entropy": h,
            "consecutive_high": consecutive_high,
            "uncertainty_evidence": "chosen_token_surprisal_proxy" if logprobs else "unavailable_neutral_zero",
        }
        self.last_state = result
        return result

    def request_finalize(self, state: AgentState) -> AgentState:
        messages = list(state["messages"]) + [HumanMessage(content=FINALIZE_PROMPT)]
        return {**state, "messages": messages, "asked_finalize": True}

    def resume_react(self, state: AgentState) -> AgentState:
        """Finalize 未产出答案时：解除收口标记，提示模型继续 Thought→Action→Answer。"""
        messages = list(state["messages"]) + [HumanMessage(content=RESUME_REACT_PROMPT)]
        return {**state, "messages": messages, "asked_finalize": False}

    def call_tools(self, state: AgentState) -> AgentState:
        self.last_state = state
        last = state["messages"][-1]
        tool_messages: List[ToolMessage] = []
        records = list(state.get("turn_records") or [])
        n_search = int(state.get("n_search") or 0)
        n_python = int(state.get("n_python") or 0)
        tool_calls = getattr(last, "tool_calls", None) or []
        result_event = None
        for call in tool_calls:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            call_id = call_id or uuid4().hex
            if self.recorder:
                self.recorder.emit(EventKind.TOOL_CALL, {"name": name, "args": args},
                                   model_call_id=self.last_model_call_id, tool_call_id=call_id)
            error = None
            tool_fn = self.tool_map.get(name or "")
            if tool_fn is None:
                error = f"Unknown tool {name}"
                content = f"Error: {error}"
            else:
                try:
                    content = tool_fn.invoke(args)
                except ArchiveWriteError:
                    raise
                except Exception as e:
                    error = self.redact(str(e))
                    content = f"Error invoking tool: {error}"
            content_s = str(content)
            if error is None and content_s.lstrip().lower().startswith(
                ("error", "[error", "traceback", "search_unavailable:")
            ):
                error = content_s
            if self.recorder:
                result_event = self.recorder.emit(EventKind.TOOL_RESULT, {
                    "name": name, "content": content_s,
                    "status": "failed" if error else "succeeded",
                    "error": error, "recoverable": bool(error),
                }, model_call_id=self.last_model_call_id, tool_call_id=call_id)
                if error:
                    self.recorder.emit(EventKind.ERROR, {
                        "stage": "tool", "error": error, "recoverable": True,
                    }, model_call_id=self.last_model_call_id, tool_call_id=call_id)
            tool_messages.append(ToolMessage(
                content=content_s, tool_call_id=call_id, name=name,
                status="error" if error else "success",
            ))
            tname = str(name or "unknown")
            if tname in SEARCH_TOOL_NAMES:
                n_search += 1
            if tname in PYTHON_TOOL_NAMES:
                n_python += 1
            records.append(
                {
                    "turn": int(state.get("num_turns") or 0),
                    "tool_name": tname,
                    "obs_hash": obs_hash(tname, content_s),
                    "prefix_message_count": len(state["messages"]) + len(tool_messages),
                }
            )
            self.last_state = {
                **state, "messages": state["messages"] + list(tool_messages),
                "turn_records": list(records), "n_search": n_search, "n_python": n_python,
            }
        new_messages = state["messages"] + tool_messages
        branch = list(state.get("branch_messages") or [])
        if tool_messages and not branch:
            branch = serialize_messages(new_messages)
        result = {
            **state,
            "messages": new_messages,
            "turn_records": records,
            "n_search": n_search,
            "n_python": n_python,
            "branch_messages": branch,
            "branch_event_id": state.get("branch_event_id") or (result_event.event_id if result_event else ""),
        }
        self.last_state = result
        return result

    def should_continue(self, state: AgentState) -> Literal["tools", "finalize", "react", "end"]:
        """ReAct 路由：tools ⇄ agent；无答案可 finalize；finalize 失败且仍有轮次则 react 续跑。"""
        last = state["messages"][-1]
        num_turns = int(state.get("num_turns") or 0)
        asked_finalize = bool(state.get("asked_finalize", False))
        turns_left = num_turns < self.max_turns
        tool_calls = getattr(last, "tool_calls", None) or []

        metadata = getattr(last, "response_metadata", {}) or {}
        finish_reason = metadata.get("finish_reason") or metadata.get("stop_reason")
        if finish_reason in {"length", "max_tokens", "content_filter"}:
            state["termination_reason"] = "token_limit" if finish_reason != "content_filter" else "content_filter"
            self.last_state = state
            return "end"
        # 收口轮次不执行工具；正常 ReAct 轮次优先走 tools。
        if tool_calls and turns_left and not asked_finalize:
            return "tools"
        if not tool_calls and extract_answer_text(str(getattr(last, "content", "") or "")) is not None:
            state["termination_reason"] = "answer"
            self.last_state = state
            return "end"
        if not turns_left:
            state["termination_reason"] = "max_turns"
            self.last_state = state
            return "end"
        # 模型停手但无答案：先强制收口一次；收口仍失败则回到带工具的 ReAct。
        if not asked_finalize:
            return "finalize"
        return "react"

    def graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("agent", self.call_model)
        builder.add_node("tools", self.call_tools)
        builder.add_node("finalize", self.request_finalize)
        builder.add_node("react", self.resume_react)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent",
            self.should_continue,
            {"tools": "tools", "finalize": "finalize", "react": "react", "end": END},
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", "agent")
        builder.add_edge("react", "agent")
        return builder.compile()


def __getattr__(name: str):
    if name == "LitTirAgent":
        from lit_tir_agent import LitTirAgent as _Cls

        return _Cls
    if name == "debug_tir_agent":
        from lit_tir_agent import debug_tir_agent as _fn

        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    from lit_tir_agent import debug_tir_agent

    debug_tir_agent()
