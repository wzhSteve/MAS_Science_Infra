"""Assistant chat engine: streaming loop, tool rounds, prompt/memory injection.

Moved verbatim from science_infra/control/assistant.py, then extended with
skill-index prompts, pinned memory injection and history compaction. The SSE
event protocol (token/status/tool_start/tool_end/done/error) is unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from ai_assistant.context import (  # noqa: F401
    ApplyBody,
    ChatBody,
    ToolSession,
    build_workspace_snapshot,
)
from ai_assistant.llm import AssistantLlm, resolve_assistant_llm  # noqa: F401
from ai_assistant.memory import store as memory_store
from ai_assistant.memory.compaction import HISTORY_LIMIT, COMPACT_TRIGGER, compress_history, summary_system_message
from ai_assistant.memory.store import MemoryStore, pinned_memory_block
from ai_assistant.prompts import build_system_prompt
from ai_assistant.tools import OPENAI_TOOLS, assistant_tools

SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{8,}|api[_-]?key\s*[:=]\s*\S+", re.IGNORECASE)
WELCOME_MARKERS = ("我只能改 user_space/", "只写 user_space/")
MAX_TOOL_ROUNDS = 16
MAX_OUTPUT_TOKENS = 4096


def _history_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    kept: List[Dict[str, str]] = []
    for item in history or []:
        role = item.get("role") or "user"
        content = (item.get("content") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        if any(marker in content for marker in WELCOME_MARKERS):
            continue
        kept.append({"role": role, "content": content})
    return kept[-HISTORY_LIMIT:]


def _local_answer(message: str, project_id: Optional[str]) -> str:
    """Offline fallback only. Never used when an assistant LLM is configured."""
    text = message.strip()
    tools = {t["name"]: t["fn"] for t in assistant_tools(project_id)}
    if any(k in text for k in ("适配", "包装", "io_module", "Mode 1", "mode 1")) and project_id:
        return tools["scaffold_io_module"]()
    if any(k in text.lower() for k in ("epc-aw", "epc_aw", "epcaw")) and project_id:
        return tools["scaffold_epc_aw_pev"]()
    if any(k in text for k in ("解析", "重构", "native", "Mode 2", "mode 2", "rollout")) and project_id:
        return tools["scaffold_native_mas"]()
    if "用户区" in text or "管理区" in text:
        return (
            "管理区是 mas/、science_infra/、rl/、webui/，只读。"
            "用户区是仓库根目录 user_space/，辅助 AI 只能改 projects/<id>/{adapted,contracts,artifacts}/。"
            "实验 YAML 必须走画布保存或 apply_to_canvas。"
        )
    if "BranchSite" in text or "采样" in text or "rollout tree" in text.lower():
        return (
            "采样站点写在 workflow.sampling.sites。锚点 after_agent_turn / after_verifier，"
            "resume_mode 用 messages。用户区不重写采样引擎，管理区 ActiveSet 与 Daemon 会建 RolloutTree。"
        )
    try:
        listing = tools["list_user_files"]()
    except Exception:
        listing = "[]"
    return (
        f"当前没有配置 AI_ASSISTANT_*，我只能做离线说明。\n"
        f"当前项目：{project_id or '未选择'}。文件：{listing}\n"
        "配置 .env 的 AI_ASSISTANT_API_KEY / AI_ASSISTANT_API_BASE / AI_ASSISTANT_MODEL 后即可流式对话。"
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _delta_tool_calls(delta: Any) -> List[Any]:
    calls = getattr(delta, "tool_calls", None)
    if calls:
        return list(calls)
    if isinstance(delta, dict):
        return list(delta.get("tool_calls") or [])
    return []


def _accumulate_stream(stream: Iterable[Any]) -> Iterator[Tuple[str, Any]]:
    calls: Dict[int, Dict[str, str]] = {}
    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = choices[0].delta if hasattr(choices[0], "delta") else {}
        content = getattr(delta, "content", None) if not isinstance(delta, dict) else delta.get("content")
        if content:
            yield "token", str(content)
        for tc in _delta_tool_calls(delta):
            idx = int(getattr(tc, "index", None) or (tc.get("index") if isinstance(tc, dict) else 0) or 0)
            slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            cid = getattr(tc, "id", None) if not isinstance(tc, dict) else tc.get("id")
            if cid:
                slot["id"] = str(cid)
            fn = getattr(tc, "function", None) if not isinstance(tc, dict) else tc.get("function")
            name = getattr(fn, "name", None) if fn is not None and not isinstance(fn, dict) else (fn or {}).get("name")
            args = getattr(fn, "arguments", None) if fn is not None and not isinstance(fn, dict) else (fn or {}).get("arguments")
            if name:
                slot["name"] += str(name)
            if args:
                slot["arguments"] += str(args)
    yield "tool_calls", [calls[i] for i in sorted(calls)]


def _non_stream_round(resp: Any) -> Iterator[Tuple[str, Any]]:
    msg = resp.choices[0].message
    content = str(getattr(msg, "content", None) or "")
    if content:
        yield "token", content
    raw_calls = getattr(msg, "tool_calls", None) or []
    packed = []
    for call in raw_calls:
        packed.append(
            {
                "id": str(getattr(call, "id", "") or ""),
                "name": str(getattr(getattr(call, "function", None), "name", "") or ""),
                "arguments": str(getattr(getattr(call, "function", None), "arguments", "") or ""),
            }
        )
    yield "tool_calls", packed


def _redact(text: str, limit: int = 2000) -> str:
    cleaned = SECRET_RE.sub("[redacted]", str(text or ""))
    if len(cleaned) > limit:
        return cleaned[:limit] + "\n...[truncated]"
    return cleaned


def _short_args(args: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in (args or {}).items():
        if key in {"content", "current_workflow", "workflow"}:
            out[key] = f"<{len(str(val))} chars>"
            continue
        text = str(val)
        out[key] = text if len(text) < 160 else text[:157] + "..."
    return out


def _run_tool(tools: Dict[str, Any], name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    if not isinstance(args, dict):
        args = {}
    fn = tools.get(name)
    if fn is None:
        return f"unknown tool {name}"
    try:
        return str(fn(**args))
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _build_messages(body: ChatBody, store: MemoryStore, scope: str) -> Tuple[List[Dict[str, Any]], str]:
    """Assemble the LLM message list.

    Returns (messages, new_summary): ``new_summary`` is non-empty only when
    this round's compaction folded overflow history into a fresh summary; the
    caller persists it via MemoryStore.save_session_summary after the turn.
    """
    messages: List[Dict[str, Any]] = [{"role": "system", "content": build_system_prompt(body.message)}]
    snapshot = build_workspace_snapshot(body)
    messages.append({"role": "system", "content": "当前工作区快照：\n" + snapshot})
    pinned = pinned_memory_block(store, scope)
    if pinned:
        messages.append({"role": "system", "content": pinned})
    previous_summary = store.latest_session_summary(scope)
    kept, summary = compress_history(_history_messages(body.history), previous_summary=previous_summary)
    summary_msg = summary_system_message(summary)
    if summary_msg is not None:
        messages.append(summary_msg)
    messages.extend(kept)
    messages.append({"role": "user", "content": body.message})
    new_summary = summary if summary and summary != previous_summary else ""
    return messages, new_summary


def iter_chat_events(body: ChatBody) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (event, data). Used by SSE and unit tests."""
    cfg = resolve_assistant_llm(body.experiment_id)
    session = ToolSession()
    store = MemoryStore()
    scope = memory_store.resolve_scope(body.experiment_id, body.project_id)
    if cfg.source == "none" or not cfg.base_url or not cfg.model:
        text = _local_answer(body.message, body.project_id)
        yield "token", {"text": text}
        yield "done", {
            "project_id": body.project_id,
            "experiment_id": body.experiment_id,
            "wrote_files": session.wrote_files,
            "can_apply": False,
            "offline": True,
        }
        return
    try:
        from openai import OpenAI
    except Exception:
        text = _local_answer(body.message, body.project_id)
        yield "token", {"text": text}
        yield "done", {
            "project_id": body.project_id,
            "experiment_id": body.experiment_id,
            "wrote_files": False,
            "can_apply": False,
            "offline": True,
        }
        return

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY")
    tool_rows = assistant_tools(
        body.project_id,
        current_workflow=body.current_workflow,
        session=session,
        experiment_id=body.experiment_id,
        memory=store,
        memory_scope=scope,
    )
    tools = {row["name"]: row["fn"] for row in tool_rows}
    messages, new_summary = _build_messages(body, store, scope)
    final_text = ""
    exhausted = False
    for _round in range(MAX_TOOL_ROUNDS):
        yield "status", {"phase": "thinking"}
        calls: List[Dict[str, str]] = []
        try:
            stream = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                tools=OPENAI_TOOLS,
                temperature=0.3,
                max_tokens=MAX_OUTPUT_TOKENS,
                stream=True,
            )
            event_iter: Iterable[Tuple[str, Any]] = _accumulate_stream(stream)
        except Exception:
            try:
                resp = client.chat.completions.create(
                    model=cfg.model,
                    messages=messages,
                    tools=OPENAI_TOOLS,
                    temperature=0.3,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                event_iter = _non_stream_round(resp)
            except Exception as exc:  # noqa: BLE001
                yield "error", {"message": _redact(str(exc), 400)}
                break
        for kind, payload in event_iter:
            if kind == "token":
                piece = str(payload)
                final_text += piece
                yield "token", {"text": piece}
            elif kind == "tool_calls":
                calls = list(payload or [])
        if not calls:
            break
        yield "status", {"phase": "calling_tools"}
        messages.append(
            {
                "role": "assistant",
                "content": final_text or None,
                "tool_calls": [
                    {
                        "id": call.get("id") or f"call_{idx}",
                        "type": "function",
                        "function": {"name": call.get("name") or "", "arguments": call.get("arguments") or "{}"},
                    }
                    for idx, call in enumerate(calls)
                ],
            }
        )
        final_text = ""
        for call in calls:
            name = call.get("name") or ""
            raw_args = call.get("arguments") or "{}"
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed = {}
            yield "tool_start", {"name": name, "args": _short_args(parsed if isinstance(parsed, dict) else {})}
            result = _run_tool(tools, name, raw_args)
            yield "tool_end", {"name": name, "result": _redact(result), "ok": not str(result).startswith("error:")}
            messages.append({"role": "tool", "tool_call_id": call.get("id") or "call", "content": _redact(result, 8000)})
    else:
        exhausted = True
        notice = "本轮工具步数已达上限。告诉我还要继续读或改哪些文件，我下一轮接着做。"
        yield "token", {"text": notice}

    if not final_text and not exhausted:
        # Tool-only last round produced no closing text.
        if session.wrote_files:
            yield "token", {"text": "已完成工具调用。需要的话我可以继续解释改动，或帮你应用到画布。"}
    if new_summary:
        store.save_session_summary(scope, new_summary, experiment_id=body.experiment_id)
    yield "done", {
        "project_id": body.project_id,
        "experiment_id": body.experiment_id,
        "wrote_files": session.wrote_files,
        "can_apply": bool(session.applied_workflow) or session.wrote_files,
        "workflow": session.applied_workflow,
        "offline": False,
    }


def _run_llm(body: ChatBody) -> str:
    """Collect streamed tokens (tests / offline). Never dumps keys."""
    parts: List[str] = []
    for event, data in iter_chat_events(body):
        if event == "token":
            parts.append(str(data.get("text") or ""))
        elif event == "error":
            parts.append(str(data.get("message") or "辅助对话失败"))
    return "".join(parts).strip()
