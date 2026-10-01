"""Floating assistant: streaming chat + user-space tools. Writes never leave user_space."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from science_infra.control.llm_config import resolve_llm_config
from science_infra.control.paths import mas_root
from science_infra.env import load_science_env
from science_infra.control.user_projects import (
    apply_project,
    get_user_project,
    ingest_upload,
    list_project_files,
    list_user_projects,
    read_project_file,
)
from science_infra.env import repo_root

router = APIRouter(prefix="/api/assistant", tags=["assistant"])

SYSTEM_PROMPT = """你是 Science Studio 里的编程向对话代理，体验应对齐 ChatGPT / Codex：先读再写，边做边说。

工作方式
- 不确定就调用 grep / read_file / list_dir，不要猜仓库里的代码。
- 改用户区文件后，用中文说明改了哪些路径、为什么，必要时给关键 diff 要点。
- 追问时利用对话历史，不要每轮重新脚手架。
- 用户明确要求封装外部 MAS 时才调用 scaffold_*：HIVE → scaffold_hive_pev，EPC-AW → scaffold_epc_aw_pev；其他上传代码用 scaffold_io_module 或 scaffold_native_mas。不要发明另一套 agent。
- 用户说「应用到画布」时调用 apply_to_canvas。实验 YAML 仍须用户点顶栏「保存实验」才会落盘。

三区规则（不可违反）
1. mas/ science_infra/ rl/ webui/ docs/ ref_Rep/ 只读，禁止修改。
2. experiments/<id>/ 只能通过 apply_to_canvas 返回草稿，禁止直接写文件。
3. 唯一可写树：user_space/projects/<id>/{adapted,contracts,artifacts}/。

合同
- 画布只产 MASSpec YAML schema 0.3，不生成 LangGraph。
- Agent kind 只有 planner / tool / verifier / blank。信封是 AgentMessage。
- 封装 HIVE / EPC-AW 必须按窗口包装 Planner / Executor / Diagnoser，保留 PEV；禁止把 Solver.solve 当采集/采样热路径。
- 采样站点写在 workflow.sampling.sites（planner after_agent_turn、router after_agent_turn、verifier after_verifier，resume_mode=messages）。
- 不要调用 register_tool_agent 覆盖内置五工具。

回答用中文。可以分步、可以引用代码块。不要输出 API 密钥。
"""

SKIP_DIR_PARTS = {
    ".venv",
    "node_modules",
    "agent-lightning",
    "LLM",
    ".git",
    "__pycache__",
    "dist",
    ".local_expansion",
}
SECRET_RE = re.compile(r"sk-[A-Za-z0-9]{8,}|api[_-]?key\s*[:=]\s*\S+", re.IGNORECASE)
WELCOME_MARKERS = ("我只能改 user_space/", "只写 user_space/")
MAX_TOOL_ROUNDS = 16
MAX_OUTPUT_TOKENS = 4096
HISTORY_LIMIT = 24


@dataclass(frozen=True)
class AssistantLlm:
    model: str
    base_url: str
    api_key: str
    source: str  # assistant_env | experiment | none

    def __repr__(self) -> str:
        return f"AssistantLlm(model={self.model!r}, base_url={self.base_url!r}, source={self.source!r})"


def resolve_assistant_llm(experiment_id: Optional[str] = None) -> AssistantLlm:
    """Prefer AI_ASSISTANT_* from repo .env. Never log the key."""
    load_science_env()
    key = os.environ.get("AI_ASSISTANT_API_KEY", "").strip()
    base = (os.environ.get("AI_ASSISTANT_API_BASE") or os.environ.get("AI_ASSISTANT_BASE_URL") or "").strip()
    model = os.environ.get("AI_ASSISTANT_MODEL", "").strip()
    if base and model:
        return AssistantLlm(model=model, base_url=base.rstrip("/"), api_key=key, source="assistant_env")
    if experiment_id:
        try:
            cfg = resolve_llm_config(experiment_id)
            if cfg.base_url and cfg.model:
                return AssistantLlm(
                    model=cfg.model,
                    base_url=str(cfg.base_url).rstrip("/"),
                    api_key=cfg.api_key or "",
                    source="experiment",
                )
        except Exception:
            pass
    return AssistantLlm("", "", "", "none")


class ChatBody(BaseModel):
    message: str
    experiment_id: Optional[str] = None
    project_id: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)
    workflow_summary: Optional[Dict[str, Any]] = None
    selected_node_id: Optional[str] = None
    current_workflow: Optional[Dict[str, Any]] = None


class ApplyBody(BaseModel):
    experiment_id: Optional[str] = None
    current_workflow: Dict[str, Any] = Field(default_factory=dict)
    replace: bool = False


@dataclass
class ToolSession:
    wrote_files: bool = False
    applied_workflow: Optional[Dict[str, Any]] = None
    notes: List[str] = field(default_factory=list)


def _ensure_mas_path() -> None:
    import sys

    mas = str(mas_root())
    if mas not in sys.path:
        sys.path.insert(0, mas)


def _paths():
    _ensure_mas_path()
    from workflow.user_gateway import paths

    return paths


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


def summarize_workflow(workflow: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    wf = workflow if isinstance(workflow, dict) else {}
    agents = []
    for agent in wf.get("agents") or []:
        if isinstance(agent, dict) and agent.get("id"):
            agents.append({"id": str(agent.get("id")), "kind": str(agent.get("kind") or "")})
    routers = [str(r.get("id")) for r in (wf.get("routers") or []) if isinstance(r, dict) and r.get("id")]
    sampling = wf.get("sampling") if isinstance(wf.get("sampling"), dict) else {}
    sites = sampling.get("sites") if isinstance(sampling, dict) else []
    return {
        "agents": agents[:24],
        "routers": routers[:12],
        "n_sites": len(sites) if isinstance(sites, list) else 0,
        "entry_agent": str(wf.get("entry_agent") or "planner"),
    }


def build_workspace_snapshot(body: ChatBody) -> str:
    summary = body.workflow_summary if isinstance(body.workflow_summary, dict) else {}
    if not summary and body.current_workflow:
        summary = summarize_workflow(body.current_workflow)
    lines = [
        f"实验: {body.experiment_id or '未绑定'}",
        f"用户项目: {body.project_id or '未选择'}",
    ]
    if body.selected_node_id:
        lines.append(f"画布选中节点: {body.selected_node_id}")
    agents = summary.get("agents") or []
    if agents:
        labels = [f"{a.get('id')}({a.get('kind') or '?'})" for a in agents if isinstance(a, dict)]
        lines.append("画布节点: " + ", ".join(labels[:20]))
    routers = summary.get("routers") or []
    if routers:
        lines.append("Router: " + ", ".join(str(r) for r in routers))
    lines.append(f"sampling.sites: {int(summary.get('n_sites') or 0)}")
    entry = summary.get("entry_agent")
    if entry:
        lines.append(f"entry_agent: {entry}")
    if body.project_id:
        try:
            files = list_project_files(body.project_id)[:40]
            names = [str(item.get("path") or "") for item in files if isinstance(item, dict)]
            if names:
                lines.append("项目文件: " + ", ".join(names))
        except Exception:
            pass
    return "\n".join(lines)


def assistant_tools(
    project_id: Optional[str] = None,
    *,
    current_workflow: Optional[Dict[str, Any]] = None,
    session: Optional[ToolSession] = None,
) -> List[Dict[str, Any]]:
    _ensure_mas_path()
    from workflow.user_gateway import paths, scaffold, validate
    from workflow.user_gateway.paths import assert_readable_repo_path, assert_writable_user_path

    sess = session or ToolSession()
    root = repo_root()

    def _resolve_read(path: str) -> Path:
        raw = Path(path) if path else root
        target = raw if raw.is_absolute() else root / raw
        return assert_readable_repo_path(target)

    def read_file(path: str, offset: int = 1, limit: int = 0) -> str:
        target = _resolve_read(path)
        if target.is_dir():
            names = sorted(p.name for p in target.iterdir())[:80]
            return "\n".join(names)
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        start = max(int(offset or 1) - 1, 0)
        chunk = lines[start : start + int(limit)] if int(limit or 0) > 0 else lines[start:]
        blob = "\n".join(chunk)
        if len(blob) > 12000:
            return blob[:12000] + "\n...[truncated]"
        return blob

    def grep(query: str, path: str = "", glob: str = "*.py", max_hits: int = 40) -> str:
        if not str(query or "").strip():
            return "(empty query)"
        start = _resolve_read(path) if path else root
        if start.is_file():
            files = [start]
        else:
            files = [p for p in start.rglob(glob or "*.py") if p.is_file()]
        hits: List[str] = []
        for file in files:
            if any(part in SKIP_DIR_PARTS for part in file.parts):
                continue
            try:
                assert_readable_repo_path(file)
                blob = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = file.relative_to(root).as_posix() if str(file).startswith(str(root)) else file.as_posix()
            for idx, line in enumerate(blob.splitlines(), 1):
                if query in line:
                    hits.append(f"{rel}:{idx}:{line.strip()[:200]}")
                    if len(hits) >= max(1, int(max_hits or 40)):
                        return "\n".join(hits)
        return "\n".join(hits) or "(no hits)"

    def search_code(query: str, glob: str = "*.py") -> str:
        return grep(query, path="", glob=glob)

    def list_dir(path: str = ".") -> str:
        target = _resolve_read(path)
        if target.is_file():
            return target.name
        names: List[str] = []
        for child in sorted(target.iterdir()):
            if child.name in SKIP_DIR_PARTS:
                continue
            suffix = "/" if child.is_dir() else ""
            names.append(child.name + suffix)
            if len(names) >= 80:
                break
        return "\n".join(names) or "(empty)"

    def write_file(path: str, content: str) -> str:
        if not project_id:
            raise PermissionError("请先选择或创建用户项目再写文件。")
        rel = path
        if not rel.startswith("adapted/") and not rel.startswith("contracts/") and not rel.startswith("artifacts/"):
            rel = f"adapted/{Path(path).name}"
        target = paths.project_dir(project_id) / rel
        locked = assert_writable_user_path(target, project_id=project_id)
        locked.parent.mkdir(parents=True, exist_ok=True)
        before = locked.read_text(encoding="utf-8") if locked.exists() else ""
        locked.write_text(content, encoding="utf-8")
        sess.wrote_files = True
        return json.dumps({"path": rel, "bytes": len(content), "changed": before != content}, ensure_ascii=False)

    def list_user_files() -> str:
        if not project_id:
            return json.dumps(list_user_projects(), ensure_ascii=False)
        return json.dumps(list_project_files(project_id), ensure_ascii=False)

    def validate_current() -> str:
        if not project_id:
            raise ValueError("没有当前项目")
        return json.dumps(validate.validate_project(project_id), ensure_ascii=False)

    def wrap_io_module() -> str:
        if not project_id:
            raise ValueError("没有当前项目")
        out = scaffold.scaffold_io_module(project_id)
        sess.wrote_files = True
        return json.dumps({k: v for k, v in out.items() if k != "workflow"} | {"ok": True}, ensure_ascii=False)

    def wrap_native_mas() -> str:
        if not project_id:
            raise ValueError("没有当前项目")
        out = scaffold.scaffold_project(project_id, "native_mas")
        sess.wrote_files = True
        return json.dumps({k: v for k, v in out.items() if k != "workflow"} | {"ok": True}, ensure_ascii=False)

    def wrap_hive() -> str:
        from workflow.user_gateway.hive import ensure_hive_project, scaffold_hive_pev

        pid = project_id or "hive"
        ensure_hive_project(pid)
        out = scaffold_hive_pev(pid)
        sess.wrote_files = True
        return json.dumps({k: v for k, v in out.items() if k != "workflow"} | {"ok": True}, ensure_ascii=False)

    def wrap_epc_aw() -> str:
        from workflow.user_gateway.epc_aw import ensure_epc_aw_project, scaffold_epc_aw_pev

        pid = project_id or "epc-aw"
        ensure_epc_aw_project(pid)
        out = scaffold_epc_aw_pev(pid)
        sess.wrote_files = True
        return json.dumps({k: v for k, v in out.items() if k != "workflow"} | {"ok": True}, ensure_ascii=False)

    def apply_to_canvas(replace: bool = False) -> str:
        if not project_id:
            raise ValueError("没有当前项目")
        if not current_workflow:
            raise ValueError("没有画布草稿，请先打开实验工作区。")
        result = apply_project(project_id, current_workflow, replace=bool(replace))
        sess.wrote_files = True
        sess.applied_workflow = result.get("workflow") if isinstance(result, dict) else None
        agents = (sess.applied_workflow or {}).get("agents") if isinstance(sess.applied_workflow, dict) else []
        return json.dumps(
            {
                "ok": True,
                "mode": result.get("mode") if isinstance(result, dict) else "",
                "n_agents": len(agents or []),
                "needs_save": True,
            },
            ensure_ascii=False,
        )

    return [
        {"name": "read_file", "fn": read_file},
        {"name": "grep", "fn": grep},
        {"name": "search_code", "fn": search_code},
        {"name": "list_dir", "fn": list_dir},
        {"name": "write_file", "fn": write_file},
        {"name": "list_user_files", "fn": list_user_files},
        {"name": "validate_project", "fn": validate_current},
        {"name": "apply_to_canvas", "fn": apply_to_canvas},
        {"name": "scaffold_io_module", "fn": wrap_io_module},
        {"name": "scaffold_native_mas", "fn": wrap_native_mas},
        {"name": "scaffold_hive_pev", "fn": wrap_hive},
        {"name": "scaffold_epc_aw_pev", "fn": wrap_epc_aw},
    ]


OPENAI_TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a repo or user-space file. Use offset/limit for large files.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search text under a path and return path:line:snippet hits.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "max_hits": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_code", "description": "Alias of grep over the repo.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "glob": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List a directory in the repo or user_space.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write only under the current user project adapted/contracts/artifacts.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "list_user_files", "description": "List user project files.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "validate_project", "description": "Validate current user project MASSpec.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "apply_to_canvas", "description": "Merge the current user project workflow into the experiment canvas draft. Call when the user asks to apply to the canvas.", "parameters": {"type": "object", "properties": {"replace": {"type": "boolean"}}}}},
    {"type": "function", "function": {"name": "scaffold_io_module", "description": "Only when the user explicitly wants Mode 1 I/O wrapping.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "scaffold_native_mas", "description": "Only when the user explicitly wants a generic Mode 2 native MAS scaffold (not HIVE/EPC-AW).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "scaffold_hive_pev", "description": "Only when the user explicitly asks to wrap ref_Rep/HIVE as PEV windows. Do not rewrite HIVE.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "scaffold_epc_aw_pev", "description": "Only when the user explicitly asks to wrap EPC-AW as PEV windows. Do not rewrite EPC-AW.", "parameters": {"type": "object", "properties": {}}}},
]


def _local_answer(message: str, project_id: Optional[str]) -> str:
    """Offline fallback only. Never used when an assistant LLM is configured."""
    text = message.strip()
    tools = {t["name"]: t["fn"] for t in assistant_tools(project_id)}
    if any(k in text for k in ("适配", "包装", "io_module", "Mode 1", "mode 1")) and project_id:
        return tools["scaffold_io_module"]()
    if any(k in text.lower() for k in ("epc-aw", "epc_aw", "epcaw")) and project_id:
        return tools["scaffold_epc_aw_pev"]()
    if any(k in text.lower() for k in ("hive", "diagnoser", "pev")) or ("封装" in text and "HIVE" in text.upper()):
        return tools["scaffold_hive_pev"]()
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


def _build_messages(body: ChatBody) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    snapshot = build_workspace_snapshot(body)
    messages.append({"role": "system", "content": "当前工作区快照：\n" + snapshot})
    messages.extend(_history_messages(body.history))
    messages.append({"role": "user", "content": body.message})
    return messages


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


def iter_chat_events(body: ChatBody) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (event, data). Used by SSE and unit tests."""
    cfg = resolve_assistant_llm(body.experiment_id)
    session = ToolSession()
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
    tool_rows = assistant_tools(body.project_id, current_workflow=body.current_workflow, session=session)
    tools = {row["name"]: row["fn"] for row in tool_rows}
    messages = _build_messages(body)
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


@router.get("/projects")
def api_list_projects(experiment_id: Optional[str] = Query(default=None)) -> Dict[str, Any]:
    return {"projects": list_user_projects(experiment_id)}


@router.get("/projects/{project_id}")
def api_get_project(project_id: str) -> Dict[str, Any]:
    try:
        return get_user_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/projects/{project_id}/files")
def api_list_files(project_id: str) -> Dict[str, Any]:
    try:
        return {"files": list_project_files(project_id)}
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/projects/{project_id}/files/{rel_path:path}")
def api_read_file(project_id: str, rel_path: str) -> Dict[str, Any]:
    try:
        return {"path": rel_path, "content": read_project_file(project_id, rel_path)}
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/projects")
async def api_upload_project(
    title: str = Form(...),
    mode: str = Form("io_module"),
    experiment_id: str = Form(""),
    project_id: str = Form(""),
    file: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    try:
        return ingest_upload(
            title=title,
            mode=mode,
            experiment_id=experiment_id,
            project_id=project_id or None,
            upload=file,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/projects/{project_id}/apply")
def api_apply(project_id: str, body: ApplyBody) -> Dict[str, Any]:
    try:
        return apply_project(project_id, body.current_workflow, replace=body.replace)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/projects/{project_id}/scaffold")
def api_scaffold(project_id: str, mode: str = Query("io_module")) -> Dict[str, Any]:
    _ensure_mas_path()
    from workflow.user_gateway.scaffold import scaffold_project

    try:
        return scaffold_project(project_id, mode)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.post("/chat")
def api_chat(body: ChatBody) -> StreamingResponse:
    def gen() -> Iterable[str]:
        try:
            for event, data in iter_chat_events(body):
                yield _sse(event, data)
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": _redact(str(exc), 400)})
            yield _sse("done", {"project_id": body.project_id, "experiment_id": body.experiment_id, "wrote_files": False})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
