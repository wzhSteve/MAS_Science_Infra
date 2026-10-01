"""Per-project tool overlay. Never mutates ``tools.tool_agents.TOOL_AGENTS``."""

from __future__ import annotations

from typing import Any, Dict, Optional

from workflow.protocol import AgentMessage, make_message

from .loader import invoke_user_window
from .paths import parse_user_agent_id
from .registry import load_manifest

_OVERLAY: Dict[str, "UserToolAgent"] = {}


class UserToolAgent:
    """Duck-typed tool-agent: ``invoke(args) -> str`` like the built-in five."""

    def __init__(self, agent_id: str, project_id: str, description: str = "") -> None:
        self.id = agent_id
        self.kind = "tool"
        self.project_id = project_id
        self.description = description or f"user tool {agent_id}"
        self.trainable = False
        self.profile = {"backend": "user_space", "user_project": project_id, "llm_required": False}
        self._kernel = None
        self._llm_invoke = None
        self.last_payload: Dict[str, Any] = {}

    @property
    def backend(self) -> str:
        return "user_space"

    @property
    def llm_required(self) -> bool:
        return False

    def invoke(self, args: Dict[str, Any], *, prefer_llm: bool = True, tier: Optional[str] = None) -> str:
        del prefer_llm, tier
        inbound = make_message(
            task_id="user-tool",
            turn=0,
            src="router",
            dst=self.id,
            kind="tool_invoke",
            payload=args if isinstance(args, dict) else {"query": str(args)},
        )
        out = invoke_user_window(self.project_id, inbound, agent_id=self.id, expected_kind="tool_result")
        self.last_payload = dict(out.payload or {})
        if out.kind == "error":
            return f"user tool error: {out.payload.get('error')}"
        return str(out.payload.get("output") or "")


def register_user_tool(agent: UserToolAgent) -> None:
    _OVERLAY[agent.id] = agent


def clear_user_tools(project_id: Optional[str] = None) -> None:
    if project_id is None:
        _OVERLAY.clear()
        return
    for key in [k for k, v in _OVERLAY.items() if v.project_id == project_id]:
        _OVERLAY.pop(key, None)


def bind_project_tools(project_id: str) -> None:
    try:
        manifest = load_manifest(project_id)
    except FileNotFoundError:
        return
    for tid in manifest.get("tool_ids") or []:
        if tid not in _OVERLAY:
            register_user_tool(UserToolAgent(str(tid), project_id))


def get_user_tool_agent(agent_id: str) -> Optional[UserToolAgent]:
    if agent_id in _OVERLAY:
        return _OVERLAY[agent_id]
    parsed = parse_user_agent_id(agent_id)
    if parsed is None:
        return None
    project_id, _ = parsed
    bind_project_tools(project_id)
    if agent_id not in _OVERLAY:
        register_user_tool(UserToolAgent(agent_id, project_id))
    return _OVERLAY.get(agent_id)


def overlay_ids() -> Dict[str, str]:
    return {k: v.project_id for k, v in _OVERLAY.items()}
