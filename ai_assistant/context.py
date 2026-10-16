"""Assistant request models and workspace snapshot (moved from control.assistant)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class ToolSession:
    """Mutable per-chat scratchpad shared with assistant tool closures."""

    wrote_files: bool = False
    applied_workflow: Optional[Dict[str, Any]] = None
    notes: List[str] = Field(default_factory=list)


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
            from science_infra.control.user_projects import list_project_files

            files = list_project_files(body.project_id)[:40]
            names = [str(item.get("path") or "") for item in files if isinstance(item, dict)]
            if names:
                lines.append("项目文件: " + ", ".join(names))
        except Exception:
            pass
    return "\n".join(lines)
