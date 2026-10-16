"""User-project validation, scaffolding and canvas apply (moved from control.assistant)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ai_assistant.tools.repo_tools import ensure_mas_path


def build_project_tools(
    project_id: Optional[str],
    *,
    current_workflow: Optional[Dict[str, Any]] = None,
    session: Any = None,
) -> List[Dict[str, Any]]:
    ensure_mas_path()
    from science_infra.control.user_projects import (
        apply_project,
        list_project_files,
        list_user_projects,
    )
    from workflow.user_gateway import scaffold, validate

    sess = session

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
        if sess is not None:
            sess.wrote_files = True
        return json.dumps({k: v for k, v in out.items() if k != "workflow"} | {"ok": True}, ensure_ascii=False)

    def wrap_native_mas() -> str:
        if not project_id:
            raise ValueError("没有当前项目")
        out = scaffold.scaffold_project(project_id, "native_mas")
        if sess is not None:
            sess.wrote_files = True
        return json.dumps({k: v for k, v in out.items() if k != "workflow"} | {"ok": True}, ensure_ascii=False)

    def wrap_epc_aw() -> str:
        from workflow.user_gateway.epc_aw import ensure_epc_aw_project, scaffold_epc_aw_pev

        pid = project_id or "epc-aw"
        ensure_epc_aw_project(pid)
        out = scaffold_epc_aw_pev(pid)
        if sess is not None:
            sess.wrote_files = True
        return json.dumps({k: v for k, v in out.items() if k != "workflow"} | {"ok": True}, ensure_ascii=False)

    def apply_to_canvas(replace: bool = False) -> str:
        if not project_id:
            raise ValueError("没有当前项目")
        if not current_workflow:
            raise ValueError("没有画布草稿，请先打开实验工作区。")
        result = apply_project(project_id, current_workflow, replace=bool(replace))
        if sess is not None:
            sess.wrote_files = True
            sess.applied_workflow = result.get("workflow") if isinstance(result, dict) else None
        agents = ((result.get("workflow") or {}) if isinstance(result, dict) else {}).get("agents") or []
        return json.dumps(
            {
                "ok": True,
                "mode": result.get("mode") if isinstance(result, dict) else "",
                "n_agents": len(agents),
                "needs_save": True,
            },
            ensure_ascii=False,
        )

    return [
        {"name": "list_user_files", "fn": list_user_files},
        {"name": "validate_project", "fn": validate_current},
        {"name": "apply_to_canvas", "fn": apply_to_canvas},
        {"name": "scaffold_io_module", "fn": wrap_io_module},
        {"name": "scaffold_native_mas", "fn": wrap_native_mas},
        {"name": "scaffold_epc_aw_pev", "fn": wrap_epc_aw},
    ]
