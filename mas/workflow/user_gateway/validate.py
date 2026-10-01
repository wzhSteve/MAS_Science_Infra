"""Validate user-zone contracts against MASSpec + compiler."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from workflow.compiler import compile_spec
from workflow.spec import MASSpec

from .paths import project_dir
from .registry import load_manifest


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def project_workflow_path(project_id: str) -> Path:
    return project_dir(project_id) / "contracts" / "workflow.yaml"


def load_project_workflow(project_id: str) -> Dict[str, Any]:
    return load_yaml(project_workflow_path(project_id))


def validate_project(project_id: str) -> Dict[str, Any]:
    issues: List[str] = []
    manifest = load_manifest(project_id)
    wf = load_project_workflow(project_id)
    if not wf:
        issues.append("缺少 contracts/workflow.yaml")
        return {"ok": False, "issues": issues, "manifest": manifest}
    try:
        spec = MASSpec.model_validate(wf)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "issues": [f"MASSpec 无效: {exc}"], "manifest": manifest}
    compiled = compile_spec(spec)
    if not compiled.ok:
        issues.append(compiled.reason or "compile_spec 失败")
    exec_ok, exec_reason = spec.is_executable()
    if not exec_ok:
        issues.append(exec_reason or "is_executable 失败")
    prefix = f"u_{project_id}__"
    role_ids = {"planner", "verifier"}
    for agent in spec.agents or []:
        if str((agent.profile or {}).get("backend") or "") != "user_space":
            continue
        if agent.id in role_ids or agent.id.startswith(prefix):
            continue
        issues.append(f"用户节点 {agent.id} 必须带前缀 {prefix}（planner/verifier 角色 id 除外）")
    sampling = wf.get("sampling") or {}
    sites = list((sampling.get("sites") or []) if isinstance(sampling, dict) else [])
    return {
        "ok": not issues,
        "issues": issues,
        "manifest": manifest,
        "executable": bool(exec_ok and compiled.ok),
        "n_sites": len(sites),
        "agent_ids": [a.id for a in spec.agents],
    }


def compile_project(project_id: str) -> Tuple[MASSpec, Any]:
    wf = load_project_workflow(project_id)
    spec = MASSpec.model_validate(wf)
    return spec, compile_spec(spec)
