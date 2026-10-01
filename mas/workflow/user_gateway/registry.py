"""Project index over ``user_space/registry.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import (
    ensure_project_layout,
    ensure_user_space,
    project_dir,
    registry_path,
    user_agent_id,
    validate_project_id,
)

VALID_MODES = ("io_module", "native_mas")
VALID_STATUS = ("uploaded", "adapting", "ready", "failed")


def _empty_registry() -> Dict[str, Any]:
    return {"version": 1, "projects": []}


def load_registry() -> Dict[str, Any]:
    ensure_user_space()
    path = registry_path()
    if not path.is_file():
        return _empty_registry()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        return _empty_registry()
    data.setdefault("version", 1)
    data.setdefault("projects", [])
    if not isinstance(data["projects"], list):
        data["projects"] = []
    return data


def save_registry(data: Dict[str, Any]) -> None:
    ensure_user_space()
    blob = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    registry_path().write_text(blob, encoding="utf-8")


def _manifest_path(project_id: str) -> Path:
    return project_dir(project_id) / "manifest.yaml"


def default_manifest(project_id: str, *, title: str, mode: str, experiment_id: str = "") -> Dict[str, Any]:
    pid = validate_project_id(project_id)
    solver = user_agent_id(pid, "solver")
    return {
        "id": pid,
        "title": title or pid,
        "mode": mode if mode in VALID_MODES else "io_module",
        "status": "uploaded",
        "entry": "adapted/entry.py:UserAgent",
        "experiment_id": experiment_id or "",
        "agent_ids": [solver],
        "tool_ids": [],
        "writable_root": ["adapted", "contracts", "artifacts"],
    }


def load_manifest(project_id: str) -> Dict[str, Any]:
    path = _manifest_path(project_id)
    if not path.is_file():
        raise FileNotFoundError(f"项目 {project_id} 没有 manifest.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("manifest.yaml 必须是对象")
    return data


def save_manifest(project_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    pid = validate_project_id(project_id)
    data = dict(data)
    data["id"] = pid
    path = _manifest_path(pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    _upsert_index(data)
    return data


def _upsert_index(manifest: Dict[str, Any]) -> None:
    reg = load_registry()
    items = [p for p in reg["projects"] if isinstance(p, dict) and p.get("id") != manifest.get("id")]
    items.append(
        {
            "id": manifest.get("id"),
            "title": manifest.get("title"),
            "mode": manifest.get("mode"),
            "status": manifest.get("status"),
            "experiment_id": manifest.get("experiment_id") or "",
        }
    )
    items.sort(key=lambda row: str(row.get("id") or ""))
    reg["projects"] = items
    save_registry(reg)


def create_project(
    project_id: str,
    *,
    title: str,
    mode: str = "io_module",
    experiment_id: str = "",
) -> Dict[str, Any]:
    pid = validate_project_id(project_id)
    root = ensure_project_layout(pid)
    if _manifest_path(pid).is_file():
        raise FileExistsError(f"项目已存在: {pid}")
    manifest = default_manifest(pid, title=title, mode=mode, experiment_id=experiment_id)
    save_manifest(pid, manifest)
    return {"root": str(root), **manifest}


def list_projects(*, experiment_id: Optional[str] = None, status: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in load_registry().get("projects") or []:
        if not isinstance(row, dict):
            continue
        if experiment_id and str(row.get("experiment_id") or "") not in ("", experiment_id):
            continue
        if status and str(row.get("status") or "") != status:
            continue
        pid = str(row.get("id") or "")
        if not pid:
            continue
        try:
            out.append(load_manifest(pid))
        except FileNotFoundError:
            out.append(row)
    return out


def get_project(project_id: str) -> Dict[str, Any]:
    return load_manifest(project_id)


def update_status(project_id: str, status: str, **extra: Any) -> Dict[str, Any]:
    if status not in VALID_STATUS:
        raise ValueError(f"未知 status: {status}")
    data = load_manifest(project_id)
    data["status"] = status
    data.update(extra)
    return save_manifest(project_id, data)
