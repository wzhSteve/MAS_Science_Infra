"""Upload / list / apply user-space projects (management API helpers)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import UploadFile

from science_infra.control.paths import mas_root


def _gw():
    import sys

    mas = str(mas_root())
    if mas not in sys.path:
        sys.path.insert(0, mas)
    from workflow.user_gateway import apply as apply_mod
    from workflow.user_gateway import paths, registry, scaffold, validate

    return paths, registry, scaffold, validate, apply_mod


def list_user_projects(experiment_id: Optional[str] = None) -> List[Dict[str, Any]]:
    _paths, registry, _s, _v, _a = _gw()
    return registry.list_projects(experiment_id=experiment_id)


def get_user_project(project_id: str) -> Dict[str, Any]:
    _paths, registry, _s, validate, _a = _gw()
    data = registry.get_project(project_id)
    data["validation"] = validate.validate_project(project_id)
    return data


def palette_user_projects() -> List[Dict[str, Any]]:
    _paths, registry, _s, validate, _a = _gw()
    items = []
    for row in registry.list_projects():
        pid = str(row.get("id") or "")
        if not pid:
            continue
        wf = {}
        try:
            wf = validate.load_project_workflow(pid)
        except Exception:
            wf = {}
        items.append(
            {
                "id": pid,
                "title": row.get("title") or pid,
                "mode": row.get("mode") or "io_module",
                "status": row.get("status") or "uploaded",
                "agent_ids": list(row.get("agent_ids") or []),
                "tool_ids": list(row.get("tool_ids") or []),
                "workflow": wf or None,
            }
        )
    return items


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for info in zf.infolist():
        name = info.filename
        if name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"非法 zip 路径: {name}")
        target = (dest / name).resolve()
        if target != dest and dest not in target.parents:
            raise ValueError(f"zip 路径逃逸: {name}")
    zf.extractall(dest)


def ingest_upload(
    *,
    title: str,
    mode: str,
    experiment_id: str = "",
    project_id: Optional[str] = None,
    upload: Optional[UploadFile] = None,
    raw: Optional[bytes] = None,
    filename: str = "",
) -> Dict[str, Any]:
    paths, registry, scaffold, _validate, _apply = _gw()
    paths.ensure_user_space()
    pid = project_id or paths.slugify_project_id(title)
    try:
        manifest = registry.create_project(pid, title=title, mode=mode, experiment_id=experiment_id)
    except FileExistsError:
        suffix = 2
        base = pid
        while True:
            cand = f"{base}-{suffix}"[:48]
            try:
                manifest = registry.create_project(cand, title=title, mode=mode, experiment_id=experiment_id)
                pid = cand
                break
            except FileExistsError:
                suffix += 1
    root = paths.project_dir(pid)
    upload_dir = root / "upload"
    payload = raw
    name = filename
    if upload is not None:
        payload = payload or upload.file.read()
        name = name or (upload.filename or "upload.bin")
    if payload:
        dest = upload_dir / Path(name).name
        if name.lower().endswith(".zip"):
            _safe_extract(zipfile.ZipFile(io.BytesIO(payload)), upload_dir)
        else:
            dest.write_bytes(payload)
    built = scaffold.scaffold_project(pid, mode if mode in ("io_module", "native_mas") else "io_module", title=title)
    manifest = registry.load_manifest(pid)
    return {"project": manifest, "scaffold": {k: v for k, v in built.items() if k != "workflow"}, "workflow": built.get("workflow")}


def apply_project(project_id: str, current: Optional[Dict[str, Any]] = None, *, replace: bool = False) -> Dict[str, Any]:
    _p, registry, scaffold, _v, apply_mod = _gw()
    manifest = registry.get_project(project_id)
    from workflow.user_gateway.detect import detect_mas_family

    family = detect_mas_family(project_id, str(manifest.get("title") or ""))
    want_native = replace or manifest.get("mode") == "native_mas"
    if want_native:
        if family == "epc_aw":
            from workflow.user_gateway.epc_aw import scaffold_epc_aw_pev

            scaffold_epc_aw_pev(project_id, title=str(manifest.get("title") or ""))
            manifest = registry.get_project(project_id)
        elif family == "hive":
            from workflow.user_gateway.hive import scaffold_hive_pev

            scaffold_hive_pev(project_id, title=str(manifest.get("title") or ""))
            manifest = registry.get_project(project_id)
    merged = apply_mod.merge_user_workflow(current or {}, project_id, replace=replace or manifest.get("mode") == "native_mas")
    return {"workflow": merged, "mode": manifest.get("mode"), "project": manifest}


def list_project_files(project_id: str) -> List[Dict[str, str]]:
    paths, _r, _s, _v, _a = _gw()
    root = paths.project_dir(project_id)
    out: List[Dict[str, str]] = []
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(root).as_posix()
        zone = rel.split("/", 1)[0] if "/" in rel else rel
        writable = zone in paths.WRITABLE_REL_DIRS
        out.append({"path": rel, "zone": zone, "writable": "true" if writable else "false"})
    return out


def read_project_file(project_id: str, rel: str) -> str:
    paths, _r, _s, _v, _a = _gw()
    path = paths.assert_in_user_space(paths.project_dir(project_id) / rel, must_exist=True)
    return path.read_text(encoding="utf-8")
