"""Upload / list / apply user-space projects (management API helpers)."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import UploadFile

from science_infra.control.paths import mas_root

DEFAULT_EPC_AW_RUNTIME: Dict[str, Any] = {
    "n": 1,
    "max_steps": 20,
    "max_time": 3000,
    "max_tokens": 4000,
    "temperature": 0.0,
    "enabled_tools": [
        "Base_Generator_Tool",
        "Python_Coder_Tool",
        "Wikipedia_Search_Tool",
        "Bing_Search_Tool",
        "Web_Fetch_Tool",
    ],
}


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
    from workflow.user_gateway.detect import detect_mas_family

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
        wraps = str(row.get("wraps") or "")
        try:
            family = detect_mas_family(pid, str(row.get("title") or ""))
        except Exception:
            family = "epc_aw" if wraps.startswith("EPC-AW") else ""
        items.append(
            {
                "id": pid,
                "title": row.get("title") or pid,
                "mode": row.get("mode") or "io_module",
                "status": row.get("status") or "uploaded",
                "agent_ids": list(row.get("agent_ids") or []),
                "tool_ids": list(row.get("tool_ids") or []),
                "workflow": wf or None,
                "family": family,
                "wraps": wraps,
            }
        )
    return items


def _runtime_path(project_id: str) -> Path:
    paths, _r, _s, _v, _a = _gw()
    return paths.project_dir(project_id) / "contracts" / "runtime.yaml"


def _clamp_runtime(data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_EPC_AW_RUNTIME)
    cfg["enabled_tools"] = list(DEFAULT_EPC_AW_RUNTIME["enabled_tools"])
    if not isinstance(data, dict):
        return cfg
    try:
        cfg["n"] = max(1, min(16, int(data.get("n", cfg["n"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("n 必须是 1–16 的整数") from exc
    try:
        cfg["max_steps"] = max(1, min(40, int(data.get("max_steps", cfg["max_steps"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_steps 必须是 1–40 的整数") from exc
    try:
        cfg["max_time"] = max(30, min(10000, int(data.get("max_time", cfg["max_time"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_time 必须是 30–10000 的整数") from exc
    try:
        cfg["max_tokens"] = max(256, min(16384, int(data.get("max_tokens", cfg["max_tokens"]))))
    except (TypeError, ValueError) as exc:
        raise ValueError("max_tokens 必须是 256–16384 的整数") from exc
    try:
        cfg["temperature"] = float(data.get("temperature", cfg["temperature"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("temperature 必须是数字") from exc
    if cfg["temperature"] < 0 or cfg["temperature"] > 2:
        raise ValueError("temperature 必须在 0–2")
    tools = data.get("enabled_tools", cfg["enabled_tools"])
    if not isinstance(tools, list) or not tools:
        raise ValueError("enabled_tools 至少勾选一项")
    allowed = set(DEFAULT_EPC_AW_RUNTIME["enabled_tools"])
    cleaned = [str(item) for item in tools if str(item) in allowed]
    if not cleaned:
        raise ValueError("enabled_tools 无效")
    cfg["enabled_tools"] = cleaned
    return cfg


def get_project_runtime(project_id: str) -> Dict[str, Any]:
    paths, registry, _s, _v, _a = _gw()
    registry.get_project(project_id)  # raises if missing
    path = _runtime_path(project_id)
    raw: Dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}
    return _clamp_runtime(raw)


def put_project_runtime(project_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    paths, registry, _s, _v, _a = _gw()
    registry.get_project(project_id)
    cfg = _clamp_runtime(data or {})
    path = _runtime_path(project_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return cfg


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
