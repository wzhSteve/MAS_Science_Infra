"""Merge a user project's workflow fragment into an experiment MASSpec dict."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from .registry import load_manifest
from .validate import load_project_workflow


def merge_user_workflow(
    current: Dict[str, Any],
    project_id: str,
    *,
    replace: bool = False,
) -> Dict[str, Any]:
    fragment = load_project_workflow(project_id)
    if not fragment:
        raise FileNotFoundError(f"项目 {project_id} 还没有 contracts/workflow.yaml")
    manifest = load_manifest(project_id)
    mode = str(manifest.get("mode") or "io_module")
    if replace or mode == "native_mas":
        merged = deepcopy(current)
        for key in ("agents", "routers", "edges", "sampling", "topology", "entry_agent", "hub", "schema_version"):
            if key in fragment:
                merged[key] = deepcopy(fragment[key])
        if fragment.get("tools"):
            merged["tools"] = list(fragment["tools"])
        return merged
    return _merge_io_module(current, fragment)


def _merge_io_module(current: Dict[str, Any], fragment: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(current) if current else {"schema_version": "0.3", "topology": "centralized", "agents": []}
    incoming = {a.get("id"): deepcopy(a) for a in fragment.get("agents") or [] if isinstance(a, dict) and a.get("id")}
    agents: List[Dict[str, Any]] = list(merged.get("agents") or [])
    existing = {a.get("id") for a in agents if isinstance(a, dict)}
    for aid, node in incoming.items():
        if aid in existing:
            agents = [node if a.get("id") == aid else a for a in agents]
        else:
            agents.append(node)
    merged["agents"] = agents
    routers = list(merged.get("routers") or [])
    if routers and incoming:
        first = dict(routers[0])
        cands = list(first.get("candidates") or [])
        for aid in incoming:
            if aid not in cands:
                cands.append(aid)
        first["candidates"] = cands
        routers[0] = first
        merged["routers"] = routers
    sites = list((merged.get("sampling") or {}).get("sites") or []) if isinstance(merged.get("sampling"), dict) else []
    frag_sites = list((fragment.get("sampling") or {}).get("sites") or []) if isinstance(fragment.get("sampling"), dict) else []
    known = {s.get("id") for s in sites if isinstance(s, dict)}
    for site in frag_sites:
        if isinstance(site, dict) and site.get("id") not in known:
            sites.append(deepcopy(site))
    if sites:
        sampling = dict(merged.get("sampling") or {})
        sampling["sites"] = sites
        merged["sampling"] = sampling
    return merged
