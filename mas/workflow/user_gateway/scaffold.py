"""Deterministic Mode 1 / Mode 2 file scaffolding (no LLM required)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .paths import ensure_project_layout, project_dir, user_agent_id
from .registry import load_manifest, save_manifest, update_status
from .tools import bind_project_tools

ENTRY_IO = '''"""Mode 1 I/O wrapper. Management runtime calls run_window only."""

from __future__ import annotations

from typing import Any


class UserAgent:
    """Black-box window: input text in, output text out."""

    def run_window(self, message: Any) -> Any:
        payload = message.payload if hasattr(message, "payload") else (message if isinstance(message, dict) else {})
        if isinstance(payload, dict) and "payload" in payload and isinstance(payload["payload"], dict):
            payload = payload["payload"]
        text = ""
        if isinstance(payload, dict):
            text = str(payload.get("input") or payload.get("output") or payload.get("query") or payload.get("question") or "")
        if not text:
            text = str(message)
        return {
            "kind": "tool_result",
            "payload": {"output": self.solve(text), "ok": True, "evidence_type": "DIRECT"},
        }

    def solve(self, text: str) -> str:
        return text.strip()
'''

ENTRY_NATIVE = '''"""Mode 2 dispatcher: planner / tool / verifier / blank windows."""

from __future__ import annotations

from typing import Any


class UserAgent:
    def run_window(self, message: Any) -> Any:
        kind = getattr(message, "kind", None) or (message.get("kind") if isinstance(message, dict) else "")
        payload = getattr(message, "payload", None) if hasattr(message, "payload") else (message.get("payload") if isinstance(message, dict) else message)
        payload = payload if isinstance(payload, dict) else {}
        dst = getattr(message, "dst", None) or (message.get("dst") if isinstance(message, dict) else "")
        if kind == "tool_invoke" or (dst and "tool" in str(dst)):
            query = str(payload.get("query") or payload.get("input") or payload.get("text") or payload)
            return {
                "kind": "tool_result",
                "payload": {"output": f"user-tool:{query}", "ok": True, "evidence_type": "DIRECT"},
            }
        if kind == "plan_step" or "planner" in str(dst):
            nxt = str(payload.get("next") or self.default_next)
            return {
                "kind": "plan_step",
                "payload": {
                    "next": nxt,
                    "args": payload.get("args") if isinstance(payload.get("args"), dict) else {"query": str(payload.get("question") or payload.get("input") or "")},
                    "sub_goal": str(payload.get("sub_goal") or payload.get("question") or "solve"),
                    "done": bool(payload.get("done")),
                },
            }
        if "verifier" in str(kind) or "verifier" in str(dst):
            return {
                "kind": "verify",
                "payload": {
                    "ok": True,
                    "reason": "user verifier pass",
                    "step_conclusion": "COMPLETE",
                    "slot_updates": [],
                },
            }
        text = str(payload.get("input") or payload.get("query") or payload)
        return {
            "kind": "tool_result",
            "payload": {"output": text, "ok": True, "evidence_type": "DIRECT"},
        }

    default_next = "__DEFAULT_NEXT__"
'''


def _dump(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _io_contract() -> Dict[str, Any]:
    return {
        "input_schema": {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
        "output_schema": {
            "type": "object",
            "properties": {"output": {"type": "string"}},
            "required": ["output"],
        },
    }


def _blank_node(agent_id: str, project_id: str, *, title: str) -> Dict[str, Any]:
    contract = _io_contract()
    return {
        "id": agent_id,
        "kind": "blank",
        "role": "agent",
        "trainable": False,
        "memory_scope": "none",
        "system_prompt": f"User-space module {title}",
        "profile": {
            "backend": "user_space",
            "user_project": project_id,
            "input_schema": contract["input_schema"],
            "output_schema": contract["output_schema"],
        },
        "meta": {"origin": "user", "user_project": project_id},
    }


def scaffold_io_module(project_id: str, *, title: Optional[str] = None) -> Dict[str, Any]:
    ensure_project_layout(project_id)
    manifest = load_manifest(project_id)
    title = title or str(manifest.get("title") or project_id)
    agent_id = user_agent_id(project_id, "solver")
    root = project_dir(project_id)
    (root / "adapted" / "entry.py").write_text(ENTRY_IO, encoding="utf-8")
    contract = _io_contract()
    _dump(root / "contracts" / "io_contract.yaml", contract)
    workflow = {
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "agents": [_blank_node(agent_id, project_id, title=title)],
        "sampling": {
            "mode": "arpo",
            "group_n": 4,
            "beam_size": 2,
            "sites": [
                {
                    "id": f"after_{agent_id}",
                    "enabled": True,
                    "anchor": {"kind": "after_agent_turn", "agent_id": agent_id},
                    "gate": {"type": "always"},
                    "fork": {"beam_size": 2, "resume_mode": "messages"},
                }
            ],
        },
    }
    _dump(root / "contracts" / "workflow.yaml", workflow)
    manifest.update(
        {
            "mode": "io_module",
            "status": "ready",
            "title": title,
            "agent_ids": [agent_id],
            "tool_ids": [],
            "entry": "adapted/entry.py:UserAgent",
        }
    )
    save_manifest(project_id, manifest)
    return {"project_id": project_id, "mode": "io_module", "agent_ids": [agent_id], "workflow": workflow}


def _sampling_sites(agent_ids: List[str]) -> List[Dict[str, Any]]:
    sites = []
    for aid in agent_ids:
        kind = "after_verifier" if aid == "verifier" else "after_agent_turn"
        sites.append(
            {
                "id": f"after_{aid}",
                "enabled": True,
                "anchor": {"kind": kind, "agent_id": aid},
                "gate": {"type": "always"},
                "fork": {"beam_size": 2, "resume_mode": "messages"},
            }
        )
    return sites


def scaffold_native_mas(project_id: str, *, title: Optional[str] = None) -> Dict[str, Any]:
    ensure_project_layout(project_id)
    manifest = load_manifest(project_id)
    title = title or str(manifest.get("title") or project_id)
    from .detect import detect_mas_family

    family = detect_mas_family(project_id, title)
    if family == "hive":
        from .hive import scaffold_hive_pev

        return scaffold_hive_pev(project_id, title=title)
    if family == "epc_aw":
        from .epc_aw import scaffold_epc_aw_pev

        return scaffold_epc_aw_pev(project_id, title=title)
    solver_id = user_agent_id(project_id, "solver")
    tool_id = user_agent_id(project_id, "tool")
    root = project_dir(project_id)
    (root / "adapted" / "entry.py").write_text(ENTRY_NATIVE.replace("__DEFAULT_NEXT__", tool_id), encoding="utf-8")
    (root / "adapted" / "tools" / "__init__.py").write_text("", encoding="utf-8")
    (root / "adapted" / "agents" / "__init__.py").write_text("", encoding="utf-8")
    contract = _io_contract()
    _dump(root / "contracts" / "io_contract.yaml", contract)
    workflow = {
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "hub": {"role": "planner", "skills": [], "max_feedback_hops": 3, "system_prompt": ""},
        "agents": [
            {
                "id": "planner",
                "kind": "planner",
                "role": "planner",
                "trainable": True,
                "system_prompt": "把题目拆成下一步。只输出 JSON：next, args, sub_goal, done。",
                "tools": [tool_id, solver_id],
            },
            _blank_node(solver_id, project_id, title=title),
            {
                "id": tool_id,
                "kind": "tool",
                "role": "tool",
                "trainable": False,
                "system_prompt": f"User tool-agent for {title}",
                "profile": {"backend": "user_space", "user_project": project_id, "llm_required": False},
                "meta": {"origin": "user", "user_project": project_id},
            },
            {
                "id": "verifier",
                "kind": "verifier",
                "trainable": False,
                "system_prompt": "根据 tool-agent 输出判断子目标是否完成，只输出 JSON。",
            },
        ],
        "routers": [
            {
                "id": "route_exec",
                "candidates": [solver_id, tool_id],
                "strategy": "from_plan",
                "output_contract": "json",
            }
        ],
        "edges": [
            {"from": "planner", "to": "route_exec", "kind": "route"},
            {"from": "route_exec", "to": "verifier", "kind": "message"},
            {"from": "verifier", "to": "planner", "kind": "feedback"},
        ],
        "sampling": {
            "mode": "arpo",
            "group_n": 4,
            "beam_size": 2,
            "sites": _sampling_sites(["planner", solver_id, "verifier"]),
        },
    }
    _dump(root / "contracts" / "workflow.yaml", workflow)
    _dump(root / "contracts" / "sampling.yaml", workflow["sampling"])
    manifest.update(
        {
            "mode": "native_mas",
            "status": "ready",
            "title": title,
            "agent_ids": [solver_id],
            "tool_ids": [tool_id],
            "entry": "adapted/entry.py:UserAgent",
        }
    )
    save_manifest(project_id, manifest)
    bind_project_tools(project_id)
    return {"project_id": project_id, "mode": "native_mas", "agent_ids": [solver_id], "tool_ids": [tool_id], "workflow": workflow}


def scaffold_project(project_id: str, mode: str, *, title: Optional[str] = None) -> Dict[str, Any]:
    if mode in ("hive_pev", "hive"):
        from .hive import scaffold_hive_pev

        return scaffold_hive_pev(project_id, title=title)
    if mode in ("epc_aw", "epc_aw_pev"):
        from .epc_aw import scaffold_epc_aw_pev

        return scaffold_epc_aw_pev(project_id, title=title)
    if mode == "native_mas":
        return scaffold_native_mas(project_id, title=title)
    return scaffold_io_module(project_id, title=title)
