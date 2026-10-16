"""Assistant tool registry: OpenAI specs + callables bound to the session."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ai_assistant.context import ToolSession
from ai_assistant.memory import store as memory_store
from ai_assistant.skills import queries as skill_queries
from ai_assistant.skills import registry as skill_registry
from ai_assistant.tools import specs as tool_specs
from ai_assistant.tools.project_tools import build_project_tools
from ai_assistant.tools.repo_tools import build_repo_tools

OPENAI_TOOLS: List[Dict[str, Any]] = (
    tool_specs.SKILL_TOOL_SPECS + tool_specs.MEMORY_TOOL_SPECS + [
        {"type": "function", "function": {"name": "read_file", "description": "Read a repo or user-space file. Use offset/limit for large files.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "grep", "description": "Search text under a path and return path:line:snippet hits.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "path": {"type": "string"}, "glob": {"type": "string"}, "max_hits": {"type": "integer"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "search_code", "description": "Alias of grep over the repo.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "glob": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {"name": "list_dir", "description": "List a directory in the repo or user_space.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "write_file", "description": "Write only under the current user project adapted/contracts/artifacts.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
        {"type": "function", "function": {"name": "list_user_files", "description": "List user project files.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "validate_project", "description": "Validate current user project MASSpec.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "apply_to_canvas", "description": "Merge the current user project workflow into the experiment canvas draft. Call when the user asks to apply to the canvas.", "parameters": {"type": "object", "properties": {"replace": {"type": "boolean"}}}}},
        {"type": "function", "function": {"name": "scaffold_io_module", "description": "Only when the user explicitly wants Mode 1 I/O wrapping.", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "scaffold_native_mas", "description": "Only when the user explicitly wants a generic Mode 2 native MAS scaffold (not EPC-AW).", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "scaffold_epc_aw_pev", "description": "Only when the user explicitly asks to wrap EPC-AW as PEV windows. Do not rewrite EPC-AW.", "parameters": {"type": "object", "properties": {}}}},
    ]
)


def assistant_tools(
    project_id: Optional[str] = None,
    *,
    current_workflow: Optional[Dict[str, Any]] = None,
    session: Optional[Any] = None,
    experiment_id: Optional[str] = None,
    memory: Optional[memory_store.MemoryStore] = None,
    memory_scope: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """All callables available to the assistant for one chat turn.

    Superset of the legacy ``control.assistant.assistant_tools``: repo and
    project tools keep their names and behaviour; skills and memory add
    list_skills / load_skill / query_* / memory_* tools.
    """
    tools: List[Dict[str, Any]] = []
    sess = session or ToolSession()
    tools.extend(build_repo_tools(project_id, sess))
    tools.extend(build_project_tools(project_id, current_workflow=current_workflow, session=sess))

    store = memory or memory_store.MemoryStore()
    scope = memory_scope or memory_store.resolve_scope(experiment_id, project_id)

    def list_skills() -> str:
        return "\n".join(skill_registry.skill_index_lines()) or "(no skills installed)"

    def load_skill(name: str) -> str:
        return skill_registry.load_skill(name)

    def memory_read(scope_override: str = "") -> str:
        if scope_override:
            use_scope = memory_store.MemoryStore._validate_scope(scope_override)
        else:
            use_scope = scope
        return json.dumps({"scope": use_scope, "facts": store.facts(use_scope)}, ensure_ascii=False)

    def memory_write(text: str, pinned: bool = False, scope_override: str = "") -> str:
        use_scope = memory_store.MemoryStore._validate_scope(scope_override) if scope_override else scope
        store.write_fact(use_scope, text, pinned=bool(pinned))
        return json.dumps({"ok": True, "scope": use_scope}, ensure_ascii=False)

    tools.extend(skill_queries.skill_queries())
    tools.extend([
        {"name": "list_skills", "fn": list_skills},
        {"name": "load_skill", "fn": load_skill},
        {"name": "memory_read", "fn": memory_read},
        {"name": "memory_write", "fn": memory_write},
    ])
    return tools
