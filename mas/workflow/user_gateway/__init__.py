"""Management-side gateway into the user-code zone."""

from .apply import merge_user_workflow
from .loader import invoke_user_window, maybe_invoke_user_node, resolve_runner
from .paths import assert_writable_user_path, user_space_root
from .protocol import UserWindowRunner
from .registry import create_project, get_project, list_projects
from .epc_aw import ensure_epc_aw_project, scaffold_epc_aw_pev
from .hive import ensure_hive_project, scaffold_hive_pev
from .scaffold import scaffold_io_module, scaffold_native_mas, scaffold_project
from .tools import get_user_tool_agent
from .validate import validate_project

__all__ = [
    "UserWindowRunner",
    "assert_writable_user_path",
    "create_project",
    "ensure_epc_aw_project",
    "ensure_hive_project",
    "get_project",
    "get_user_tool_agent",
    "invoke_user_window",
    "list_projects",
    "maybe_invoke_user_node",
    "merge_user_workflow",
    "resolve_runner",
    "scaffold_epc_aw_pev",
    "scaffold_hive_pev",
    "scaffold_io_module",
    "scaffold_native_mas",
    "scaffold_project",
    "user_space_root",
    "validate_project",
]
