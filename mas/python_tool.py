# Copyright (c) Microsoft. All rights reserved.

"""Restricted Python code execution tool for the math GSM agent."""

from __future__ import annotations

import ast
import math
import operator
import threading
import traceback
from io import StringIO
from typing import Any, Dict, Set

# Allowed built-ins for sandbox execution.
_SAFE_BUILTINS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "round": round,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

_FORBIDDEN_NAMES: Set[str] = {
    "os",
    "sys",
    "subprocess",
    "shutil",
    "pathlib",
    "socket",
    "requests",
    "urllib",
    "ctypes",
    "importlib",
    "builtins",
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "vars",
    "dir",
    "input",
    "help",
    "breakpoint",
    "memoryview",
}


class _Timeout(Exception):
    pass


def _check_ast_safety(code: str) -> None:
    """Reject imports and calls to forbidden names via AST inspection."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Import statements are not allowed.")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise ValueError(f"Use of '{node.id}' is not allowed.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Access to dunder attributes is not allowed.")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _FORBIDDEN_NAMES:
                raise ValueError(f"Call to '{func.id}' is not allowed.")


def execute_python(code: str, timeout_seconds: float = 5.0) -> str:
    """Execute a restricted Python snippet and return stdout / result.

    The sandbox allows the ``math`` module and a small set of built-ins.
    Assign the final answer to a variable named ``result`` when possible.

    Args:
        code: Python source to execute.
        timeout_seconds: Wall-clock timeout for execution.

    Returns:
        A human-readable string with stdout and/or the ``result`` variable,
        or an error message.
    """
    if not code or not code.strip():
        return "Error: empty code."

    try:
        _check_ast_safety(code)
    except ValueError as e:
        return f"Error: {e}"

    stdout_buf = StringIO()
    local_vars: Dict[str, Any] = {}
    global_vars: Dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "operator": operator,
    }

    result_holder: Dict[str, Any] = {"error": None}

    def _run() -> None:
        import sys

        old_stdout = sys.stdout
        sys.stdout = stdout_buf
        try:
            exec(code, global_vars, local_vars)  # noqa: S102 — intentional sandbox
        except Exception:
            result_holder["error"] = traceback.format_exc(limit=3)
        finally:
            sys.stdout = old_stdout

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        return f"Error: execution timed out after {timeout_seconds}s."

    if result_holder["error"]:
        return f"Error during execution:\n{result_holder['error']}"

    parts = []
    stdout_text = stdout_buf.getvalue().strip()
    if stdout_text:
        parts.append(f"stdout:\n{stdout_text}")
    if "result" in local_vars:
        parts.append(f"result = {local_vars['result']!r}")
    elif local_vars:
        # Fall back to last assigned simple value names.
        simple = {k: v for k, v in local_vars.items() if not k.startswith("_")}
        if simple:
            parts.append(f"locals: {simple!r}")

    if not parts:
        return "Execution finished with no output. Assign the answer to `result` or use print()."
    return "\n".join(parts)
