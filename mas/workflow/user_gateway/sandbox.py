"""Timeout + exception isolation around user-window calls."""

from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Iterable, List

FORBIDDEN_USER_IMPORTS = {
    "science_infra",
    "rl",
    "agentlightning",
    "verl",
    "ray",
    "lit_tir_agent",
    "tools",
}


def scan_user_imports(path: Path) -> List[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
            found.append(node.module)
    return found


def assert_user_module_imports(path: Path) -> None:
    mods = scan_user_imports(path)
    for name in mods:
        top = name.split(".")[0]
        if top in FORBIDDEN_USER_IMPORTS:
            raise PermissionError(f"{path.name} 禁止 import {name}")
        if name.startswith("workflow.") and not name.startswith("workflow.user_gateway.public") and name != "workflow.protocol":
            if name.startswith("workflow.user_gateway") and name != "workflow.user_gateway.public":
                raise PermissionError(f"{path.name} 只能 import public 合同，不能 import {name}")


def run_isolated(fn: Callable[[], Any], *, timeout_s: float = 20.0) -> Any:
    if timeout_s <= 0:
        return fn()
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=timeout_s)
    except FuturesTimeout as exc:
        raise TimeoutError(f"user window timed out after {timeout_s}s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def scan_tree(root: Path, pattern: str = "*.py") -> Iterable[Path]:
    if not root.is_dir():
        return []
    return root.rglob(pattern)
