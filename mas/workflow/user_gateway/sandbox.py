"""Timeout + exception isolation around user-window calls."""

from __future__ import annotations

import ast
import contextvars
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

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


class EvalCancelled(Exception):
    """The eval thread should leave the current question."""


_EVAL_CANCEL: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("science_eval_cancel", default=None)


def bind_eval_cancel(event: Optional[Any]) -> contextvars.Token:
    return _EVAL_CANCEL.set(event)


def reset_eval_cancel(token: contextvars.Token) -> None:
    _EVAL_CANCEL.reset(token)


def eval_cancelled() -> bool:
    event = _EVAL_CANCEL.get()
    return bool(event is not None and event.is_set())


def reap_search_browsers() -> int:
    """Kill Chrome and ChromeDriver left behind by a hung Google navigation.

    ``driver.get`` does not return when www.google.com never connects, and a
    timed-out worker thread cannot reap its own browser. Match only the
    profile directory this wrap creates and undetected-chromedriver.
    """
    proc = Path("/proc")
    if not proc.is_dir():
        return 0
    killed = 0
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if "chrome_profile/" not in raw and "undetected_chromedriver" not in raw:
            continue
        try:
            os.kill(int(entry.name), signal.SIGKILL)
            killed += 1
        except OSError:
            pass
    return killed


def run_isolated(fn: Callable[[], Any], *, timeout_s: float = 20.0) -> Any:
    if timeout_s <= 0:
        return fn()
    pool = ThreadPoolExecutor(max_workers=1)
    # Copy ContextVar bindings (eval window logger) into the worker thread.
    ctx = contextvars.copy_context()
    fut = pool.submit(ctx.run, fn)
    deadline = time.monotonic() + float(timeout_s)
    try:
        while True:
            if eval_cancelled():
                reap_search_browsers()
                raise EvalCancelled("用户停止")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reap_search_browsers()
                raise TimeoutError(f"user window timed out after {timeout_s}s")
            try:
                return fut.result(timeout=min(0.4, remaining))
            except FuturesTimeout:
                continue
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def scan_tree(root: Path, pattern: str = "*.py") -> Iterable[Path]:
    if not root.is_dir():
        return []
    return root.rglob(pattern)
