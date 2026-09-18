#!/usr/bin/env python3
"""Assert workflow/ does not import agentlightning or verl."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / "workflow"
FORBIDDEN = {"agentlightning", "verl", "ray", "lit_tir_agent"}


def imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def main() -> None:
    errs: list[str] = []
    for path in WF.rglob("*.py"):
        mods = imports_of(path)
        for f in FORBIDDEN:
            if f in mods:
                errs.append(f"{path.relative_to(ROOT)} imports {f}")
    if errs:
        print("FAIL check_workflow_deps:")
        for e in errs:
            print(" ", e)
        sys.exit(1)
    print("OK check_workflow_deps")


if __name__ == "__main__":
    main()
