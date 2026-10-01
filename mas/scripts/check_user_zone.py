#!/usr/bin/env python3
"""Assert management / user-zone import isolation."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAS = ROOT / "mas"
WF = MAS / "workflow"
USER_SPACE = Path(
    __import__("os").environ.get("SCIENCE_USER_SPACE_DIR", "") or (ROOT / "user_space")
).expanduser()

FORBIDDEN_IN_MAS = {"user_space"}
GATEWAY = WF / "user_gateway"
ALLOWED_USER_FROM = {
    "science_user",
    "workflow.protocol",
    "workflow.user_gateway.public",
}
FORBIDDEN_USER_TOP = {"science_infra", "rl", "agentlightning", "verl", "ray", "lit_tir_agent", "tools"}


def imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def main() -> None:
    errs: list[str] = []
    for path in WF.rglob("*.py"):
        if GATEWAY in path.parents or path.parent == GATEWAY:
            continue
        for mod in imports_of(path):
            if mod.split(".")[0] in FORBIDDEN_IN_MAS or mod == "user_space":
                errs.append(f"{path.relative_to(ROOT)} imports {mod}")
    adapted_root = USER_SPACE / "projects"
    if adapted_root.is_dir():
        for path in adapted_root.rglob("adapted/**/*.py"):
            if not path.is_file():
                continue
            for mod in imports_of(path):
                top = mod.split(".")[0]
                if top in FORBIDDEN_USER_TOP:
                    errs.append(f"{path} imports {mod}")
                if mod.startswith("workflow.user_gateway") and mod not in ALLOWED_USER_FROM:
                    errs.append(f"{path} imports {mod}")
    if errs:
        print("FAIL check_user_zone:")
        for e in errs:
            print(" ", e)
        sys.exit(1)
    print("OK check_user_zone")


if __name__ == "__main__":
    main()
