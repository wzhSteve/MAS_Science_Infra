"""Detect uploaded MAS family so Mode 2 can pick a real PEV wrap."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .paths import project_dir

PEV_FAMILIES = ("epc_aw", "agentflow")


def _walk_markers(root: Path) -> str:
    if not root.is_dir():
        return ""
    parts: list[str] = []
    for path in root.rglob("*"):
        rel = path.as_posix().lower()
        parts.append(rel)
        if len(parts) >= 400:
            break
    return "\n".join(parts)


def detect_mas_family(project_id: str, title: str = "", wraps: str = "") -> str:
    """Return ``epc_aw``, ``agentflow``, or ``""``."""
    hint = f"{project_id} {title} {wraps}".lower().replace("-", "_")
    upload = project_dir(project_id) / "upload"
    blob = _walk_markers(upload)
    if "agentflow" in hint or "agent_flow" in hint or "agentflow" in blob:
        return "agentflow"
    if "epc_aw" in blob or "epc_aw" in hint or "epcaw" in hint.replace("_", ""):
        return "epc_aw"
    if "diagnoser" in blob and "planner" in blob and "executor" in blob and "solver.py" in blob:
        return "epc_aw"
    return ""


def find_epc_aw_import_root(project_id: Optional[str] = None) -> Optional[Path]:
    """Directory that contains the ``MAS`` package (so ``MAS.epc_aw`` imports).

    Only the project's ``upload/`` or ``SCIENCE_EPC_AW_MAS_DIR``. Does not walk
    the gateway tree or the process cwd.
    """
    import os

    env = os.environ.get("SCIENCE_EPC_AW_MAS_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path if (path / "MAS").is_dir() else path.parent
    if not project_id:
        return None
    upload = project_dir(project_id) / "upload"
    if not upload.is_dir():
        return None
    for solver in upload.rglob("epc_aw/solver.py"):
        mas = solver.parent.parent
        if mas.name == "MAS":
            return mas.parent
    return None
