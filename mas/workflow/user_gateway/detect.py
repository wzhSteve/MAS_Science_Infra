"""Detect uploaded MAS family so Mode 2 can pick a real PEV wrap."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .paths import project_dir

PEV_FAMILIES = ("epc_aw", "hive")


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


def detect_mas_family(project_id: str, title: str = "") -> str:
    """Return ``epc_aw``, ``hive``, or ``""``."""
    hint = f"{project_id} {title}".lower().replace("-", "_")
    upload = project_dir(project_id) / "upload"
    blob = _walk_markers(upload)
    if "epc_aw" in blob or "epc_aw" in hint or "epcaw" in hint.replace("_", ""):
        return "epc_aw"
    if "/hive/" in blob or blob.endswith("hive") or " hive/" in blob or "hive" in hint:
        if "diagnoser" in blob or "planner" in blob or "hive" in hint:
            return "hive"
    if "diagnoser" in blob and "planner" in blob and "executor" in blob and "solver.py" in blob:
        return "epc_aw"
    return ""


def find_epc_aw_import_root(project_id: Optional[str] = None) -> Optional[Path]:
    """Directory that contains the ``MAS`` package (so ``MAS.epc_aw`` imports)."""
    import os

    env = os.environ.get("SCIENCE_EPC_AW_MAS_DIR", "").strip()
    if env:
        path = Path(env).expanduser()
        if path.is_dir():
            return path if (path / "MAS").is_dir() else path.parent
    search: list[Path] = []
    if project_id:
        search.append(project_dir(project_id) / "upload")
    search.append(Path(__file__).resolve())
    search.append(Path.cwd())
    for start in search:
        if start.is_dir():
            for solver in start.rglob("epc_aw/solver.py"):
                mas = solver.parent.parent
                if mas.name == "MAS":
                    return mas.parent
        for parent in [start, *getattr(start, "parents", [])]:
            cand = parent / "ref_Rep" / "EPC-AW"
            if (cand / "MAS" / "epc_aw" / "solver.py").is_file():
                return cand
    return None
