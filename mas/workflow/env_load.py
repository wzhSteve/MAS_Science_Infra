"""Load Agent_Science_Infra/.env without importing science_infra (layer-safe).

workflow/ must stay free of agentlightning; this only touches os.environ.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED = False


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    override = os.environ.get("SCIENCE_INFRA_ENV", "").strip()
    if override:
        paths.append(Path(override).expanduser().resolve())
    # workflow/collector.py → …/MAS_Science_Infra
    here = Path(__file__).resolve()
    # …/mas/workflow/env_load.py → parents[2] = MAS_Science_Infra
    if len(here.parents) >= 3:
        paths.append(here.parents[2] / ".env")
    cwd = Path.cwd()
    paths.append(cwd / ".env")
    # walk up a few levels from cwd
    for p in cwd.parents:
        paths.append(p / ".env")
        if p.name == "Agent_Science_Infra" or (p / "science_infra").is_dir():
            break
    # dedupe
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        try:
            r = p.resolve()
        except OSError:
            continue
        if r in seen:
            continue
        seen.add(r)
        out.append(r)
    return out


def _strip_quotes(value: str) -> str:
    v = value.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _parse(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, _, rest = line.partition("=")
        key = key.strip()
        if not key:
            continue
        val = rest.strip()
        if val and val[0] not in ("'", '"') and " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        out[key] = _strip_quotes(val)
    return out


def load_repo_dotenv(*, override: bool = False) -> Optional[Path]:
    """Find and load the first existing ``.env``. Existing non-empty env wins."""
    global _LOADED
    if _LOADED and not override:
        return None
    for path in _candidate_paths():
        if not path.is_file():
            continue
        # Prefer a file that sits next to science_infra/ when multiple exist
        data = _parse(path)
        for key, value in data.items():
            if override:
                os.environ[key] = value
            elif key not in os.environ or os.environ.get(key, "") == "":
                os.environ[key] = value
        _LOADED = True
        return path
    _LOADED = True
    return None
