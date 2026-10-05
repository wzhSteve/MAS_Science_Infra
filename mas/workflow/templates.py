"""Load MAS canvas templates from ``mas/specs/templates/*.yaml``.

Each file stem is the template id. An optional top-level ``label`` is stripped
before the remaining mapping is a MASSpec workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "specs" / "templates"

# Display order in the GraphPalette 模板 tab (must match files on disk).
TEMPLATE_ORDER = [
    "centralized",
    "tir_five_tools",
    "pev_python",
    "pev_search",
    "fanout_parallel",
    "blank_set",
    "score_robin",
]

TEMPLATE_LABELS = {
    "tir_five_tools": "单策略五工具（APPO / ARPO）",
    "centralized": "中心化 PEV + Router",
    "pev_python": "最小 PEV（python_coder）",
    "pev_search": "检索 PEV（wiki / google / web）",
    "fanout_parallel": "双 Router 并行 fan-out",
    "blank_set": "Router 候选含空白 Agent",
    "score_robin": "score 专家组 + round_robin",
}


def _read_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        from .spec import _parse_simple_yaml

        raw = _parse_simple_yaml(text)
    else:
        raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"template must be a mapping: {path}")
    return raw


def load_template_workflow(template_id: str) -> Dict[str, Any]:
    """Return a workflow dict (no ``label``) for ``MASSpec.model_validate``."""
    path = TEMPLATES_DIR / f"{template_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"MAS template not found: {path}")
    raw = dict(_read_yaml(path))
    raw.pop("label", None)
    return raw


def list_palette_templates() -> List[Dict[str, Any]]:
    """Palette entries: ``{id, label, workflow}`` in ``TEMPLATE_ORDER``."""
    found = {p.stem: p for p in TEMPLATES_DIR.glob("*.yaml") if p.stem != "_index"}
    out: List[Dict[str, Any]] = []
    seen = set()
    for tid in TEMPLATE_ORDER:
        if tid not in found:
            continue
        raw = dict(_read_yaml(found[tid]))
        label = str(raw.pop("label", None) or TEMPLATE_LABELS.get(tid) or tid)
        out.append({"id": tid, "label": label, "workflow": raw})
        seen.add(tid)
    for tid, path in sorted(found.items()):
        if tid in seen:
            continue
        raw = dict(_read_yaml(path))
        label = str(raw.pop("label", None) or TEMPLATE_LABELS.get(tid) or tid)
        out.append({"id": tid, "label": label, "workflow": raw})
    return out
