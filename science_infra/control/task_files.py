"""Load MAS task rows from parquet or HIVE-style JSON."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List


def resolve_task_path(raw: str | Path) -> Path:
    """Find a dataset file, including repo ``data/`` and MAS-relative paths."""
    from science_infra.control.paths import repo_data_dir, tir_agent_root

    path = Path(raw).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.append(tir_agent_root() / path)
        candidates.append(tir_agent_root() / "data" / path.name)
        candidates.append(repo_data_dir() / path)
        if path.parts and path.parts[0] == "data":
            candidates.append(repo_data_dir() / Path(*path.parts[1:]))
        candidates.append(repo_data_dir() / path.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"dataset not found: {raw}")


def _default_source(path: Path) -> str:
    # hotpotqa/data/data.json keeps the dataset folder name.
    # data/hotpotqa.json is named by the file.
    if path.suffix.lower() == ".json" and path.name == "data.json" and path.parent.name == "data":
        return path.parent.parent.name or path.stem
    if path.suffix.lower() == ".json":
        return path.stem
    return "gsm8k"


def _answers(raw: Any) -> tuple[str, List[str]]:
    if isinstance(raw, list):
        texts = [str(item) for item in raw if str(item).strip()]
        return (texts[0] if texts else ""), texts
    if raw is None:
        return "", []
    text = str(raw)
    return text, [text] if text else []


def _task_id(row: Dict[str, Any]) -> str:
    for key in ("id", "task_id", "pid", "idx"):
        if key not in row or row[key] is None or str(row[key]) == "":
            continue
        return str(row[key])
    return ""


def normalize_task_row(row: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Map one parquet or JSON record onto id/question/answer/answers/source."""
    question = str(row.get("question") or row.get("query") or "")
    listed = row.get("answers")
    if isinstance(listed, str):
        try:
            listed = ast.literal_eval(listed)
        except (SyntaxError, ValueError):
            listed = None
    if isinstance(listed, list) and any(str(item).strip() for item in listed):
        answers = [str(item) for item in listed if str(item).strip()]
        raw_answer = row.get("answer")
        if isinstance(raw_answer, list):
            answer, _ = _answers(raw_answer)
        else:
            answer = str(raw_answer or answers[0])
    else:
        answer, answers = _answers(row.get("answer"))
    source = str(row.get("source") or "").strip() or _default_source(path)
    task = dict(row)
    task.update({
        "id": _task_id(row),
        "question": question,
        "answer": answer,
        "answers": answers,
        "source": source,
    })
    return task


def load_task_rows(raw: str | Path) -> List[Dict[str, Any]]:
    """Read every row from a ``.parquet`` or ``.json`` task file."""
    path = resolve_task_path(raw)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        records = pd.read_parquet(path).to_dict(orient="records")
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("data") or payload.get("items") or []
        if not isinstance(payload, list):
            raise ValueError(f"JSON dataset must be a list: {path}")
        records = payload
    else:
        raise ValueError(f"unsupported dataset file: {path}")
    return [normalize_task_row(row, path) for row in records if isinstance(row, dict)]
