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


def _as_mapping(value: Any) -> Dict[str, Any]:
    """Accept a dict or a stringified dict from a parquet struct column."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return {}
            if isinstance(parsed, dict):
                return parsed
    return {}


def _gold_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else " | ".join(parts)
    return str(value).strip()


def fill_open_task_fields(row: Dict[str, Any]) -> None:
    """Fill empty question/answer/source from the verl 10k schema.

    ``train_10k.parquet`` and ``valid.parquet`` store the question in
    ``extra_info.question`` and the label in ``reward_model.ground_truth``.
    Rows that already carry those fields are left alone. The UI test path
    and the training dataloader both go through this mapping.
    """
    extra = _as_mapping(row.get("extra_info"))
    reward = _as_mapping(row.get("reward_model"))
    if not str(row.get("question") or row.get("query") or "").strip():
        question = str(extra.get("question") or "").strip()
        if question:
            row["question"] = question
    if not str(row.get("answer") or "").strip():
        gold = _gold_text(reward.get("ground_truth"))
        if gold:
            row["answer"] = gold
    if not str(row.get("source") or "").strip():
        source = str(row.get("data_source") or "").strip()
        if source:
            row["source"] = source


def normalize_task_row(row: Dict[str, Any], path: Path) -> Dict[str, Any]:
    """Map one parquet or JSON record onto id/question/answer/answers/source."""
    row = dict(row)
    fill_open_task_fields(row)
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
    task = row
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
