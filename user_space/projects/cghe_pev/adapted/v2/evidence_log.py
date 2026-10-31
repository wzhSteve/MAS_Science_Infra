"""SQLite trace for CGHE-v2. Separate from the v1 evidence tables."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS observation (
    observation_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL DEFAULT '',
    environment_id TEXT NOT NULL DEFAULT '',
    round_id TEXT NOT NULL DEFAULT '',
    tool TEXT NOT NULL DEFAULT '',
    resource TEXT NOT NULL DEFAULT '',
    identity TEXT NOT NULL DEFAULT '',
    parameter TEXT NOT NULL DEFAULT '',
    argument TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypothesis (
    hypothesis_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL DEFAULT '',
    environment_id TEXT NOT NULL DEFAULT '',
    round_id TEXT NOT NULL DEFAULT '',
    tool TEXT NOT NULL DEFAULT '',
    argument TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS feedback_prediction (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL DEFAULT '',
    round_id TEXT NOT NULL DEFAULT '',
    action_id TEXT NOT NULL DEFAULT '',
    epv REAL,
    cost REAL,
    risk REAL,
    score REAL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS prune_decision (
    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL DEFAULT '',
    environment_id TEXT NOT NULL DEFAULT '',
    round_id TEXT NOT NULL DEFAULT '',
    constraint_id TEXT NOT NULL DEFAULT '',
    tool TEXT NOT NULL DEFAULT '',
    resource TEXT NOT NULL DEFAULT '',
    identity TEXT NOT NULL DEFAULT '',
    parameter TEXT NOT NULL DEFAULT '',
    argument TEXT NOT NULL DEFAULT '',
    active_before INTEGER NOT NULL,
    active_after INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class V2EvidenceLog:
    """Append-only audit log. Cross-task reads must match every supplied scope field."""

    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def record_observation(self, **fields: Any) -> None:
        self._insert("observation", "observation_id", fields)

    def record_hypothesis(self, **fields: Any) -> None:
        self._insert("hypothesis", "hypothesis_id", fields)

    def record_feedback(self, **fields: Any) -> None:
        payload = dict(fields)
        self._conn.execute(
            """INSERT INTO feedback_prediction
               (task_id, round_id, action_id, epv, cost, risk, score, created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(payload.get("task_id") or ""),
                str(payload.get("round_id") or ""),
                str(payload.get("action_id") or ""),
                payload.get("epv"),
                payload.get("cost"),
                payload.get("risk"),
                payload.get("score"),
                _now(),
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()

    def record_prune(self, **fields: Any) -> None:
        payload = dict(fields)
        self._conn.execute(
            """INSERT INTO prune_decision
               (task_id, environment_id, round_id, constraint_id, tool, resource, identity,
                parameter, argument, active_before, active_after, created_at, payload)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                str(payload.get("task_id") or ""),
                str(payload.get("environment_id") or ""),
                str(payload.get("round_id") or ""),
                str(payload.get("constraint_id") or ""),
                str(payload.get("tool") or ""),
                str(payload.get("resource") or ""),
                str(payload.get("identity") or ""),
                str(payload.get("parameter") or ""),
                str(payload.get("argument") or ""),
                int(payload.get("active_before") or 0),
                int(payload.get("active_after") or 0),
                _now(),
                json.dumps(payload, ensure_ascii=False, default=str),
            ),
        )
        self._conn.commit()

    def _insert(self, table: str, key: str, fields: dict[str, Any]) -> None:
        payload = dict(fields)
        columns = {
            "observation": (
                "observation_id", "task_id", "environment_id", "round_id", "tool",
                "resource", "identity", "parameter", "argument", "source", "created_at", "payload",
            ),
            "hypothesis": (
                "hypothesis_id", "task_id", "environment_id", "round_id", "tool",
                "argument", "status", "created_at", "payload",
            ),
        }[table]
        values = [str(payload.get(column) or "") for column in columns[:-2]]
        values.append(_now())
        values.append(json.dumps(payload, ensure_ascii=False, default=str))
        placeholders = ", ".join("?" for _ in columns)
        names = ", ".join(columns)
        self._conn.execute(
            f"INSERT OR REPLACE INTO {table} ({names}) VALUES ({placeholders})",
            values,
        )
        self._conn.commit()

    def matching_constraints(self, **scope: str) -> list[dict[str, Any]]:
        """Return prune rows whose stored scope equals every non-empty query field."""
        rows = self._conn.execute("SELECT payload FROM prune_decision").fetchall()
        found = []
        for (raw,) in rows:
            item = json.loads(raw)
            if _scope_matches(item, scope):
                found.append(item)
        return found


def _scope_matches(item: dict[str, Any], scope: dict[str, str]) -> bool:
    for key, expected in scope.items():
        if not str(expected or "").strip():
            continue
        if str(item.get(key) or "") != str(expected):
            return False
    return True
