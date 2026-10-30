"""SQLite memory for verified facts, verified counterexamples, and hypotheses.

The connection stays on this object and is omitted from ``export_state``.
Callers persist ``db_path`` and reopen the file. ``m_cf`` rows
(``UNVERIFIED_HYPOTHESIS``) are stored for diagnosis and are never returned
by the pruning query.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from cghe_types import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    Constraint,
    EvidenceOrigin,
    EvidenceStatus,
    ExecutionObservation,
    Scope,
    constraint_from_dict,
    new_id,
    norm,
    to_plain,
)
from failure_analyzer import FailureAnalysis


_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence_record (
    record_id TEXT PRIMARY KEY,
    evidence_status TEXT NOT NULL,
    verification_status TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    task_type TEXT NOT NULL DEFAULT '',
    task_goal TEXT NOT NULL DEFAULT '',
    environment TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    tool TEXT NOT NULL DEFAULT '',
    command TEXT NOT NULL DEFAULT '',
    observation_json TEXT NOT NULL,
    constraint_json TEXT,
    scope_json TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    origin TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0,
    direct INTEGER NOT NULL DEFAULT 0,
    note TEXT NOT NULL DEFAULT '',
    round_id TEXT NOT NULL DEFAULT '',
    plan_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_status
    ON evidence_record(evidence_status);
CREATE INDEX IF NOT EXISTS idx_evidence_environment
    ON evidence_record(environment);
"""

_INFERENCE_SOURCES = frozenset({"llm_hypothesis", "llm", "inference"})


def default_evidence_db_path() -> Path:
    """``<project>/artifacts/cghe_evidence.sqlite``."""
    project = Path(__file__).resolve().parent.parent
    return project / "artifacts" / "cghe_evidence.sqlite"


def _json_dump(value: Any) -> str:
    return json.dumps(to_plain(value), ensure_ascii=False, sort_keys=True, default=str)


def _json_load(value: Optional[str]) -> Any:
    if not value:
        return None
    return json.loads(value)


@dataclass
class EvidenceRecord:
    """One stored execution fact. ``constraint`` is empty for a bare success."""

    record_id: str
    evidence_status: EvidenceStatus
    verification_status: str
    task_id: str
    task_type: str
    task_goal: str
    environment: str
    action: str
    tool: str
    command: str
    observation: dict[str, Any]
    constraint: Optional[Constraint]
    scope: Scope
    source: str
    origin: EvidenceOrigin
    confidence: float
    direct: bool
    note: str
    round_id: str = ""
    plan_id: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.evidence_status, str):
            self.evidence_status = EvidenceStatus(self.evidence_status)
        if isinstance(self.origin, str):
            self.origin = EvidenceOrigin(self.origin)
        self.confidence = float(self.confidence)
        self.direct = bool(self.direct)


@dataclass(frozen=True)
class PlanningRecords:
    """Records a planner may read before generating a universe.

    ``pruning_constraints`` is only the verified counterexamples. Hypotheses
    stay in ``diagnostic_hypotheses`` and are not constraints for elimination.
    """

    verified_successes: tuple[EvidenceRecord, ...] = ()
    verified_counterexamples: tuple[EvidenceRecord, ...] = ()
    diagnostic_hypotheses: tuple[EvidenceRecord, ...] = ()

    def pruning_constraints(self) -> tuple[Constraint, ...]:
        found: list[Constraint] = []
        for record in self.verified_counterexamples:
            if record.evidence_status != EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
                continue
            if record.constraint is None:
                continue
            if record.constraint.evidence_status != EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
                continue
            found.append(record.constraint)
        return tuple(found)


class EvidenceMemory:
    """Append-only evidence store. Reopen it from ``db_path`` after a restore."""

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        path = Path(db_path) if db_path else default_evidence_db_path()
        self.db_path = str(path)
        self.confidence_threshold = float(confidence_threshold)
        self._conn: Optional[sqlite3.Connection] = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connect()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_conn"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def export_state(self) -> dict[str, Any]:
        """Path plus plain records. The SQLite connection is not included."""
        return {
            "db_path": self.db_path,
            "confidence_threshold": self.confidence_threshold,
            "records": self.export_records(),
        }

    def export_records(self) -> list[dict[str, Any]]:
        return [to_plain(record) for record in self.records()]

    def records(self) -> tuple[EvidenceRecord, ...]:
        rows = self._connect().execute(
            "SELECT * FROM evidence_record ORDER BY created_at, record_id"
        ).fetchall()
        return tuple(self._row_to_record(row) for row in rows)

    def record_success(
        self,
        *,
        task_id: str = "",
        task_type: str = "",
        task_goal: str = "",
        environment: str = "",
        action: str = "",
        tool: str = "",
        command: str = "",
        scope: Optional[Scope] = None,
        observation: Optional[ExecutionObservation | Mapping[str, Any]] = None,
        source: str = "verifier",
        note: str = "",
        round_id: str = "",
        plan_id: str = "",
        subgoal: str = "",
    ) -> EvidenceRecord:
        """Store ``m+``. A success is not a pruning constraint."""
        env = norm(environment)
        if observation is not None and not env:
            env = norm(_obs_get(observation, "environment"))
        tool_name = norm(tool or _obs_get(observation, "tool_name"))
        domain = scope if isinstance(scope, Scope) else Scope(
            environment=env,
            tool=tool_name,
            resource=norm(_obs_get(observation, "resource")),
            identity=norm(_obs_get(observation, "identity")),
            task_type=norm(task_type),
        )
        return self._insert(
            evidence_status=EvidenceStatus.VERIFIED_SUCCESS,
            task_id=task_id,
            task_type=task_type,
            task_goal=task_goal or subgoal,
            environment=env or domain.environment,
            action=action or subgoal or tool_name,
            tool=tool_name or domain.tool,
            command=command or str(_obs_get(observation, "command") or ""),
            observation=_observation_payload(observation),
            constraint=None,
            scope=domain,
            source=source,
            origin=EvidenceOrigin.DIRECT_OBSERVATION,
            confidence=1.0,
            direct=True,
            note=note or "verified success",
            round_id=round_id,
            plan_id=plan_id or str(_obs_get(observation, "plan_id") or ""),
        )

    def record_counterexample(
        self,
        constraint: Constraint,
        *,
        task_id: str = "",
        task_type: str = "",
        task_goal: str = "",
        environment: str = "",
        action: str = "",
        tool: str = "",
        command: str = "",
        observation: Optional[ExecutionObservation | Mapping[str, Any]] = None,
        round_id: str = "",
        plan_id: str = "",
        note: str = "",
        record_id: str = "",
    ) -> EvidenceRecord:
        """Store ``m-``. The row is rejected unless the constraint is verified."""
        if constraint.evidence_status != EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
            raise ValueError("m- requires VERIFIED_COUNTEREXAMPLE")
        if not constraint.direct or constraint.origin != EvidenceOrigin.DIRECT_OBSERVATION:
            raise ValueError("m- requires a direct observation")
        env = norm(environment or constraint.scope.environment)
        return self._insert(
            evidence_status=EvidenceStatus.VERIFIED_COUNTEREXAMPLE,
            task_id=task_id,
            task_type=task_type,
            task_goal=task_goal,
            environment=env,
            action=action or constraint.condition.predicate,
            tool=norm(tool or constraint.scope.tool or constraint.condition.tool),
            command=command or str(_obs_get(observation, "command") or ""),
            observation=_observation_payload(observation, constraint=constraint),
            constraint=constraint,
            scope=constraint.scope,
            source=constraint.source,
            origin=constraint.origin,
            confidence=constraint.confidence,
            direct=True,
            note=note or constraint.note,
            round_id=round_id,
            plan_id=plan_id or str(_obs_get(observation, "plan_id") or ""),
            record_id=record_id or constraint.evidence_id or constraint.constraint_id,
        )

    def record_hypothesis(
        self,
        *,
        task_id: str = "",
        task_type: str = "",
        task_goal: str = "",
        environment: str = "",
        action: str = "",
        tool: str = "",
        command: str = "",
        observation: Optional[ExecutionObservation | Mapping[str, Any]] = None,
        constraint: Optional[Constraint] = None,
        scope: Optional[Scope] = None,
        source: str = "",
        origin: EvidenceOrigin = EvidenceOrigin.INFERENCE,
        note: str = "",
        round_id: str = "",
        plan_id: str = "",
        record_id: str = "",
    ) -> EvidenceRecord:
        """Store ``m_cf``. Status is forced to unverified."""
        stored = _as_hypothesis(constraint)
        env = norm(environment or (stored.scope.environment if stored else "") or _obs_get(observation, "environment"))
        domain = stored.scope if stored is not None else (scope or Scope(environment=env, tool=norm(tool)))
        return self._insert(
            evidence_status=EvidenceStatus.UNVERIFIED_HYPOTHESIS,
            task_id=task_id,
            task_type=task_type,
            task_goal=task_goal,
            environment=env,
            action=action or (stored.condition.predicate if stored else ""),
            tool=norm(tool or domain.tool or _obs_get(observation, "tool_name")),
            command=command or str(_obs_get(observation, "command") or ""),
            observation=_observation_payload(observation, constraint=stored),
            constraint=stored,
            scope=domain,
            source=source or (stored.source if stored else ""),
            origin=origin if stored is None else stored.origin,
            confidence=0.0,
            direct=False,
            note=note or (stored.note if stored else "unverified hypothesis"),
            round_id=round_id,
            plan_id=plan_id or str(_obs_get(observation, "plan_id") or ""),
            record_id=record_id,
        )

    def record_analysis(
        self,
        analysis: FailureAnalysis,
        *,
        task_id: str = "",
        task_type: str = "",
        task_goal: str = "",
        environment: str = "",
        action: str = "",
        command: str = "",
        observation: Optional[ExecutionObservation] = None,
        round_id: str = "",
    ) -> EvidenceRecord:
        """Persist analyzer output without promoting a hypothesis."""
        evidence = analysis.evidence
        env = norm(environment or evidence.environment)
        shared = dict(
            task_id=task_id,
            task_type=task_type,
            task_goal=task_goal,
            environment=env,
            action=action or evidence.observation_code or evidence.tool,
            tool=evidence.tool,
            command=command,
            observation=observation,
            round_id=round_id,
            plan_id=evidence.plan_id,
            record_id=evidence.evidence_id,
        )
        if analysis.verified_counterexample and analysis.constraint is not None:
            return self.record_counterexample(analysis.constraint, note=evidence.note, **shared)
        return self.record_hypothesis(
            constraint=analysis.constraint,
            source=evidence.source,
            origin=evidence.origin,
            note=evidence.note,
            **shared,
        )

    def constraint_applies(
        self,
        constraint: Optional[Constraint],
        *,
        environment: str,
        context: Optional[Scope] = None,
        task_type: str = "",
        record_task_type: str = "",
    ) -> bool:
        """Re-check a stored counterexample against the current environment.

        A field left empty on ``context`` does not reject the row. A field
        that the caller did set must agree with the constraint scope. An
        unverified hypothesis never applies.
        """
        if constraint is None:
            return False
        if constraint.evidence_status != EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
            return False
        if not constraint.direct or constraint.origin != EvidenceOrigin.DIRECT_OBSERVATION:
            return False
        if norm(constraint.source) in _INFERENCE_SOURCES:
            return False
        if constraint.confidence < self.confidence_threshold:
            return False
        if constraint.parameter_level and constraint.disables_tool:
            return False
        scope = constraint.scope.normalized()
        if not scope.environment or not scope.tool:
            return False
        current_env = norm(environment) or (context.normalized().environment if context is not None else "")
        if not current_env or scope.environment != current_env:
            return False
        recorded_task = norm(record_task_type)
        current_task = norm(task_type)
        if recorded_task and current_task and recorded_task != current_task:
            return False
        if scope.task_type and current_task and scope.task_type != current_task:
            return False
        if context is None:
            return True
        ctx = context.normalized()
        if ctx.environment and ctx.environment != scope.environment:
            return False
        for name in (
            "tool",
            "resource",
            "identity",
            "parameter",
            "argument",
            "schema_signature",
            "task_type",
            "call_instance",
        ):
            required = getattr(scope, name)
            actual = getattr(ctx, name)
            if required and actual and actual != required:
                return False
        return True

    def constraints_for_pruning(
        self,
        *,
        environment: str,
        context: Optional[Scope] = None,
        task_type: str = "",
    ) -> tuple[Constraint, ...]:
        """Verified, scope-matched counterexamples only.

        ``UNVERIFIED_HYPOTHESIS`` and ``VERIFIED_SUCCESS`` are omitted.
        """
        found: list[Constraint] = []
        for record in self.records():
            if record.evidence_status != EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
                continue
            if not self.constraint_applies(
                record.constraint,
                environment=environment,
                context=context,
                task_type=task_type,
                record_task_type=record.task_type,
            ):
                continue
            assert record.constraint is not None
            found.append(record.constraint)
        return tuple(found)

    def records_for_planning(
        self,
        *,
        environment: str,
        context: Optional[Scope] = None,
        task_type: str = "",
    ) -> PlanningRecords:
        """Relevant memory before a new universe is generated.

        Hypotheses are returned separately so a planner can use them as
        diagnostic candidates. They are not pruning constraints.
        """
        successes: list[EvidenceRecord] = []
        counterexamples: list[EvidenceRecord] = []
        hypotheses: list[EvidenceRecord] = []
        for record in self.records():
            if not _same_environment(record.environment, environment):
                continue
            if record.task_type and task_type and norm(record.task_type) != norm(task_type):
                continue
            if record.evidence_status == EvidenceStatus.VERIFIED_SUCCESS:
                if context is not None and not _scope_compatible(record.scope, context):
                    continue
                successes.append(record)
            elif record.evidence_status == EvidenceStatus.VERIFIED_COUNTEREXAMPLE:
                if self.constraint_applies(
                    record.constraint,
                    environment=environment,
                    context=context,
                    task_type=task_type,
                    record_task_type=record.task_type,
                ):
                    counterexamples.append(record)
            elif record.evidence_status == EvidenceStatus.UNVERIFIED_HYPOTHESIS:
                hypotheses.append(record)
        return PlanningRecords(
            verified_successes=tuple(successes),
            verified_counterexamples=tuple(counterexamples),
            diagnostic_hypotheses=tuple(hypotheses),
        )

    def _insert(
        self,
        *,
        evidence_status: EvidenceStatus,
        task_id: str,
        task_type: str,
        task_goal: str,
        environment: str,
        action: str,
        tool: str,
        command: str,
        observation: dict[str, Any],
        constraint: Optional[Constraint],
        scope: Scope,
        source: str,
        origin: EvidenceOrigin,
        confidence: float,
        direct: bool,
        note: str,
        round_id: str,
        plan_id: str,
        record_id: str = "",
    ) -> EvidenceRecord:
        if isinstance(evidence_status, str):
            evidence_status = EvidenceStatus(evidence_status)
        if evidence_status == EvidenceStatus.UNVERIFIED_HYPOTHESIS:
            direct = False
            confidence = 0.0
            constraint = _as_hypothesis(constraint)
        record = EvidenceRecord(
            record_id=record_id or new_id("m"),
            evidence_status=evidence_status,
            verification_status=evidence_status.value,
            task_id=str(task_id or ""),
            task_type=norm(task_type),
            task_goal=str(task_goal or ""),
            environment=norm(environment),
            action=str(action or ""),
            tool=norm(tool),
            command=str(command or ""),
            observation=dict(observation or {}),
            constraint=constraint,
            scope=scope.normalized() if isinstance(scope, Scope) else Scope(),
            source=str(source or ""),
            origin=origin if isinstance(origin, EvidenceOrigin) else EvidenceOrigin(origin),
            confidence=float(confidence),
            direct=bool(direct),
            note=str(note or ""),
            round_id=str(round_id or ""),
            plan_id=str(plan_id or ""),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._connect().execute(
            """
            INSERT OR REPLACE INTO evidence_record (
                record_id, evidence_status, verification_status, task_id, task_type,
                task_goal, environment, action, tool, command, observation_json,
                constraint_json, scope_json, source, origin, confidence, direct,
                note, round_id, plan_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.record_id,
                record.evidence_status.value,
                record.verification_status,
                record.task_id,
                record.task_type,
                record.task_goal,
                record.environment,
                record.action,
                record.tool,
                record.command,
                _json_dump(record.observation),
                _json_dump(record.constraint) if record.constraint is not None else None,
                _json_dump(record.scope),
                record.source,
                record.origin.value,
                record.confidence,
                1 if record.direct else 0,
                record.note,
                record.round_id,
                record.plan_id,
                record.created_at,
            ),
        )
        self._connect().commit()
        return record

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        return self._conn

    def _row_to_record(self, row: sqlite3.Row) -> EvidenceRecord:
        raw_constraint = _json_load(row["constraint_json"])
        raw_scope = _json_load(row["scope_json"]) or {}
        constraint = constraint_from_dict(raw_constraint) if isinstance(raw_constraint, dict) else None
        return EvidenceRecord(
            record_id=row["record_id"],
            evidence_status=EvidenceStatus(row["evidence_status"]),
            verification_status=row["verification_status"],
            task_id=row["task_id"],
            task_type=row["task_type"],
            task_goal=row["task_goal"],
            environment=row["environment"],
            action=row["action"],
            tool=row["tool"],
            command=row["command"],
            observation=_json_load(row["observation_json"]) or {},
            constraint=constraint,
            scope=Scope(**{key: raw_scope.get(key, "") for key in Scope.__dataclass_fields__}),
            source=row["source"],
            origin=EvidenceOrigin(row["origin"]),
            confidence=row["confidence"],
            direct=bool(row["direct"]),
            note=row["note"],
            round_id=row["round_id"],
            plan_id=row["plan_id"],
            created_at=row["created_at"],
        )


def _as_hypothesis(constraint: Optional[Constraint]) -> Optional[Constraint]:
    """Copy a constraint and force the unverified status. The input is unchanged."""
    if constraint is None:
        return None
    return replace(
        constraint,
        evidence_status=EvidenceStatus.UNVERIFIED_HYPOTHESIS,
        confidence=0.0,
        direct=False,
        disables_tool=False,
    )


def _observation_payload(
    observation: Optional[ExecutionObservation | Mapping[str, Any]],
    *,
    constraint: Optional[Constraint] = None,
) -> dict[str, Any]:
    if observation is None:
        payload: dict[str, Any] = {}
    elif isinstance(observation, ExecutionObservation):
        payload = {
            "tool_name": observation.tool_name,
            "resource": observation.resource,
            "identity": observation.identity,
            "environment": observation.environment,
            "parameters": dict(observation.parameters),
            "tool_result": observation.tool_result,
            "execution_log": observation.execution_log,
            "llm_hypothesis": observation.llm_hypothesis,
            "verifier_label": observation.verifier_label,
            "plan_id": observation.plan_id,
            "node_id": observation.node_id,
            "command": observation.command,
            "call_instance": observation.call_instance,
        }
    else:
        payload = dict(observation)
    if constraint is not None:
        payload.setdefault("observation_code", constraint.observation_code)
        payload.setdefault("source", constraint.source)
    return payload


def _obs_get(observation: Optional[ExecutionObservation | Mapping[str, Any]], key: str) -> Any:
    if observation is None:
        return ""
    if isinstance(observation, Mapping):
        return observation.get(key, "")
    return getattr(observation, key, "")


def _same_environment(recorded: str, current: str) -> bool:
    left = norm(recorded)
    right = norm(current)
    if not left or not right:
        return False
    return left == right


def _scope_compatible(scope: Scope, context: Scope) -> bool:
    """Reject a success whose stated scope contradicts the current context."""
    recorded = scope.normalized()
    current = context.normalized()
    for name in Scope.__dataclass_fields__:
        required = getattr(recorded, name)
        actual = getattr(current, name)
        if required and actual and required != actual:
            return False
    return True
