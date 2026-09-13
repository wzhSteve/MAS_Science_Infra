"""Single-rollout HTTP adapter and local run storage, without reward or training."""

from __future__ import annotations

import errno
import base64
import binascii
import copy
import json
import logging
import os
import re
import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache
from threading import RLock
from typing import Any, Iterator, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from .experiments import exp_dir
from .llm_config import public_endpoint, resolve_llm_config
from .readiness import ensure_workflow_path, require_live

logger = logging.getLogger(__name__)
ExecutionMode = Literal["mock", "live"]
_INSTANCE_ID = uuid4().hex
_PRIVATE_FIELDS = {"api_key", "openai_api_key", "authorization", "password"}
_INDEX_LOCK = RLock()


class RunError(BaseModel):
    stage: str
    code: str
    message: str


class RunModel(BaseModel):
    source: str
    name: str
    policy_version: Optional[str] = None


class RolloutRunSummary(BaseModel):
    run_id: str
    trajectory_id: Optional[str] = None
    status: Literal["running", "succeeded", "failed", "interrupted"] = "running"
    execution: ExecutionMode
    started_at: str
    finished_at: Optional[str] = None
    model: Optional[RunModel] = None
    final_answer: Optional[str] = None
    termination_reason: Optional[str] = None
    format_ok: Optional[bool] = None
    model_call_count: int = 0
    tool_call_count: int = 0
    trace_status: Literal["complete", "partial", "unavailable"] = "unavailable"
    error: Optional[RunError] = None


class RunRecord(BaseModel):
    schema_version: Literal["1"] = "1"
    experiment_id: str
    server_instance_id: str
    workflow: dict[str, Any]
    task: dict[str, str]
    model_config_revision: Optional[str] = None
    model_binding: Optional[dict[str, Any]] = None
    redaction_applied: bool = False
    summary: RolloutRunSummary


class RolloutStorageError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_dir(experiment_id: str, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("无效的运行 ID。")
    root = exp_dir(experiment_id).resolve()
    path = (root / "artifacts" / "rollout-runs" / run_id).resolve()
    if not path.is_relative_to(root):
        raise ValueError("运行记录不属于当前实验。")
    return path


def _redact(value: Any, key: str = "") -> Any:
    if isinstance(value, str):
        return value.replace(key, "[redacted]") if key else value
    if isinstance(value, list):
        return [_redact(item, key) for item in value]
    if isinstance(value, dict):
        return {
            field: "[redacted]" if field.lower() in _PRIVATE_FIELDS else _redact(item, key)
            for field, item in value.items()
        }
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    except (OSError, ValueError, TypeError) as error:
        logger.error("Rollout record write failed: %s (%s)", path.name, type(error).__name__)
        raise RolloutStorageError("运行记录写入失败，不能确认轨迹已完整保存。") from error
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _run_lease(path: Path) -> Iterator[bool]:
    """An OS-held lease survives multiple workers and releases on process exit."""
    stream = path.open("r+b")
    try:
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                raise
            yield False
            return
        try:
            yield True
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        stream.close()


def _load_record(path: Path, experiment_id: str, run_id: str) -> RunRecord:
    try:
        record = RunRecord.model_validate_json((path / "run.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError("运行记录不存在。") from None
    except (ValidationError, OSError) as error:
        raise RolloutStorageError("运行记录无法读取或已损坏。") from error
    if record.experiment_id != experiment_id or record.summary.run_id != run_id:
        raise RolloutStorageError("运行记录的身份信息不匹配。")
    return record


@contextmanager
def _index_connection(experiment_id: str) -> Iterator[sqlite3.Connection]:
    root = exp_dir(experiment_id).resolve()
    if not (root / "experiment.yaml").is_file():
        raise FileNotFoundError("实验不存在。")
    directory = (root / "artifacts" / "rollout-runs").resolve()
    if not directory.is_relative_to(root):
        raise ValueError("运行索引不属于当前实验。")
    directory.mkdir(parents=True, exist_ok=True)
    with _INDEX_LOCK:
        try:
            with closing(sqlite3.connect(directory / "index.sqlite3", timeout=10)) as connection, connection:
                connection.row_factory = sqlite3.Row
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, "
                    "question_preview TEXT NOT NULL, record_error TEXT)"
                )
                connection.execute("CREATE INDEX IF NOT EXISTS runs_order ON runs(started_at DESC, run_id DESC)")
                connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                if connection.execute("SELECT 1 FROM metadata WHERE key='backfilled'").fetchone() is None:
                    # One-time migration reads small summaries only, never full trajectories.
                    for path in directory.iterdir():
                        if not re.fullmatch(r"[0-9a-f]{32}", path.name) or not path.is_dir():
                            continue
                        try:
                            record = _load_record(_run_dir(experiment_id, path.name), experiment_id, path.name)
                            row = (path.name, record.summary.started_at, record.task.get("question", "")[:160], None)
                        except (FileNotFoundError, RolloutStorageError, ValueError) as error:
                            row = (path.name, "", "", str(error))
                        connection.execute("INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?)", row)
                    connection.execute("INSERT INTO metadata VALUES ('backfilled', '1')")
                yield connection
        except (sqlite3.Error, OSError) as error:
            raise RolloutStorageError("运行索引不可用，请检查本地存储。") from error


def _persist_record(path: Path, record: RunRecord) -> None:
    _write_json(path / "run.json", record.model_dump(mode="json"))
    with _index_connection(record.experiment_id) as connection:
        connection.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, NULL)",
            (record.summary.run_id, record.summary.started_at, record.task.get("question", "")[:160]),
        )


def list_rollouts(experiment_id: str, cursor: Optional[str] = None, limit: int = 15) -> dict[str, Any]:
    if not 1 <= limit <= 50:
        raise ValueError("每页数量必须在 1–50 之间。")
    boundary = None
    if cursor:
        try:
            boundary = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")).decode("utf-8"))
            if not isinstance(boundary, list) or len(boundary) != 2 or not all(isinstance(item, str) for item in boundary):
                raise ValueError()
        except (ValueError, UnicodeError, binascii.Error):
            raise ValueError("运行列表游标无效，请刷新列表。") from None
    query = "SELECT * FROM runs"
    parameters: list[Any] = []
    if boundary:
        query += " WHERE (started_at, run_id) < (?, ?)"
        parameters.extend(boundary)
    query += " ORDER BY started_at DESC, run_id DESC LIMIT ?"
    parameters.append(limit + 1)
    with _index_connection(experiment_id) as connection:
        rows = connection.execute(query, parameters).fetchall()
    items = []
    for row in rows[:limit]:
        try:
            record = get_rollout(experiment_id, row["run_id"])
            items.append({
                "run_id": row["run_id"], "run": record.summary.model_dump(mode="json"),
                "question_preview": record.task.get("question", "")[:160], "record_error": None,
            })
        except (FileNotFoundError, RolloutStorageError, ValueError) as error:
            items.append({"run_id": row["run_id"], "run": None, "question_preview": row["question_preview"], "record_error": str(error)})
    next_cursor = None
    if len(rows) > limit:
        last = rows[limit - 1]
        next_cursor = base64.urlsafe_b64encode(json.dumps([last["started_at"], last["run_id"]]).encode()).decode("ascii")
    return {"items": items, "next_cursor": next_cursor}


def _validated_spec(workflow: dict[str, Any]):
    ensure_workflow_path()
    from workflow.runtime import validate_execution_spec
    from workflow.spec import MASSpec

    try:
        spec = MASSpec.model_validate(workflow)
    except ValidationError as error:
        fields = [".".join(str(part) for part in issue["loc"]) for issue in error.errors(include_input=False)]
        raise ValueError("Workflow 配置格式无效：" + "、".join(fields)) from None
    try:
        validate_execution_spec(spec)
    except KeyError as error:
        raise ValueError(str(error)) from None
    return spec


def run_rollout(
    experiment_id: str, workflow: dict[str, Any], task: dict[str, str], execution: ExecutionMode,
) -> RolloutRunSummary:
    root = exp_dir(experiment_id)
    if not (root / "experiment.yaml").is_file():
        raise FileNotFoundError("实验不存在。")
    spec = _validated_spec(workflow)
    from workflow.archive import ArchiveWriteError

    run_id = uuid4().hex
    path = _run_dir(experiment_id, run_id)
    task = {"id": task.get("id") or uuid4().hex, "question": task["question"]}
    snapshot = spec.model_dump(mode="json", by_alias=True)
    snapshot["llm"]["base_url"] = public_endpoint(snapshot["llm"].get("base_url") or "")
    summary = RolloutRunSummary(run_id=run_id, execution=execution, started_at=_now())
    safe_snapshot = _redact(snapshot)
    record = RunRecord(
        experiment_id=experiment_id, server_instance_id=_INSTANCE_ID,
        workflow=safe_snapshot, task=task, summary=summary,
        redaction_applied=safe_snapshot != snapshot,
    )
    try:
        path.mkdir(parents=True, exist_ok=False)
        (path / ".lease").write_bytes(b"\0")
        with _run_lease(path / ".lease") as acquired:
            if not acquired:
                raise RolloutStorageError("无法取得本次运行的记录锁。")
            stage = "configuration"
            secret = ""
            try:
                llm = None
                config = None
                if execution == "live":
                    config = resolve_llm_config(experiment_id)
                    secret = config.api_key
                    record.workflow = _redact(record.workflow, secret)
                    record.task = _redact(record.task, secret)
                    summary.model = RunModel(source=config.kind, name=config.model)
                    record.model_config_revision = config.revision
                    record.model_binding = config.public()
                    record.redaction_applied = record.redaction_applied or record.task != task or record.workflow != snapshot
                _persist_record(path, record)
                if config is not None:
                    require_live(config)
                    from workflow.runtime import LLMConfig

                    llm = LLMConfig(
                        endpoint=config.base_url, model=config.model,
                        api_key=config.api_key, source=config.kind,
                        resource_id=config.resource_id, resource_revision=config.resource_revision,
                        resource_name=config.resource_name,
                    )
                stage = "runtime"
                from workflow.runtime import ExecutionService

                service = ExecutionService(
                    mock=execution == "mock", spec=spec, llm=llm,
                    archive_root=str(path / "archives"), run_id=run_id,
                )
                trajectory = service.run(task)
                raw_payload = trajectory.model_dump(mode="json")
                payload = _redact(raw_payload, secret)
                record.redaction_applied = record.redaction_applied or payload != raw_payload
                stage = "storage"
                _write_json(path / "trajectory.json", payload)
                meta = payload.get("meta", {})
                summary.trajectory_id = trajectory.trajectory_id
                summary.final_answer = payload.get("final_answer")
                summary.format_ok = trajectory.format_ok
                summary.model_call_count = sum(1 for event in payload["events"] if event["kind"] == "model_call")
                summary.tool_call_count = sum(1 for event in payload["events"] if event["kind"] == "tool_call")
                summary.trace_status = "complete" if meta.get("trace_status") == "complete" and not record.redaction_applied else "partial"
                summary.termination_reason = meta.get("termination_reason")
                if meta.get("status") == "succeeded" and not meta.get("error"):
                    summary.status = "succeeded"
                else:
                    summary.status = "failed"
                    summary.error = RunError(
                        stage=meta.get("error_stage") or "runtime",
                        code=meta.get("error_code") or "rollout_failed",
                        message=meta.get("error") or "运行未正常完成，请查看轨迹与终止原因。",
                    )
            except RolloutStorageError:
                raise
            except (ArchiveWriteError, OSError) as error:
                raise RolloutStorageError("执行轨迹写入失败，不能确认记录已完整保存。") from error
            except Exception as error:
                logger.warning("Rollout %s failed during %s (%s)", run_id, stage, type(error).__name__)
                summary.status = "failed"
                summary.termination_reason = f"{stage}_error"
                summary.error = RunError(stage=stage, code=type(error).__name__, message=_redact(str(error), secret))
            summary.finished_at = _now()
            record.summary = summary
            _persist_record(path, record)
    except RolloutStorageError as error:
        raise RolloutStorageError(f"运行 {run_id}：{error}") from error
    except OSError as error:
        raise RolloutStorageError(f"运行 {run_id} 的存储不可用，不能确认采集完成。") from error
    return summary


def get_rollout(experiment_id: str, run_id: str) -> RunRecord:
    path = _run_dir(experiment_id, run_id)
    record = _load_record(path, experiment_id, run_id)
    if record.summary.status != "running":
        return record
    try:
        with _run_lease(path / ".lease") as acquired:
            if not acquired:
                return record
            # Re-read under the lease: a writer may have just committed its result.
            record = _load_record(path, experiment_id, run_id)
            if record.summary.status == "running":
                record.summary.status = "interrupted"
                record.summary.finished_at = _now()
                record.summary.termination_reason = "executor_interrupted"
                record.summary.error = RunError(
                    stage="runtime", code="executor_interrupted",
                    message="执行进程已退出，运行被中断；不会自动重跑。",
                )
                _persist_record(path, record)
    except OSError as error:
        raise RolloutStorageError("无法读取运行存储状态。") from error
    return record


@lru_cache(maxsize=4)
def _read_trajectory(path: Path, modified: int, size: int) -> dict[str, Any]:
    del modified, size
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError("本次运行尚无已保存的轨迹，请查看运行摘要。") from None
    except (ValueError, OSError) as error:
        raise RolloutStorageError("轨迹记录无法读取或已损坏。") from error
    if (not isinstance(payload, dict) or not isinstance(payload.get("events", []), list)
            or not isinstance(payload.get("messages", []), list)
            or not isinstance(payload.get("branch_points", []), list)):
        raise RolloutStorageError("轨迹记录格式无效。")
    return payload


def _trajectory_payload(experiment_id: str, record: RunRecord) -> dict[str, Any]:
    path = _run_dir(experiment_id, record.summary.run_id) / "trajectory.json"
    try:
        stat = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError("本次运行尚无已保存的轨迹，请查看运行摘要。") from None
    except OSError as error:
        raise RolloutStorageError("轨迹文件无法读取。") from error
    payload = _read_trajectory(path, stat.st_mtime_ns, stat.st_size)
    if payload.get("trajectory_id") != record.summary.trajectory_id:
        raise RolloutStorageError("轨迹记录与本次运行不匹配。")
    return payload


def _preview(value: Any, depth: int = 0) -> tuple[Any, bool]:
    if isinstance(value, str):
        return (value[:800] + "…" if len(value) > 800 else value), len(value) > 800
    if isinstance(value, (dict, list)) and depth >= 8:
        return "[展开查看完整内容]", True
    if isinstance(value, list):
        values = [_preview(item, depth + 1) for item in value[:20]]
        return [item for item, _ in values], len(value) > 20 or any(truncated for _, truncated in values)
    if isinstance(value, dict):
        values = [(key, *_preview(item, depth + 1)) for key, item in list(value.items())[:60]]
        return {key: item for key, item, _ in values}, len(value) > 60 or any(truncated for _, _, truncated in values)
    return value, False


def _snapshot_summaries(experiment_id: str, run_id: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    path = _run_dir(experiment_id, run_id)
    archive = payload.get("archive") or {}
    archive_id = archive.get("archive_id") if isinstance(archive, dict) else None
    if not isinstance(archive_id, str) or not re.fullmatch(r"[0-9a-f]{32}", archive_id):
        return []
    result = []
    for branch in payload.get("branch_points", [])[:50]:
        if not isinstance(branch, dict):
            continue
        snapshot_id = branch.get("snapshot_id")
        if not isinstance(snapshot_id, str) or not re.fullmatch(r"[0-9a-f]{32}", snapshot_id):
            result.append({"snapshot_id": str(snapshot_id), "coverage": None, "error": "快照标识无效。"})
            continue
        file = (path / "archives" / archive_id / "snapshots" / f"{snapshot_id}.json").resolve()
        if not file.is_relative_to(path):
            raise RolloutStorageError("快照不属于当前运行。")
        try:
            snapshot = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(snapshot, dict) or snapshot.get("snapshot_id") != snapshot_id or snapshot.get("archive_id") != archive_id:
                raise ValueError("快照身份不匹配")
            coverage = snapshot.get("coverage") if isinstance(snapshot, dict) else None
            result.append({
                "snapshot_id": snapshot_id,
                "coverage": coverage if isinstance(coverage, dict) else None,
                "error": None if isinstance(coverage, dict) else "旧快照未记录状态覆盖范围。",
            })
        except (OSError, ValueError):
            result.append({"snapshot_id": snapshot_id, "coverage": None, "error": "快照无法读取或已损坏。"})
    return result


def get_trajectory(experiment_id: str, run_id: str, *, preview: bool = False, offset: int = 0, limit: int = 100) -> dict[str, Any]:
    record = get_rollout(experiment_id, run_id)
    payload = _trajectory_payload(experiment_id, record)
    extra: dict[str, Any] = {}
    if preview:
        events = payload.get("events", [])
        messages = payload.get("messages", [])
        if not 0 <= offset <= len(events) or not 1 <= limit <= 200:
            raise ValueError("轨迹分页参数无效。")
        rendered = []
        for event in events[offset:offset + limit]:
            if not isinstance(event, dict):
                raise RolloutStorageError("轨迹包含无效事件。")
            body, truncated = _preview(event.get("payload", {}))
            rendered.append({**event, "payload": body, "preview_truncated": truncated})
        metadata, metadata_truncated = _preview(payload.get("meta", {}))
        message_preview, message_truncated = _preview(messages[:20])
        payload = {
            **{key: value for key, value in payload.items() if key not in ("events", "messages", "meta", "task")},
            "task": record.task, "events": rendered, "messages": message_preview, "meta": metadata,
        }
        extra = {
            "preview": True,
            "metadata_truncated": metadata_truncated, "messages_truncated": message_truncated or len(messages) > 20,
            "event_page": {"offset": offset, "total": len(events), "next_offset": offset + limit if offset + limit < len(events) else None},
            "message_page": {"offset": 0, "total": len(messages), "next_offset": 20 if len(messages) > 20 else None},
        }
    return {
        "run": record.summary.model_dump(mode="json"),
        "workflow": record.workflow,
        "task": record.task,
        "model_binding": record.model_binding,
        "trajectory": payload,
        "redaction_applied": record.redaction_applied,
        "snapshots": _snapshot_summaries(experiment_id, run_id, payload),
        **extra,
    }


def get_rollout_context(experiment_id: str, run_id: str) -> dict[str, Any]:
    record = get_rollout(experiment_id, run_id)
    return {
        "run": record.summary.model_dump(mode="json"), "workflow": record.workflow, "task": record.task,
        "model_binding": record.model_binding, "redaction_applied": record.redaction_applied,
    }


def get_trajectory_event(experiment_id: str, run_id: str, event_id: str) -> dict[str, Any]:
    record = get_rollout(experiment_id, run_id)
    payload = _trajectory_payload(experiment_id, record)
    matches = [event for event in payload.get("events", []) if isinstance(event, dict) and event.get("event_id") == event_id]
    if not matches:
        raise FileNotFoundError("事件不存在。")
    if len(matches) != 1:
        raise RolloutStorageError("事件标识重复，无法确定对应记录。")
    return {"event": matches[0]}


def get_trajectory_messages(experiment_id: str, run_id: str, offset: int = 0, limit: int = 20) -> dict[str, Any]:
    record = get_rollout(experiment_id, run_id)
    messages = _trajectory_payload(experiment_id, record).get("messages", [])
    if not 0 <= offset <= len(messages) or not 1 <= limit <= 50:
        raise ValueError("消息分页参数无效。")
    return {"messages": messages[offset:offset + limit], "offset": offset, "total": len(messages),
            "next_offset": offset + limit if offset + limit < len(messages) else None}


def export_rollout(experiment_id: str, run_id: str) -> dict[str, Any]:
    record = get_rollout(experiment_id, run_id)
    result: dict[str, Any] = {
        "schema_version": "1", "run": record.summary.model_dump(mode="json"),
        "workflow": record.workflow, "task": record.task, "model_binding": record.model_binding,
        "redaction_applied": record.redaction_applied,
    }
    if record.summary.trajectory_id:
        result["trajectory"] = copy.deepcopy(_trajectory_payload(experiment_id, record))
        result["snapshots"] = _snapshot_summaries(experiment_id, run_id, result["trajectory"])
    else:
        result["trajectory"] = None
    return _redact(result)
