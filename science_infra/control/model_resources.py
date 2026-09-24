"""Personal model resources and experiment bindings."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping
from uuid import uuid4

from .paths import model_resources_db


class ResourceError(ValueError):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@contextmanager
def transaction(*, write: bool = False) -> Iterator[sqlite3.Connection]:
    path = model_resources_db()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    connection = sqlite3.connect(path, timeout=10, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS model_resources (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                revision INTEGER NOT NULL,
                config TEXT NOT NULL,
                credential_mode TEXT NOT NULL,
                api_key TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_binding_states (
                experiment_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_bindings (
                experiment_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                PRIMARY KEY (experiment_id, purpose)
            );
            CREATE INDEX IF NOT EXISTS model_binding_resource
                ON model_bindings(resource_id);
            """
        )
        connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        yield connection
        connection.commit()
    except (sqlite3.Error, OSError):
        if connection.in_transaction:
            connection.rollback()
        raise ResourceError("模型资源存储暂不可用，请稍后重试。", 409) from None
    finally:
        connection.close()


def _purpose(value: Any) -> str:
    if value not in ("inference", "training"):
        raise ResourceError("模型用途必须是 inference 或 training。")
    return value


def _revision(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise ResourceError("请提供有效的整数修订号。")
    return value


def _text(
    value: Any,
    label: str,
    *,
    maximum: int = 4096,
    required: bool = False,
) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise ResourceError(f"{label} 必须是有效字符串。")
    value = value.strip()
    if any(ord(char) < 32 for char in value):
        raise ResourceError(f"{label} 包含无效控制字符。")
    if required and not value:
        raise ResourceError(f"请填写{label}。")
    return value


def validate_resource(body: Mapping[str, Any]) -> dict[str, Any]:
    from .llm_config import endpoint_issue

    allowed_body = {
        "name",
        "type",
        "config",
        "credential_mode",
        "api_key",
        "clear_api_key",
        "revision",
    }
    if set(body) - allowed_body:
        raise ResourceError("资源包含不支持的字段。")
    purpose = _purpose(body.get("type"))
    name = _text(body.get("name"), "资源名称", maximum=200, required=True)
    mode = body.get("credential_mode")
    if mode not in ("saved", "service", "none"):
        raise ResourceError("请选择有效的凭据来源。")
    raw = body.get("config")
    if not isinstance(raw, dict):
        raise ResourceError("请提供模型配置。")
    allowed_config = {
        "kind",
        "model",
        "base_url",
        "model_path",
        "port",
        "gpu_memory_utilization",
    }
    if set(raw) - allowed_config:
        raise ResourceError("模型配置包含不支持的字段。")
    config = {
        key: _text(raw[key], key)
        for key in ("kind", "model", "base_url", "model_path")
        if key in raw
    }
    if "port" in raw:
        if type(raw["port"]) is not int or not 1 <= raw["port"] <= 65535:
            raise ResourceError("端口必须是 1 到 65535 的整数。")
        config["port"] = raw["port"]
    if "gpu_memory_utilization" in raw:
        memory = raw["gpu_memory_utilization"]
        if type(memory) not in (int, float) or not 0 < memory <= 1:
            raise ResourceError("GPU 显存使用比例必须大于 0 且不超过 1。")
        config["gpu_memory_utilization"] = memory
    key = _text(body.get("api_key") or "", "API Key", maximum=16384)
    clear = body.get("clear_api_key", False)
    if type(clear) is not bool or (clear and key):
        raise ResourceError("API Key 清除参数无效。")
    if purpose == "training":
        if mode != "none" or key:
            raise ResourceError("训练模型来源不接受 API 凭据。")
        if set(config) - {"model_path"} or not config.get("model_path"):
            raise ResourceError("训练模型来源只接受非空 model_path。")
    else:
        config.setdefault("kind", "api")
        if config["kind"] not in ("api", "local"):
            raise ResourceError("推理资源只支持 API 或本地服务。")
        if not config.get("model"):
            raise ResourceError("请填写模型名称。")
        issue = endpoint_issue(config.get("base_url", ""))
        if issue:
            raise ResourceError(issue)
        config["base_url"] = config["base_url"].rstrip("/")
        if config["kind"] == "local":
            config.setdefault("port", 8000)
            config.setdefault("gpu_memory_utilization", 0.45)
    return {
        "name": name,
        "type": purpose,
        "config": config,
        "credential_mode": mode,
        "api_key": key,
        "clear_api_key": clear,
    }


@dataclass(frozen=True)
class ResourceSnapshot:
    data: dict[str, Any]
    api_key: str = field(repr=False)

    def public(self) -> dict[str, Any]:
        from .llm_config import _SERVICE_KEY

        result = {**self.data, "config": dict(self.data["config"])}
        mode = result["credential_mode"]
        result["api_key_set"] = bool(
            self.api_key if mode == "saved" else _SERVICE_KEY if mode == "service" else ""
        )
        return result


def _resource(connection: sqlite3.Connection, resource_id: str) -> ResourceSnapshot:
    row = connection.execute(
        "SELECT * FROM model_resources WHERE id = ?", (resource_id,)
    ).fetchone()
    if row is None:
        raise ResourceError("模型资源不存在。", 404)
    data = dict(row)
    api_key = data.pop("api_key")
    try:
        data["config"] = json.loads(data["config"])
    except (TypeError, json.JSONDecodeError):
        raise ResourceError("模型资源配置损坏，请修复或解除绑定。", 409) from None
    return ResourceSnapshot(data=data, api_key=api_key)


def get_resource(
    resource_id: str, *, revision: int | None = None
) -> ResourceSnapshot:
    with transaction() as connection:
        resource = _resource(connection, resource_id)
        if revision is not None and resource.data["revision"] != _revision(revision):
            raise ResourceError("模型资源已更新，请刷新后重试。", 409)
        return resource


def _references(
    connection: sqlite3.Connection, resource_id: str
) -> list[dict[str, str]]:
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT experiment_id, purpose
            FROM model_bindings
            WHERE resource_id = ?
            ORDER BY experiment_id, purpose
            """,
            (resource_id,),
        )
    ]


def resource_detail(resource_id: str) -> dict[str, Any]:
    with transaction() as connection:
        return {
            **_resource(connection, resource_id).public(),
            "references": _references(connection, resource_id),
        }


def list_resources(
    *,
    type: str | None = None,
    offset: int = 0,
    limit: int = 20,
    q: str = "",
) -> dict[str, Any]:
    if type is not None:
        _purpose(type)
    if offset < 0 or not 1 <= limit <= 100:
        raise ResourceError("分页要求 offset >= 0 且 limit 在 1 到 100 之间。")
    q = _text(q, "搜索内容", maximum=200)
    filters: list[str] = []
    values: list[Any] = []
    if type is not None:
        filters.append("type = ?")
        values.append(type)
    if q:
        filters.append("(instr(lower(name), lower(?)) > 0 OR instr(lower(config), lower(?)) > 0)")
        values.extend([q, q])
    where = " WHERE " + " AND ".join(filters) if filters else ""
    with transaction() as connection:
        total = connection.execute(
            "SELECT COUNT(*) FROM model_resources" + where, values
        ).fetchone()[0]
        rows = connection.execute(
            "SELECT id FROM model_resources"
            + where
            + " ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
            [*values, limit, offset],
        ).fetchall()
        return {
            "items": [_resource(connection, row["id"]).public() for row in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }


def _create(
    connection: sqlite3.Connection, clean: dict[str, Any]
) -> ResourceSnapshot:
    resource_id = uuid4().hex
    now = datetime.now(timezone.utc).isoformat()
    connection.execute(
        "INSERT INTO model_resources VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)",
        (
            resource_id,
            clean["name"],
            clean["type"],
            json.dumps(clean["config"], ensure_ascii=False),
            clean["credential_mode"],
            clean["api_key"],
            now,
            now,
        ),
    )
    return _resource(connection, resource_id)


def create_resource(body: Mapping[str, Any]) -> dict[str, Any]:
    clean = validate_resource(body)
    with transaction(write=True) as connection:
        return _create(connection, clean).public()


def update_resource(resource_id: str, body: Mapping[str, Any]) -> dict[str, Any]:
    revision = _revision(body.get("revision"))
    clean = validate_resource(body)
    with transaction(write=True) as connection:
        old = _resource(connection, resource_id)
        if old.data["revision"] != revision:
            raise ResourceError("模型资源已更新，请刷新后重试。", 409)
        if old.data["type"] != clean["type"]:
            raise ResourceError("不能修改模型资源的用途。")
        key = "" if clean["clear_api_key"] else clean["api_key"] or old.api_key
        connection.execute(
            """
            UPDATE model_resources
            SET name = ?, revision = revision + 1, config = ?,
                credential_mode = ?, api_key = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                clean["name"],
                json.dumps(clean["config"], ensure_ascii=False),
                clean["credential_mode"],
                key,
                datetime.now(timezone.utc).isoformat(),
                resource_id,
            ),
        )
        return _resource(connection, resource_id).public()


def delete_resource(resource_id: str, revision: int) -> dict[str, bool]:
    with transaction(write=True) as connection:
        resource = _resource(connection, resource_id)
        if resource.data["revision"] != _revision(revision):
            raise ResourceError("模型资源已更新，请刷新后重试。", 409)
        if _references(connection, resource_id):
            raise ResourceError("模型资源仍被实验引用，请先解除绑定。", 409)
        connection.execute(
            "DELETE FROM model_resources WHERE id = ?", (resource_id,)
        )
    return {"ok": True}


def require_experiment(exp_id: str) -> None:
    from .experiments import exp_dir

    if not (exp_dir(exp_id) / "experiment.yaml").is_file():
        raise ResourceError("实验不存在，请先创建或选择实验。", 404)


def _binding_state(
    connection: sqlite3.Connection, exp_id: str
) -> dict[str, Any]:
    row = connection.execute(
        "SELECT revision FROM model_binding_states WHERE experiment_id = ?",
        (exp_id,),
    ).fetchone()
    result: dict[str, Any] = {"revision": row["revision"] if row else 0}
    bindings = dict(
        connection.execute(
            "SELECT purpose, resource_id FROM model_bindings WHERE experiment_id = ?",
            (exp_id,),
        )
    )
    for purpose in ("inference", "training"):
        resource_id = bindings.get(purpose)
        item: dict[str, Any] = {"resource_id": resource_id, "resource": None}
        if resource_id is not None:
            try:
                resource = _resource(connection, resource_id)
                if resource.data["type"] != purpose:
                    raise ResourceError("资源类型与实验绑定用途不一致。", 409)
                item["resource"] = resource.public()
            except ResourceError as error:
                item["error"] = str(error)
        result[purpose] = item
    return result


def get_bindings(exp_id: str) -> dict[str, Any]:
    require_experiment(exp_id)
    with transaction() as connection:
        return _binding_state(connection, exp_id)


def binding_context(
    exp_id: str,
) -> tuple[dict[str, Any], ResourceSnapshot | None]:
    require_experiment(exp_id)
    with transaction() as connection:
        state = _binding_state(connection, exp_id)
        inference = state["inference"]
        resource = (
            _resource(connection, inference["resource_id"])
            if inference["resource"] is not None
            else None
        )
        return state, resource


def resolve_binding(exp_id: str, purpose: str) -> ResourceSnapshot | None:
    require_experiment(exp_id)
    _purpose(purpose)
    with transaction() as connection:
        row = connection.execute(
            """
            SELECT resource_id
            FROM model_bindings
            WHERE experiment_id = ? AND purpose = ?
            """,
            (exp_id, purpose),
        ).fetchone()
        if row is None:
            return None
        resource = _resource(connection, row["resource_id"])
        if resource.data["type"] != purpose:
            raise ResourceError("资源类型与实验绑定用途不一致，请修复绑定。", 409)
        return resource


def _check_binding_revision(
    connection: sqlite3.Connection, exp_id: str, revision: int
) -> None:
    row = connection.execute(
        "SELECT revision FROM model_binding_states WHERE experiment_id = ?",
        (exp_id,),
    ).fetchone()
    if (row["revision"] if row else 0) != _revision(revision):
        raise ResourceError("实验模型绑定已更新，请刷新后重试。", 409)


def _bind(
    connection: sqlite3.Connection,
    exp_id: str,
    purpose: str,
    resource_id: str | None,
) -> None:
    if resource_id is None:
        connection.execute(
            "DELETE FROM model_bindings WHERE experiment_id = ? AND purpose = ?",
            (exp_id, purpose),
        )
    else:
        resource = _resource(connection, resource_id)
        if resource.data["type"] != purpose:
            raise ResourceError("资源类型与实验绑定用途不一致。")
        connection.execute(
            "INSERT OR REPLACE INTO model_bindings VALUES (?, ?, ?)",
            (exp_id, purpose, resource_id),
        )
    connection.execute(
        """
        INSERT INTO model_binding_states VALUES (?, 1)
        ON CONFLICT(experiment_id) DO UPDATE SET revision = revision + 1
        """,
        (exp_id,),
    )


def put_binding(
    exp_id: str,
    *,
    revision: int,
    purpose: str,
    resource_id: str | None,
) -> dict[str, Any]:
    require_experiment(exp_id)
    _purpose(purpose)
    if resource_id is not None:
        _text(resource_id, "资源 ID", maximum=128, required=True)
    with transaction(write=True) as connection:
        _check_binding_revision(connection, exp_id, revision)
        _bind(connection, exp_id, purpose, resource_id)
        return _binding_state(connection, exp_id)


def register_and_bind_training(
    exp_id: str,
    *,
    revision: int,
    name: str,
    model_path: str,
) -> dict[str, Any]:
    require_experiment(exp_id)
    clean = validate_resource(
        {
            "name": name,
            "type": "training",
            "config": {"model_path": model_path},
            "credential_mode": "none",
        }
    )
    with transaction(write=True) as connection:
        _check_binding_revision(connection, exp_id, revision)
        existing = connection.execute(
            """
            SELECT id FROM model_resources
            WHERE type = 'training' AND json_extract(config, '$.model_path') = ?
            ORDER BY created_at LIMIT 1
            """,
            (model_path,),
        ).fetchone()
        resource = (
            _resource(connection, existing["id"])
            if existing
            else _create(connection, clean)
        )
        _bind(connection, exp_id, "training", resource.data["id"])
        return {
            "resource": resource.public(),
            "bindings": _binding_state(connection, exp_id),
        }


def ensure_local_training_resource(*, name: str, model_path: str) -> dict[str, Any]:
    """Return the training resource for a discovered local model, creating it once."""
    from .training import local_model_candidate

    candidate = local_model_candidate(model_path)
    canonical_path = candidate["path"]
    clean = validate_resource(
        {
            "name": name or candidate["name"],
            "type": "training",
            "config": {"model_path": canonical_path},
            "credential_mode": "none",
        }
    )
    with transaction(write=True) as connection:
        existing = connection.execute(
            """
            SELECT id FROM model_resources
            WHERE type = 'training' AND json_extract(config, '$.model_path') = ?
            ORDER BY created_at LIMIT 1
            """,
            (canonical_path,),
        ).fetchone()
        return (
            _resource(connection, existing["id"])
            if existing
            else _create(connection, clean)
        ).public()


def migrate_binding(
    exp_id: str,
    *,
    revision: int,
    purpose: str,
    name: str,
    credential_mode: str,
) -> dict[str, Any]:
    from .experiments import load_bundle, load_secrets_env
    from .llm_config import resolve_legacy_llm_config

    require_experiment(exp_id)
    _purpose(purpose)
    if credential_mode not in ("copy", "service"):
        raise ResourceError("迁移凭据来源必须是 copy 或 service。")
    bundle = load_bundle(exp_id)
    if purpose == "inference":
        effective = resolve_legacy_llm_config(exp_id, llm=bundle["llm"])
        config = {
            key: bundle["llm"][key]
            for key in ("model_path", "port", "gpu_memory_utilization")
            if key in bundle["llm"]
        }
        config.update(
            kind=effective.kind, model=effective.model, base_url=effective.base_url
        )
        mode = "saved" if credential_mode == "copy" else "service"
        api_key = (
            load_secrets_env(exp_id).get("OPENAI_API_KEY", "")
            if credential_mode == "copy"
            else ""
        )
    else:
        rl = bundle["rl"]
        config = {
            "model_path": rl.get("model_path")
            or (rl.get("actor_rollout_ref") or {}).get("model", {}).get("path")
            or ""
        }
        mode = "none"
        api_key = ""
    clean = validate_resource(
        {
            "name": name,
            "type": purpose,
            "config": config,
            "credential_mode": mode,
            "api_key": api_key,
        }
    )
    with transaction(write=True) as connection:
        _check_binding_revision(connection, exp_id, revision)
        existing = connection.execute(
            "SELECT 1 FROM model_bindings WHERE experiment_id = ? AND purpose = ?",
            (exp_id, purpose),
        ).fetchone()
        if existing:
            raise ResourceError("该用途已经绑定资源，不能重复迁移。", 409)
        resource = _create(connection, clean)
        _bind(connection, exp_id, purpose, resource.data["id"])
        return {
            "resource": resource.public(),
            "bindings": _binding_state(connection, exp_id),
        }
