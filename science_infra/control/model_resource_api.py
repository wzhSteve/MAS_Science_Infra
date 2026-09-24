"""Model resource and experiment binding API."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query

from . import model_resources as catalog
from .llm_config import resource_llm_config
from .readiness import probe_llm

router = APIRouter()


def _call(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except catalog.ResourceError as error:
        raise HTTPException(error.status, str(error)) from None


def _fields(
    body: dict[str, Any], required: set[str], optional: set[str] | None = None
) -> None:
    if set(body) - required - (optional or set()) or required - set(body):
        raise HTTPException(400, "请求字段不完整或包含不支持的字段。")


@router.get("/api/model-resources")
def list_model_resources(
    type: str | None = None,
    offset: int = Query(0),
    limit: int = Query(20),
    q: str = "",
) -> dict[str, Any]:
    return _call(catalog.list_resources, type=type, offset=offset, limit=limit, q=q)


@router.post("/api/model-resources")
def create_model_resource(body: dict[str, Any]) -> dict[str, Any]:
    _fields(
        body,
        {"name", "type", "config", "credential_mode"},
        {"api_key", "clear_api_key"},
    )
    return _call(catalog.create_resource, body)


@router.get("/api/model-resources/discover-local")
def discover_local_models() -> dict[str, Any]:
    from .training import discover_local_models as discover

    return {"items": discover()}


@router.post("/api/model-resources/register-local")
def register_local_model(body: dict[str, Any]) -> dict[str, Any]:
    _fields(body, {"name", "model_path"})
    return _call(
        catalog.ensure_local_training_resource,
        name=body["name"],
        model_path=body["model_path"],
    )


@router.get("/api/model-resources/{resource_id}")
def get_model_resource(resource_id: str) -> dict[str, Any]:
    return _call(catalog.resource_detail, resource_id)


@router.put("/api/model-resources/{resource_id}")
def update_model_resource(
    resource_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    _fields(
        body,
        {"name", "type", "config", "credential_mode", "revision"},
        {"api_key", "clear_api_key"},
    )
    return _call(catalog.update_resource, resource_id, body)


@router.delete("/api/model-resources/{resource_id}")
def delete_model_resource(
    resource_id: str, revision: int = Query(...)
) -> dict[str, Any]:
    return _call(catalog.delete_resource, resource_id, revision)


@router.post("/api/model-resources/{resource_id}/probe")
async def probe_model_resource(
    resource_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    _fields(body, {"revision"})
    resource = _call(
        catalog.get_resource,
        resource_id,
        revision=_call(catalog._revision, body["revision"]),
    )
    config = _call(resource_llm_config, resource)
    result = await probe_llm(config)
    return {
        **result,
        "resource_id": config.resource_id,
        "resource_revision": config.resource_revision,
    }


@router.get("/api/experiments/{exp_id}/model-bindings")
def get_model_bindings(exp_id: str) -> dict[str, Any]:
    return _call(catalog.get_bindings, exp_id)


@router.put("/api/experiments/{exp_id}/model-bindings")
def put_model_binding(exp_id: str, body: dict[str, Any]) -> dict[str, Any]:
    _fields(body, {"revision", "purpose", "resource_id"})
    return _call(catalog.put_binding, exp_id, **body)


@router.post("/api/experiments/{exp_id}/model-bindings/register-local")
def register_local_training_model(
    exp_id: str, body: dict[str, Any]
) -> dict[str, Any]:
    from .training import local_model_candidate

    _fields(body, {"revision", "name", "model_path"})
    candidate = _call(local_model_candidate, body["model_path"])
    return _call(
        catalog.register_and_bind_training,
        exp_id,
        revision=body["revision"],
        name=body["name"] or candidate["name"],
        model_path=candidate["path"],
    )


@router.post("/api/experiments/{exp_id}/model-bindings/migrate")
def migrate_model_binding(exp_id: str, body: dict[str, Any]) -> dict[str, Any]:
    _fields(body, {"revision", "purpose", "name", "credential_mode"})
    return _call(catalog.migrate_binding, exp_id, **body)
