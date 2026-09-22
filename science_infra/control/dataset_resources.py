"""Shared server-side dataset paths; experiments save the selected path."""

from __future__ import annotations

import threading
from uuid import uuid4

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from science_infra.env import repo_root
from .paths import server_data_path

router = APIRouter(prefix="/api/dataset-resources")
_LOCK = threading.Lock()


class DatasetFields(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    path: str = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("请输入数据名称。")
        return value.strip()

    @field_validator("path")
    @classmethod
    def clean_path(cls, value: str) -> str:
        path = server_data_path(value)
        if not path.endswith(".parquet"):
            raise ValueError("请选择服务器上的 Parquet 文件。")
        return path


class Dataset(DatasetFields):
    id: str


class DatasetCatalog(BaseModel):
    revision: int = Field(ge=1)
    items: list[Dataset]


class DatasetWrite(DatasetFields):
    revision: int = Field(ge=1)


def _read() -> DatasetCatalog:
    path = repo_root() / "resources" / "datasets.yaml"
    return DatasetCatalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _write(catalog: DatasetCatalog) -> None:
    path = repo_root() / "resources" / "datasets.yaml"
    temporary = path.with_suffix(".yaml.tmp")
    temporary.write_text(
        yaml.safe_dump(catalog.model_dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    temporary.replace(path)


@router.get("")
def list_datasets() -> DatasetCatalog:
    with _LOCK:
        return _read()


def _save(body: DatasetWrite, dataset_id: str | None = None) -> DatasetCatalog:
    with _LOCK:
        catalog = _read()
        if body.revision != catalog.revision:
            raise HTTPException(409, "数据目录已更新，请刷新后重新编辑。")
        if dataset_id and not any(item.id == dataset_id for item in catalog.items):
            raise HTTPException(404, "数据资源不存在。")
        if any(item.path == body.path and item.id != dataset_id for item in catalog.items):
            raise HTTPException(409, "此服务器路径已经登记。")
        item = Dataset(id=dataset_id or uuid4().hex, name=body.name, path=body.path)
        items = [item if old.id == dataset_id else old for old in catalog.items]
        if dataset_id is None:
            items.append(item)
        updated = DatasetCatalog(revision=catalog.revision + 1, items=items)
        _write(updated)
        return updated


@router.post("")
def create_dataset(body: DatasetWrite) -> DatasetCatalog:
    return _save(body)


@router.put("/{dataset_id}")
def update_dataset(dataset_id: str, body: DatasetWrite) -> DatasetCatalog:
    return _save(body, dataset_id)
