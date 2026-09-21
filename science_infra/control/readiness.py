"""Cached local dependency checks and explicit model endpoint probes."""

from __future__ import annotations

import importlib
import logging
import socket
import ssl
import sys
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock
from typing import Any, Literal, Optional

import httpx

from .llm_config import EffectiveLlmConfig, public_endpoint, resolve_llm_config
from .paths import tir_agent_root

logger = logging.getLogger(__name__)
_LIVE_IMPORTS = {
    "langchain": "langchain",
    "langchain_core": "langchain-core",
    "langchain_openai": "langchain-openai",
    "langchain_community": "langchain-community",
    "langgraph.graph": "langgraph",
    "openai": "openai",
    "requests": "requests",
    "bs4": "beautifulsoup4",
}
_DEPENDENCY_LOCK = Lock()
_PROBE_LOCK = Lock()
_LAST_PROBES: dict[str, dict[str, Any]] = {}
LIVE_HINT = '在 Control 使用的 Python 环境安装 live 依赖：python -m pip install -e ".[live]"。'
PARQUET_HINT = '在 Control 使用的 Python 环境安装数据依赖：python -m pip install -e ".[data]"。'


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_workflow_path() -> None:
    path = str(tir_agent_root())
    if path not in sys.path:
        sys.path.insert(0, path)


@lru_cache(maxsize=2)
def _dependencies(group: Literal["live", "parquet"]) -> dict[str, Any]:
    required = _LIVE_IMPORTS if group == "live" else {"pandas": "pandas"}
    missing: list[str] = []
    failures: list[str] = []
    for module, package in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
        except Exception as error:
            logger.warning("Dependency import failed: %s (%s)", module, type(error).__name__)
            failures.append(f"{package} 无法加载（{type(error).__name__}）")
    if group == "parquet":
        if not any(importlib.util.find_spec(engine) for engine in ("pyarrow", "fastparquet")):
            missing.append("pyarrow 或 fastparquet")
    elif not missing and not failures:
        ensure_workflow_path()
        try:
            importlib.import_module("tir_agent")
        except Exception as error:
            failures.append(f"TirAgent 无法加载（{type(error).__name__}）")
    return {
        "available": not missing and not failures,
        "missing": missing,
        "error": "；".join(failures) or None,
        "hint": LIVE_HINT if group == "live" else PARQUET_HINT,
    }


def execution_dependencies(group: Literal["live", "parquet"]) -> dict[str, Any]:
    with _DEPENDENCY_LOCK:
        return _dependencies(group)


def require_live(config: EffectiveLlmConfig) -> None:
    issues = config.issues()
    if issues:
        raise ValueError("；".join(issue["message"] for issue in issues))
    dependency = execution_dependencies("live")
    if not dependency["available"]:
        reason = dependency["error"] or "缺少依赖：" + "、".join(dependency["missing"])
        raise RuntimeError(f"{reason} {dependency['hint']}")


def require_parquet() -> None:
    dependency = execution_dependencies("parquet")
    if not dependency["available"]:
        reason = dependency["error"] or "缺少依赖：" + "、".join(dependency["missing"])
        raise RuntimeError(f"{reason} {dependency['hint']}")


def model_readiness(exp_id: str) -> dict[str, Any]:
    config = resolve_llm_config(exp_id)
    live = execution_dependencies("live")
    parquet = execution_dependencies("parquet")
    issues = config.issues()
    warnings: list[dict[str, str]] = []
    if not live["available"]:
        issues.append(
            {
                "code": "live_dependencies",
                "message": live["error"] or "缺少真实推理依赖：" + "、".join(live["missing"]),
                "hint": live["hint"],
            }
        )
    if not parquet["available"]:
        warnings.append(
            {
                "code": "parquet_dependencies",
                "message": "parquet 读取依赖未就绪，不影响示例任务。",
                "hint": parquet["hint"],
            }
        )
    if not config.api_key:
        warnings.append(
            {
                "code": "no_credential",
                "message": "未提供 API Key；仅无鉴权的兼容服务可直接使用。",
            }
        )
    if config.kind == "local":
        warnings.append(
            {
                "code": "local_service",
                "message": "需要已启动的本地模型服务；本检查不会自动启动模型。",
            }
        )
    with _PROBE_LOCK:
        probe = _LAST_PROBES.get(exp_id)
    if probe and probe["config_revision"] != config.revision:
        probe = None
    return {
        "experiment_id": exp_id,
        "checked_at": _now(),
        "ready": not issues,
        "model": config.public(),
        "blocking_issues": issues,
        "warnings": warnings,
        "dependencies": {"live": live, "parquet": parquet},
        "capabilities": {
            name: "unknown"
            for name in ("tool_calling", "token_ids", "logprobs", "policy_version")
        },
        "probe": probe,
    }


async def probe_llm(
    config: EffectiveLlmConfig, *, experiment_id: Optional[str] = None
) -> dict[str, Any]:
    endpoint = public_endpoint(config.base_url)
    result: dict[str, Any] = {
        "ok": False,
        "models": [],
        "url": f"{endpoint}/models" if endpoint else "",
        "checked_at": _now(),
        "config_revision": config.revision,
        "probe_type": "models",
        "inference_verified": False,
        "tool_calling_verified": False,
    }

    def finish(status: str, message: str) -> dict[str, Any]:
        result.update(status=status, message=message, ok=status == "reachable")
        if status != "reachable":
            result["error"] = message
        if experiment_id is not None:
            with _PROBE_LOCK:
                _LAST_PROBES[experiment_id] = dict(result)
        return result

    issues = config.issues(require_model=False)
    if issues:
        return finish("configuration_error", "；".join(issue["message"] for issue in issues))
    headers = {"Authorization": f"Bearer {config.api_key}"} if config.api_key else {}
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=False) as client:
            response = await client.get(f"{config.base_url}/models", headers=headers)
    except httpx.TimeoutException:
        return finish("timeout", "模型列表请求超时，请检查网络和服务状态。")
    except httpx.ConnectError as error:
        cause: Optional[BaseException] = error
        while cause is not None:
            if isinstance(cause, ssl.SSLError):
                return finish("network_error", "模型端点 TLS 连接失败。")
            if isinstance(cause, socket.gaierror):
                return finish("network_error", "模型端点 DNS 解析失败。")
            cause = cause.__cause__
        return finish("network_error", "无法连接模型端点。")
    except httpx.HTTPError:
        return finish("network_error", "模型端点网络请求失败。")

    status = response.status_code
    result["status_code"] = status
    if status in (401, 403):
        return finish("authentication_failed", "模型服务拒绝鉴权或访问权限。")
    if status == 429:
        return finish("rate_limited", "模型服务限流，请稍后重试。")
    if status in (404, 405):
        return finish("unsupported", "服务不支持模型列表探测；这不代表推理不可用。")
    if status >= 500:
        return finish("server_error", f"模型服务返回错误（HTTP {status}）。")
    if not 200 <= status < 300:
        return finish("request_failed", f"模型列表请求未成功（HTTP {status}）。")
    try:
        body = response.json()
    except ValueError:
        return finish("invalid_response", "模型列表返回不是有效 JSON。")
    if not isinstance(body, dict) or not isinstance(body.get("data"), list):
        return finish("invalid_response", "模型列表格式不符合 OpenAI 兼容协议。")
    if any(
        not isinstance(item, dict) or not isinstance(item.get("id"), str)
        for item in body["data"]
    ):
        return finish("invalid_response", "模型列表包含无效条目。")
    result["models"] = [item["id"] for item in body["data"]]
    return finish("reachable", "模型列表可访问；尚未验证生成或工具调用。")
