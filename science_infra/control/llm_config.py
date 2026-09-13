"""Request-scoped model configuration and safe public connection summaries."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from science_infra.env import load_science_env

# Resolve service defaults once, before requests or workflow imports can change env.
load_science_env()
_SERVICE_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
_SERVICE_BASE = (os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "").strip()
_SERVICE_MODEL = (os.environ.get("OPENAI_MODEL") or os.environ.get("MODEL") or "").strip()
_REVISION_KEY = secrets.token_bytes(32)
CredentialSource = Literal["request", "experiment", "resource", "service", "none"]


def public_endpoint(base_url: str) -> str:
    """Exclude embedded credentials, query parameters and fragments from responses."""
    try:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return ""
        host = parts.hostname
        if ":" in host:
            host = f"[{host}]"
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path.rstrip("/"), "", ""))
    except ValueError:
        return ""


def endpoint_issue(base_url: str) -> Optional[str]:
    if not base_url:
        return "请填写模型 API 端点。"
    try:
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.port == 0:
            return "模型端点必须是有效的 HTTP 或 HTTPS 地址。"
        if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
            return "请使用不含认证信息、查询参数或片段的 API 基础地址，密钥单独配置。"
        if any(char.isspace() for char in base_url):
            return "模型端点不能包含空白字符。"
    except ValueError:
        return "模型端点格式无效。"
    return None


def resolve_credential(exp_id: str, api_key: Optional[str] = None) -> tuple[str, CredentialSource]:
    from .experiments import load_secrets_env

    if api_key is not None and api_key.strip():
        return api_key.strip(), "request"
    saved = load_secrets_env(exp_id).get("OPENAI_API_KEY", "").strip()
    if saved:
        return saved, "experiment"
    if _SERVICE_KEY:
        return _SERVICE_KEY, "service"
    return "", "none"


@dataclass(frozen=True)
class EffectiveLlmConfig:
    kind: str
    model: str
    base_url: str = field(repr=False)
    api_key: str = field(repr=False)
    credential_source: CredentialSource
    resource_id: Optional[str] = None
    resource_revision: Optional[int] = None
    resource_name: Optional[str] = None

    @property
    def revision(self) -> str:
        # HMAC avoids exposing a guessable hash of the user's API key.
        payload = json.dumps(
            [self.kind, self.model, self.base_url, self.api_key, self.credential_source,
             self.resource_id, self.resource_revision],
            ensure_ascii=False,
        ).encode("utf-8")
        return hmac.new(_REVISION_KEY, payload, hashlib.sha256).hexdigest()

    def public(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model": self.model,
            "base_url": public_endpoint(self.base_url),
            "api_key_set": bool(self.api_key),
            "credential_source": self.credential_source,
            "config_revision": self.revision,
            "resource_id": self.resource_id,
            "resource_revision": self.resource_revision,
            "resource_name": self.resource_name,
        }

    def issues(self, *, require_model: bool = True) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        if self.kind == "rl_endpoint":
            issues.append({
                "code": "training_endpoint",
                "message": "训练端点需要由训练过程注入，不能用于独立运行。",
                "hint": "独立运行请选择 API 或已启动的本地模型服务。",
            })
        elif self.kind not in ("api", "local"):
            issues.append({"code": "invalid_kind", "message": "请选择 API 或本地模型模式。"})
        error = endpoint_issue(self.base_url)
        if error:
            issues.append({"code": "invalid_endpoint", "message": error})
        if require_model and not self.model:
            issues.append({"code": "missing_model", "message": "请填写实际调用的模型名称。"})
        if any(char in self.api_key for char in ("\r", "\n", "\x00")):
            issues.append({"code": "invalid_credential", "message": "API Key 包含无效控制字符，请重新配置。"})
        return issues

    def subprocess_env(self) -> dict[str, str]:
        env = {"OPENAI_API_KEY": self.api_key}
        if self.base_url:
            env.update(OPENAI_API_BASE=self.base_url, OPENAI_BASE_URL=self.base_url)
        if self.model:
            env.update(OPENAI_MODEL=self.model, MODEL=self.model)
        return env


def resolve_llm_config(
    exp_id: str,
    *,
    llm: Optional[Mapping[str, Any]] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    kind: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EffectiveLlmConfig:
    from .model_resources import ResourceError, resolve_binding

    resource = resolve_binding(exp_id, "inference")
    if resource is not None:
        effective = resource_llm_config(resource, api_key=api_key)
        for field_name, value in (("base_url", base_url), ("model", model), ("kind", kind)):
            if value is not None and value.strip().rstrip("/") != getattr(effective, field_name).rstrip("/"):
                raise ResourceError("当前实验使用模型资源，请修改资源或显式解除绑定，不能覆盖资源配置。", 409)
        return effective
    return resolve_legacy_llm_config(exp_id, llm=llm, base_url=base_url, model=model, kind=kind, api_key=api_key)


def resource_llm_config(resource: Any, *, api_key: Optional[str] = None) -> EffectiveLlmConfig:
    from .model_resources import ResourceError

    data = resource.data
    if data["type"] != "inference":
        raise ResourceError("训练模型来源不能用于推理或连接检查。")
    mode = data["credential_mode"]
    if api_key is not None and api_key.strip():
        key, source = api_key.strip(), "request"
    elif mode == "saved":
        key, source = resource.api_key, "resource" if resource.api_key else "none"
    elif mode == "service":
        key, source = _SERVICE_KEY, "service" if _SERVICE_KEY else "none"
    else:
        key, source = "", "none"
    config = data["config"]
    return EffectiveLlmConfig(
        kind=config.get("kind", "api"), model=config["model"], base_url=config["base_url"],
        api_key=key, credential_source=source,
        resource_id=data["id"], resource_revision=data["revision"], resource_name=data["name"],
    )


def resolve_legacy_llm_config(
    exp_id: str,
    *,
    llm: Optional[Mapping[str, Any]] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    kind: Optional[str] = None,
    api_key: Optional[str] = None,
) -> EffectiveLlmConfig:
    if llm is None:
        from .experiments import load_llm_section

        llm = load_llm_section(exp_id)
    key, source = resolve_credential(exp_id, api_key)
    resolved_kind = str(kind if kind is not None else llm.get("kind") or "api").strip()
    # Explicit empty UI overrides stay empty; saved empty API fields use service defaults.
    saved_base = str(llm.get("base_url") or "").strip()
    saved_model = str(llm.get("model") or "").strip()
    if resolved_kind == "api":
        saved_base = saved_base or _SERVICE_BASE
        saved_model = saved_model or _SERVICE_MODEL
    return EffectiveLlmConfig(
        kind=resolved_kind,
        model=(model if model is not None else saved_model).strip(),
        base_url=(base_url if base_url is not None else saved_base).strip().rstrip("/"),
        api_key=key,
        credential_source=source,
    )
