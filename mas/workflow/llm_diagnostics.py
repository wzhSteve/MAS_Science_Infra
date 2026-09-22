"""Small, redacted checkpoints for server-side model requests."""

from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import urlsplit, urlunsplit
from urllib.request import getproxies, proxy_bypass

logger = logging.getLogger(__name__)
_failure_logged = False


def safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if parts.port:
            host += f":{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "<invalid-url>"


def _summary(value: str, limit: int = 1000) -> str:
    for key, secret in os.environ.items():
        if secret and re.search(r"(?:^|_)(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|TOKEN|SECRET|PASSWORD)$", key, re.I):
            value = value.replace(secret, "[redacted]")
    value = re.sub(r"https?://[^\s<>\"']+", lambda match: safe_url(match.group()), value)
    value = re.sub(r"(?i)bearer\s+[^\s,\"'<>]+", "Bearer [redacted]", value)
    value = re.sub(
        r"""(?i)(["']?(?:api[_-]?key|access[_-]?token|password|secret)["']?\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;<]+)""",
        r"\1[redacted]",
        value,
    )
    return " ".join(value.split())[:limit]


def log_model_route(label: str, endpoint: str, model: str) -> None:
    proxies = getproxies()
    host = urlsplit(endpoint).hostname or ""
    logger.warning(
        "[TIR-DIAG route] %s pid=%s model=%s endpoint=%s env_proxies=%s "
        "no_proxy_set=%s env_bypass_hint=%s",
        label, os.getpid(), _summary(model, 160), safe_url(endpoint),
        json.dumps({key: safe_url(value) for key, value in proxies.items()
                    if key in ("http", "https", "all")}),
        bool(proxies.get("no")), proxy_bypass(host),
    )


def log_first_model_failure(error: Exception, endpoint: str, model: str) -> None:
    global _failure_logged
    if _failure_logged:
        return
    _failure_logged = True
    log_model_route("failed-request", endpoint, model)
    response = getattr(error, "response", None)
    request = getattr(error, "request", None)
    headers = {}
    body = ""
    if response is not None:
        headers = {name: _summary(response.headers[name], 160)
                   for name in ("server", "via", "content-type", "x-request-id",
                                "x-litellm-model-id", "x-litellm-response-cost")
                   if name in response.headers}
        # HTTP status errors contain an already-read response; never log request bodies.
        if response.is_stream_consumed:
            body = _summary(response.text)
    logger.error(
        "[TIR-DIAG failure] type=%s status=%s url=%s headers=%s body=%r cause=%s message=%r",
        type(error).__name__, getattr(error, "status_code", None),
        safe_url(str(request.url)) if request is not None else safe_url(endpoint),
        json.dumps(headers), body or "<empty-or-unread>",
        type(error.__cause__).__name__ if error.__cause__ else None,
        _summary(str(error)),
    )
