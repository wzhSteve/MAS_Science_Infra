"""Shared HTTP(S) proxy helpers for MAS tools.

Preference (first non-empty wins):
1. ``GOOGLE_CHROME_PROXY`` / ``CHROME_PROXY`` (SSH RemoteForward to local Clash)
2. ``WEB_SEARCH_PROXY``
3. Process ``HTTPS_PROXY`` / ``HTTP_PROXY`` / lowercase variants

During VERL RL training, prefer **not** publishing process-wide ``HTTP_PROXY``:
OpenAI / LiteLLM clients would otherwise route local vLLM traffic through Clash
(502). Tools that need a proxy should call ``get_requests_proxies()`` or
``wikipedia_proxy_env()`` instead.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


def normalize_proxy_url(proxy: str) -> str:
    proxy = (proxy or "").strip()
    if not proxy:
        return ""
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def resolve_proxy_url() -> str:
    for key in (
        "GOOGLE_CHROME_PROXY",
        "CHROME_PROXY",
        "WEB_SEARCH_PROXY",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "https_proxy",
        "http_proxy",
    ):
        value = normalize_proxy_url(os.getenv(key, ""))
        if value:
            return value
    return ""


def get_requests_proxies() -> Optional[Dict[str, str]]:
    """Return ``{"http": ..., "https": ...}`` for ``requests``, or ``None``."""
    proxy = resolve_proxy_url()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def ensure_process_proxy_env() -> str:
    """Optionally mirror tool proxy into process-wide HTTP(S)_PROXY.

    Skipped when ``MAS_NO_PROCESS_PROXY=1`` (recommended for VERL training so
    OpenAI clients do not send local LLM traffic through Clash).
    """
    flag = (os.getenv("MAS_NO_PROCESS_PROXY") or "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        return ""
    proxy = resolve_proxy_url()
    if not proxy:
        return ""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        if not (os.environ.get(key) or "").strip():
            os.environ[key] = proxy
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    extras = ["127.0.0.1", "localhost", "0.0.0.0"]
    parts = [p.strip() for p in existing.split(",") if p.strip()]
    for host in extras:
        if host not in parts:
            parts.append(host)
    joined = ",".join(parts)
    os.environ["NO_PROXY"] = joined
    os.environ["no_proxy"] = joined
    return proxy


@contextmanager
def wikipedia_proxy_env() -> Iterator[str]:
    """Temporarily set HTTP(S)_PROXY for ``wikipedia``/urllib only."""
    proxy = resolve_proxy_url()
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
    prev = {k: os.environ.get(k) for k in keys}
    try:
        if proxy:
            for k in keys:
                os.environ[k] = proxy
        yield proxy
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
