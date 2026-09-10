"""Wikipedia REST/API lookup without an in-tool LLM filter."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from .search import proxy_candidates

_MAX_CHARS = int(os.getenv("TIR_WIKI_MAX_CHARS", "2000"))
_TOP_K = int(os.getenv("TIR_WIKI_TOP_K", "3"))


def wikipedia_search(query: str, top_k: int = _TOP_K) -> str:
    """Return Wikipedia search hits plus short extracts."""
    query = (query or "").strip()
    if not query:
        return "Error: empty wikipedia query."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"

    headers = {"User-Agent": "tir-agent/1.0 (agent-lightning; educational)"}
    last_err = "no proxy candidate"
    hits: List[Dict[str, Any]] = []
    used_proxies: Optional[Dict[str, str]] = None
    for proxies in proxy_candidates():
        label = (proxies or {}).get("http") or "direct"
        try:
            search_resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "srlimit": top_k,
                    "format": "json",
                },
                headers=headers,
                timeout=15,
                proxies=proxies,
            )
            if search_resp.status_code >= 400:
                last_err = f"{label}: HTTP {search_resp.status_code}"
                continue
            hits = search_resp.json().get("query", {}).get("search", [])
            used_proxies = proxies
            if hits:
                break
            last_err = f"{label}: 0 hits"
        except Exception as e:
            last_err = f"{label}: {type(e).__name__}: {e}"
            continue

    if not hits:
        return f"search_unavailable: {last_err}"

    parts: List[str] = []
    for i, hit in enumerate(hits[:top_k], 1):
        title = str(hit.get("title") or "")
        snippet = str(hit.get("snippet") or "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        extract = ""
        try:
            sum_resp = requests.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}",
                headers=headers,
                timeout=15,
                proxies=used_proxies,
            )
            if sum_resp.status_code < 400:
                extract = str(sum_resp.json().get("extract") or "")
        except Exception:
            extract = ""
        body = extract or snippet
        parts.append(f"{i}. {title}\n{body}")

    text = "\n\n".join(parts)
    if len(text) > _MAX_CHARS:
        text = text[:_MAX_CHARS] + "\n...[truncated]"
    return text
