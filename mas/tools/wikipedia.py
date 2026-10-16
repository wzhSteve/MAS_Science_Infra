"""Wikipedia REST/API lookup. Titles + urls only, no page content."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from .search import _UA_HEADERS, _cache_query_key, proxy_candidates, truncate_description
from .search_cache import GAP_SENTINEL, cache_lookup, cache_store

_TOP_K = int(os.getenv("TIR_WIKI_TOP_K", "10"))


def _cache_consume_query(query: str) -> Optional[str]:
    """Cache-first lookup mirroring search.py's three-branch pattern.

    hit -> cached text verbatim; gap -> clean sentinel WITHOUT a network
    retry (the producer already recorded this query as network-failed);
    miss -> None (existing network logic runs).
    """
    try:
        if not os.getenv("MAS_SEARCH_CACHE", "").strip():
            return None
        text, hit_type = cache_lookup("query", _cache_query_key(query))
        if hit_type == "hit":
            return text
        if hit_type == "gap":
            return f"search_unavailable: {GAP_SENTINEL}"
        return None
    except Exception:
        return None


def _cache_store_query(query: str, value: str) -> None:
    try:
        if not os.getenv("MAS_SEARCH_CACHE", "").strip():
            return
        cache_store("query", _cache_query_key(query), value)
    except Exception:
        pass


def wikipedia_search(query: str, top_k: int = _TOP_K) -> str:
    """Return top-k Wikipedia hits as numbered title + URL entries.

    Does NOT fetch page content. Call fetch_page(url) for that.
    """
    query = (query or "").strip()
    if not query:
        return "Error: empty wikipedia query."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"

    cached = _cache_consume_query(query)
    if cached is not None:
        return cached

    headers = {"User-Agent": _UA_HEADERS["User-Agent"]}
    last_err = "no proxy candidate"
    hits: List[Dict[str, Any]] = []
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
            if hits:
                break
            last_err = f"{label}: 0 hits"
        except Exception as e:
            last_err = f"{label}: {type(e).__name__}: {e}"
            continue

    if not hits:
        return f"search_unavailable: {last_err}"

    lines: List[str] = [f"wikipedia results for: {query}"]
    for i, hit in enumerate(hits[:top_k], 1):
        title = str(hit.get("title") or "").strip()
        if not title:
            continue
        page = f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
        lines.append(f"{i}. {title}")
        lines.append(f"   URL: {page}")
        desc = truncate_description(str(hit.get("snippet") or ""))
        if desc:
            lines.append(f"   {desc}")
    if len(lines) <= 1:
        return f"search_unavailable: {label}: 0 hits"
    out = "\n".join(lines)
    _cache_store_query(query, out)
    return out
