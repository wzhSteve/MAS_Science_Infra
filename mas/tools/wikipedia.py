"""Wikipedia REST/API lookup. Titles + urls only, no page content."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from .search import _cache_query_key, truncate_description, wiki_get, wiki_proxy_map
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

    if wiki_proxy_map() is None:
        return "search_unavailable: proxy_down"
    body, err = wiki_get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(top_k),
            "format": "json",
        },
        timeout=15.0,
    )
    hits: List[Dict[str, Any]] = []
    last_err = err or "0 hits"
    if body and not err:
        try:
            hits = (json.loads(body).get("query") or {}).get("search") or []
        except json.JSONDecodeError as exc:
            last_err = f"JSONDecodeError: {exc}"
        if not hits and "0 hits" not in last_err:
            last_err = "0 hits"

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
        return "search_unavailable: 0 hits"
    out = "\n".join(lines)
    _cache_store_query(query, out)
    return out
