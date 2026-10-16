"""Cache-first lookup kernel for search tools (read-only consumer).

Shared JSON contract with scripts/build_search_cache.py (the producer):
  - Path: data/search_cache.json (env MAS_SEARCH_CACHE overrides).
  - Top level keys:
      "query::<condensed-query>" -> SERP text with the normal return
                                    format's header, e.g.
                                    "wikipedia results for: <q>\\n1. <title>\\n   URL: <url>..."
      "url::<page-url>"          -> page body plain text (<=2000 chars).
  - "search_unavailable_in_cache" is the EXPLICIT-MISS sentinel value:
    it distinguishes "we already tried and the network failed" (gap) from
    "key simply not in cache" (miss).

Simplified from the APPO search_tool pattern (ref_Rep/APPO-main/
ARPO/verl_arpo_entropy/verl/workers/agent/tools/search_tool.py): no
concurrent writers in local training rollouts, so writes are a plain
synchronous read-modify-write of the whole JSON guarded by an fcntl lock
(APPO's _acquire_file_lock idea, but synchronous). Store failures only
log — never raise into a rollout.

Pure-function API with one module-level lazily-loaded cache dict.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:  # fcntl is POSIX-only; imports must stay safe on any platform.
    import fcntl
except ImportError:  # pragma: no cover - windows / non-posix fallback
    fcntl = None

GAP_SENTINEL = "search_unavailable_in_cache"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _REPO_ROOT / "data" / "search_cache.json"

# Module-level cache dict, lazily loaded once per process. Resetting via
# reload_search_cache(path) in tests; runtime code never reloads.
_cache: Optional[Dict[str, str]] = None
_cache_path: Optional[str] = None


def _log(msg: str) -> None:
    """Log to stderr only; never raise and never pollute tool output."""
    try:
        print(f"[search_cache] {msg}", file=sys.stderr)
    except Exception:
        pass


def resolve_cache_path(path: Optional[str] = None) -> Optional[str]:
    """Resolve the active cache file path or None when caching is disabled.

    Precedence: explicit arg > env MAS_SEARCH_CACHE > default
    data/search_cache.json. An empty/whitespace env value DISABLES the
    cache (so unset and explicitly-off behave the same for callers that
    gate on this function's truthiness).
    """
    if path is not None:
        p = path.strip()
        return p or None
    env = os.getenv("MAS_SEARCH_CACHE")
    if env is not None and env.strip():
        return env.strip()
    return str(_DEFAULT_CACHE)


def load_search_cache(path: Optional[str] = None) -> Dict[str, str]:
    """Load (lazily, once) and return the module-level cache dict.

    Missing file -> empty dict (no error, cache simply has no hits yet).
    Invalid JSON -> empty dict + one stderr log (never raise into rollouts).
    """
    global _cache, _cache_path
    resolved = resolve_cache_path(path)
    if _cache is not None and _cache_path == resolved:
        return _cache
    data: Dict[str, str] = {}
    try:
        if resolved and Path(resolved).is_file():
            with open(resolved, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                data = {str(k): str(v) for k, v in raw.items()}
    except Exception as exc:  # corrupt file: treat as empty, do not raise
        _log(f"load failed for {resolved}: {type(exc).__name__}: {exc}")
        data = {}
    _cache = data
    _cache_path = resolved
    return _cache


def reload_search_cache(path: Optional[str] = None) -> Dict[str, str]:
    """Force a fresh read from disk (tests / post-build refresh)."""
    global _cache
    _cache = None
    return load_search_cache(path)


def cache_enabled(path: Optional[str] = None) -> bool:
    """True when a cache path is active (env MAS_SEARCH_CACHE or default).

    Only the env-var form gates callers: when MAS_SEARCH_CACHE is UNSET
    the whole cache layer stays off (bypass flag for zero-cost rollouts).
    """
    if not os.getenv("MAS_SEARCH_CACHE"):
        return False
    return bool(resolve_cache_path(path))


def cache_lookup(category: str, key: str) -> Tuple[Optional[str], str]:
    """Look up "::<key>" in the cache. hit_type: "hit" | "miss" | "gap".

    - "hit":  normal cached text is returned.
    - "miss": key not present (fresh-network path should run).
    - "gap":  key present with the GAP_SENTINEL mark ("search failed before");
              callers should NOT retry the network for this key.
    """
    if category not in ("query", "url"):
        raise ValueError(f"category must be 'query' or 'url', got {category!r}")
    cache = load_search_cache()
    if cache is None:  # pragma: no cover - load never returns None
        return None, "miss"
    full = f"{category}::{key}"
    if full not in cache:
        # Name collision safety: a raw SERP text could contain the sentinel
        # only as a whole value, never as a key, so key presence is enough.
        return None, "miss"
    value = cache.get(full)
    if value == GAP_SENTINEL:
        return None, "gap"
    return value, "hit"


def cache_store(category: str, key: str, value: str) -> bool:
    """Incrementally write back one entry ("::<key>" = value) to disk.

    Read-modify-write of the whole JSON under an fcntl lock on the
    sidecar <path>.lock file (synchronous version of APPO's
    _acquire_file_lock). Keeps recently written entries in the module
    cache. Any failure is logged and swallowed: a rollout must never
    crash because the cache could not be persisted. The producer
    (scripts/build_search_cache.py) merges entries under the same lock so
    concurrent-writer lost updates cannot happen.
    """
    global _cache
    if category not in ("query", "url"):
        raise ValueError(f"category must be 'query' or 'url', got {category!r}")
    resolved = resolve_cache_path()
    if not resolved:
        _log("store skipped: no cache path configured")
        return False
    full = f"{category}::{key}"
    try:
        target = Path(resolved)
        lock_path = target.with_name(target.name + ".lock")
        with open(lock_path, "a+b") as lock_fh:
            if fcntl is not None:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                merged: Dict[str, str] = {}
                if target.is_file():
                    with open(target, "r", encoding="utf-8") as fh:
                        raw = json.load(fh)
                    if isinstance(raw, dict):
                        merged = {str(k): str(v) for k, v in raw.items()}
                merged[full] = str(value)
                tmp = target.with_name(target.name + ".tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(merged, fh, ensure_ascii=False, indent=2)
                os.replace(str(tmp), str(target))
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        if isinstance(_cache, dict):
            _cache[full] = str(value)
        return True
    except Exception as exc:
        # Write failure must never break a live search call.
        _log(f"store of {full!r} failed: {type(exc).__name__}: {exc}")
        return False
