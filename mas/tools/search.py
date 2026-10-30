"""Bing search (title, url, truncated description) and page fetch kernels.

Isolated search tools:
  - wikipedia_search(query, top_k=10) -> titles, urls, truncated snippets
  - bing_search(query, top_k=10)      -> titles, urls, truncated descriptions
  - fetch_page(url) / fetch_by_type   -> open ONE url and extract its content

Bing and Wikipedia never open result pages. Description length is
``TIR_SEARCH_DESC_CHARS`` (default 200). Page-body length stays
``TIR_SEARCH_MAX_CHARS``.

Bright Data SERP API:
  When BRIGHT_DATA_API_KEY is set, bing_search uses the parsed Bing SERP.
  Zones rotate through BRIGHT_DATA_SERP_ZONES. Set BRIGHT_DATA_MODE=0 to disable.
"""

from __future__ import annotations

import fcntl
import html
import json
import os
import re
import socket
import threading
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from .search_cache import GAP_SENTINEL, cache_lookup, cache_store  # noqa: E402

_MAX_SNIPPET_CHARS = int(os.getenv("TIR_SEARCH_MAX_CHARS", "2000"))
_MAX_RESULTS = int(os.getenv("TIR_SEARCH_MAX_RESULTS", "10"))
# Last-resort default when .env / SCIENCE_INFRA_PROXY unset.
# Matches AutoDL SSH RemoteForward 17890 → local Clash (e.g. 12001). Override via .env.
_DEFAULT_PROXY = "http://127.0.0.1:17890"

# --- Bright Data SERP API config ---
BRIGHT_DATA_API_KEY = os.getenv("BRIGHT_DATA_API_KEY", "").strip()
BRIGHT_DATA_SERP_ZONE = os.getenv("BRIGHT_DATA_SERP_ZONE", "serp_api1").strip()
_zone_idx = 0
_zone_lock = threading.Lock()
BRIGHT_DATA_BASE_URL = "https://api.brightdata.com/request"
# 0/False/no = disabled; auto = Bright Data first, fallback to local; force = Bright Data only
BRIGHT_DATA_MODE = os.getenv("BRIGHT_DATA_MODE", "auto").strip().lower()
BRIGHT_DATA_TIMEOUT = int(os.getenv("BRIGHT_DATA_TIMEOUT", "45"))
BRIGHT_DATA_MAX_RETRIES = int(os.getenv("BRIGHT_DATA_MAX_RETRIES", "2"))

_UA_HEADERS = {
    # Do NOT use a Chrome UA here: DDG's bot risk control fingerprints it on
    # datacenter proxies and serves the 202 "anomaly" page on every endpoint.
    # The Firefox UA passed 5/5 probes via the academic proxy.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
        "Gecko/20100101 Firefox/121.0"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_WIKI_API_HEADERS = {
    # Wikimedia blocks browser UAs on api.php with HTTP 429. Identify as a bot.
    "User-Agent": "MAS-Science-Infra/1.0 (academic-rl-training; wikipedia-api)",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}
_WIKI_GATE = os.getenv("TIR_WIKI_GATE", "/tmp/mas_wiki_gate.lock")
_BING_GATE = os.getenv("TIR_BING_GATE", "/tmp/mas_bing_gate.lock")


def _normalize_proxy(proxy: str) -> str:
    proxy = proxy.strip()
    if proxy and "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def _env_proxy() -> str:
    return _normalize_proxy(
        os.getenv("SCIENCE_INFRA_PROXY", "").strip()
        or os.getenv("WEB_SEARCH_PROXY", "").strip()
        or _DEFAULT_PROXY
    )


_proxy_alive_lock = threading.Lock()
_proxy_alive_memo: Dict[str, object] = {"t": 0.0, "ok": False}


def _proxy_host_port(proxy: str) -> Tuple[str, int]:
    parsed = urllib.parse.urlparse(_normalize_proxy(proxy))
    host = parsed.hostname or "127.0.0.1"
    if parsed.port:
        port = int(parsed.port)
    else:
        port = 443 if parsed.scheme == "https" else 80
    return host, port


def proxy_alive(ttl: float = 15.0) -> bool:
    """Cheap TCP probe of SCIENCE_INFRA_PROXY. Cached so rollouts do not stall."""
    force = os.getenv("TIR_PROXY_FORCE", "").strip().lower()
    if force in ("0", "down", "off", "false", "no"):
        return False
    if force in ("1", "up", "on", "true", "yes"):
        return True
    now = time.time()
    with _proxy_alive_lock:
        if now - float(_proxy_alive_memo["t"]) < ttl:
            return bool(_proxy_alive_memo["ok"])
        proxy = _env_proxy()
        ok = False
        if proxy:
            host, port = _proxy_host_port(proxy)
            try:
                sock = socket.create_connection((host, port), timeout=0.25)
                sock.close()
                ok = True
            except OSError:
                ok = False
        _proxy_alive_memo["t"] = now
        _proxy_alive_memo["ok"] = ok
        return ok


def wiki_proxy_map() -> Optional[Dict[str, str]]:
    """Wikipedia-only outbound: reverse proxy, or None if it is down."""
    if not proxy_alive():
        return None
    proxy = _env_proxy()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def wiki_proxy_candidates() -> List[Dict[str, str]]:
    mapped = wiki_proxy_map()
    return [mapped] if mapped else []


def _wiki_min_interval() -> float:
    try:
        return max(0.0, float(os.getenv("TIR_WIKI_MIN_INTERVAL", "0.25")))
    except ValueError:
        return 0.25


def wiki_throttle(extra_wait: float = 0.0) -> None:
    """Serialize Wikipedia calls across runners so one SSH IP is not 429-stormed."""
    wait_extra = max(0.0, float(extra_wait))
    interval = _wiki_min_interval()
    path = _WIKI_GATE
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0
            now = time.time()
            wait = max(last + interval - now, wait_extra)
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            handle.seek(0)
            handle.truncate()
            handle.write(f"{now:.6f}")
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _bing_min_interval() -> float:
    try:
        return max(0.0, float(os.getenv("TIR_BING_MIN_INTERVAL", "0.15")))
    except ValueError:
        return 0.15


def bing_throttle(extra_wait: float = 0.0) -> None:
    """Serialize Bright Data SERP calls across runners."""
    wait_extra = max(0.0, float(extra_wait))
    interval = _bing_min_interval()
    path = _BING_GATE
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.seek(0)
            raw = handle.read().strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0
            now = time.time()
            wait = max(last + interval - now, wait_extra)
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            handle.seek(0)
            handle.truncate()
            handle.write(f"{now:.6f}")
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _bd_timeout() -> Optional[float]:
    """Read deadline in seconds. Unset, 0, or invalid means wait until the body finishes."""
    raw = os.getenv("BRIGHT_DATA_TIMEOUT", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _bd_max_retries() -> int:
    raw = os.getenv("BRIGHT_DATA_MAX_RETRIES", "").strip()
    try:
        return max(1, int(raw or BRIGHT_DATA_MAX_RETRIES or 2))
    except (TypeError, ValueError):
        return 2


def wiki_get(url: str, *, params: Optional[Dict[str, str]] = None, timeout: float = 15.0) -> Tuple[str, str]:
    """GET wikipedia.org via the reverse proxy, with UA + 429 backoff."""
    proxies = wiki_proxy_map()
    if proxies is None:
        return "", "proxy_down"
    last_err = "HTTP 429"
    for attempt in range(4):
        wiki_throttle()
        try:
            resp = requests.get(
                url,
                params=params,
                headers=_WIKI_API_HEADERS,
                proxies=proxies,
                timeout=timeout,
            )
        except Exception as exc:
            return "", f"{type(exc).__name__}: {exc}"
        if resp.status_code in (429, 503):
            raw_retry = str(resp.headers.get("Retry-After") or "").strip()
            try:
                wait = float(raw_retry) if raw_retry else (1.0 * (2 ** attempt))
            except ValueError:
                wait = 1.0 * (2 ** attempt)
            wait = min(max(wait, 0.5), 8.0)
            last_err = f"HTTP {resp.status_code}"
            wiki_throttle(extra_wait=wait)
            continue
        if resp.status_code >= 400:
            return "", f"HTTP {resp.status_code}"
        return resp.text or "", ""
    return "", last_err


def proxy_candidates() -> List[Optional[Dict[str, str]]]:
    """Deprecated mixed list. Prefer wiki_proxy_candidates or direct-only."""
    out: List[Optional[Dict[str, str]]] = []
    mapped = wiki_proxy_map()
    if mapped:
        out.append(mapped)
    out.append(None)
    return out


def is_wikipedia_host(url: str) -> bool:
    host = _url_host(url)
    return host.endswith("wikipedia.org") or host.endswith("wikimedia.org")


def _request(method: str, url: str, headers: Optional[Dict[str, str]] = None,
             proxies: Optional[Dict[str, str]] = None, data: Optional[Dict[str, str]] = None,
             timeout: float = 8.0) -> Tuple[str, str]:
    try:
        if method.upper() == "POST":
            resp = requests.post(url, headers=headers, proxies=proxies, data=data, timeout=timeout)
        else:
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=timeout)
        if resp.status_code >= 400:
            return "", f"{method} {resp.status_code}"
        return resp.text or "", ""
    except Exception as e:
        return "", f"{method} {type(e).__name__}: {e}"


# --- Bright Data SERP API helper ---------------------------------------------

def desc_char_limit() -> int:
    """Per-entry description cap. Read on each call so .env and tests can change it."""
    raw = os.getenv("TIR_SEARCH_DESC_CHARS", "200").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 200


def truncate_description(text: str, limit: Optional[int] = None) -> str:
    """Strip tags, collapse whitespace, then cut to ``TIR_SEARCH_DESC_CHARS``."""
    plain = html.unescape(re.sub(r"<[^>]+>", " ", str(text or "")))
    plain = re.sub(r"\s+", " ", plain).strip()
    n = desc_char_limit() if limit is None else max(0, int(limit))
    if n <= 0 or not plain:
        return ""
    if len(plain) <= n:
        return plain
    return plain[:n] + " ...[truncated]"


def _serp_zones() -> List[str]:
    raw = os.getenv("BRIGHT_DATA_SERP_ZONES", "").strip()
    zones = [part.strip() for part in raw.split(",") if part.strip()]
    if zones:
        return zones
    one = (os.getenv("BRIGHT_DATA_SERP_ZONE", "") or BRIGHT_DATA_SERP_ZONE or "serp_api1").strip()
    return [one or "serp_api1"]


def _next_zone() -> str:
    global _zone_idx
    zones = _serp_zones()
    with _zone_lock:
        zone = zones[_zone_idx % len(zones)]
        _zone_idx += 1
        return zone


def _organic_hits(body: object, max_results: int) -> List[Dict[str, str]]:
    if not isinstance(body, dict):
        return []
    organic = body.get("organic") or []
    hits: List[Dict[str, str]] = []
    for item in organic[:max_results]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("link") or item.get("url") or "").strip()
        if title and url:
            hits.append({
                "title": title,
                "url": url,
                "description": str(item.get("description") or item.get("snippet") or ""),
            })
    return hits


def _hits_from_brightdata_response(resp: object, max_results: int) -> Tuple[List[Dict[str, str]], str]:
    """Parse a Bright Data /request envelope. Empty body is not JSON."""
    text = str(getattr(resp, "text", None) or "")
    try:
        envelope = resp.json()  # type: ignore[union-attr]
    except Exception:
        hits = _parse_bing(text, max_results)
        if hits:
            return hits, ""
        return [], f"non-json len={len(text)}"
    if not isinstance(envelope, dict):
        return [], "unexpected envelope"
    if envelope.get("organic"):
        hits = _organic_hits(envelope, max_results)
        if hits:
            return hits, ""
    upstream = envelope.get("status_code")
    try:
        up_code = int(upstream) if upstream is not None else 200
    except (TypeError, ValueError):
        up_code = 200
    headers = envelope.get("headers") if isinstance(envelope.get("headers"), dict) else {}
    proxy_status = str(headers.get("proxy-status") or headers.get("Proxy-Status") or "")
    body = envelope.get("body")
    if up_code in (407, 401, 403) or "http_request_denied" in proxy_status.lower():
        return [], f"upstream {up_code} denied"
    if up_code in (429, 503) or up_code >= 500:
        return [], f"upstream {up_code}"
    if isinstance(body, dict):
        hits = _organic_hits(body, max_results)
        if hits:
            return hits, ""
        html = body.get("html") if isinstance(body.get("html"), str) else ""
        if html:
            hits = _parse_bing(html, max_results)
            if hits:
                return hits, ""
    elif isinstance(body, str) and body.strip():
        stripped = body.strip()
        if stripped[:1] in "{[":
            try:
                inner = json.loads(stripped)
            except json.JSONDecodeError:
                inner = None
            if isinstance(inner, dict):
                hits = _organic_hits(inner, max_results)
                if hits:
                    return hits, ""
        hits = _parse_bing(body, max_results)
        if hits:
            return hits, ""
        return [], f"empty organic body_len={len(body)}"
    if up_code != 200:
        return [], f"upstream {up_code} empty body"
    return [], "empty body"


def _brightdata_search(query: str, max_results: int = 10) -> Tuple[List[Dict[str, str]], str]:
    """Call Bright Data Bing SERP and return (hits, error).

    Hits are ``{"title", "url", "description"}``. ``description`` is not truncated
    here; ``format_title_url`` applies ``TIR_SEARCH_DESC_CHARS``.
    """
    api_key = (os.getenv("BRIGHT_DATA_API_KEY", "") or BRIGHT_DATA_API_KEY).strip()
    if not api_key:
        return [], "BRIGHT_DATA_API_KEY not set"

    search_url = (
        "https://www.bing.com/search?q="
        + urllib.parse.quote_plus(query)
        + f"&count={max_results}&mkt=en-US&setlang=en"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    formats = ("parsed_bing", "parsed")
    last_err = ""
    fmt_idx = 0
    retries = _bd_max_retries()
    timeout = _bd_timeout()
    proxy = _env_proxy()
    proxies = {"http": proxy, "https": proxy} if proxy else None
    for attempt in range(retries):
        bing_throttle()
        payload = {
            "zone": _next_zone(),
            "url": search_url,
            "format": "json",
            "data_format": formats[min(fmt_idx, len(formats) - 1)],
        }
        try:
            resp = requests.post(
                BRIGHT_DATA_BASE_URL,
                headers=headers,
                json=payload,
                # Connect still fails fast. Read timeout is optional so a slow SERP body is not cut off.
                timeout=(30.0, timeout),
                proxies=proxies,
            )
            if resp.status_code in (429, 503) or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                wait = min(8.0, 1.0 * (2 ** attempt))
                bing_throttle(extra_wait=wait)
                continue
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                continue
            hits, err = _hits_from_brightdata_response(resp, max_results)
            if hits:
                return hits, ""
            last_err = err or "empty organic results"
            if "denied" in last_err:
                break
            fmt_idx += 1
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < retries - 1:
            time.sleep(1.5 * (attempt + 1))

    return [], f"brightdata_failed: {last_err}"


_BD_ZONE_DENIED = False


def _brightdata_enabled() -> bool:
    mode = (os.getenv("BRIGHT_DATA_MODE", "") or BRIGHT_DATA_MODE or "auto").strip().lower()
    api_key = (os.getenv("BRIGHT_DATA_API_KEY", "") or BRIGHT_DATA_API_KEY).strip()
    if _BD_ZONE_DENIED and mode in ("auto",):
        return False
    return mode in ("auto", "force", "1", "true", "yes", "on") and bool(api_key)


# --- Search-result cache (cache-first wiring) ------------------------------
#
# Contract shared with scripts/build_search_cache.py (producer) and
# mas/tools/search_cache.py (consumer kernel): JSON with keys
#   "query::<condensed-query>" -> SERP text with the per-tool return header
#   "url::<page-url>"          -> page plain text (<=2000 chars)
# and the value "search_unavailable_in_cache" marking an explicit
# network-failed gap (distinct from a plain miss). Enabled ONLY when env
# MAS_SEARCH_CACHE is set (path or "default"); when unset the whole layer
# is bypassed so non-cache deployments are byte-identical to before.
# TIR_OFFLINE_SEARCH=1 keeps priority over the cache (checked first in
# each tool, unchanged).

# Instruction tails appended by datasets, e.g. "?When ready, output the
# final answer enclosed in <answer> and </answer> tags. Do not generate
# any content after the </answer> tag." Never valid in a search query.
_CACHE_TAIL_RE = re.compile(
    r"(?i)\??\s*when\s+ready\s*,?\s*output\s+the\s+final\s+answer.*$"
    r"|\?\s*when\s+ready\b.*$"
    r"|do\s+not\s+generate\s+any\s+content\b.*$",
    re.S,
)

# Imperative prefixes the planner prepends ("Search the web for: ...").
_CACHE_PREFIX_RE = re.compile(
    r"(?i)^\s*(?:please\s+)?"
    r"(?:search\s+(?:the\s+web|wikipedia|the\s+internet|online)?\s*for"
    r"|look\s+up"
    r"|find\s+(?:information\s+)?(?:about|on)"
    r"|fetch\s+(the\s+)?(?:url|page|website)"
    r"|compute"
    r"|calculate"
    r")\s*[:：]?\s*"
)

_CACHE_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "was", "are", "were", "be", "been", "his", "her", "its", "their",
    "before", "after", "answering", "please", "prefer",
    # Imperative/instruction-tail vocabulary: never useful as search keys.
    "search", "web", "internet", "online", "look", "up", "find", "about",
    "when", "ready", "output", "final", "answer", "generate", "any",
    "content", "tag", "tags", "enclosed", "after", "do", "not", "did",
    "what", "which", "who", "where", "whom", "whose", "how",
}


def _cache_query_key(query: str) -> str:
    """Normalise a raw planner query into the cache's "query::" key.

    Equivalent behavioural copy of user_space/projects/agentflow/adapted/
    context.py::_condense_query (which this module must NOT import to
    avoid coupling the RL kernel to user_space). Any change here MUST be
    mirrored in the producer's key normalisation; consistency is locked
    by mas/tests/test_search_cache.py.
    """
    q = str(query or "").strip()
    if not q:
        return q
    # 1) Strip dataset instruction tails BEFORE the short-circuit.
    text = _CACHE_TAIL_RE.sub("", q)
    # 2) Strip planner imperative prefixes.
    text = _CACHE_PREFIX_RE.sub("", text).strip()
    # 3) Already short: assume it is a keyword query.
    if len(text) <= 48 and "?" not in text:
        return text
    # 4) Cut at the first "?"; drop interrogative + filler patterns.
    text = text.split("?", 1)[0]
    text = re.sub(r"(?i)^(what|which|who|where|when|whom|whose|how)\s+(is|was|are|were|did|does|do|has|have|had)\b", " ", text)
    text = re.sub(r"(?i)\b(the actor who|the actress who|the person who|who played|that appeared in|which appeared in|use (wikipedia|web|google)[^.]*\.?)", " ", text)
    text = re.sub(r"[?.!]+$", " ", text)
    text = re.sub(r"^(?:What|Which|Who|Where|When|Whom|Whose|How)\s+", "", text, flags=re.I)
    # Entity-first keyword window.
    words = text.split()
    keep: List[str] = []
    for i, w in enumerate(words):
        lowered = w.lower().strip(".,;:!?\"'()")
        if lowered in _CACHE_STOPWORDS:
            continue
        is_entity = bool(re.match(r"^[A-Z0-9]", w)) or w.isdigit()
        is_year = bool(re.match(r"^\d{4}$", w))
        if is_entity and not is_year:
            keep.append(w)
        elif is_year and keep:
            keep.append(w)
        elif keep and len(keep) <= 5 and w not in ("played", "role", "title"):
            continue
    keep2 = [w for w in (keep or words) if w.lower().strip(".,;:!?\"'()") not in _CACHE_STOPWORDS]
    if len(keep2) > 8:
        keep2 = keep2[:8]
    out = " ".join(keep2).strip()
    return out or q


def _cache_active() -> bool:
    """Cache layer on/off: ONLY the MAS_SEARCH_CACHE env var decides."""
    return bool(os.getenv("MAS_SEARCH_CACHE", "").strip())


def _cache_consume_query(query: str) -> Optional[str]:
    """Three-branch cache consumption for SERP queries. Never raises.

    hit -> cached text verbatim.
    gap -> clean "search_unavailable: <sentinel>" WITHOUT retrying the
           network: the producer already recorded this query as
           network-failed, and burning proxy-hops on a known failure is
           exactly what the cache exists to prevent.
    miss -> None (call existing network logic).
    """
    try:
        if not _cache_active():
            return None
        text, hit_type = cache_lookup("query", _cache_query_key(query))
        if hit_type == "hit":
            return text
        if hit_type == "gap":
            return f"search_unavailable: {GAP_SENTINEL}"
        return None
    except Exception:
        # Cache failure must never break a live search.
        return None


def _cache_consume_url(url: str) -> Optional[str]:
    """Cached page body on hit; None on miss; gap short-circuits (see above).

    NOTE: returns the body text WITHOUT the "url: <url>" header — the
    header is added by fetch_page so cached and fresh outputs are
    byte-identical.
    """
    try:
        if not _cache_active():
            return None
        text, hit_type = cache_lookup("url", url)
        if hit_type == "hit":
            return text
        if hit_type == "gap":
            return f"fetch_unavailable: {GAP_SENTINEL}"
        return None
    except Exception:
        return None


def _cache_store_query(query: str, value: str) -> None:
    """Write-back a successful fresh SERP result. Never store failures as gaps."""
    try:
        if not _cache_active():
            return
        text = str(value or "")
        low = text.lower()
        if (
            not text
            or text == GAP_SENTINEL
            or low.startswith(("search_unavailable", "bing unreachable", "error"))
        ):
            return
        cache_store("query", _cache_query_key(query), text)
    except Exception:
        pass


def _cache_store_url(url: str, value: str) -> None:
    """Write-back a successful fresh page body under "url::<url>"."""
    try:
        if not _cache_active():
            return
        cache_store("url", url, value)
    except Exception:
        pass


# --- Google-style SERP kernels (DuckDuckGo endpoints; Bing removed) ---------

def _parse_ddg_lite(html: str, limit: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    hits: List[Dict[str, str]] = []
    for link in soup.select("a.result-link"):
        title = " ".join(link.get_text(" ", strip=True).split())
        href = link.get("href") or ""
        if href.startswith("//duckduckgo.com/l/?uddg="):
            try:
                href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
            except Exception:
                pass
        if title and href.startswith("http"):
            hits.append({"title": title, "url": href})
        if len(hits) >= limit:
            break
    return hits


def _parse_ddg_html(html: str, limit: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    hits: List[Dict[str, str]] = []
    for res in soup.select("div.result, div.web-result"):
        a = res.select_one("a.result__a")
        if a is None:
            continue
        title = " ".join(a.get_text(" ", strip=True).split())
        href = a.get("href") or ""
        if href.startswith("//duckduckgo.com/l/?uddg=") or "uddg=" in href:
            try:
                href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
            except Exception:
                pass
        if title and href.startswith("http"):
            hits.append({"title": title, "url": href})
        if len(hits) >= limit:
            break
    return hits


def _is_anomaly_page(html: str) -> bool:
    """DDG bot risk-control page: 200 OK but zero organic results."""
    head = (html or "")[:5000]
    return bool(re.search(r"anomaly|challenge-detected|captcha|blocked", head, re.I))


def _parse_bing(html: str, limit: int) -> List[Dict[str, str]]:
    """Bing SERP: li.b_algo > h2 a is the canonical organic-result shape.

    Bing wraps outbound links in /ck/a?...&u=a1<base64url>; decode back so
    the agent gets a fetchable URL instead of a redirector.
    """
    soup = BeautifulSoup(html or "", "html.parser")
    hits: List[Dict[str, str]] = []
    for li in soup.select("li.b_algo"):
        a = li.select_one("h2 a")
        if a is None:
            continue
        title = " ".join(a.get_text(" ", strip=True).split())
        href = _unpack_bing_ck(a.get("href") or "")
        cap = li.select_one(".b_caption p, .b_caption")
        desc = " ".join(cap.get_text(" ", strip=True).split()) if cap is not None else ""
        if title and href.startswith("http"):
            hits.append({"title": title, "url": href, "description": desc})
        if len(hits) >= limit:
            break
    return hits


def _unpack_bing_ck(href: str) -> str:
    """Decode bing.com/ck/a redirector URLs to the real target."""
    if "bing.com/ck" not in (href or ""):
        return href or ""
    try:
        u = urllib.parse.parse_qs(urllib.parse.urlparse(href).query).get("u", [""])[0]
    except Exception:
        return ""
    if u.startswith("a1"):
        import base64

        try:
            payload = u[2:]
            pad = "=" * (-len(payload) % 4)
            return base64.urlsafe_b64decode(payload + pad).decode("utf-8", "ignore")
        except Exception:
            return ""
    return u


_RELEVANCE_STOP = {
    "from", "with", "that", "this", "film", "home", "search", "about",
    "which", "where", "when", "what", "have", "been", "were", "your",
    "into", "over", "after", "before", "their", "there", "would",
    "could", "should", "wikipedia",
}


def _query_tokens(query: str) -> set:
    """Distinctive tokens. Short words like ``ada`` match unrelated ADA sites."""
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]{4,}", query or "")
        if token.lower() not in _RELEVANCE_STOP
    }


def _relevant_hits(query: str, hits: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop SERP cards that share no distinctive token with the query.

    Datacenter Bing sometimes returns ``li.b_algo`` cards for hotels, VPNs, or
    other injected links. Those must not count as a successful search.
    Queries with no Latin token of length >= 4 keep the parsed hits.
    """
    tokens = _query_tokens(query)
    if not tokens:
        return list(hits)
    kept: List[Dict[str, str]] = []
    for hit in hits:
        blob = " ".join(
            str(hit.get(key) or "") for key in ("title", "description", "url")
        ).lower()
        if any(token in blob for token in tokens):
            kept.append(hit)
    return kept


def _is_empty_bing_serp(html: str) -> bool:
    low = (html or "").lower()
    return "there are no results" in low or 'class="b_no"' in low


def format_title_url(hits: List[Dict[str, str]], header: str = "") -> str:
    """Numbered title, URL, and truncated description. Does not include page text."""
    lines: List[str] = []
    if header:
        lines.append(header)
    for i, hit in enumerate(hits, 1):
        lines.append(f"{i}. {hit.get('title') or ''}")
        lines.append(f"   URL: {hit.get('url') or ''}")
        desc = truncate_description(str(hit.get("description") or ""))
        if desc:
            lines.append(f"   {desc}")
    return "\n".join(lines)


def bing_search(query: str, top_k: int = _MAX_RESULTS) -> str:
    """Bing search kernel: top_k entries of title, URL, and truncated description.

    Does not open pages and does not fall back to Wikipedia.
    """
    query = (query or "").strip()
    if not query:
        return "Error: empty search query."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"
    cached = _cache_consume_query(query)
    if cached is not None:
        return cached

    # Bright Data SERP API (primary when enabled)
    bd_err = ""
    if _brightdata_enabled():
        hits, err = _brightdata_search(query, max_results=top_k)
        if hits:
            out = format_title_url(hits[:top_k], header=f"bing results for: {query}")
            _cache_store_query(query, out)
            return out
        bd_err = err or "brightdata_failed"
        mode = (os.getenv("BRIGHT_DATA_MODE", "") or BRIGHT_DATA_MODE or "auto").strip().lower()
        # 407 is a zone denial. Timeouts are the same for this process:
        # retrying api.brightdata.com on every query stalls the rollout.
        low_err = bd_err.lower()
        if mode == "auto" and any(
            marker in low_err for marker in ("407", "denied", "timeout", "connectionerror")
        ):
            global _BD_ZONE_DENIED
            _BD_ZONE_DENIED = True
        if mode in ("force", "1", "true", "yes", "on"):
            return f"search_unavailable: {bd_err}"

    # Last-resort datacenter HTML. Do not send Bing through SCIENCE_INFRA_PROXY.
    # This egress gets an explicit empty SERP ("There are no results") or
    # li.b_algo cards that do not mention the query. Only token-overlapping
    # cards are returned; otherwise Wikipedia is the fallback.
    q = re.sub(r"\s+", " ", query)[:160]
    encoded = urllib.parse.quote(q)
    errors: List[str] = []
    for proxies in (None,):
        label = "direct"
        bing_throttle()
        hdrs = dict(_UA_HEADERS)
        hdrs["Referer"] = "https://www.bing.com/"
        html, err = _request(
            "GET",
            "https://www.bing.com/search?q=" + encoded + "&count=20&mkt=en-US&setlang=en",
            headers=hdrs,
            proxies=proxies,
            timeout=6.0,
        )
        if err:
            errors.append(f"{label}: {err}")
            continue
        parsed = _parse_bing(html, top_k)
        organic = _relevant_hits(q or query, parsed)
        if organic:
            out = format_title_url(organic[:top_k], header=f"bing results for: {query}")
            _cache_store_query(query, out)
            return out
        if parsed:
            why = f"irrelevant organic n={len(parsed)} html={len(html)}"
        elif _is_empty_bing_serp(html):
            why = f"empty_serp html={len(html)}"
        elif _is_anomaly_page(html):
            why = "anomaly page (bot risk control)"
        else:
            why = f"0 results html={len(html)}"
        errors.append(f"{label}: {why}")
    wiki_hits = _wikipedia_search_hits("en", q or query)
    if wiki_hits:
        out = format_title_url(wiki_hits[:top_k], header=f"wikipedia results for: {query}")
        _cache_store_query(query, out)
        return out
    detail = "; ".join(errors[-4:])
    if bd_err:
        return f"search_unavailable: {bd_err}; Bing unreachable: {detail}"
    return f"search_unavailable: Bing unreachable: {detail}"


_WIKI_URL_RE = re.compile(
    r"^https?://([a-z]{2,3}(?:-[a-z]+)?)(?:\.m)?\.wikipedia\.org/wiki/([^?#]+)",
    re.I,
)
_BLOCKED_FETCH_HOSTS = frozenset({
    "imdb.com",
    "allmusic.com",
    "discogs.com",
    "peakbagger.com",
    "rottentomatoes.com",
})
_BOT_WALL_MARKERS = (
    "bot verification",
    "verifying that you are not a robot",
    "are you a robot",
    "access denied",
    "pardon our interruption",
    "enable javascript and cookies to continue",
)


def _url_host(url: str) -> str:
    try:
        host = (urllib.parse.urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def is_blocked_fetch_host(url: str) -> bool:
    host = _url_host(url)
    if not host:
        return False
    return any(host == blocked or host.endswith("." + blocked) for blocked in _BLOCKED_FETCH_HOSTS)


def wikipedia_url_parts(url: str) -> Optional[Tuple[str, str]]:
    match = _WIKI_URL_RE.match((url or "").strip())
    if not match:
        return None
    title = urllib.parse.unquote(match.group(2)).replace("_", " ").strip()
    if not title:
        return None
    return match.group(1).lower(), title


def query_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url or "")
    path = urllib.parse.unquote(parsed.path or "").rstrip("/")
    slug = path.split("/")[-1] if path else ""
    slug = slug.replace("_", " ").replace("-", " ").strip()
    host = _url_host(url)
    brand = host.split(".")[0] if host else ""
    parts = [part for part in (slug, brand) if part and part not in {"wiki", "www"}]
    return " ".join(parts).strip()


def _visible_text(html_text: str, max_chars: int) -> str:
    soup = BeautifulSoup(html_text or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())
    if len(text) > max_chars:
        return text[:max_chars] + " ...[truncated]"
    return text


def _is_bot_wall(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in _BOT_WALL_MARKERS)


def _wikipedia_search_hits(lang: str, query: str, *, _follow_suggest: bool = True) -> List[Dict[str, str]]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    suggestion = ""
    if wiki_proxy_map() is None:
        return []
    body, err = wiki_get(api, params={
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": "5",
        "format": "json",
        "utf8": "1",
    }, timeout=10.0)
    if not err and body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {}
        query_block = payload.get("query") or {}
        hits = query_block.get("search") or []
        out: List[Dict[str, str]] = []
        for hit in hits:
            title = str(hit.get("title") or "").strip()
            if not title:
                continue
            page = f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
            out.append({"title": title, "url": page})
        if out:
            return out
        suggestion = str((query_block.get("searchinfo") or {}).get("suggestion") or "").strip()
    if _follow_suggest and suggestion and suggestion.lower() != query.lower():
        return _wikipedia_search_hits(lang, suggestion, _follow_suggest=False)
    return []


def _fetch_search_fallback(url: str, *, reason: str) -> str:
    query = query_from_url(url)
    if not query:
        return f"fetch_unavailable: {reason}"
    snippets = bing_search(query, top_k=5)
    return (
        f"fetch_blocked: {url}\n"
        f"reason: {reason}\n"
        f"used_search_fallback: bing query={query}\n"
        f"{snippets}"
    )


def _repair_failed_fetch(url: str, max_chars: int, errors: List[str]) -> str:
    reason = "; ".join(errors[-4:]) if errors else "blocked_host"
    parts = wikipedia_url_parts(url)
    if parts:
        lang, title = parts
        for hit in _wikipedia_search_hits(lang, title):
            canonical = hit["url"]
            if canonical.rstrip("/") == url.rstrip("/"):
                continue
            if title.lower() == hit["title"].lower():
                continue
            page = fetch_page(canonical, max_chars=max_chars, _repair=False)
            if page.startswith("url:"):
                body = page.split("\n", 1)[-1]
                _cache_store_url(canonical, body)
                return f"url: {canonical}\ncorrected_from: {url}\n{body}"
        return _fetch_search_fallback(url, reason=f"wikipedia_miss after {reason}")
    return _fetch_search_fallback(url, reason=reason)


_HTTP_URL_RE = re.compile(r"https?://[^\s\"'<>\]\)]+", re.I)
_FETCH_URL_CAP = 3


def extract_http_urls(text: str, limit: int = _FETCH_URL_CAP) -> List[str]:
    """Unique http(s) URLs in first-seen order, capped at ``limit``."""
    seen: List[str] = []
    for match in _HTTP_URL_RE.findall(str(text or "")):
        url = match.rstrip(".,);]")
        if url and url not in seen:
            seen.append(url)
        if len(seen) >= max(0, int(limit)):
            break
    return seen


def fetch_pages(urls: List[str], max_chars: int = _MAX_SNIPPET_CHARS, *, limit: int = _FETCH_URL_CAP) -> str:
    """Open up to ``limit`` pages. One failure does not drop the others.

    Duplicate URLs in this call are skipped. Hosts on the block list are
    skipped instead of falling through to a search. A cached page is still
    returned, so the caller sees the text.
    """
    chosen: List[str] = []
    for raw in urls or []:
        for url in extract_http_urls(str(raw), limit=limit):
            if url in chosen:
                continue
            chosen.append(url)
            if len(chosen) >= max(1, int(limit)):
                break
        if len(chosen) >= max(1, int(limit)):
            break
    if not chosen:
        return "fetch_unavailable: no urls"
    blocks: List[str] = []
    for url in chosen:
        if is_blocked_fetch_host(url):
            blocks.append(f"skipped_blocked: {url}")
            continue
        blocks.append(fetch_page(url, max_chars=max_chars))
    return "\n\n".join(blocks)


def fetch_page(url: str, max_chars: int = _MAX_SNIPPET_CHARS, *, _repair: bool = True) -> str:
    """Fetch a URL and return truncated visible text (web_fetch kernel).

    Wikipedia 404s are retried via MediaWiki search on the title. Sites that
    routinely 403 (IMDb / AllMusic / Discogs / Peakbagger) skip the GET and
    fall back to Bing snippets.
    """
    url = (url or "").strip()
    if not url:
        return "Error: empty url."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"
    cached = _cache_consume_url(url)
    if cached is not None:
        return f"url: {url}\n{cached}"
    if _repair and is_blocked_fetch_host(url):
        return _repair_failed_fetch(url, max_chars, ["blocked_host"])
    if is_wikipedia_host(url):
        html_text, err = wiki_get(url, timeout=8.0)
        proxy_tries: List[Optional[Dict[str, str]]] = []
        errors: List[str] = []
        if err:
            if err == "proxy_down":
                return "fetch_unavailable: proxy_down"
            errors.append(err)
        else:
            text = _visible_text(html_text, max_chars)
            if not text:
                errors.append("empty page")
            elif _is_bot_wall(text):
                errors.append("bot_wall")
            else:
                _cache_store_url(url, text)
                return f"url: {url}\n{text}"
        if _repair:
            return _repair_failed_fetch(url, max_chars, errors)
        return "fetch_unavailable: " + "; ".join(errors[-4:])
    proxy_tries = [None]
    errors: List[str] = []
    for proxies in proxy_tries:
        html_text, err = _request("GET", url, headers=_UA_HEADERS, proxies=proxies, timeout=8.0)
        if err:
            errors.append(err)
            continue
        text = _visible_text(html_text, max_chars)
        if not text:
            errors.append("empty page")
            continue
        if _is_bot_wall(text):
            errors.append("bot_wall")
            continue
        _cache_store_url(url, text)
        return f"url: {url}\n{text}"
    if _repair:
        return _repair_failed_fetch(url, max_chars, errors)
    return "fetch_unavailable: " + "; ".join(errors[-4:])


def fetch_by_type(url: str, query: str = "", max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Open one URL and branch on Content-Type. Does not search."""
    target = (url or "").strip()
    if not target:
        return "Error: empty url."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"
    if is_blocked_fetch_host(target):
        return _repair_failed_fetch(target, max_chars, ["blocked_host"])
    errors: List[str] = []
    path = target.lower().split("?", 1)[0]
    if is_wikipedia_host(target):
        body, err = wiki_get(target, timeout=8.0)
        if err == "proxy_down":
            return "fetch_unavailable: proxy_down"
        if err:
            return f"fetch_unavailable: {err}"
        text = _visible_text(body, max_chars)
        if not text:
            return "fetch_unavailable: empty page"
        return f"url: {target}\n{text}"
    proxy_tries: List[Optional[Dict[str, str]]] = [None]
    for proxies in proxy_tries:
        try:
            resp = requests.get(target, headers=_UA_HEADERS, timeout=8.0, proxies=proxies, stream=True)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        try:
            if resp.status_code >= 400:
                errors.append(f"GET {resp.status_code}")
                continue
            ctype = str(resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if ctype.startswith("image/") or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
                size = str(resp.headers.get("Content-Length") or "unknown")
                return (
                    f"type: image\nurl: {target}\ncontent_type: {ctype or 'image'}\nbytes: {size}"
                )
            if "pdf" in ctype or path.endswith(".pdf"):
                data = resp.content[:200000]
                text = _pdf_text(data)
                if not text:
                    return f"type: pdf\nurl: {target}\nPDF has no text layer; it looks like a scan."
                return f"type: pdf\nurl: {target}\n{text[:max_chars]}"
            body = resp.text or ""
            if "html" in ctype or body.lstrip().startswith("<"):
                text = _visible_text(body, max_chars)
                kind = "html"
            else:
                text = " ".join(body.split())
                if len(text) > max_chars:
                    text = text[:max_chars] + " ...[truncated]"
                kind = "text"
            if _is_bot_wall(text):
                errors.append("bot_wall")
                continue
            lines = [f"type: {kind}", f"url: {target}"]
            q = (query or "").strip()
            if q:
                lines.append(f"query: {q}")
            lines.append(text or "(empty page)")
            return "\n".join(lines)
        finally:
            resp.close()
    return _repair_failed_fetch(target, max_chars, errors)


def _pdf_text(data: bytes) -> str:
    try:
        import io

        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(data or b""))
        parts: List[str] = []
        for page in list(reader.pages)[:3]:
            parts.append(page.extract_text() or "")
        return " ".join(" ".join(parts).split())
    except Exception:
        return ""
