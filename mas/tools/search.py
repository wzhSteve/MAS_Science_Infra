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

import html
import html
import json
import os
import re
import threading
import time
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from .search_cache import GAP_SENTINEL, cache_lookup, cache_store  # noqa: E402

_MAX_SNIPPET_CHARS = int(os.getenv("TIR_SEARCH_MAX_CHARS", "2000"))
_MAX_RESULTS = int(os.getenv("TIR_SEARCH_MAX_RESULTS", "10"))
_DEFAULT_PROXY = "http://127.0.0.1:7890"

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


def _normalize_proxy(proxy: str) -> str:
    proxy = proxy.strip()
    if proxy and "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def _env_proxy() -> str:
    return _normalize_proxy(
        os.getenv("WEB_SEARCH_PROXY", "").strip()
        or os.getenv("SCIENCE_INFRA_PROXY", "").strip()
        or _DEFAULT_PROXY
    )


def proxy_candidates() -> List[Optional[Dict[str, str]]]:
    """Proxy first, then direct. First candidate that answers wins."""
    out: List[Optional[Dict[str, str]]] = []
    p = _env_proxy()
    if p:
        out.append({"http": p, "https": p})
    out.append(None)
    return out


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


def _brightdata_search(query: str, max_results: int = 10) -> Tuple[List[Dict[str, str]], str]:
    """Call Bright Data Bing SERP and return (hits, error).

    Hits are ``{"title", "url", "description"}``. ``description`` is not truncated
    here; ``format_title_url`` applies ``TIR_SEARCH_DESC_CHARS``.
    """
    if not BRIGHT_DATA_API_KEY:
        return [], "BRIGHT_DATA_API_KEY not set"

    search_url = (
        "https://www.bing.com/search?q="
        + urllib.parse.quote_plus(query)
        + f"&count={max_results}&mkt=en-US&setlang=en"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BRIGHT_DATA_API_KEY}",
    }
    formats = ("parsed", "parsed_bing")
    last_err = ""
    fmt_idx = 0
    for attempt in range(BRIGHT_DATA_MAX_RETRIES):
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
                timeout=BRIGHT_DATA_TIMEOUT,
            )
            if resp.status_code != 200:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                continue
            envelope = resp.json()
            body_str = envelope.get("body", "")
            body = json.loads(body_str) if isinstance(body_str, str) else body_str
            hits = _organic_hits(body, max_results)
            if hits:
                return hits, ""
            last_err = "empty organic results"
            fmt_idx += 1
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < BRIGHT_DATA_MAX_RETRIES - 1:
            time.sleep(1.5 * (attempt + 1))

    return [], f"brightdata_failed: {last_err}"


def _brightdata_enabled() -> bool:
    return BRIGHT_DATA_MODE in ("auto", "force", "1", "true", "yes", "on") and bool(BRIGHT_DATA_API_KEY)


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
    """Write-back a successful fresh SERP result for the query key."""
    try:
        if not _cache_active():
            return
        cache_store("query", _cache_query_key(query), value)
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
    if _brightdata_enabled():
        hits, err = _brightdata_search(query, max_results=top_k)
        if hits:
            out = format_title_url(hits[:top_k], header=f"bing results for: {query}")
            _cache_store_query(query, out)
            return out
        if BRIGHT_DATA_MODE == "force":
            return f"Bing unreachable: Bright Data force mode failed: {err}"

    # Local fallback
    q = re.sub(r"\s+", " ", query)[:160]
    encoded = urllib.parse.quote(q)
    errors: List[str] = []
    for proxies in proxy_candidates():
        label = (proxies or {}).get("http") or "direct"
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
        organic = _parse_bing(html, top_k)
        if organic:
            out = format_title_url(organic[:top_k], header=f"bing results for: {query}")
            _cache_store_query(query, out)
            return out
        why = "anomaly page (bot risk control)" if _is_anomaly_page(html) else f"0 results html={len(html)}"
        errors.append(f"{label}: {why}")
    return "Bing unreachable: " + "; ".join(errors[-4:])


def fetch_page(url: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Fetch a URL and return truncated visible text (web_fetch kernel)."""
    url = (url or "").strip()
    if not url:
        return "Error: empty url."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"
    cached = _cache_consume_url(url)
    if cached is not None:
        return f"url: {url}\n{cached}"
    errors: List[str] = []
    for proxies in proxy_candidates():
        html, err = _request("GET", url, headers=_UA_HEADERS, proxies=proxies, timeout=8.0)
        if err:
            errors.append(err)
            continue
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        if not text:
            return f"fetch_unavailable: empty page at {url}"
        _cache_store_url(url, text)
        return f"url: {url}\n{text}"
    return "fetch_unavailable: " + "; ".join(errors[-4:])


def fetch_by_type(url: str, query: str = "", max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Open one URL and branch on Content-Type. Does not search."""
    target = (url or "").strip()
    if not target:
        return "Error: empty url."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"
    errors: List[str] = []
    path = target.lower().split("?", 1)[0]
    for proxies in proxy_candidates():
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
                soup = BeautifulSoup(body, "html.parser")
                for tag in soup(["script", "style", "noscript"]):
                    tag.decompose()
                text = " ".join(soup.get_text(" ", strip=True).split())
                kind = "html"
            else:
                text = " ".join(body.split())
                kind = "text"
            if len(text) > max_chars:
                text = text[:max_chars] + " ...[truncated]"
            lines = [f"type: {kind}", f"url: {target}"]
            q = (query or "").strip()
            if q:
                lines.append(f"query: {q}")
            lines.append(text or "(empty page)")
            return "\n".join(lines)
        finally:
            resp.close()
    return "fetch_unavailable: " + "; ".join(errors[-4:])


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
