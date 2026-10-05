"""Chrome-free web search via DuckDuckGo HTML (adapted from epc_aw local_google_scraper).

No Gemini, Playwright, or in-tool LLM summarizer — returns raw snippets for RL.
"""

from __future__ import annotations

import os
import re
import urllib.parse
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

_MAX_SNIPPET_CHARS = int(os.getenv("TIR_SEARCH_MAX_CHARS", "2000"))
_MAX_RESULTS = int(os.getenv("TIR_SEARCH_MAX_RESULTS", "5"))
_DEFAULT_PROXY = "http://127.0.0.1:7890"


def _normalize_proxy(proxy: str) -> str:
    proxy = proxy.strip()
    if proxy and "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def _env_proxy() -> str:
    return _normalize_proxy(
        os.getenv("WEB_SEARCH_PROXY", "").strip()
        or os.getenv("GOOGLE_CHROME_PROXY", "").strip()
        or os.getenv("CHROME_PROXY", "").strip()
    )


def proxy_candidates() -> List[Optional[Dict[str, str]]]:
    """Try explicit env proxy, then local Clash (7890), then direct."""
    seen: List[str] = []
    ordered: List[Optional[Dict[str, str]]] = []
    for raw in (_env_proxy(), _DEFAULT_PROXY):
        if not raw or raw in seen:
            continue
        seen.append(raw)
        ordered.append({"http": raw, "https": raw})
    ordered.append(None)
    return ordered


def _proxies() -> Optional[Dict[str, str]]:
    """First proxy candidate (env or default Clash). Kept for wikipedia import."""
    cands = proxy_candidates()
    return cands[0] if cands else None


def _unwrap_ddg_href(href: str) -> str:
    href = (href or "").strip()
    if "uddg=" in href:
        parsed = urllib.parse.urlparse(href if href.startswith("http") else "https://duckduckgo.com" + href)
        qs = urllib.parse.parse_qs(parsed.query)
        vals = qs.get("uddg") or qs.get("u") or []
        if vals:
            return urllib.parse.unquote(vals[0])
    return href


def _parse_ddg_html(html: str, max_results: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    organic: List[Dict[str, str]] = []
    seen: set[str] = set()

    for result in soup.select("div.result, div.web-result, article"):
        link_el = result.select_one("a.result__a, a.result-link, a[href]")
        if not link_el:
            continue
        href = _unwrap_ddg_href(link_el.get("href") or "")
        if not href.startswith("http") or "duckduckgo.com" in href:
            continue
        title = link_el.get_text(" ", strip=True)
        snip_el = result.select_one("a.result__snippet, td.result-snippet, div.result__snippet")
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
        if not title or href in seen:
            continue
        seen.add(href)
        organic.append({"title": title, "link": href, "snippet": snippet})
        if len(organic) >= max_results:
            return organic
    return organic


def _format_organic(organic: List[Dict[str, str]]) -> str:
    lines = []
    for i, item in enumerate(organic, 1):
        lines.append(f"{i}. {item['title']}\n   {item['link']}\n   {item['snippet']}")
    text = "\n".join(lines)
    if len(text) > _MAX_SNIPPET_CHARS:
        text = text[:_MAX_SNIPPET_CHARS] + "\n...[truncated]"
    return text


def _parse_bing_html(html: str, max_results: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    organic: List[Dict[str, str]] = []
    seen: set[str] = set()
    for li in soup.select("li.b_algo, li.b_algoSlash"):
        link_el = li.select_one("h2 a, a[href]")
        if not link_el:
            continue
        href = (link_el.get("href") or "").strip()
        if not href.startswith("http"):
            continue
        title = link_el.get_text(" ", strip=True)
        snip_el = li.select_one("p, .b_caption p, .b_lineclamp2, .b_lineclamp3")
        snippet = snip_el.get_text(" ", strip=True) if snip_el else ""
        if not title or href in seen:
            continue
        seen.add(href)
        organic.append({"title": title, "link": href, "snippet": snippet})
        if len(organic) >= max_results:
            return organic
    return organic


def _parse_ddg_lite(html: str, max_results: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html or "", "html.parser")
    organic: List[Dict[str, str]] = []
    seen: set[str] = set()
    for a in soup.select("a.result-link"):
        href = _unwrap_ddg_href(a.get("href") or "")
        if not href.startswith("http") or "duckduckgo.com" in href:
            continue
        title = a.get_text(" ", strip=True)
        snippet = ""
        td = a.find_parent("tr")
        if td:
            snip_el = td.find_next("td", class_="result-snippet")
            if snip_el:
                snippet = snip_el.get_text(" ", strip=True)
        if not title or href in seen:
            continue
        seen.add(href)
        organic.append({"title": title, "link": href, "snippet": snippet})
        if len(organic) >= max_results:
            return organic
    return organic


def _request(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    proxies: Optional[Dict[str, str]],
    data: Optional[Dict[str, str]] = None,
    timeout: float = 8.0,
) -> Tuple[str, str]:
    try:
        if method == "POST":
            resp = requests.post(url, data=data, headers=headers, timeout=timeout, proxies=proxies)
        else:
            resp = requests.get(url, headers=headers, timeout=timeout, proxies=proxies)
        if resp.status_code >= 400:
            return "", f"{method} {resp.status_code}"
        return resp.text or "", ""
    except Exception as e:
        return "", f"{method} {type(e).__name__}: {e}"


def _wikipedia_fallback(query: str, max_results: int) -> str:
    """Last resort when SERPs are firewalled: Wikipedia search hits as snippets."""
    try:
        from .wikipedia import wikipedia_search
    except ImportError:
        from tools.wikipedia import wikipedia_search

    text = wikipedia_search(query, top_k=max_results)
    if text.lower().startswith("search_unavailable") or text.lower().startswith("error"):
        return ""
    return f"(web_search fallback via Wikipedia)\n{text}"


def search_organic(query: str, max_results: int = _MAX_RESULTS) -> Tuple[List[Dict[str, str]], str]:
    """Return SERP hits (title, link, snippet) and a short error if none were parsed."""
    query = (query or "").strip()
    if not query:
        return [], "Error: empty search query."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return [], "search_unavailable: TIR_OFFLINE_SEARCH=1"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://duckduckgo.com/",
    }
    q = re.sub(r"\s+", " ", query)[:160]
    encoded = urllib.parse.quote(q)
    endpoints = (
        ("cn_bing", "GET", "https://cn.bing.com/search?q=" + encoded, None, _parse_bing_html, "https://cn.bing.com/"),
        ("bing", "GET", "https://www.bing.com/search?q=" + encoded, None, _parse_bing_html, "https://www.bing.com/"),
        ("ddg_lite", "GET", "https://lite.duckduckgo.com/lite/?q=" + encoded, None, _parse_ddg_lite, "https://lite.duckduckgo.com/"),
        ("ddg_post", "POST", "https://html.duckduckgo.com/html/", {"q": q}, _parse_ddg_html, "https://duckduckgo.com/"),
        ("ddg_get", "GET", "https://html.duckduckgo.com/html/?q=" + encoded, None, _parse_ddg_html, "https://duckduckgo.com/"),
    )
    errors: List[str] = []
    for proxies in proxy_candidates():
        label = (proxies or {}).get("http") or "direct"
        for name, method, url, data, parser, referer in endpoints:
            hdrs = dict(headers)
            hdrs["Referer"] = referer
            html, err = _request(method, url, headers=hdrs, proxies=proxies, data=data, timeout=5.0)
            if err:
                errors.append(f"{label}/{name}: {err}")
                continue
            organic = parser(html, max_results)
            if organic:
                return organic, ""
            errors.append(f"{label}/{name}: 0 results html={len(html)}")
    return [], "search_unavailable: " + "; ".join(errors[-8:])


def web_search(query: str, max_results: int = _MAX_RESULTS) -> str:
    """Search the open web and return truncated title/snippet lines."""
    query = (query or "").strip()
    if not query:
        return "Error: empty search query."
    organic, err = search_organic(query, max_results)
    if organic:
        return _format_organic(organic)
    wiki = ""
    if not err.startswith("Error:") and "TIR_OFFLINE_SEARCH" not in err:
        wiki = _wikipedia_fallback(query, max_results)
    if wiki:
        return wiki
    return err or "search_unavailable"


def fetch_page(url: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Fetch a URL and return truncated visible text (web_search kernel)."""
    url = (url or "").strip()
    if not url:
        return "Error: empty url."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    errors: List[str] = []
    for proxies in proxy_candidates():
        html, err = _request("GET", url, headers=headers, proxies=proxies, timeout=8.0)
        if err:
            errors.append(err)
            continue
        soup = BeautifulSoup(html or "", "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = " ".join(soup.get_text(" ", strip=True).split())
        if not text:
            continue
        if len(text) > max_chars:
            text = text[:max_chars] + " ...[truncated]"
        return text
    return "fetch_unavailable: " + "; ".join(errors[-4:])


def _pdf_text(data: bytes) -> str:
    if not data or not data.startswith(b"%PDF"):
        return ""
    try:
        import io

        from PyPDF2 import PdfReader

        reader = PdfReader(io.BytesIO(data))
        parts = [(page.extract_text() or "") for page in list(reader.pages)[:3]]
        return " ".join(" ".join(parts).split())
    except Exception:
        return ""


def fetch_by_type(url: str, query: str = "", max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    """Open one URL and branch on Content-Type. Does not search."""
    target = (url or "").strip()
    if not target:
        return "Error: empty url."
    if os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes"):
        return "search_unavailable: TIR_OFFLINE_SEARCH=1"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    errors: List[str] = []
    path = target.lower().split("?", 1)[0]
    for proxies in proxy_candidates():
        try:
            resp = requests.get(target, headers=headers, timeout=8.0, proxies=proxies, stream=True)
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

