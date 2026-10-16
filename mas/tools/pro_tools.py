"""HIVE-style retrieval for tool-agents marked tier=pro.

Tool isolation contract (lite/pro tiers):
  - ``wikipedia_search`` / ``bing_search``: both tiers return title, URL, and a
    truncated description. Neither tier opens a result page.
  - ``web_fetch`` / ``web_search``: url -> page content. Pro tier adds
    query-relevant sentence highlights. NEVER searches.

Uses the existing HTTP kernels. Vendored imports are late (function-body only)
via :func:`tools.tool_agents._import_epc_aw_tool` so a missing legacy tree
degrades to fetch_page excerpts instead of crashing the module (isolation
semantics locked by mas/tests/test_legacy_isolation.py).
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

Kernel = Optional[Callable[[Dict[str, Any]], str]]


def _offline() -> bool:
    return os.getenv("TIR_OFFLINE_SEARCH", "").strip().lower() in ("1", "true", "yes")


def _terms(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def select_titles(query: str, titles: List[str], limit: int = 3) -> List[str]:
    """Pick titles that share terms with the query. Empty overlap keeps the first title."""
    scored = []
    query_terms = _terms(query)
    for index, title in enumerate(titles):
        scored.append((len(query_terms & _terms(title)), -index, title))
    scored.sort(reverse=True)
    picked = [title for score, _index, title in scored if score > 0][:limit]
    if not picked and titles:
        return titles[:1]
    return picked


def wiki_page_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote((title or "").replace(" ", "_"))


def _fetch_top_n(payload: Dict[str, Any]) -> int:
    """Pro scan depth: args.fetch_top_n -> env MAS_PRO_FETCH_N -> 3; clamp 1-10."""
    raw = payload.get("fetch_top_n") or os.getenv("MAS_PRO_FETCH_N", "3")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 3
    return max(1, min(10, value))


# Module-level singleton for the vendored Web_Search_Tool (design §5.4):
# __init__ builds a ChatOpenAI client + embedding config on every call, and
# tool_agents gets reloaded by test_legacy_isolation.py attribute snapshots —
# the cache therefore lives HERE, not there.
_RAG_TOOL: Any = None
_RAG_TOOL_FAILED = False  # sticky failure: never retry the vendored import


def _ensure_mas_alias() -> None:
    """Register ``sys.modules["MAS"]`` for the vendored engine, lazily.

    The vendored tools import each other via ``from MAS.epc_aw...``; the only
    sanctioned registration point is ``mas_agent.bootstrap_mas_package``
    (which imports agentlightning — too heavy for the product stack). This
    mirrors its anti-hijack rules exactly: never overwrite an existing
    registration that already points at legacy/structagent, never register
    when the vendored tree is absent (keeping the explicit-ImportError
    degradation semantics of test_legacy_isolation intact).
    """
    import sys
    import types
    from pathlib import Path

    struct_root = Path(__file__).resolve().parents[2] / "legacy" / "structagent"
    if not (struct_root / "epc_aw").is_dir():
        return
    existing = sys.modules.get("MAS")
    if existing is not None:
        paths = [str(Path(p).resolve()) for p in (getattr(existing, "__path__", []) or [])]
        if paths == [str(struct_root.resolve())]:
            return  # already ours; foreign registrations are left alone here
        # a foreign MAS pointing elsewhere stays untouched (anti-hijack is
        # bootstrap's job); importing through it would be wrong, so bail out
        # and let the ImportError fallback handle the degradation.
        if paths:
            return
    pkg = types.ModuleType("MAS")
    pkg.__file__ = str(struct_root / "__init__.py")
    pkg.__path__ = [str(struct_root)]  # type: ignore[attr-defined]
    sys.modules["MAS"] = pkg


def _get_rag_tool() -> Any:
    """Lazy singleton around legacy Web_Search_Tool. Never raises."""
    global _RAG_TOOL, _RAG_TOOL_FAILED
    if _RAG_TOOL is not None or _RAG_TOOL_FAILED:
        return _RAG_TOOL
    try:
        from .tool_agents import _import_epc_aw_tool, _model_name

        _ensure_mas_alias()
        web_search_tool = _import_epc_aw_tool("epc_aw.tools.web_search.tool", "Web_Search_Tool")
        _RAG_TOOL = web_search_tool(model_string=_model_name())
    except Exception:  # noqa: BLE001 — ImportError semantics: degrade, never crash
        _RAG_TOOL = None
        _RAG_TOOL_FAILED = True
    return _RAG_TOOL


def _raw_excerpt(query: str, url: str) -> str:
    """fetch_page fallback excerpt, truncated. Returns a marked string."""
    try:
        from .search import fetch_page

        page = fetch_page(url, max_chars=800)
        text = str(page or "").strip()
        if not text or text.startswith(("fetch_unavailable", "Error", "search_unavailable")):
            return f"[rag_failed, raw_excerpt] {text or 'empty page'}"
        return "[rag_failed, raw_excerpt] " + text
    except Exception as exc:  # noqa: BLE001
        return f"[rag_failed, raw_excerpt] fetch_page error: {exc}"


def _rag_summary(query: str, url: str) -> str:
    """Web RAG summary of one URL: vendored Web_Search_Tool pipeline
    (fetch -> chunk 200/20 -> embedding rank -> LLM summary).

    Never raises: on vendored-import failure, instantiation failure or an
    execute exception it falls back to a truncated ``fetch_page`` excerpt
    marked ``[rag_failed, raw_excerpt]``. The vendored ``execute`` already
    swallows its own errors into an ``Error processing request: ...`` string,
    which we pass through instead of double-fetching.
    """
    query = str(query or "").strip()
    url = str(url or "").strip()
    if not url:
        return "[rag_failed] empty url"
    tool = _get_rag_tool()
    if tool is None:
        return _raw_excerpt(query, url)
    try:
        summary = str(tool.execute(query=query, url=url) or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"[rag_failed] {type(exc).__name__}: {exc}"
    if not summary:
        return _raw_excerpt(query, url)
    return summary


def _rag_section(query: str, entries: List[Dict[str, Any]], fetch_top_n: int) -> str:
    """``retrieved_information:`` block over the first ``fetch_top_n`` entries.

    ``entries`` items provide ``index`` (1-based SERP rank) and ``url``.
    A single-URL failure never aborts the section (per-entry degrade, mirrors
    the EPC-AW google tool's ``abstract_fetch_failed`` semantics). Returns an
    empty string when no fetchable URL is in range (no dangling header).
    """
    picked = [
        entry
        for entry in entries[: max(1, fetch_top_n)]
        if str(entry.get("url") or "").strip()
    ]
    if not picked:
        return ""
    lines = ["retrieved_information:"]
    for entry in picked:
        url = str(entry.get("url") or "").strip()
        lines.append(f"[{entry.get('index')}] URL: {url}")
        lines.append(f"    rag: {_rag_summary(query, url)}")
    return "\n".join(lines)


def _lite(kernel: Kernel, args: Dict[str, Any]) -> str:
    if kernel is None:
        return ""
    try:
        return str(kernel(args))
    except Exception as exc:  # noqa: BLE001
        return f"lite error: {exc}"


def _fallback(kernel: Kernel, args: Dict[str, Any], body: str = "") -> str:
    lite = body or _lite(kernel, args)
    return f"tier=pro pro_fallback\n{lite}".rstrip()


def _web_pro(query: str, url: str, kernel: Kernel, args: Dict[str, Any]) -> str:
    """Fetch one url and highlight query-relevant sentences. Never searches."""
    if _offline():
        return _fallback(kernel, args)
    try:
        from .search import fetch_page
    except Exception as exc:  # noqa: BLE001
        return _fallback(kernel, args, f"pro setup failed: {exc}")
    page = fetch_page(url, max_chars=4000)
    if not page or page.startswith(("fetch_unavailable", "Error", "search_unavailable")):
        return _fallback(kernel, args, page)
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", page) if part.strip()]
    query_terms = _terms(query)
    matches = [sentence for sentence in sentences if query_terms & _terms(sentence)][:5]
    body = page
    if matches:
        body = page + "\n\nquery_matches:\n" + "\n".join(matches)
    return f"tier=pro\n{body}"


def run_pro(name: str, args: Dict[str, Any], kernel: Kernel) -> str:
    """Run the strengthened tool. Failures keep the lite text and say pro_fallback."""
    payload = args if isinstance(args, dict) else {}
    try:
        if name in ("wikipedia_search", "bing_search"):
            query = str(payload.get("query") or "")
            if not query.strip():
                label = "wikipedia" if name == "wikipedia_search" else "search"
                return _fallback(kernel, payload, f"Error: empty {label} query.")
            lite = _lite(kernel, payload)
            return f"tier=pro\n{lite}".rstrip()
        if name in ("web_search", "web_fetch"):
            return _web_pro(str(payload.get("query") or ""), str(payload.get("url") or ""), kernel, payload)
        if name == "python_coder":
            lite = _lite(kernel, payload)
            return f"tier=pro\n{lite}".rstrip()
        if name in ("think", "base_generator"):
            text = str(payload.get("text") or payload.get("query") or payload.get("input") or "")
            return f"tier=pro\n{text}".rstrip()
    except Exception as exc:  # noqa: BLE001
        return _fallback(kernel, payload, f"pro error: {exc}")
    return _fallback(kernel, payload, f"unknown tool-agent {name}")
