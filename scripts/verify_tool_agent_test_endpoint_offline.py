"""Offline end-to-end check of POST /api/mas/tool-agents/test (agent task B).

THROWAWAY verification script (NOT part of the test suite). It exercises the
real FastAPI app -> services.run_tool_agent_test -> ToolAgent.invoke ->
pro_tools.run_pro chain over httpx/TestClient with EVERY network boundary
mocked.

Network guard design (critical — an earlier draft of this script really
called Bright Data and learned the hard way):

  * The repo root holds a real ``.env`` with ``BRIGHT_DATA_API_KEY`` /
    ``OPENAI_API_KEY`` / ``WEB_SEARCH_PROXY`` credentials, and the control
    app import chain (``mas/workflow/collector.py: load_repo_dotenv()``)
    injects it into ``os.environ``. ``tools/search.py`` snapshots
    ``BRIGHT_DATA_API_KEY`` at module-import time, and ``_google_pro``'s
    Bright Data branch posts via ``requests.post`` directly — NOT through
    ``tools.search._request`` — so patching ``_request`` alone cannot stop
    a real SERP call once the key is loaded.
  * Therefore this script (a) strips every network credential from
    ``os.environ`` BEFORE any project import, and (b) additionally patches
    ``tools.search._brightdata_enabled`` to False inside each request
    context, plus (c) patches ``tools.search._request`` (fixed DDG-lite
    SERP HTML) and ``tools.pro_tools._rag_summary`` (fixed text — the
    vendored Web_Search_Tool pipeline with its requests.head probing and
    real embedding endpoint never runs).

Verified contract points:
  1. bing_search tier=pro -> 200, ok=true,
     requested_tier=pro, resolved_tier=pro (kernel switch cleared), output
     has tier=pro first line, SERP title+URL lines and a
     retrieved_information: section whose [i] entries cover exactly the
     first 2 SERP URLs.
  2. same request under SCIENCE_INFRA_TOOL_KERNEL=1 -> resolved_tier=lite
     (forced downgrade honoured by the endpoint), no RAG section.
  3. unknown agent_id -> HTTP 404.

Run:  cd <repo root> && .venv/bin/python scripts/verify_tool_agent_test_endpoint_offline.py
"""

from __future__ import annotations

import os

# --- network guard: strip credentials BEFORE any project import ------------
_CREDENTIAL_KEYS = (
    "BRIGHT_DATA_API_KEY",
    "BRIGHT_DATA_SERP_ZONE",
    "OPENAI_API_KEY",
    "OPENAI_API_BASE",
    "OPENAI_API_BASE_URL",
    "EMBEDDING_MODEL_API_KEY",
    "EMBEDDING_MODEL_URL",
    "WEB_SEARCH_PROXY",
    "SCIENCE_INFRA_PROXY",
    "MODEL_Name",
    "OPENAI_MODEL",
)
for _k in _CREDENTIAL_KEYS:
    os.environ.pop(_k, None)
os.environ["BRIGHT_DATA_MODE"] = "0"  # hard-disable even if re-loaded later
os.environ.pop("MAS_SEARCH_CACHE", None)

import sys  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest.mock import patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TIR = REPO / "mas"
for _p in (str(REPO), str(TIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DDG_LITE_HTML = """\
<html><body><ol>
  <li><a rel="nofollow" class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fada&amp;rut=abc">Ada Lovelace</a></li>
  <li><a rel="nofollow" class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Ftwo&amp;rut=def">Second</a></li>
  <li><a rel="nofollow" class="result-link" href="https://example.com/three">Third</a></li>
</ol></body></html>
"""


def _fixed_serp(method, url, **_kwargs):
    return DDG_LITE_HTML, ""


def _rag_stub(query, url):
    return f"rag evidence about {query} from {url}"


def _network_patches(rag_spy: list[str]):
    """All patches that keep the endpoint fully offline."""
    return [
        patch("tools.search._request", side_effect=_fixed_serp),
        patch("tools.search._brightdata_enabled", return_value=False),
        patch("tools.search._brightdata_search", side_effect=AssertionError("Bright Data must not be called")),
        patch(
            "tools.pro_tools._rag_summary",
            side_effect=lambda q, u: (rag_spy.append(u), _rag_stub(q, u))[1],
        ),
    ]


def main() -> int:
    failures: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        status = "PASS" if cond else "FAIL"
        print(f"[{status}] {label}{(' :: ' + detail) if detail and not cond else ''}")
        if not cond:
            failures.append(label)

    ambient = [k for k in _CREDENTIAL_KEYS if os.environ.get(k)]
    print(f"ambient network env after guard: {ambient or 'clean'}")

    from fastapi.testclient import TestClient

    from science_infra.control.app import create_app
    from science_infra.control.experiments import ensure_experiment

    # import-time sanity: the control chain loads repo .env (which re-fills
    # BRIGHT_DATA_API_KEY since we popped it); the module snapshot picks that
    # up, BUT BRIGHT_DATA_MODE was pinned to "0" in this process before any
    # import, so the effective gate stays closed:
    import tools.search as ts

    check(
        "brightdata gate closed in-process",
        not ts._brightdata_enabled(),
        f"key={bool(ts.BRIGHT_DATA_API_KEY)} mode={ts.BRIGHT_DATA_MODE}",
    )

    ensure_experiment("demo")
    client = TestClient(create_app())

    # ---- 1. pro + fetch_top_n=2, kernel switch cleared -------------------
    rag_calls: list[str] = []
    saved_kernel = os.environ.pop("SCIENCE_INFRA_TOOL_KERNEL", None)
    try:
        patches = _network_patches(rag_calls)
        for p in patches:
            p.start()
        try:
            r = client.post(
                "/api/mas/tool-agents/test",
                json={
                    "agent_id": "bing_search",
                    "tier": "pro",
                    "args": {"query": "Ada Lovelace"},
                    "fetch_top_n": 2,
                },
            )
        finally:
            for p in patches:
                p.stop()
    finally:
        if saved_kernel is not None:
            os.environ["SCIENCE_INFRA_TOOL_KERNEL"] = saved_kernel

    check("status_code == 200", r.status_code == 200, f"got {r.status_code}: {r.text[:200]}")
    body = r.json()
    print("  request: agent_id=bing_search tier=pro")
    print(
        f"  response: requested_tier={body.get('requested_tier')} resolved_tier={body.get('resolved_tier')} "
        f"ok={body.get('ok')} elapsed_ms={body.get('elapsed_ms')}"
    )
    check("requested_tier == 'pro'", body.get("requested_tier") == "pro", repr(body.get("requested_tier")))
    check("resolved_tier == 'pro'", body.get("resolved_tier") == "pro", repr(body.get("resolved_tier")))
    check("ok is True", body.get("ok") is True)
    check("elapsed_ms is int >= 0", isinstance(body.get("elapsed_ms"), int) and body["elapsed_ms"] >= 0)
    out = str(body.get("output") or "")
    print("  --- output (first 12 lines) ---")
    for line in out.splitlines()[:12]:
        print("  | " + line)
    check("output starts with tier=pro", out.startswith("tier=pro"))
    check("SERP title present", "Ada Lovelace" in out)
    check("SERP url present", "https://example.com/ada" in out)
    check("no retrieved_information section", "retrieved_information:" not in out)
    check("RAG not invoked", rag_calls == [], f"rag_calls={rag_calls}")

    # ---- 2. kernel switch forces lite even when pro is requested ----------
    rag_calls.clear()
    with patch.dict(os.environ, {"SCIENCE_INFRA_TOOL_KERNEL": "1"}):
        patches = [
            patch("tools.search._request", side_effect=_fixed_serp),
            patch("tools.search._brightdata_enabled", return_value=False),
            patch("tools.search._brightdata_search", side_effect=AssertionError("Bright Data must not be called")),
            patch(
                "tools.pro_tools._rag_summary",
                side_effect=AssertionError("RAG must not run when kernel forces lite"),
            ),
        ]
        for p in patches:
            p.start()
        try:
            r2 = client.post(
                "/api/mas/tool-agents/test",
                json={
                    "agent_id": "bing_search",
                    "tier": "pro",
                    "args": {"query": "Ada Lovelace"},
                    "fetch_top_n": 2,
                },
            )
        finally:
            for p in patches:
                p.stop()
    body2 = r2.json()
    print(
        f"kernel=1 -> requested={body2.get('requested_tier')} resolved={body2.get('resolved_tier')} ok={body2.get('ok')}"
    )
    check("kernel=1 status 200", r2.status_code == 200)
    check("kernel=1 resolved_tier == 'lite'", body2.get("resolved_tier") == "lite", repr(body2.get("resolved_tier")))
    check("kernel=1 ok True", body2.get("ok") is True)
    out2 = str(body2.get("output") or "")
    check("kernel=1 output has tier=lite", out2.startswith("tier=lite"))
    check("kernel=1 no retrieved_information", "retrieved_information:" not in out2)
    check("kernel=1 no rag calls", rag_calls == [])

    # ---- 3. think tool: no kernel, no network — full success path --------
    r_think = client.post(
        "/api/mas/tool-agents/test",
        json={"agent_id": "think", "tier": "pro", "args": {"text": "unit test thought"}},
    )
    body_t = r_think.json()
    print(
        f"think pro -> requested={body_t.get('requested_tier')} resolved={body_t.get('resolved_tier')} ok={body_t.get('ok')}"
    )
    check("think status 200", r_think.status_code == 200)
    check("think requested_tier == 'pro'", body_t.get("requested_tier") == "pro")
    check("think resolved_tier == 'pro'", body_t.get("resolved_tier") == "pro")
    check("think ok True", body_t.get("ok") is True)
    check("think output echoes text", "unit test thought" in str(body_t.get("output") or ""))
    check("think output starts tier=pro", str(body_t.get("output") or "").startswith("tier=pro"))

    # ---- 4. unknown agent -> 404 -----------------------------------------
    r3 = client.post(
        "/api/mas/tool-agents/test",
        json={"agent_id": "no_such_agent", "tier": "lite", "args": {"query": "x"}},
    )
    print(f"unknown agent -> status={r3.status_code} detail={r3.text[:80]!r}")
    check("unknown agent 404", r3.status_code == 404)
    check("404 mentions unknown tool-agent", "unknown tool-agent" in r3.text)

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
        return 1
    print("RESULT: all offline endpoint checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
