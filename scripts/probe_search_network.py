"""Live probe of Bing SERP / fetch. Prints timings only; never prints secrets."""

from __future__ import annotations

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mas"))

from science_infra.env import load_science_env  # noqa: E402

load_science_env()

from tools.search import bing_search, fetch_page  # noqa: E402


def _kind(text: str) -> str:
    low = (text or "").lower()
    if text.startswith("bing results for:"):
        return "ok"
    if "html=" in low:
        return "html_scrape"
    if "search_unavailable" in low or "bing unreachable" in low:
        return "unavailable"
    return "other"


def _one(query: str) -> dict:
    t0 = time.time()
    out = bing_search(query)
    dt = time.time() - t0
    kind = _kind(out)
    return {
        "query": query,
        "kind": kind,
        "seconds": round(dt, 3),
        "chars": len(out or ""),
        "prefix": (out or "").split("\n", 1)[0][:80],
    }


def main() -> int:
    key_on = bool(os.getenv("BRIGHT_DATA_API_KEY", "").strip())
    mode = os.getenv("BRIGHT_DATA_MODE", "auto")
    cache = os.getenv("MAS_SEARCH_CACHE", "")
    print(f"bd_key={key_on} mode={mode} cache_set={bool(cache.strip())}", flush=True)

    sequential = [
        "Ada Lovelace",
        "ICML 2024",
        "Python programming language",
    ]
    seq_ok = 0
    print("=== sequential ===", flush=True)
    for q in sequential:
        row = _one(q)
        seq_ok += int(row["kind"] == "ok")
        print(row, flush=True)
    print(f"sequential_ok={seq_ok}/{len(sequential)}", flush=True)

    parallel = [f"{q} wikipedia" for q in sequential] + [
        "Alan Turing",
        "Grace Hopper",
        "John von Neumann",
        "Claude Shannon",
        "Donald Knuth",
    ]
    print("=== parallel 8 ===", flush=True)
    times: list[float] = []
    kinds: list[str] = []
    html_hits = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(_one, q) for q in parallel[:8]]
        for fut in as_completed(futs):
            row = fut.result()
            times.append(row["seconds"])
            kinds.append(row["kind"])
            html_hits += int(row["kind"] == "html_scrape")
            print(row, flush=True)
    ok = sum(k == "ok" for k in kinds)
    times.sort()
    p50 = statistics.median(times) if times else 0.0
    p95 = times[min(len(times) - 1, max(0, int(round(0.95 * (len(times) - 1)))))] if times else 0.0
    print(f"parallel_ok={ok}/8 p50={p50:.3f} p95={p95:.3f} html_scrape={html_hits}", flush=True)

    print("=== html contrast BRIGHT_DATA_MODE=0 ===", flush=True)
    os.environ["BRIGHT_DATA_MODE"] = "0"
    contrast = _one("Ada Lovelace contrast")
    print(contrast, flush=True)
    os.environ["BRIGHT_DATA_MODE"] = mode or "auto"

    print("=== fetch ===", flush=True)
    for url in (
        "https://en.wikipedia.org/wiki/Ada_Lovelace",
        "https://www.imdb.com/title/tt0073639/",
    ):
        t0 = time.time()
        page = fetch_page(url)
        dt = time.time() - t0
        head = (page or "").split("\n", 1)[0][:100]
        print({"url": url, "seconds": round(dt, 3), "chars": len(page or ""), "head": head}, flush=True)

    if seq_ok < 2:
        print("FAIL sequential < 2/3", flush=True)
        return 1
    if html_hits:
        print("FAIL parallel still scraping bing HTML", flush=True)
        return 1
    print(f"html_contrast_kind={contrast['kind']}", flush=True)
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
