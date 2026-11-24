#!/usr/bin/env python3
"""Live probe: Bright Data Bing latency / success vs timeout & retries.

Fast default: 3 queries × timeout {15,45} + retries@45 + small concurrent.
  PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/probe_brightdata_bing.py
  PROBE_FULL=1 ...  # 10 queries × 15/30/45 + retries + 8x5 concurrent
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "mas"))
sys.path.insert(0, str(REPO))

os.environ.setdefault("SCIENCE_INFRA_TOOL_KERNEL", "1")
os.environ["MAS_SEARCH_CACHE"] = ""
os.environ.setdefault("TIR_OFFLINE_SEARCH", "0")
os.environ.setdefault("BRIGHT_DATA_MODE", "auto")
os.environ.setdefault("TIR_BING_MIN_INTERVAL", "0.15")

FULL = os.environ.get("PROBE_FULL", "").strip() in ("1", "true", "yes")
QUERIES_FULL = [
    "Ada Lovelace biography",
    "capital of New Zealand",
    "Odoardo Farnese date of death",
    "binomial coefficient parity 1000",
    "Princess Maria Felicita of Savoy mother",
    "gunfight at the OK Corral date",
    "Curly Seckler lifespan",
    "An Phu Quang Tri Vietnam",
    "James Flack Norris death date",
    "Hereditary Prince of Parma father",
]
QUERIES = QUERIES_FULL if FULL else QUERIES_FULL[:3]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = min(len(ys) - 1, max(0, int(round((p / 100.0) * (len(ys) - 1)))))
    return ys[k]


def _run_one(query: str) -> Tuple[bool, float, str, bool]:
    import tools.search as search_mod
    from tools.search import bing_search

    t0 = time.time()
    out = bing_search(query)
    dt = time.time() - t0
    ok = bool(out) and not str(out).startswith("search_unavailable") and not str(out).startswith("Error:")
    err = "" if ok else str(out)[:180]
    denied = bool(getattr(search_mod, "_BD_ZONE_DENIED", False))
    return ok, dt, err, denied


def _bench(label: str, timeout: str, retries: str, queries: List[str]) -> Dict[str, Any]:
    import tools.search as search_mod

    os.environ["BRIGHT_DATA_TIMEOUT"] = timeout
    os.environ["BRIGHT_DATA_MAX_RETRIES"] = retries
    search_mod._BD_ZONE_DENIED = False

    lats: List[float] = []
    oks = 0
    timeouts = 0
    denied_end = False
    errors: List[str] = []
    for i, q in enumerate(queries):
        search_mod._BD_ZONE_DENIED = False
        _log(f"  [{label}] {i+1}/{len(queries)} q={q!r} ...")
        ok, dt, err, denied = _run_one(q)
        _log(f"  [{label}] -> {'OK' if ok else 'FAIL'} {dt:.2f}s denied={denied}" + (f" err={err[:100]}" if err else ""))
        lats.append(dt)
        if ok:
            oks += 1
        else:
            errors.append(err)
            if "timeout" in err.lower() or "ReadTimeout" in err:
                timeouts += 1
        denied_end = denied_end or denied

    n = max(1, len(queries))
    return {
        "label": label,
        "timeout": timeout,
        "retries": retries,
        "n": len(queries),
        "success_rate": oks / n,
        "timeout_frac": timeouts / n,
        "p50": _pct(lats, 50),
        "p95": _pct(lats, 95),
        "p99": _pct(lats, 99),
        "mean": statistics.mean(lats) if lats else float("nan"),
        "denied_seen": denied_end,
        "errors": errors[:3],
    }


def _concurrent(timeout: str, retries: str, workers: int = 4, per_worker: int = 2) -> Dict[str, Any]:
    import tools.search as search_mod

    os.environ["BRIGHT_DATA_TIMEOUT"] = timeout
    os.environ["BRIGHT_DATA_MAX_RETRIES"] = retries
    search_mod._BD_ZONE_DENIED = False

    jobs = [QUERIES[i % len(QUERIES)] for i in range(workers * per_worker)]
    lats: List[float] = []
    oks = 0
    timeouts = 0

    def work(q: str) -> Tuple[bool, float, str]:
        ok, dt, err, _ = _run_one(q)
        return ok, dt, err

    _log(f"  [concurrent] start workers={workers} jobs={len(jobs)}")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(work, q) for q in jobs]
        for fut in as_completed(futs):
            ok, dt, err = fut.result()
            lats.append(dt)
            if ok:
                oks += 1
            elif "timeout" in err.lower() or "ReadTimeout" in err:
                timeouts += 1
            _log(f"  [concurrent] done ok={ok} {dt:.2f}s")
    wall = time.time() - t0
    n = max(1, len(jobs))
    return {
        "label": f"concurrent_w{workers}x{per_worker}",
        "timeout": timeout,
        "retries": retries,
        "n": len(jobs),
        "success_rate": oks / n,
        "timeout_frac": timeouts / n,
        "p50": _pct(lats, 50),
        "p95": _pct(lats, 95),
        "p99": _pct(lats, 99),
        "mean": statistics.mean(lats) if lats else float("nan"),
        "wall": wall,
        "denied_seen": bool(search_mod._BD_ZONE_DENIED),
    }


def _print_row(r: Dict[str, Any]) -> None:
    line = (
        f"{r['label']:<28} to={r['timeout']:>3} r={r['retries']}  "
        f"ok={r['success_rate']:.0%}  tmo={r.get('timeout_frac', 0):.0%}  "
        f"p50={r['p50']:.2f}s  p95={r['p95']:.2f}s  p99={r['p99']:.2f}s  "
        f"mean={r['mean']:.2f}s  denied={r.get('denied_seen')}"
    )
    if "wall" in r:
        line += f"  wall={r['wall']:.1f}s"
    _log(line)
    for e in r.get("errors") or []:
        _log(f"    err: {e[:160]}")


def main() -> int:
    key = (os.environ.get("BRIGHT_DATA_API_KEY") or "").strip()
    if not key:
        _log("ERROR: BRIGHT_DATA_API_KEY not set")
        return 2
    _log(
        f"probe Bright Data Bing  full={FULL} n_queries={len(QUERIES)}  "
        f"key=…{key[-6:]}  zone={os.environ.get('BRIGHT_DATA_SERP_ZONE') or os.environ.get('BRIGHT_DATA_SERP_ZONES')}"
    )
    rows: List[Dict[str, Any]] = []

    timeouts = ("15", "30", "45") if FULL else ("15", "45")
    _log(f"=== single-shot timeout sweep {timeouts} (retries=1) ===")
    for to in timeouts:
        row = _bench(f"single_to{to}", to, "1", QUERIES)
        rows.append(row)
        _print_row(row)

    _log("=== retries at timeout=45 ===")
    for rtry in ("1", "2"):
        row = _bench(f"retry{rtry}_to45", "45", rtry, QUERIES)
        rows.append(row)
        _print_row(row)

    if FULL:
        _log("=== concurrent 8x5 at timeout=45 retries=2 ===")
        crow = _concurrent("45", "2", workers=8, per_worker=5)
    else:
        _log("=== concurrent 4x2 at timeout=45 retries=2 ===")
        crow = _concurrent("45", "2", workers=4, per_worker=2)
    rows.append(crow)
    _print_row(crow)

    by_to = {r["timeout"]: r for r in rows if r["label"].startswith("single_to")}
    r15, r45 = by_to.get("15"), by_to.get("45")
    _log("\n=== recommendation ===")
    if r15 and r45:
        if r45["success_rate"] - r15["success_rate"] >= 0.15 or (r15["p95"] >= 14 and r45["success_rate"] > r15["success_rate"]):
            _log("TRAIN: raise BRIGHT_DATA_TIMEOUT 15 → 45 (p95/success improve with longer read deadline).")
        elif r45["p95"] < 12 and r15["success_rate"] >= 0.9:
            _log("TRAIN: timeout=15 may be enough (p95<<15); keep 15 or use 30 for headroom.")
        else:
            _log("TRAIN: prefer BRIGHT_DATA_TIMEOUT=45 as safe default (module default).")
    retry_rows = [r for r in rows if r["label"].startswith("retry")]
    if len(retry_rows) >= 2 and retry_rows[1]["success_rate"] > retry_rows[0]["success_rate"] + 0.05:
        _log("TRAIN: set BRIGHT_DATA_MAX_RETRIES=2 (retries recover transient timeouts).")
    else:
        _log("TRAIN: MAX_RETRIES=2 still recommended for transient 429/5xx; low downside.")
    if r45 and r45["success_rate"] < 0.5:
        _log("WARN: timeout=45 still weak — check zone/quota/proxy before pushing timeout to 180.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
