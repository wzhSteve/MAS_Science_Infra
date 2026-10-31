"""Paired Hotpot dev run on one shared search cache.

Order is pure PEV, then v1 CGHE, then v1 with shared-node priority off.
Later runs reuse SERP and page text already stored for the same query or URL.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path("/root/autodl-tmp/MAS_Science_Infra")
OUT = ROOT / "experiments" / "cghe_v2"
CACHE = OUT / "paired_search_cache.json"
RUN = ROOT / "experiments" / "cghe_hotpot" / "run_pair.py"


def _run(system: str, ablation: str, destination: Path) -> None:
    env = os.environ.copy()
    env["MAS_SEARCH_CACHE"] = str(CACHE)
    env["CGHE_ALGORITHM"] = "v1"
    env.pop("CGHE_ABLATION", None)
    command = [
        sys.executable,
        str(RUN),
        "worker",
        "--system",
        system,
        "--split",
        "dev",
        "--ablation",
        ablation,
        "--reset-memory",
        "--out",
        str(destination),
    ]
    subprocess.run(command, cwd=str(ROOT), env=env, check=True)


def _first_url(row: dict) -> str:
    for step in row["steps"]:
        text = str(step.get("sub_goal") or "")
        if step.get("phase") == "plan" and "open http" in text:
            return text
    return ""


def _unrelated(url: str, question: str) -> bool:
    if not url:
        return False
    low = url.lower()
    if "wikipedia.org" in low:
        return False
    words = [word for word in "".join(ch.lower() if ch.isalnum() else " " for ch in question).split() if len(word) >= 5]
    return not any(word in low for word in words[:8])


def _summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    unrelated = 0
    for row in report["rows"]:
        if _unrelated(_first_url(row), row["question"]):
            unrelated += 1
    return {
        "correct": report["correct"],
        "mean_tool_calls": report["mean_tool_calls"],
        "unrelated_urls": unrelated,
        "path": str(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [
        ("pure_pev", "full", OUT / "paired_pure.json"),
        ("cghe_pev", "full", OUT / "paired_v1.json"),
        ("cghe_pev", "no_shared", OUT / "paired_no_shared.json"),
    ]
    for system, ablation, destination in runs:
        _run(system, ablation, destination)
    summary = {
        "pure": _summarize(runs[0][2]),
        "v1": _summarize(runs[1][2]),
        "no_shared": _summarize(runs[2][2]),
    }
    gap = abs(summary["no_shared"]["correct"] - summary["v1"]["correct"])
    pure_gap = max(
        abs(summary["pure"]["correct"] - summary["v1"]["correct"]),
        abs(summary["pure"]["correct"] - summary["no_shared"]["correct"]),
        gap,
    )
    demote = (
        summary["no_shared"]["correct"] >= summary["v1"]["correct"] + 2
        and summary["no_shared"]["unrelated_urls"] <= summary["v1"]["unrelated_urls"]
    )
    stop = pure_gap <= 1
    payload = {
        "cache": str(CACHE),
        "systems": summary,
        "demote_sharing": demote,
        "stop_hotpot_tuning": stop and not demote,
        "heldout_rerun": False,
    }
    (OUT / "paired_dev.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
