#!/usr/bin/env python3
"""RolloutTree visualization acceptance (B3).

After an ARPO/RAE training run (B2) or any run that produced
``mas/.local_expansion/*.json``:

1. GET /api/mas/rollout-trees and assert:
   - trees >= 1
   - nodes carry event_kind / h_tool metrics
   - (RAE) verdict fields when reward scheme is rae_adjudicate
2. Print manual UI steps for the RolloutTree page (React Flow + SSE badge).

Exit 0 = acceptance passed. Requires the Control UI running
(``./run.sh ui --daemon``) — or pass ``--offline`` to scan the local
expansion directory directly without the API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from branch_rollout_ui_test import req, wait_health  # noqa: E402


def scan_trees_local(root: Path) -> List[Dict[str, Any]]:
    exp_dir = root / "mas" / ".local_expansion"
    trees: List[Dict[str, Any]] = []
    if not exp_dir.is_dir():
        return trees
    for f in sorted(exp_dir.glob("*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        tree = payload.get("tree")
        if isinstance(tree, dict):
            trees.append({"file": f.name, "tree": tree})
    return trees


def check_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Structural assertions on one RolloutTree payload."""
    nodes = list(tree.get("nodes") or [])
    out: Dict[str, Any] = {
        "n_nodes": len(nodes),
        "has_event_kind": False,
        "has_h_tool": False,
        "has_verdict": False,
        "kinds": set(),
    }
    for n in nodes:
        meta = n.get("meta") or {}
        # Canonical contract (workflow/contracts.py RolloutTreeNode): metrics
        # is a top-level field on the node. Keep legacy meta fallbacks for
        # older payloads.
        metrics = n.get("metrics") or meta.get("metrics") or {}
        ek = n.get("event_kind") or meta.get("event_kind") or metrics.get("event_kind")
        if ek:
            out["has_event_kind"] = True
            out["kinds"].add(str(ek))
        if "h_tool" in metrics or "h_tool" in meta or "h_tool" in n:
            out["has_h_tool"] = True
        if any(k in n for k in ("verdict", "rae_verdict")) or any(
            k in meta for k in ("verdict", "rae_verdict")
        ):
            out["has_verdict"] = True
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="RolloutTree visualization acceptance (B3)")
    p.add_argument("--base", default="http://127.0.0.1:8787")
    p.add_argument("--experiment", default="arpo_e2e")
    p.add_argument("--offline", action="store_true", help="scan mas/.local_expansion directly")
    p.add_argument("--require-verdict", action="store_true", help="also assert RAE verdict fields")
    args = p.parse_args()

    failed = 0

    def ok(name: str, cond: bool, detail: Any = "") -> None:
        nonlocal failed
        mark = "PASS" if cond else "FAIL"
        if not cond:
            failed += 1
        print(f"  [{mark}] {name}{('  ' + str(detail)) if detail not in ('', None) else ''}")

    print(f"RolloutTree verify → {'offline scan' if args.offline else args.base}")
    if args.offline:
        trees = scan_trees_local(REPO)
        ok("local expansion scan found trees", len(trees) >= 1, f"n={len(trees)} (run a train first)")
    else:
        wait_health(args.base)
        code, body = req(args.base, "GET", f"/api/mas/rollout-trees?experiment_id={args.experiment}")
        ok("GET /api/mas/rollout-trees", code == 200 and isinstance(body, dict))
        trees = list((body or {}).get("trees") or []) if isinstance(body, dict) else []
        ok("trees >= 1", len(trees) >= 1, f"n={len(trees)} (train with branches first: ./run.sh arpo-train-test)")

    if trees:
        t0 = trees[0].get("tree") or {}
        stats = check_tree(t0)
        ok("tree has nodes", stats["n_nodes"] >= 1, f"n={stats['n_nodes']}")
        ok("nodes carry event_kind", stats["has_event_kind"], sorted(stats["kinds"]))
        ok("nodes carry h_tool metrics", stats["has_h_tool"])
        if args.require_verdict:
            ok("RAE verdict fields present", stats["has_verdict"])

    print()
    print("Manual UI steps (docs/MAS_AGENT_FRAMEWORK_TEST.md §B3):")
    print("  1. ./run.sh ui --port 8787 --daemon  (if not running)")
    print("  2. Open http://127.0.0.1:8787/#rollout-tree")
    print("  3. Pick a tree from the list → React Flow renders root/child layers")
    print("  4. Node badges show event_kind + h_tool; RAE trees show verdicts")
    print("  5. Start a train (./run.sh arpo-train-test) → SSE badge turns LIVE on node_added")

    if failed:
        print(f"\nRolloutTree verify FAILED ({failed} checks)")
        return 1
    print("\nRolloutTree verify OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
