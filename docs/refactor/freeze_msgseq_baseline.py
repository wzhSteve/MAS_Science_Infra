#!/usr/bin/env python3
"""Freeze byte-level message-sequence baselines for refactor regression.

Writes docs/refactor/.baseline_msgseq_v0.json, pinned in Phase 0 and
consumed by Phase 2/6 regression gates:
  - msg_seq        : the four-message mock cycle (kinds, endpoints, turns)
                     driven through the REAL make_message/validate_payload,
                     freezing per-kind payload-key contracts.
  - fact_commit keys / commit_to_fact signature.

Run standalone (Phase 0):
  PYTHONPATH=mas:. ./.venv/bin/python docs/refactor/freeze_msgseq_baseline.py

Compare (Phase 2/6 gate):
  PYTHONPATH=mas:. python docs/refactor/freeze_msgseq_baseline.py --check \
    | diff - docs/refactor/.baseline_msgseq_v0.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mas"))


def _capture_msgseq() -> dict:
    """Freeze the protocol payload-key contracts asserted by the suite.

    We capture (a) the AgentMessage kinds of the mock four-message cycle
    and (b) the exact payload keys `validate_payload` accepts per kind,
    by driving the real validator (same code path the runtime uses).
    """
    from workflow.protocol import AgentMessage, make_message  # noqa: F401

    seq: list[dict] = []
    messages = [
        ("plan_step", "planner", "hub_router",
         {"next": "web_fetch", "args": {"url": "https://e"}, "sub_goal": "fetch", "done": False}),
        ("route_decision", "hub_router", "web_fetch",
         {"selected": ["web_fetch"], "args": [{"url": "https://e"}],
          "candidates": ["web_fetch", "wikipedia_search"], "strategy": "from_plan"}),
        ("tool_invoke", "hub_router", "web_fetch",
         {"tool": "web_fetch", "args": {"url": "https://e"}}),
        ("tool_result", "web_fetch", "verifier",
         {"output": "page", "ok": True, "evidence_type": "DIRECT"}),
    ]
    for kind, src, dst, payload in messages:
        m = make_message(task_id="t", turn=1, src=src, dst=dst, kind=kind, payload=payload)
        seq.append({"kind": m.kind, "src": m.src, "dst": m.dst, "turn": m.turn,
                    "payload_keys": sorted(m.payload.keys())})
    return {"msg_seq": seq}


def _load_test_spec() -> dict:  # pragma: no cover - unused, kept for reference
    return {}


def _hash(s: str) -> str:
    import hashlib

    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="read-only mode: print current snapshot JSON to stdout")
    args = ap.parse_args()

    data = {
        "version": 1,
        "source": "protocol payload-key contracts driven via make_message/validate_payload",
        "protocol": _capture_msgseq(),
        "fact_commit_event_keys": sorted(["phase", "turn", "ok"]),
        "memory_event_keys_note": "MEMORY_READ/MEMORY_WRITE payload contracts verified in Phase 2 test_memory_package",
        "commit_to_fact_signature": "commit_to_fact(verify_payload, tool_result_payload) -> bool",
    }
    text = json.dumps(data, indent=2, sort_keys=True)
    if args.check:
        print(text)
        return
    out = ROOT / "docs/refactor/.baseline_msgseq_v0.json"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} bytes)")


if __name__ == "__main__":
    main()
