#!/usr/bin/env python3
"""Probe EVALUATION_MODEL_* — try gpt-4o then gpt-4o-mini; write winner to .env."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "mas"))

from rl.rewards.llm_judge import probe_evaluation_model, resolve_evaluation_endpoint  # noqa: E402


CANDIDATES = ("gpt-4o", "gpt-4o-mini")


def _write_env_model(model: str) -> None:
    env_path = REPO / ".env"
    text = env_path.read_text(encoding="utf-8")
    if re.search(r"^EVALUATION_MODEL=", text, flags=re.MULTILINE):
        text = re.sub(
            r"^EVALUATION_MODEL=.*$",
            f'EVALUATION_MODEL="{model}"',
            text,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        text = text.rstrip() + f'\nEVALUATION_MODEL="{model}"\n'
    env_path.write_text(text, encoding="utf-8")


def main() -> int:
    key, base, current = resolve_evaluation_endpoint()
    print(f"base={base!r} key_set={bool(key)} current_model={current!r}")
    if not base:
        print("FAIL: EVALUATION_MODEL_API_BASE unset")
        return 1

    order = []
    if current and current not in CANDIDATES:
        order.append(current)
    for c in CANDIDATES:
        if c not in order:
            order.append(c)

    for cand in order:
        print(f"probing {cand} ...")
        result = probe_evaluation_model(cand)
        print(f"  -> {result}")
        if result.get("ok") and float(result.get("score") or 0) >= 1.0:
            _write_env_model(cand)
            print(f"LOCKED EVALUATION_MODEL={cand}")
            return 0
        if result.get("ok"):
            print(f"  warn: model responded but score={result.get('score')} (expected 1.0)")

    print("FAIL: no candidate worked")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
