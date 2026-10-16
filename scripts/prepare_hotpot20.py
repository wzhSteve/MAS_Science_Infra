#!/usr/bin/env python3
"""Build HotpotQA train/val parquet from the local data/hotpotqa.json pool.

Train = first 20 rows, val = next 5 rows. `answers` (JSON-ish list string) is
parsed into a proper list so the outcome reward can use alias-aware token F1.
Existing parquet files are backed up as <name>.gsm8k.bak (once).
"""

from __future__ import annotations

import ast
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
POOL = DATA / "hotpotqa.json"
TRAIN_N = 20
VAL_N = 5


def parse_answers(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(x) for x in raw]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        for parser in (json.loads, ast.literal_eval):
            try:
                out = parser(text)
                if isinstance(out, (list, tuple)):
                    return [str(x) for x in out]
            except Exception:
                continue
    return [text]


def main() -> int:
    rows = json.loads(POOL.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < TRAIN_N + VAL_N:
        print(f"pool too small: need {TRAIN_N + VAL_N}, got {len(rows) if isinstance(rows, list) else type(rows)}")
        return 1

    def to_row(rec: dict, split: str, i: int) -> dict:
        answers = parse_answers(rec.get("answers") or rec.get("answer"))
        answer = answers[0] if answers else ""
        return {
            "id": f"hotpot-{split}-{i}",
            "question": str(rec.get("question") or rec.get("query") or "").strip(),
            "answer": answer,
            "answers": answers,
            "source": "hotpot",
            "split": split,
        }

    train = [to_row(r, "train", i) for i, r in enumerate(rows[:TRAIN_N])]
    val = [to_row(r, "val", TRAIN_N + i) for i, r in enumerate(rows[TRAIN_N : TRAIN_N + VAL_N])]

    try:
        import pandas as pd
    except ImportError:
        print("pandas required: pip install pandas pyarrow")
        return 1

    for name, frame in (("train.parquet", train), ("val.parquet", val)):
        path = DATA / name
        backup = DATA / (name.replace(".parquet", ".gsm8k.bak"))
        if path.exists() and not backup.exists():
            shutil.copyfile(path, backup)
        pd.DataFrame(frame).to_parquet(path, index=False)
        print(f"wrote {path} rows={len(frame)} (backup: {backup.name})")

    print("sample question:", train[0]["question"][:80])
    print("sample answers:", train[0]["answers"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
