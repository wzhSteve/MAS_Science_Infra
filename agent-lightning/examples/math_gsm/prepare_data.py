# Copyright (c) Microsoft. All rights reserved.

"""Download GSM8K and write train/val parquet files for math_gsm training."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

# Prefer China-friendly HF mirror when HF_ENDPOINT is unset.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_gsm8k_answer(raw: str) -> str:
    """Extract the final answer after #### from a GSM8K solution string."""
    match = re.search(r"####\s*(.+)", raw)
    if match:
        return match.group(1).strip().replace(",", "")
    return raw.strip()


def convert_split(split_name: str, limit: int | None = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split_name)
    rows: List[Dict[str, Any]] = []
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        rows.append(
            {
                "id": f"{split_name}-{i}",
                "question": item["question"].strip(),
                "answer": parse_gsm8k_answer(item["answer"]),
            }
        )
    return rows


def load_gsmhard_fallback(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Fallback: convert local GSM-hard jsonl (unsloth example) to our schema."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            rows.append(
                {
                    "id": f"gsmhard-{i}",
                    "question": str(obj["input"]).strip(),
                    "answer": str(obj["target"]).strip(),
                }
            )
    return rows


def synthetic_fallback(n_train: int = 64, n_val: int = 16) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Tiny synthetic arithmetic set when Hub is unreachable."""
    train: List[Dict[str, Any]] = []
    for i in range(n_train):
        a, b = 3 + i, 5 + (i % 7)
        train.append({"id": f"syn-train-{i}", "question": f"What is {a} multiplied by {b}?", "answer": str(a * b)})
    val: List[Dict[str, Any]] = []
    for i in range(n_val):
        a, b = 10 + i, 2 + (i % 5)
        val.append({"id": f"syn-val-{i}", "question": f"What is {a} plus {b}?", "answer": str(a + b)})
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare GSM8K parquet data for math_gsm")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional cap on train size")
    parser.add_argument("--val-limit", type=int, default=200, help="Validation subset size")
    parser.add_argument(
        "--gsmhard-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "unsloth", "data_gsmhard.jsonl"),
        help="Fallback local GSM-hard jsonl if Hub download fails",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        print(f"Downloading GSM8K via HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} ...")
        train_rows = convert_split("train", limit=args.train_limit)
        val_rows = convert_split("test", limit=args.val_limit)
    except Exception as e:
        print(f"WARNING: failed to download GSM8K ({e})")
        gsmhard = os.path.abspath(args.gsmhard_path)
        if os.path.exists(gsmhard):
            print(f"Falling back to local GSM-hard: {gsmhard}")
            all_rows = load_gsmhard_fallback(gsmhard)
            n_val = min(args.val_limit, max(1, len(all_rows) // 5))
            val_rows = all_rows[:n_val]
            train_rows = all_rows[n_val:]
            if args.train_limit is not None:
                train_rows = train_rows[: args.train_limit]
        else:
            print("Falling back to synthetic arithmetic dataset.")
            n_train = args.train_limit or 64
            train_rows, val_rows = synthetic_fallback(n_train=n_train, n_val=args.val_limit)

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"Wrote {len(train_rows)} train rows -> {train_path}")
    print(f"Wrote {len(val_rows)} val rows -> {val_path}")
    print("Sample train row:", train_rows[0] if train_rows else None)


if __name__ == "__main__":
    main()
