"""Build a mixed GSM8K + HotpotQA (optional NQ) parquet dataset for tir_agent."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def parse_gsm8k_answer(raw: str) -> str:
    match = re.search(r"####\s*(.+)", raw)
    if match:
        return match.group(1).strip().replace(",", "")
    return raw.strip()


def _row(
    *,
    row_id: str,
    question: str,
    answer: str,
    answers: Sequence[str],
    source: str,
    split: str,
) -> Dict[str, Any]:
    uniq = []
    for a in [answer, *answers]:
        s = str(a).strip()
        if s and s not in uniq:
            uniq.append(s)
    return {
        "id": row_id,
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "answers": json.dumps(uniq, ensure_ascii=False),
        "source": source,
        "split": split,
    }


def load_gsm8k(split_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split_name)
    rows: List[Dict[str, Any]] = []
    out_split = "train" if split_name == "train" else "val"
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        ans = parse_gsm8k_answer(item["answer"])
        rows.append(
            _row(
                row_id=f"gsm8k-{out_split}-{i}",
                question=item["question"],
                answer=ans,
                answers=[ans],
                source="gsm8k",
                split=out_split,
            )
        )
    return rows


def load_hotpot(split_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    hf_split = "validation" if split_name in ("test", "validation", "val") else "train"
    ds = load_dataset("hotpot_qa", "distractor", split=hf_split)
    rows: List[Dict[str, Any]] = []
    out_split = "train" if hf_split == "train" else "val"
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        ans = str(item.get("answer") or "").strip()
        rows.append(
            _row(
                row_id=f"hotpot-{out_split}-{i}",
                question=item["question"],
                answer=ans,
                answers=[ans],
                source="hotpot",
                split=out_split,
            )
        )
    return rows


def load_nq(split_name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    hf_split = "validation" if split_name in ("test", "validation", "val") else "train"
    ds = load_dataset("nq_open", split=hf_split)
    rows: List[Dict[str, Any]] = []
    out_split = "train" if hf_split == "train" else "val"
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        answers = item.get("answer") or item.get("answers") or []
        if isinstance(answers, str):
            answers = [answers]
        answers = [str(a).strip() for a in answers if str(a).strip()]
        if not answers:
            continue
        rows.append(
            _row(
                row_id=f"nq-{out_split}-{i}",
                question=item["question"],
                answer=answers[0],
                answers=answers,
                source="nq",
                split=out_split,
            )
        )
    return rows


def load_local_qa_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            ans = str(obj.get("answer") or obj.get("target") or "").strip()
            aliases = obj.get("answers") or [ans]
            if isinstance(aliases, str):
                aliases = [aliases]
            rows.append(
                _row(
                    row_id=str(obj.get("id") or f"local-qa-{i}"),
                    question=str(obj.get("question") or obj.get("input") or ""),
                    answer=ans,
                    answers=[str(a) for a in aliases],
                    source=str(obj.get("source") or "hotpot"),
                    split=str(obj.get("split") or "train"),
                )
            )
    return rows


def load_gsmhard_fallback(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            obj = json.loads(line)
            ans = str(obj["target"]).strip()
            rows.append(
                _row(
                    row_id=f"gsmhard-{i}",
                    question=str(obj["input"]).strip(),
                    answer=ans,
                    answers=[ans],
                    source="gsm8k",
                    split="train",
                )
            )
    return rows


def synthetic_gsm(n: int, split: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        if split == "train":
            a, b = 3 + i, 5 + (i % 7)
            q, ans = f"What is {a} multiplied by {b}?", str(a * b)
        else:
            a, b = 10 + i, 2 + (i % 5)
            q, ans = f"What is {a} plus {b}?", str(a + b)
        rows.append(_row(row_id=f"syn-gsm-{split}-{i}", question=q, answer=ans, answers=[ans], source="gsm8k", split=split))
    return rows


def synthetic_qa(n: int, split: str) -> List[Dict[str, Any]]:
    facts = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Hamlet?", "William Shakespeare"),
        ("What planet is known as the Red Planet?", "Mars"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
        ("What is the chemical symbol for gold?", "Au"),
        ("Which continent is Egypt in?", "Africa"),
    ]
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        q, ans = facts[i % len(facts)]
        if i >= len(facts):
            q = f"{q} (copy {i})"
        rows.append(
            _row(row_id=f"syn-qa-{split}-{i}", question=q, answer=ans, answers=[ans], source="hotpot", split=split)
        )
    return rows


def mix_sources(
    gsm: List[Dict[str, Any]],
    qa: List[Dict[str, Any]],
    *,
    total_limit: Optional[int],
    gsm_ratio: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    gsm = list(gsm)
    qa = list(qa)
    rng.shuffle(gsm)
    rng.shuffle(qa)
    if total_limit is None:
        mixed = gsm + qa
        rng.shuffle(mixed)
        return mixed
    n_gsm = int(round(total_limit * gsm_ratio))
    n_qa = max(0, total_limit - n_gsm)
    if n_gsm > len(gsm):
        extra = n_gsm - len(gsm)
        n_gsm = len(gsm)
        n_qa = min(len(qa), n_qa + extra)
    if n_qa > len(qa):
        extra = n_qa - len(qa)
        n_qa = len(qa)
        n_gsm = min(len(gsm), n_gsm + extra)
    mixed = gsm[:n_gsm] + qa[:n_qa]
    rng.shuffle(mixed)
    return mixed[:total_limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare mixed GSM8K + HotpotQA parquet for tir_agent")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--val-limit", type=int, default=200)
    parser.add_argument("--gsm-ratio", type=float, default=0.5, help="Fraction of GSM8K in the mix")
    parser.add_argument("--include-nq", action="store_true", help="Also mix Natural Questions Open")
    parser.add_argument("--offline", action="store_true", help="Skip Hub; use synthetic / local files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gsmhard-path",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "unsloth", "data_gsmhard.jsonl"),
    )
    parser.add_argument(
        "--local-qa",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data", "local_qa.jsonl"),
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    fetch_cap_train = None if args.train_limit is None else max(args.train_limit * 2, args.train_limit)
    fetch_cap_val = None if args.val_limit is None else max(args.val_limit * 2, args.val_limit)

    gsm_train: List[Dict[str, Any]] = []
    gsm_val: List[Dict[str, Any]] = []
    qa_train: List[Dict[str, Any]] = []
    qa_val: List[Dict[str, Any]] = []

    if not args.offline:
        try:
            print(f"Downloading GSM8K via HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} ...")
            gsm_train = load_gsm8k("train", limit=fetch_cap_train)
            gsm_val = load_gsm8k("test", limit=fetch_cap_val)
        except Exception as e:
            print(f"WARNING: GSM8K download failed ({e})")
        try:
            print("Downloading HotpotQA distractor split ...")
            qa_train = load_hotpot("train", limit=fetch_cap_train)
            qa_val = load_hotpot("val", limit=fetch_cap_val)
        except Exception as e:
            print(f"WARNING: HotpotQA download failed ({e})")
        if args.include_nq:
            try:
                print("Downloading nq_open ...")
                qa_train.extend(load_nq("train", limit=fetch_cap_train))
                qa_val.extend(load_nq("val", limit=fetch_cap_val))
            except Exception as e:
                print(f"WARNING: NQ download failed ({e})")

    if not gsm_train:
        gsmhard = os.path.abspath(args.gsmhard_path)
        if os.path.exists(gsmhard):
            print(f"Falling back to local GSM-hard: {gsmhard}")
            all_rows = load_gsmhard_fallback(gsmhard)
            n_val = min(args.val_limit or 16, max(1, len(all_rows) // 5))
            gsm_val = all_rows[:n_val]
            for r in gsm_val:
                r["split"] = "val"
            gsm_train = all_rows[n_val:]
        else:
            print("Falling back to synthetic GSM8K-style arithmetic.")
            n_train = args.train_limit or 64
            gsm_train = synthetic_gsm(max(n_train, 8), "train")
            gsm_val = synthetic_gsm(args.val_limit or 16, "val")

    if not qa_train:
        local_qa = os.path.abspath(args.local_qa)
        if os.path.exists(local_qa):
            print(f"Falling back to local QA jsonl: {local_qa}")
            loaded = load_local_qa_jsonl(local_qa)
            qa_train = [r for r in loaded if r.get("split") != "val"] or loaded
            qa_val = [r for r in loaded if r.get("split") == "val"] or loaded[: max(1, len(loaded) // 5)]
        else:
            print("Falling back to synthetic QA facts.")
            n_train = args.train_limit or 64
            qa_train = synthetic_qa(max(n_train, 8), "train")
            qa_val = synthetic_qa(args.val_limit or 16, "val")

    train_rows = mix_sources(gsm_train, qa_train, total_limit=args.train_limit, gsm_ratio=args.gsm_ratio, rng=rng)
    val_rows = mix_sources(gsm_val, qa_val, total_limit=args.val_limit, gsm_ratio=args.gsm_ratio, rng=rng)
    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"

    train_path = os.path.join(args.output_dir, "train.parquet")
    val_path = os.path.join(args.output_dir, "val.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    def _count(rows: List[Dict[str, Any]], src: str) -> int:
        return sum(1 for r in rows if r.get("source") == src)

    print(f"Wrote {len(train_rows)} train rows -> {train_path} (gsm8k={_count(train_rows, 'gsm8k')} qa={len(train_rows) - _count(train_rows, 'gsm8k')})")
    print(f"Wrote {len(val_rows)} val rows -> {val_path} (gsm8k={_count(val_rows, 'gsm8k')} qa={len(val_rows) - _count(val_rows, 'gsm8k')})")
    print("Sample train row:", train_rows[0] if train_rows else None)


if __name__ == "__main__":
    main()
