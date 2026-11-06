#!/usr/bin/env python3
"""Pair CGHE SoftBing jsonl results with pure_pev full_qa and print a summary."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_cg(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            rows[int(o["index"])] = o
    return rows


def load_pure(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open() as f:
        for line in f:
            o = json.loads(line)
            rows[int(o["idx"])] = o
    return rows


def pair(name: str, pure: dict[int, dict], cg: dict[int, dict]) -> dict:
    common = sorted(set(pure) & set(cg))
    both = only_c = only_p = both_w = 0
    for i in common:
        pc = bool(pure[i].get("correct"))
        cc = bool(cg[i].get("correct"))
        if pc and cc:
            both += 1
        elif (not pc) and cc:
            only_c += 1
        elif pc and not cc:
            only_p += 1
        else:
            both_w += 1
    regen = [int(cg[i].get("regen_count") or 0) for i in common]
    pruned = [int(cg[i].get("pruned_plan_count") or 0) for i in common]
    out = {
        "name": name,
        "n": len(common),
        "both_ok": both,
        "only_cg": only_c,
        "only_pure": only_p,
        "both_wrong": both_w,
        "pure_acc": sum(1 for i in common if pure[i].get("correct")) / max(1, len(common)),
        "cg_acc": sum(1 for i in common if cg[i].get("correct")) / max(1, len(common)),
        "cg_hit": sum(1 for i in common if cg[i].get("hit")) / max(1, len(common)),
        "regen_mean": statistics.mean(regen) if regen else 0.0,
        "pruned_mean": statistics.mean(pruned) if pruned else 0.0,
    }
    print(
        f"{name}: n={out['n']} pure={out['pure_acc']:.3f} cg={out['cg_acc']:.3f} "
        f"hit={out['cg_hit']:.3f} only_cg={only_c} only_pure={only_p} "
        f"regen={out['regen_mean']:.2f} pruned={out['pruned_mean']:.2f}"
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cghe-dir", required=True, help="dir with hotpot.jsonl / bamboogle.jsonl")
    ap.add_argument(
        "--pure-dir",
        default="/root/autodl-tmp/MAS_Science_Infra/user_space/projects/pure_pev/artifacts/full_qa_20261104_205608",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cg_dir = Path(args.cghe_dir)
    pure_dir = Path(args.pure_dir)
    summary = {"pairs": []}
    mapping = [
        ("hotpot", pure_dir / "hotpotqa.jsonl", cg_dir / "hotpot.jsonl"),
        ("bamboogle", pure_dir / "bamboogle.jsonl", cg_dir / "bamboogle.jsonl"),
    ]
    for name, pp, cp in mapping:
        if not pp.exists() or not cp.exists():
            print(f"skip {name}: missing {pp if not pp.exists() else cp}")
            continue
        summary["pairs"].append(pair(name, load_pure(pp), load_cg(cp)))
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
