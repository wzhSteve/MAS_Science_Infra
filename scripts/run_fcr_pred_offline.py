#!/usr/bin/env python3
"""Offline H1 probe: train YES/NO prediction under class imbalance without Branch/Bing.

A1: Brier only, online batch (no replay)
A5: BCE + 0.1 Brier + persistent class-balanced replay

Writes metrics JSONL under experiments/fcr_short/offline_pred/.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.fcr import (
    PersistentPredReplay,
    PredReplayItem,
    binary_bce_from_scores,
    binary_brier_from_scores,
    combine_pred_losses,
)

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path[:0] = [str(ROOT), str(ROOT / "rl" / "verl_backend"), str(ROOT / "mas")]
OUT = ROOT / "experiments" / "fcr_short" / "offline_pred"


def _prefix_text(seed: int, idx: int) -> str:
    return (
        f"You are a planner. Subgoal {idx} (seed={seed}). "
        f"Decide if the executor will satisfy the verifier.\n"
        f'{{"thought":"check tools","prediction": "'
    )


def _build_items(tokenizer, n: int, pos_rate: float, seed: int) -> list[PredReplayItem]:
    random.seed(seed)
    yes_ids = tokenizer.encode("YES", add_special_tokens=False)
    no_ids = tokenizer.encode("NO", add_special_tokens=False)
    items: list[PredReplayItem] = []
    for i in range(n):
        label = random.random() < pos_rate
        prefix = tokenizer.encode(_prefix_text(seed, i), add_special_tokens=True)
        items.append(
            PredReplayItem(
                prefix_ids=prefix,
                satisfy_ids=yes_ids,
                unsatisfy_ids=no_ids,
                label=label,
                policy_version=0,
            )
        )
    return items


def _score_pair(model, device, item: PredReplayItem, temperature: float = 1.0):
    scores = []
    for candidate in (item.satisfy_ids, item.unsatisfy_ids):
        ids = torch.tensor(item.prefix_ids + list(candidate), dtype=torch.long, device=device).unsqueeze(0)
        mask = torch.ones_like(ids)
        pos = torch.arange(ids.size(1), device=device).unsqueeze(0)
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False)
        logits = out.logits.float() / float(temperature)
        start = len(item.prefix_ids) - 1
        token_logits = logits[0, start : start + len(candidate)]
        targets = torch.tensor(candidate, dtype=torch.long, device=device)
        scores.append(
            torch.log_softmax(token_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).sum()
        )
    return scores[0], scores[1]


def _metrics(ps: list[float], ys: list[bool]) -> dict:
    import numpy as np

    probs = np.asarray(ps, dtype=np.float64)
    labels = np.asarray(ys, dtype=np.bool_)
    errors = (probs - labels.astype(np.float64)) ** 2
    base = float(labels.mean()) if len(labels) else 0.0
    bs_c = float(((base - labels.astype(np.float64)) ** 2).mean()) if len(labels) else 0.0
    bs = float(errors.mean()) if len(errors) else 0.0
    guesses = probs >= 0.5
    tp = int(((guesses) & labels).sum())
    fn = int(((~guesses) & labels).sum())
    return {
        "brier": bs,
        "bs_constant": bs_c,
        "bss": float(1.0 - bs / bs_c) if bs_c > 1e-12 else 0.0,
        "accuracy": float((guesses == labels).mean()) if len(labels) else 0.0,
        "mean_p": float(probs.mean()) if len(probs) else 0.0,
        "mean_p_given_pos": float(probs[labels].mean()) if labels.any() else 0.0,
        "mean_p_given_neg": float(probs[~labels].mean()) if (~labels).any() else 0.0,
        "positive_recall": float(tp / max(1, int(labels.sum()))),
        "y_pos_rate": base,
        "unexpected_proxy": float(((probs >= 0.5) & (~labels)).mean()) if len(labels) else 0.0,
    }


def run_arm(args) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    pool = _build_items(tokenizer, args.n_pool, args.pos_rate, args.seed)
    replay = PersistentPredReplay(capacity=4096)
    replay.add_many(pool)

    metrics_path = OUT / f"{args.name}_metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    t0 = time.time()
    for step in range(1, args.steps + 1):
        if args.mode == "a1":
            # Raw online mini-batch: mostly negatives — no replay rebalance.
            batch = random.sample(pool, k=min(args.batch, len(pool)))
            use_bce, brier_w = False, 1.0
        else:
            batch = replay.sample_balanced(k_pos=args.k_pos, k_neg=args.k_neg)
            use_bce, brier_w = True, 0.1

        optim.zero_grad(set_to_none=True)
        losses = []
        ps, ys = [], []
        for item in batch:
            s_yes, s_no = _score_pair(model, device, item)
            brier, p = binary_brier_from_scores(s_yes, s_no, item.label)
            bce = None
            if use_bce:
                bce, p = binary_bce_from_scores(s_yes, s_no, item.label)
            loss = combine_pred_losses(bce, brier, bce_weight=1.0 if use_bce else 0.0, brier_weight=brier_w)
            if loss is not None:
                losses.append(loss)
            ps.append(float(p.detach().item()))
            ys.append(bool(item.label))
        if not losses:
            continue
        total = torch.stack(losses).mean()
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        optim.step()
        row = {
            "step": step,
            "name": args.name,
            "mode": args.mode,
            "loss": float(total.detach().item()),
            "grad_norm": grad_norm,
            **_metrics(ps, ys),
            "elapsed_s": time.time() - t0,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    return metrics_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=["a1", "a5"], required=True)
    parser.add_argument("--model", default=str(ROOT / "LLM" / "Qwen3-4B"))
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--k-pos", type=int, default=2)
    parser.add_argument("--k-neg", type=int, default=2)
    parser.add_argument("--n-pool", type=int, default=256)
    parser.add_argument("--pos-rate", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    path = run_arm(args)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
