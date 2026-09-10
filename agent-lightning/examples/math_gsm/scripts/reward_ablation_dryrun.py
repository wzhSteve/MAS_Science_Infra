#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.
"""Offline reward-mode dry run for interview ablation (no VERL / no GPU).

Scores canned assistant transcripts under binary|shaped|format_only so you can
explain reward hacking without launching training.

Usage:
    python scripts/reward_ablation_dryrun.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

# Allow `python scripts/reward_ablation_dryrun.py` from repo example root or scripts/.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from math_agent import RewardMode, compute_reward, extract_final_answer  # noqa: E402

# Canned cases: (name, ground_truth, assistant_text, note)
CASES: List[Dict[str, Any]] = [
    {
        "name": "correct_strict_format",
        "gt": "42",
        "text": "I used the tool.\n### 42 ###",
        "note": "正确 + 严格格式：shaped 应略高于 binary",
    },
    {
        "name": "correct_loose_number",
        "gt": "42",
        "text": "The answer is 42",
        "note": "正确但无 ###：binary=1；shaped 无格式分；format_only=0",
    },
    {
        "name": "wrong_but_pretty_format",
        "gt": "42",
        "text": "### 41 ###",
        "note": "格式漂亮但算错：format_only 会被 hacking 成 1",
    },
    {
        "name": "tool_result_leak",
        "gt": "180",
        "text": "result = 180",
        "note": "像 tool 输出；extract 可能拿到数，但无 ### 格式",
    },
    {
        "name": "verbose_correct",
        "gt": "16",
        "text": ("step " * 800) + "\n### 16 ###",  # ~4000 chars → clear length penalty
        "note": "又对又长：shaped 应被长度惩罚压低（相对 binary 的 1.0）",
    },
    {
        "name": "empty",
        "gt": "1",
        "text": "",
        "note": "空回复：各 mode 都应为 0",
    },
]


def score_case(case: Dict[str, Any], mode: RewardMode) -> Dict[str, Any]:
    text = case["text"]
    pred = extract_final_answer(text)
    reward = compute_reward(
        pred,
        case["gt"],
        mode=mode,
        raw_assistant_text=text,
        num_assistant_chars=len(text),
    )
    return {
        "name": case["name"],
        "mode": mode,
        "prediction": pred,
        "reward": reward,
        "note": case["note"],
    }


def main() -> None:
    modes: List[RewardMode] = ["binary", "shaped", "format_only"]
    rows = []
    for case in CASES:
        for mode in modes:
            rows.append(score_case(case, mode))

    # Pretty table
    print(f"{'case':<28} {'mode':<12} {'pred':<12} {'reward':>7}  note")
    print("-" * 100)
    for row in rows:
        pred = repr(row["prediction"]) if row["prediction"] is not None else "None"
        print(
            f"{row['name']:<28} {row['mode']:<12} {pred:<12} {row['reward']:>7.3f}  {row['note']}"
        )

    out_path = os.path.join(_ROOT, "docs", "interview", "reward_ablation_dryrun.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {out_path}")

    # Interview takeaways printed for copy-paste into notes
    print("\nTakeaways:")
    print("1. format_only 把『答对』偷换成『长得像答案』→ 经典 reward hacking。")
    print("2. shaped 用格式奖励鼓励可解析输出，但要用长度惩罚防止废话刷分。")
    print("3. 正式训准确率时仍以 binary 为主；shaped/format_only 用于消融与讲述。")


if __name__ == "__main__":
    main()
