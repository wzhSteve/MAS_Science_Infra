#!/usr/bin/env python3
"""确认 ARPO SFT 只在 assistant 回复上计算 loss。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llamafactory.data import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import DataArguments, ModelArguments
from llamafactory.model import load_tokenizer


def _load_rows(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--template", default="qwen")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="传给 DataArguments.enable_thinking；qwen3 慢思考时应为 true",
    )
    parser.add_argument(
        "--require-think",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="断言 answer 可训区间含 <think>；默认在 template=qwen3 时开启",
    )
    args = parser.parse_args()
    require_think = args.require_think if args.require_think is not None else args.template == "qwen3"

    data_path = Path(args.data)
    dataset_dir = data_path.parent.parent / "dataset_info"
    rows = _load_rows(data_path, args.num_samples)
    if not rows:
        raise SystemExit(f"数据为空: {data_path}")

    model_args = ModelArguments(model_name_or_path=args.model, trust_remote_code=True, flash_attn="disabled")
    data_args = DataArguments(
        template=args.template,
        dataset_dir=str(dataset_dir),
        cutoff_len=15000,
        enable_thinking=args.enable_thinking,
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    print(
        f"template={args.template} enable_thinking={args.enable_thinking} "
        f"require_think={require_think} template_cls={type(template).__name__}"
    )

    for index, row in enumerate(rows):
        messages = []
        for turn in row["conversations"]:
            role = {"human": "user", "gpt": "assistant"}.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})
        prompt_ids, answer_ids = template.encode_oneturn(tokenizer, messages, row.get("system"), None)
        if not prompt_ids or not answer_ids:
            raise SystemExit(f"样本 {index} 编码为空 prompt={len(prompt_ids)} answer={len(answer_ids)}")

        labels = [IGNORE_INDEX] * len(prompt_ids) + list(answer_ids)
        masked = sum(token == IGNORE_INDEX for token in labels)
        supervised = len(labels) - masked
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        answer_text = tokenizer.decode(answer_ids, skip_special_tokens=False)
        question = row["conversations"][0]["value"][:80]
        reply = row["conversations"][1]["value"][:80]
        if question not in prompt_text:
            raise SystemExit(f"样本 {index} 的用户问题没有落在被 mask 的 prompt 里")
        if reply not in answer_text:
            raise SystemExit(f"样本 {index} 的 assistant 回复没有落在 loss 区间里")
        if question in answer_text and question not in reply:
            raise SystemExit(f"样本 {index} 的用户问题被算进了 loss")
        if require_think and "<think>" not in answer_text:
            raise SystemExit(f"样本 {index} 的 loss 区间缺少 <think>（enable_thinking 可能关了或被剥掉）")
        print(
            f"sample {index}: prompt_tokens={len(prompt_ids)} answer_tokens={len(answer_ids)} "
            f"masked={masked} supervised={supervised} has_think={'<think>' in answer_text}"
        )
        print(f"  prompt_head={prompt_text[:120]!r}")
        print(f"  answer_head={answer_text[:120]!r}")

    print("loss_mask_ok")


if __name__ == "__main__":
    main()
