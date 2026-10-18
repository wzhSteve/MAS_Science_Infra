#!/usr/bin/env python3
"""SFT 训练中周期性 generate，把当前模型输出写到 output_dir 供看板展示。

推荐配合 ZeRO-2（参数完整）：rank0 直接 generate，其它 rank barrier 等待。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _is_rank0() -> bool:
    return _rank() == 0


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _load_prompts(data_path: Path, n: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            user = ""
            gold = ""
            for turn in row.get("conversations") or []:
                if turn.get("from") == "human" and not user:
                    user = turn.get("value") or ""
                elif turn.get("from") == "gpt" and not gold:
                    gold = turn.get("value") or ""
            rows.append({"system": row.get("system") or "", "user": user, "gold": gold})
            if len(rows) >= n:
                break
    return rows


class LiveGenCallback(TrainerCallback):
    def __init__(
        self,
        data_path: str | Path,
        every_n_steps: int = 20,
        max_new_tokens: int = 256,
        n_samples: int = 2,
        enabled: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.every_n_steps = max(1, int(every_n_steps))
        self.max_new_tokens = int(max_new_tokens)
        self.n_samples = max(1, int(n_samples))
        self.enabled = enabled
        pool = max(self.n_samples * 4, 24)
        self.prompts = _load_prompts(self.data_path, pool) if enabled else []
        self.stop_strings = ["</search>", "</python>", "</answer>"]
        self._last_step = -1

    @classmethod
    def from_env(cls) -> "LiveGenCallback":
        return cls(
            data_path=os.environ.get(
                "ARPO_LIVE_GEN_DATA",
                "/root/autodl-tmp/MAS_Science_Infra/agent-lightning/examples/ARPO/"
                "LLaMA-Factory/arpo_train_sft/dataset/final_5w4_still.jsonl",
            ),
            every_n_steps=int(os.environ.get("ARPO_LIVE_GEN_EVERY", "20")),
            max_new_tokens=int(os.environ.get("ARPO_LIVE_GEN_MAX_NEW", "512")),
            n_samples=int(os.environ.get("ARPO_LIVE_GEN_N_SAMPLES", "2")),
            enabled=os.environ.get("ARPO_LIVE_GEN", "1") not in ("0", "false", "False"),
        )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        if not self.enabled or not _is_rank0():
            return
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "live_generations.jsonl").write_text("", encoding="utf-8")
        meta = {
            "every_n_steps": self.every_n_steps,
            "max_new_tokens": self.max_new_tokens,
            "n_samples": len(self.prompts),
            "note": "training-time model.generate written by LiveGenCallback (ZeRO-2 friendly)",
        }
        (out / "live_gen_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[live_gen] enabled every={self.every_n_steps} max_new={self.max_new_tokens} "
            f"per_round={self.n_samples} pool={len(self.prompts)} -> {out / 'live_generations.jsonl'}",
            flush=True,
        )

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: Any = None,
        processing_class: Any = None,
        tokenizer: Any = None,
        **kwargs,
    ) -> None:
        if not self.enabled or model is None or not self.prompts:
            return
        step = int(state.global_step)
        if step <= 0 or step == self._last_step or step % self.every_n_steps != 0:
            return
        self._last_step = step

        tok = processing_class or tokenizer or kwargs.get("tokenizer")
        _barrier()
        record: dict | None = None
        if _is_rank0() and tok is not None:
            try:
                record = self._generate_rank0(step, float(state.epoch or 0), model, tok)
                self._write(args.output_dir, record)
                heads = [(s.get("model_output") or "").replace("\n", " ")[:80] for s in record.get("samples") or []]
                print(
                    f"[live_gen] step={step} ok secs={record.get('secs')} "
                    f"has_think={any(s.get('has_think') for s in record.get('samples') or [])} "
                    f"preview={heads!r}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "step": step,
                    "epoch": float(state.epoch or 0),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "secs": None,
                    "samples": [],
                }
                self._write(args.output_dir, record)
                print(f"[live_gen] step={step} FAILED {record['error']}", flush=True)
        _barrier()

    def _unwrap(self, model: Any) -> Any:
        m = model
        for _ in range(6):
            if hasattr(m, "module"):
                m = m.module
            else:
                break
        return m

    def _generate_rank0(self, step: int, epoch: float, model: Any, tok: Any) -> dict:
        was_training = model.training
        model.eval()
        t0 = time.time()
        samples_out: list[dict] = []
        unwrapped = self._unwrap(model)
        device = next(unwrapped.parameters()).device
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            wave = max(step // self.every_n_steps - 1, 0)
            start = (wave * self.n_samples) % max(len(self.prompts), 1)
            batch = [
                self.prompts[(start + i) % len(self.prompts)]
                for i in range(min(self.n_samples, len(self.prompts)))
            ]
            for idx, prompt in enumerate(batch):
                messages = []
                if prompt["system"]:
                    messages.append({"role": "system", "content": prompt["system"]})
                # 软约束：缩短 think，尽快发工具/答案（硬停靠 stop_strings）
                user = prompt["user"]
                user = (
                    user
                    + "\n\nKeep <think> to at most 3 short sentences, then close </think> "
                    "and immediately emit <search>, <python>, or <answer>."
                )
                messages.append({"role": "user", "content": user})
                try:
                    text_in = tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except TypeError:
                    text_in = tok.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                inputs = tok(text_in, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                gen_kwargs = dict(
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
                try:
                    gen_kwargs.update(stop_strings=self.stop_strings, tokenizer=tok)
                    with torch.no_grad():
                        out = unwrapped.generate(**inputs, **gen_kwargs)
                except TypeError:
                    gen_kwargs.pop("stop_strings", None)
                    gen_kwargs.pop("tokenizer", None)
                    with torch.no_grad():
                        out = unwrapped.generate(**inputs, **gen_kwargs)
                new_tokens = out[0][inputs["input_ids"].shape[1] :]
                text = tok.decode(new_tokens, skip_special_tokens=True)
                stopped = next((s for s in self.stop_strings if s in text), None)
                truncated = stopped is None and int(new_tokens.numel()) >= self.max_new_tokens
                samples_out.append(
                    {
                        "index": idx,
                        "pool_index": (start + idx) % len(self.prompts),
                        "user": prompt["user"],
                        "system": prompt["system"],
                        "gold_head": (prompt["gold"] or "")[:240],
                        "model_output": text,
                        "has_think": "<think>" in text,
                        "stopped_at": stopped,
                        "truncated": truncated,
                        "n_tokens": int(new_tokens.numel()),
                    }
                )
        finally:
            if was_training:
                model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return {
            "step": step,
            "epoch": epoch,
            "ok": True,
            "secs": round(time.time() - t0, 2),
            "max_new_tokens": self.max_new_tokens,
            "samples": samples_out,
        }

    def _write(self, output_dir: str, record: dict) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "live_generations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (out / "live_generation_latest.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
