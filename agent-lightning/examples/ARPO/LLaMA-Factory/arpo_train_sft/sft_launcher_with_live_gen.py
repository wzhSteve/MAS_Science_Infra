#!/usr/bin/env python3
"""LLaMA-Factory 入口：注入 LiveGenCallback，训练中写模型 generate。"""

from __future__ import annotations

from live_gen_callback import LiveGenCallback
from llamafactory.train.tuner import run_exp


def launch() -> None:
    run_exp(callbacks=[LiveGenCallback.from_env()])


if __name__ == "__main__":
    launch()
