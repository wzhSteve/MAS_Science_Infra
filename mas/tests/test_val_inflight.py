"""Validation planner calls share eight slots. Tool time and training do not."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

from val_inflight import bind_val_mode, hold_planner_generation


def _generate(directory: str, mode: str, seconds: float, stamp: str) -> None:
    with bind_val_mode(mode):
        with hold_planner_generation(slots=1, directory=Path(directory)):
            Path(stamp).write_text(f"{time.time():.6f}\n", encoding="utf-8")
            time.sleep(seconds)


def _generate_then_tool(directory: str, gen_stamp: str, tool_stamp: str) -> None:
    with bind_val_mode("val"):
        with hold_planner_generation(slots=1, directory=Path(directory)):
            Path(gen_stamp).write_text(f"{time.time():.6f}\n", encoding="utf-8")
            time.sleep(0.4)
        Path(tool_stamp).write_text(f"{time.time():.6f}\n", encoding="utf-8")
        time.sleep(0.4)


def test_second_validation_waits_for_the_only_slot(tmp_path: Path) -> None:
    directory = tmp_path / "slots"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    ctx = mp.get_context("spawn")
    holder = ctx.Process(target=_generate, args=(str(directory), "val", 0.4, str(first)))
    waiter = ctx.Process(target=_generate, args=(str(directory), "val", 0.0, str(second)))
    holder.start()
    time.sleep(0.1)
    waiter.start()
    holder.join(timeout=5)
    waiter.join(timeout=5)
    assert holder.exitcode == 0
    assert waiter.exitcode == 0
    first_at = float(first.read_text(encoding="utf-8"))
    second_at = float(second.read_text(encoding="utf-8"))
    assert second_at >= first_at + 0.3


def test_training_does_not_wait_for_a_held_validation_slot(tmp_path: Path) -> None:
    directory = tmp_path / "slots"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    ctx = mp.get_context("spawn")
    holder = ctx.Process(target=_generate, args=(str(directory), "val", 0.6, str(first)))
    trainer = ctx.Process(target=_generate, args=(str(directory), "train", 0.0, str(second)))
    holder.start()
    time.sleep(0.1)
    trainer.start()
    trainer.join(timeout=2)
    assert trainer.exitcode == 0
    assert second.exists()
    holder.join(timeout=5)
    assert holder.exitcode == 0


def test_tool_section_releases_the_slot_for_the_next_generation(tmp_path: Path) -> None:
    directory = tmp_path / "slots"
    first_gen = tmp_path / "first_gen.txt"
    tool = tmp_path / "tool.txt"
    second_gen = tmp_path / "second_gen.txt"
    ctx = mp.get_context("spawn")
    holder = ctx.Process(
        target=_generate_then_tool, args=(str(directory), str(first_gen), str(tool))
    )
    waiter = ctx.Process(target=_generate, args=(str(directory), "val", 0.0, str(second_gen)))
    holder.start()
    time.sleep(0.1)
    waiter.start()
    waiter.join(timeout=5)
    holder.join(timeout=5)
    assert holder.exitcode == 0
    assert waiter.exitcode == 0
    first_at = float(first_gen.read_text(encoding="utf-8"))
    second_at = float(second_gen.read_text(encoding="utf-8"))
    tool_at = float(tool.read_text(encoding="utf-8"))
    assert second_at >= first_at + 0.3
    assert second_at < tool_at + 0.2
