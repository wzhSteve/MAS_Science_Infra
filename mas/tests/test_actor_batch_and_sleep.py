"""Short actor batches still update, and planner sleep waits for the GPU."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from rl.hooks.actor_batch import actor_batch_keep_count, effective_ppo_mini_batch_size

REPO = Path(__file__).resolve().parents[2]


def test_step12_triplet_count_is_kept_for_one_update():
    # The failed run had 1975 triplets and a 256*8 floor, which dropped the step.
    kept = actor_batch_keep_count(1975, mini_times_n=2048, world_size=2, micro_per_gpu=1)
    assert kept == 1974
    assert effective_ppo_mini_batch_size(kept // 2, mini=1024, micro=1) == 987


def test_full_mini_batch_still_floors_to_the_configured_size():
    assert actor_batch_keep_count(5000, 2048, 2, 1) == 4096
    assert effective_ppo_mini_batch_size(1024, mini=1024, micro=1) == 1024


def test_too_few_rows_for_one_per_rank_stays_empty():
    assert actor_batch_keep_count(1, 2048, 2, 1) == 0
    with pytest.raises(RuntimeError, match="empty batch"):
        effective_ppo_mini_batch_size(0, mini=1024, micro=1)


def test_trainer_uses_the_keep_count():
    text = (REPO / "agent-lightning/agentlightning/verl/trainer.py").read_text(encoding="utf-8")
    assert "actor_batch_keep_count" in text
    exit_text = (REPO / "rl/verl_backend/verl/workers/sharding_manager/fsdp_vllm.py").read_text(encoding="utf-8")
    sync_at = exit_text.find("get_torch_device().synchronize()")
    sleep_at = exit_text.find("self.inference_engine.sleep(level=planner_sleep_level())")
    assert 0 <= sync_at < sleep_at
    assert exit_text.count("get_torch_device().synchronize()") >= 2
    server = (REPO / "rl/verl_backend/verl/workers/rollout/vllm_rollout/vllm_async_server.py").read_text(encoding="utf-8")
    assert "continuing so the actor update is not blocked" not in server
    assert "refusing to update or save on a live engine" in server


def test_drain_waits_until_requests_finish_before_sleep():
    from rl.verl_backend.verl.workers.rollout.vllm_rollout.vllm_async_server import drain_then_sleep

    class Processor:
        def __init__(self):
            self.request_states = {"rid": object()}

        def has_unfinished_requests(self):
            return bool(self.request_states)

    class Engine:
        def __init__(self):
            self.output_processor = Processor()
            self.slept = False

        async def abort(self, request_id):
            await asyncio.sleep(0)
            self.output_processor.request_states.pop(request_id, None)

        async def sleep(self):
            assert self.output_processor.has_unfinished_requests() is False
            self.slept = True

    engine = Engine()
    asyncio.run(drain_then_sleep(engine, timeout=2))
    assert engine.slept is True


def test_drain_refuses_to_sleep_while_requests_remain():
    from rl.verl_backend.verl.workers.rollout.vllm_rollout.vllm_async_server import drain_then_sleep

    class Processor:
        request_states: dict = {}

        def has_unfinished_requests(self):
            return True

    class Engine:
        def __init__(self):
            self.output_processor = Processor()
            self.slept = False

        async def abort(self, request_id):
            return None

        async def sleep(self):
            self.slept = True

    engine = Engine()
    with pytest.raises(RuntimeError, match="unfinished requests"):
        asyncio.run(drain_then_sleep(engine, timeout=0.1))
    assert engine.slept is False


def test_level2_unmap_during_a_live_kernel_is_the_save_fault():
    """Same allocator call as planner sleep(level=2): unmap while a kernel runs.

    Synchronizing first leaves CUDA usable. Unmapping first raises the illegal
    access that killed save_checkpoint. The allocator's destructor can abort
    the interpreter after a successful run, so the child exits with os._exit.
    """
    import os
    import subprocess
    import sys

    if subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).returncode != 0:
        pytest.skip("no GPU")
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    good = r"""
import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
import torch
from vllm.device_allocator.cumem import CuMemAllocator
alloc = CuMemAllocator.get_instance()
with alloc.use_memory_pool(tag="planner"):
    a = torch.randn(1024, 1024, device="cuda")
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    b = a
    for _ in range(4):
        b = b @ a
torch.cuda.synchronize()
alloc.sleep(offload_tags=tuple())
torch.cuda.synchronize()
c = torch.zeros(4, device="cuda")
torch.cuda.synchronize()
print("GOOD", float(c.sum()), flush=True)
os._exit(0)
"""
    bad = r"""
import os
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
import torch
from vllm.device_allocator.cumem import CuMemAllocator
alloc = CuMemAllocator.get_instance()
with alloc.use_memory_pool(tag="planner"):
    a = torch.randn(8192, 8192, device="cuda")
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    b = a
    for _ in range(20):
        b = b @ a
alloc.sleep(offload_tags=tuple())
torch.cuda.synchronize()
print("BAD_SURVIVED", flush=True)
os._exit(0)
"""
    good_proc = subprocess.run([sys.executable, "-c", good], env=env, capture_output=True, text=True)
    assert good_proc.returncode == 0, good_proc.stderr[-800:]
    assert "GOOD" in good_proc.stdout
    bad_proc = subprocess.run([sys.executable, "-c", bad], env=env, capture_output=True, text=True)
    combined = bad_proc.stdout + bad_proc.stderr
    assert "BAD_SURVIVED" not in combined
    assert "illegal memory access" in combined


def test_sleep_timeout_is_not_swallowed():
    from rl.verl_backend.verl.workers.rollout.vllm_rollout.vllm_async_server import drain_then_sleep

    class Engine:
        output_processor = None

        async def sleep(self):
            await asyncio.sleep(1)

    with pytest.raises(RuntimeError, match="refusing to update or save"):
        asyncio.run(drain_then_sleep(Engine(), timeout=0.05))
