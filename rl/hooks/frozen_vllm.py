"""Sleep and wake the dedicated frozen executor/verifier vLLM.

The OpenAI server in this environment exposes:

- ``POST /wake_up``
- ``POST /sleep?level=2``
- ``GET /is_sleeping``

``FCR_FROZEN_VLLM_URL`` empty, unset, or one of ``0/false/off/none`` disables
both calls. The training yaml sets it to ``http://127.0.0.1:8001`` (no ``/v1``).
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Callable

_OFF = {"", "0", "false", "no", "off", "none"}


def frozen_base() -> str | None:
    raw = os.environ.get("FCR_FROZEN_VLLM_URL")
    if raw is None:
        return None
    text = str(raw).strip().rstrip("/")
    if text.lower() in _OFF:
        return None
    return text


def enabled() -> bool:
    return frozen_base() is not None


def planner_sleep_level() -> int:
    """Level-2 drops the planner KV pool. Other runs stay at level 1."""
    return 2 if enabled() else 1


def before_planner_wake(tp_rank: int, barrier: Callable[[], None]) -> None:
    """Rank 0 wakes the frozen engine before the planner vLLM maps its pool.

    The barrier runs even when the HTTP call fails. Skipping it leaves the
    other rank inside the collective and the next allgather never matches.
    After that barrier, every rank raises together so generation does not
    start against a dead engine.
    """
    if not enabled():
        return
    failed = False
    try:
        if int(tp_rank) == 0:
            wake()
    except Exception:
        failed = True
        print("frozen vLLM wake failed; both ranks still leave the barrier")
    finally:
        barrier()
    if _ranks_agree_failure(failed):
        raise RuntimeError("frozen vLLM wake failed; refusing to generate against a dead engine")


def after_planner_sleep(tp_rank: int, barrier: Callable[[], None]) -> None:
    """Rank 0 releases the frozen pool after the planner has slept.

    A failed sleep still enters the barrier. Both ranks then raise, so the
    actor update does not run while the frozen pool is still mapped.
    """
    if not enabled():
        return
    failed = False
    try:
        if int(tp_rank) == 0:
            sleep()
    except Exception:
        failed = True
        print("frozen vLLM sleep failed; both ranks still leave the barrier")
    finally:
        barrier()
    if _ranks_agree_failure(failed):
        raise RuntimeError("frozen vLLM sleep failed; refusing to update on a live or dead engine")


def _ranks_agree_failure(failed: bool) -> bool:
    """True when any rank failed and the process group can see that flag."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return False
    import torch

    flag = torch.tensor([1 if failed else 0], device=torch.cuda.current_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return int(flag.item()) != 0


def wake(timeout: float = 180.0) -> None:
    base = frozen_base()
    if base is None:
        return
    _request(base + "/wake_up", method="POST", timeout=timeout)
    _chat_ping(base, timeout=min(timeout, 30.0))


def sleep(timeout: float = 180.0) -> None:
    base = frozen_base()
    if base is None:
        return
    # EngineCore.sleep unmaps the pool as soon as POST /sleep returns. A forward
    # still inside execute_model then hits an illegal memory access and the
    # process dies on the next step. Wait until the server itself reports idle.
    wait_until_idle(base)
    _request(base + "/sleep?level=2", method="POST", timeout=timeout)
    # vLLM 0.9.2 can return from POST /sleep before the engine is idle, and
    # then GET /is_sleeping blocks the same HTTP thread. A timeout here must
    # not skip the caller barrier.
    try:
        body = _request(base + "/is_sleeping", method="GET", timeout=30.0)
        payload = json.loads(body or "{}")
    except (RuntimeError, ValueError) as exc:
        print(f"frozen vLLM is_sleeping check skipped: {exc}")
        return
    if not payload.get("is_sleeping"):
        print(f"frozen vLLM at {base} did not report is_sleeping after level-2 sleep")


_GAUGE_VALUE = re.compile(r"^(?:\{[^}]*\})?\s+([-+0-9.eE]+)\s*$")


def prometheus_gauge_sum(text: str, name: str) -> float | None:
    """Sum one Prometheus gauge. Missing series returns None."""
    total: float | None = None
    prefix_labeled = name + "{"
    prefix_plain = name + " "
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith(prefix_labeled):
            rest = line[len(name) :]
        elif line.startswith(prefix_plain):
            rest = line[len(name) :]
        else:
            continue
        match = _GAUGE_VALUE.match(rest)
        if match is None:
            continue
        value = float(match.group(1))
        total = value if total is None else total + value
    return total


def wait_until_idle(base: str) -> None:
    """Block until vLLM reports no running and no waiting requests.

    The settle window covers the gap between the scheduler dropping a request
    and the last GPU kernel finishing. Sleeping inside that gap is the
    level-2 illegal memory access.
    """
    timeout = float(os.environ.get("FCR_FROZEN_IDLE_TIMEOUT", "1800"))
    settle = float(os.environ.get("FCR_FROZEN_SLEEP_SETTLE_S", "3"))
    poll = float(os.environ.get("FCR_FROZEN_IDLE_POLL_S", "1"))
    deadline = time.monotonic() + timeout
    idle_since: float | None = None
    last_notice = 0.0
    running = waiting = None
    while True:
        body = _request(base + "/metrics", method="GET", timeout=30.0)
        running = prometheus_gauge_sum(body, "vllm:num_requests_running")
        waiting = prometheus_gauge_sum(body, "vllm:num_requests_waiting")
        if running is None or waiting is None:
            raise RuntimeError(
                "frozen vLLM metrics have no request gauges; refusing to sleep"
            )
        now = time.monotonic()
        if running == 0 and waiting == 0:
            if idle_since is None:
                idle_since = now
            if now - idle_since >= settle:
                print(f"frozen vLLM idle for {settle:.1f}s; posting level-2 sleep")
                return
        else:
            idle_since = None
            if now - last_notice >= 30.0:
                print(
                    "frozen vLLM still busy "
                    f"(running={running}, waiting={waiting}); delaying level-2 sleep"
                )
                last_notice = now
        if now >= deadline:
            raise RuntimeError(
                "frozen vLLM still has requests "
                f"(running={running}, waiting={waiting}); refusing to sleep"
            )
        time.sleep(poll)


def _chat_ping(base: str, timeout: float) -> None:
    model = str(os.environ.get("AGENTFLOW_FIXED_MODEL") or os.environ.get("FCR_FROZEN_VLLM_MODEL") or "Qwen3-4B")
    raw = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
    ).encode("utf-8")
    body = _request(
        base + "/v1/chat/completions",
        method="POST",
        timeout=timeout,
        data=raw,
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(body or "{}")
    if not payload.get("choices"):
        raise RuntimeError(f"frozen vLLM at {base} woke but did not generate")


def _request(
    url: str,
    method: str,
    timeout: float,
    data: bytes | None = None,
    headers: dict | None = None,
) -> str:
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"frozen vLLM {method} {url} failed: {exc}") from exc
