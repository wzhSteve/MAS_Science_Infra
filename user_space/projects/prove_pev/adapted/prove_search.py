"""Synchronous Best-of-N search mirroring MAS-ProVe ``llm_parallel_search_decorator``.

PEV windows are sync, so candidates are sampled sequentially (or via a small
thread pool). Judge selection matches the decorator: take ``rankings[0]``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence

from prove_judge import judge_inline, judge_tcp
from prove_types import ProveCandidate, ProveDecision


def _flatten_to_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (list, tuple)):
        return "\n".join(_flatten_to_str(item) for item in obj)
    if isinstance(obj, dict):
        if "response" in obj and obj["response"]:
            return _flatten_to_str(obj["response"])
        if "agents" in obj and obj["agents"]:
            return _flatten_to_str(obj["agents"])
        for value in obj.values():
            if value:
                return _flatten_to_str(value)
        return str(obj)
    return str(obj)


def build_partial_trajectories(
    candidates: Sequence[ProveCandidate],
    *,
    context: str,
) -> List[Dict[str, str]]:
    return [
        {"context": context, "current-step": _flatten_to_str(c.text)}
        for c in candidates
    ]


def sample_candidates(
    sampler: Callable[[int], str],
    n: int,
    *,
    parallel: bool = False,
) -> List[ProveCandidate]:
    """Call ``sampler(index)`` ``n`` times; failures become a policy-safe stub."""
    n = max(1, int(n))
    out: List[Optional[ProveCandidate]] = [None] * n

    def _one(i: int) -> ProveCandidate:
        try:
            text = sampler(i)
        except Exception as exc:  # noqa: BLE001
            text = f"I cannot respond to this request due to content policy restrictions. ({type(exc).__name__})"
        return ProveCandidate(index=i, text=str(text or ""))

    if parallel and n > 1:
        with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
            futures = {pool.submit(_one, i): i for i in range(n)}
            for fut in as_completed(futures):
                cand = fut.result()
                out[cand.index] = cand
    else:
        for i in range(n):
            out[i] = _one(i)
    return [c for c in out if c is not None]


def select_best(
    candidates: Sequence[ProveCandidate],
    *,
    question: str,
    context: str,
    task_type: str,
    site: str,
    judge_backend: str,
    llm_call: Optional[Callable[[str, str], str]] = None,
    tcp_host: str = "127.0.0.1",
    tcp_port: int = 5555,
) -> ProveDecision:
    n = len(candidates)
    if n == 0:
        return ProveDecision(
            enabled=True,
            n=0,
            best_idx=0,
            rankings=[],
            skipped_reason="no_candidates",
            site=site,
        )
    if n == 1:
        return ProveDecision(
            enabled=True,
            n=1,
            best_idx=0,
            rankings=[0],
            skipped_reason="n_eq_1",
            site=site,
            candidates=[candidates[0].text],
            context=context,
        )

    partial = build_partial_trajectories(candidates, context=context)
    try:
        if judge_backend == "tcp":
            result = judge_tcp(
                partial,
                question=question,
                task_type=task_type,
                host=tcp_host,
                port=tcp_port,
            )
        else:
            if llm_call is None:
                raise RuntimeError("inline judge requires llm_call")
            result = judge_inline(
                partial,
                question=question,
                task_type=task_type,
                llm_call=llm_call,
            )
        rankings = list(result.get("rankings") or list(range(n)))
        feedback = str(result.get("judge-feedback") or "")
    except Exception as exc:  # noqa: BLE001
        rankings = list(range(n))
        feedback = f"judge_error: {type(exc).__name__}: {exc}"

    if not rankings or rankings[0] < 0 or rankings[0] >= n:
        rankings = list(range(n))
    best_idx = int(rankings[0])
    return ProveDecision(
        enabled=True,
        n=n,
        best_idx=best_idx,
        rankings=rankings,
        judge_feedback=feedback,
        context=context,
        candidates=[c.text for c in candidates],
        site=site,
    )
