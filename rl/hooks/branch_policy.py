"""Pure branch / budget helpers aligned with official ARPO + MAS AEPO-lite.

Official ARPO (vllm_rollout_with_tools) after tool write-back::

    entropy_delta = entropy_now - entropy_init
    prob = clamp(random() - entropy_weight * entropy_delta)
    fork if prob <= branch_probability

MAS historically used ``p = alpha + gamma * ΔH`` then ``p > tau``.
Both gates live here; ActiveSetSession / Daemon share them.
"""

from __future__ import annotations

import random
from typing import List, Optional, Sequence

from rl.hooks.arpo_rollout import (  # re-export for single import surface
    aepo_global_budget,
    branch_probability,
    should_branch,
)

__all__ = [
    "aepo_global_budget",
    "allocate_forks",
    "arpo_fork_acceptance_prob",
    "arpo_should_fork",
    "branch_probability",
    "distribute_root_budgets",
    "should_branch",
]


def arpo_fork_acceptance_prob(
    entropy_now: float,
    entropy_root: float,
    *,
    entropy_weight: float = 0.5,
    rng: Optional[random.Random] = None,
) -> float:
    """Official-style acceptance probability before comparing to branch_probability.

    Lower returned ``prob`` ⇒ easier to pass ``prob <= branch_probability``.
    Larger ΔH lowers ``prob`` (more forks).
    """
    delta = float(entropy_now) - float(entropy_root)
    draw = (rng.random() if rng is not None else random.random())
    raw = draw - float(entropy_weight) * delta
    return max(0.0, min(1.0, raw))


def arpo_should_fork(
    entropy_now: float,
    entropy_root: float,
    *,
    branch_probability: float = 0.5,
    entropy_weight: float = 0.5,
    rng: Optional[random.Random] = None,
) -> bool:
    """Return True when official ARPO would copy ``curr_inputs`` and append a branch."""
    prob = arpo_fork_acceptance_prob(
        entropy_now,
        entropy_root,
        entropy_weight=entropy_weight,
        rng=rng,
    )
    return prob <= float(branch_probability)


def allocate_forks(
    *,
    remaining: int,
    beam_size: int,
    n_sources: int = 1,
) -> List[int]:
    """Per-source fork counts: each source at most ``beam_size - 1``, sum ≤ remaining.

    Mirrors official::

        branches_per_idx = min(beam_size - 1, remaining_slots - branches_created)
    """
    rem = max(0, int(remaining))
    per_cap = max(0, int(beam_size) - 1)
    n_src = max(0, int(n_sources))
    if rem <= 0 or per_cap <= 0 or n_src <= 0:
        return [0] * n_src
    out: List[int] = []
    created = 0
    for _ in range(n_src):
        n = min(per_cap, rem - created)
        if n <= 0:
            out.append(0)
            continue
        out.append(n)
        created += n
    return out


def distribute_root_budgets(group_n: int, n_roots: int) -> List[int]:
    """Split ``group_n`` completed-traj slots across ``n_roots`` tree-root tasks."""
    g = max(1, int(group_n))
    r = max(1, int(n_roots))
    r = min(r, g)
    base, extra = divmod(g, r)
    return [base + (1 if i < extra else 0) for i in range(r)]
