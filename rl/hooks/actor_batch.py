"""Keep a non-empty actor batch when a step has fewer triplets than one full mini-batch."""


def actor_batch_keep_count(
    n_transition: int,
    mini_times_n: int,
    world_size: int,
    micro_per_gpu: int = 1,
) -> int:
    """Rows to keep so every rank can run one or more Adam steps.

    A full step uses ``mini_times_n`` rows (yaml mini-batch times rollout.n).
    When the step collected fewer triplets than that, keep the largest multiple
    of ``world_size * micro_per_gpu`` so the update is one shorter step instead
    of an empty batch.
    """
    world = max(1, int(world_size))
    micro = max(1, int(micro_per_gpu))
    unit = world * micro
    full = max(unit, int(mini_times_n))
    if full % unit != 0:
        full = (full // unit) * unit
    n = max(0, int(n_transition))
    if n >= full:
        return (n // full) * full
    return (n // unit) * unit


def effective_ppo_mini_batch_size(n_rows: int, mini: int, micro: int) -> int:
    """Local mini-batch size. A short rank batch is one optimizer step."""
    rows = int(n_rows)
    mini_size = int(mini)
    micro_size = max(1, int(micro))
    if rows <= 0:
        raise RuntimeError("actor update received an empty batch")
    if rows < mini_size:
        if rows % micro_size != 0:
            raise RuntimeError(
                f"actor batch {rows} is smaller than mini {mini_size} "
                f"and not divisible by micro {micro_size}"
            )
        return rows
    if rows % mini_size != 0:
        raise RuntimeError(
            f"actor batch {rows} is not a multiple of mini {mini_size}"
        )
    return mini_size
