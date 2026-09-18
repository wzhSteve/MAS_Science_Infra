"""RL hooks that subclass / configure Agent-lightning without forking it.

Importing this package must NOT pull VERL/AGL; use ``rl.hooks.trainer`` explicitly.
"""

from __future__ import annotations

from rl.hooks.overlay import VALID_ALGOS, apply_algo_overlay, apply_sample_policy, apply_train_signal

__all__ = [
    "VALID_ALGOS",
    "apply_algo_overlay",
    "apply_sample_policy",
    "apply_train_signal",
]


def __getattr__(name: str):
    if name in ("TirAgentLightningTrainer", "bound_daemon_cls"):
        from rl.hooks.trainer import TirAgentLightningTrainer, bound_daemon_cls

        return {"TirAgentLightningTrainer": TirAgentLightningTrainer, "bound_daemon_cls": bound_daemon_cls}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
