"""Compatibility shim: prefer ``rl.hooks``."""

from rl.hooks.overlay import VALID_ALGOS, apply_algo_overlay, apply_sample_policy, apply_train_signal

__all__ = [
    "VALID_ALGOS",
    "apply_algo_overlay",
    "apply_sample_policy",
    "apply_train_signal",
    "TirAgentLightningTrainer",
    "bound_daemon_cls",
]


def __getattr__(name: str):
    if name in ("TirAgentLightningTrainer", "bound_daemon_cls"):
        from rl.hooks.trainer import TirAgentLightningTrainer, bound_daemon_cls

        return {"TirAgentLightningTrainer": TirAgentLightningTrainer, "bound_daemon_cls": bound_daemon_cls}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
