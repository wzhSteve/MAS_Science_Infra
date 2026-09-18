"""RL hooks layer: reward / loss / algo overlays. Depends on AGL for training hooks only."""

from __future__ import annotations

__all__ = [
    "AdvantageSpec",
    "LossSpec",
    "TrainSignal",
    "VALID_ALGOS",
    "apply_algo_overlay",
    "apply_train_signal",
    "batch_to_train_signal",
]


def __getattr__(name: str):
    if name in ("VALID_ALGOS", "apply_algo_overlay", "apply_train_signal"):
        from rl.hooks.overlay import VALID_ALGOS, apply_algo_overlay, apply_train_signal

        return {
            "VALID_ALGOS": VALID_ALGOS,
            "apply_algo_overlay": apply_algo_overlay,
            "apply_train_signal": apply_train_signal,
        }[name]
    if name in ("AdvantageSpec", "LossSpec"):
        from rl.loss import AdvantageSpec, LossSpec

        return {"AdvantageSpec": AdvantageSpec, "LossSpec": LossSpec}[name]
    if name in ("TrainSignal", "batch_to_train_signal"):
        from rl.train_signal import TrainSignal, batch_to_train_signal

        return {"TrainSignal": TrainSignal, "batch_to_train_signal": batch_to_train_signal}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
