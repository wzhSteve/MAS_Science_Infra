"""Shim: TrainSignal / LossSpec owned by ``rl``."""

from rl.loss import AdvantageSpec, LossSpec
from rl.train_signal import TrainSignal, batch_to_train_signal

__all__ = ["AdvantageSpec", "LossSpec", "TrainSignal", "batch_to_train_signal"]
