"""Shim → ``rl.hooks.trainer`` (imports AGL)."""

from rl.hooks.trainer import TirAgentLightningTrainer, bound_daemon_cls

__all__ = ["TirAgentLightningTrainer", "bound_daemon_cls"]
