"""Sampling Strategy Adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

from workflow.contracts import BranchSite
from workflow.sampling.contracts import (
    SamplingDecision,
    SamplingOpportunity,
    SamplingWindow,
)


class SamplingStrategyAdapter(ABC):
    id: str
    aliases: tuple[str, ...] = ()

    @abstractmethod
    def opportunities(self, workflow: Any) -> Iterable[SamplingOpportunity]:
        """Return design-time opportunities supported by this adapter."""

    @abstractmethod
    def supports_window(self, window: SamplingWindow) -> bool:
        """Whether this adapter can make a Decision for the Window kind."""

    @abstractmethod
    def decide(
        self,
        window: SamplingWindow,
        site: BranchSite,
        context: Dict[str, Any],
    ) -> SamplingDecision:
        """Evaluate one matching Window."""

    @abstractmethod
    def allocate(
        self,
        decision: SamplingDecision,
        *,
        remaining: int,
        site: BranchSite,
        default_beam: int,
    ) -> int:
        """Allocate child count for a passed Decision."""

    def training_overlay(self, config: Dict[str, Any], policy: Any) -> Dict[str, Any]:
        return config
