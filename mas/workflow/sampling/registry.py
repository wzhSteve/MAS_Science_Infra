"""Sampling Strategy Adapter registry."""

from __future__ import annotations

from typing import Dict

from workflow.sampling.adapters.base import SamplingStrategyAdapter
from workflow.sampling.adapters.arpo import ArpoSamplingAdapter
from workflow.sampling.adapters.configured import ConfiguredGateAdapter
from workflow.sampling.adapters.independent import IndependentSamplingAdapter


class SamplingAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, SamplingStrategyAdapter] = {}

    def register(self, adapter: SamplingStrategyAdapter) -> None:
        for name in (adapter.id, *adapter.aliases):
            self._adapters[name] = adapter

    def resolve(self, name: str) -> SamplingStrategyAdapter:
        try:
            return self._adapters[name]
        except KeyError as error:
            raise ValueError(f"Unknown sampling strategy adapter: {name}") from error


sampling_adapters = SamplingAdapterRegistry()
sampling_adapters.register(IndependentSamplingAdapter())
sampling_adapters.register(ConfiguredGateAdapter())
sampling_adapters.register(ArpoSamplingAdapter())
