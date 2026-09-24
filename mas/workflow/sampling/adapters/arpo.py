"""ARPO adapter for entropy-gated Tool Result Window expansion."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable

from workflow.contracts import BranchSite
from workflow.sampling.adapters.configured import ConfiguredGateAdapter
from workflow.sampling.adapters.tool_result import tool_result_opportunities
from workflow.sampling.contracts import (
    SamplingDecision,
    SamplingOpportunity,
    SamplingWindow,
)


class ArpoSamplingAdapter(ConfiguredGateAdapter):
    id = "arpo"
    aliases: tuple[str, ...] = ()
    allowed_gates = ("entropy_delta", "arpo", "always")

    def opportunities(self, workflow: Any) -> Iterable[SamplingOpportunity]:
        return tool_result_opportunities(
            workflow,
            allowed_gates=self.allowed_gates,
            message="Tool Result 后保留完整消息前缀，重新采样上游 Agent 的后续决策。",
        )

    def decide(
        self,
        window: SamplingWindow,
        site: BranchSite,
        context: Dict[str, Any],
    ) -> SamplingDecision:
        if site.gate.type not in self.allowed_gates:
            return SamplingDecision(
                window_id=window.window_id,
                site_id=site.id,
                passed=False,
                reason="unsupported_arpo_gate",
                metadata={"gate": site.gate.type},
            )
        return super().decide(window, site, context)

    def training_overlay(self, config: Dict[str, Any], policy: Any) -> Dict[str, Any]:
        output = deepcopy(config)
        algorithm = dict(output.get("algorithm") or {})
        tir = dict(algorithm.get("tir") or {})
        tir["sampling_strategy"] = self.id
        algorithm["tir"] = tir
        output["algorithm"] = algorithm
        return output
