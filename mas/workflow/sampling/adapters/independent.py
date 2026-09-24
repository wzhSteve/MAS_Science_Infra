"""Independent rollout strategy: no branch opportunities or expansion."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from workflow.contracts import BranchSite
from workflow.sampling.adapters.base import SamplingStrategyAdapter
from workflow.sampling.contracts import SamplingDecision, SamplingOpportunity, SamplingWindow


class IndependentSamplingAdapter(SamplingStrategyAdapter):
    id = "independent"
    aliases = ("grpo", "grpo_n", "igpo", "gigpo")

    def opportunities(self, workflow: Any) -> Iterable[SamplingOpportunity]:
        return ()

    def supports_window(self, window: SamplingWindow) -> bool:
        return False

    def decide(
        self,
        window: SamplingWindow,
        site: BranchSite,
        context: Dict[str, Any],
    ) -> SamplingDecision:
        return SamplingDecision(
            window_id=window.window_id,
            site_id=site.id,
            passed=False,
            reason="independent_sampling",
        )

    def allocate(
        self,
        decision: SamplingDecision,
        *,
        remaining: int,
        site: BranchSite,
        default_beam: int,
    ) -> int:
        return 0
