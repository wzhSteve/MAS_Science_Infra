"""Compatibility adapter for the currently configured Gate-based execution."""

from __future__ import annotations

from typing import Any, Dict, Iterable

from rl.hooks.branch_policy import allocate_forks
from workflow.contracts import BranchSite
from workflow.gates import GateContext, evaluate_gate
from workflow.sampling.adapters.base import SamplingStrategyAdapter
from workflow.sampling.adapters.tool_result import tool_result_opportunities
from workflow.sampling.contracts import (
    SamplingDecision,
    SamplingOpportunity,
    SamplingWindow,
    WindowKind,
)


class ConfiguredGateAdapter(SamplingStrategyAdapter):
    id = "configured_gate"
    aliases = ("aepo", "appo", "rae")
    allowed_gates = ("entropy_delta", "arpo", "always", "dual_entropy")

    def supports_window(self, window: SamplingWindow) -> bool:
        return window.kind == WindowKind.TOOL_RESULT

    def opportunities(self, workflow: Any) -> Iterable[SamplingOpportunity]:
        return tool_result_opportunities(
            workflow,
            allowed_gates=self.allowed_gates,
            message="Tool Result 后保留 messages 前缀并扩展后续路径。",
        )

    def decide(
        self,
        window: SamplingWindow,
        site: BranchSite,
        context: Dict[str, Any],
    ) -> SamplingDecision:
        metrics = window.metrics
        missing = self._missing_inputs(site, metrics, context)
        if missing:
            return SamplingDecision(
                window_id=window.window_id,
                site_id=site.id,
                passed=False,
                reason="missing_gate_inputs",
                metadata={"missing": missing},
            )
        gate_context = GateContext(
            h_root=float(metrics.get("h_root") or 0.0),
            h_tool=float(metrics.get("h_tool") or 0.0),
            consecutive_high=int(metrics.get("consecutive_high", 0) or 0),
            tool_ok=metrics.get("tool_ok", context.get("tool_ok")),
            verifier_ok=metrics.get("verifier_ok", context.get("verifier_ok")),
            final_failed=bool(metrics.get("final_failed", context.get("final_failed", False))),
            h_branch=context.get("h_branch"),
            rng=context.get("rng"),
        )
        result = evaluate_gate(site.gate, gate_context)
        return SamplingDecision(
            window_id=window.window_id,
            site_id=site.id,
            passed=result.passed,
            score=result.score,
            reason=result.reason,
            metadata={
                **dict(result.meta or {}),
                "h_branch": context.get("h_branch"),
            },
        )

    @staticmethod
    def _missing_inputs(
        site: BranchSite,
        metrics: Dict[str, Any],
        context: Dict[str, Any],
    ) -> list[str]:
        gate = str(site.gate.type or "").lower()
        missing: list[str] = []
        if gate in ("entropy_delta", "arpo", "dual_entropy"):
            missing.extend(
                key for key in ("h_root", "h_tool")
                if key not in metrics or metrics[key] is None
            )
        if gate == "dual_entropy" and context.get("h_branch") is None:
            missing.append("h_branch")
        return missing

    def allocate(
        self,
        decision: SamplingDecision,
        *,
        remaining: int,
        site: BranchSite,
        default_beam: int,
    ) -> int:
        if not decision.passed:
            return 0
        beam = int(site.fork.beam_size or default_beam)
        counts = allocate_forks(remaining=remaining, beam_size=beam, n_sources=1)
        return counts[0] if counts else 0
