"""Algorithm-neutral matching, hit policy, budget, and expansion planning."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Sequence

from workflow.contracts import BranchSite
from workflow.sampling.adapters.base import SamplingStrategyAdapter
from workflow.sampling.compat import selector_from_anchor, selector_matches_window
from workflow.sampling.contracts import ExpansionPlan, SamplingWindow


@dataclass(frozen=True)
class ResumableWindow:
    window: SamplingWindow
    resume_messages: Sequence[Dict[str, Any]]


@dataclass(frozen=True)
class PlannedExpansion:
    plan: ExpansionPlan
    site: BranchSite
    window: SamplingWindow
    resume_messages: List[Dict[str, Any]]


ContextFactory = Callable[[BranchSite, ResumableWindow], Dict[str, Any]]


class SamplingCore:
    def __init__(self, adapter: SamplingStrategyAdapter) -> None:
        self.adapter = adapter
        self._hits: Dict[str, int] = {}

    @staticmethod
    def _applies(site: BranchSite, hit_count: int) -> bool:
        when = str(site.when or "first").lower()
        if when == "first":
            return hit_count == 0
        if when == "every":
            return True
        if when == "nth":
            return hit_count + 1 == max(1, int(site.nth or 1))
        return False

    def plan(
        self,
        *,
        parent_rollout_id: str,
        sites: Iterable[BranchSite],
        windows: Iterable[ResumableWindow],
        remaining: int,
        depth: int,
        max_depth: int,
        default_beam: int,
        context_factory: ContextFactory,
    ) -> List[PlannedExpansion]:
        if remaining <= 0 or depth >= max_depth:
            return []
        available = list(windows)
        output: List[PlannedExpansion] = []
        budget = remaining
        for site in sites:
            selector = selector_from_anchor(site.anchor)
            for resolved in available:
                if budget <= 0:
                    return output
                window = resolved.window
                if not self.adapter.supports_window(window):
                    continue
                if not selector_matches_window(selector, window):
                    continue
                hit_count = self._hits.get(site.id, 0)
                self._hits[site.id] = hit_count + 1
                if not self._applies(site, hit_count):
                    continue
                if not window.snapshot_ref or not resolved.resume_messages:
                    continue
                decision = self.adapter.decide(
                    window,
                    site,
                    context_factory(site, resolved),
                )
                count = self.adapter.allocate(
                    decision,
                    remaining=budget,
                    site=site,
                    default_beam=default_beam,
                )
                if count <= 0:
                    continue
                plan = ExpansionPlan(
                    parent_rollout_id=parent_rollout_id,
                    window_id=window.window_id,
                    snapshot_ref=window.snapshot_ref,
                    site_id=site.id,
                    count=count,
                    depth=depth + 1,
                    decision=decision,
                    metadata={
                        "window_kind": window.kind.value,
                        "window_sequence": window.sequence,
                        "interaction": deepcopy(window.interaction),
                    },
                )
                output.append(
                    PlannedExpansion(
                        plan=plan,
                        site=site,
                        window=window,
                        resume_messages=deepcopy(list(resolved.resume_messages)),
                    )
                )
                budget -= count
        return output
