"""Competitive, testable explanations of one failure.

SUPPORTED is not permission to prune. Only an admissible verified
counterexample can be passed to HypothesisPruner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


STATUSES = ("PROPOSED", "UNDER_TEST", "SUPPORTED", "REFUTED", "INCONCLUSIVE")


@dataclass
class CounterfactualHypothesis:
    hypothesis_id: str
    statement: str
    predicted_observations: list[str]
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)
    affected_plan_ids: list[str] = field(default_factory=list)
    scope: dict[str, str] = field(default_factory=dict)
    status: str = "PROPOSED"
    confidence: float = 0.0
    discriminative_action: str = ""

    @property
    def testable(self) -> bool:
        return bool(self.predicted_observations and self.scope.get("tool") and self.discriminative_action)


def _hypothesis(
    hypothesis_id: str,
    statement: str,
    prediction: str,
    action: str,
    *,
    tool: str,
    argument: str = "",
    plans: Sequence[str] = (),
) -> CounterfactualHypothesis:
    return CounterfactualHypothesis(
        hypothesis_id=hypothesis_id,
        statement=statement,
        predicted_observations=[prediction],
        affected_plan_ids=list(plans),
        scope={"tool": tool, "argument": argument},
        discriminative_action=action,
        status="PROPOSED",
    )


def generate_hypotheses(
    *,
    tool: str,
    argument: str = "",
    failure_type: str = "",
    plans: Sequence[str] = (),
) -> list[CounterfactualHypothesis]:
    """Several rival explanations. A bare failure is not rewritten as a planner error."""
    if not tool or failure_type in {"", "VERIFIER_UNCERTAIN", "TRANSIENT_ERROR", "GENERIC_FAILURE"}:
        return []
    rivals = [
        _hypothesis(
            "h-tool",
            f"{tool} cannot obtain the requested information",
            "alternate_tool_succeeds",
            "try_other_tool",
            tool=tool,
            argument=argument,
            plans=plans,
        ),
        _hypothesis(
            "h-query",
            "the current arguments do not name the intended resource",
            "revised_arguments_succeed",
            "revise_arguments",
            tool=tool,
            argument=argument,
            plans=plans,
        ),
        _hypothesis(
            "h-snippet",
            "the tool result is present but too incomplete to satisfy the subgoal",
            "reading_the_linked_resource_succeeds",
            "read_linked_resource",
            tool=tool,
            argument=argument,
            plans=plans,
        ),
    ]
    return deduplicate(rivals)


def deduplicate(hypotheses: Sequence[CounterfactualHypothesis]) -> list[CounterfactualHypothesis]:
    seen: set[str] = set()
    kept: list[CounterfactualHypothesis] = []
    for item in hypotheses:
        key = (item.statement.strip().casefold(), item.discriminative_action)
        if not item.testable or key in seen:
            continue
        seen.add(key)
        kept.append(item)
    return kept


def update_hypothesis(
    hypothesis: CounterfactualHypothesis,
    *,
    observation: str,
    verified_counterexample: bool = False,
) -> CounterfactualHypothesis:
    """Record new evidence. Verification still belongs to the pruner's admission check."""
    hypothesis.status = "UNDER_TEST"
    if observation in hypothesis.predicted_observations and verified_counterexample:
        hypothesis.supporting_evidence.append(observation)
        hypothesis.status = "SUPPORTED"
        return hypothesis
    if observation.startswith("contradict:"):
        hypothesis.contradicting_evidence.append(observation)
        hypothesis.status = "REFUTED"
        return hypothesis
    hypothesis.status = "INCONCLUSIVE"
    return hypothesis


def may_prune(hypothesis: CounterfactualHypothesis) -> bool:
    """Counterfactual state alone never authorizes elimination."""
    return False


def feedback_partition(
    hypotheses: Sequence[CounterfactualHypothesis],
    plan_ids: Sequence[str],
) -> Optional[dict[str, list[str]]]:
    """Group plans by the discriminative action that would test their hypothesis."""
    if not hypotheses:
        return None
    partition: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for item in hypotheses:
        bucket = [plan_id for plan_id in item.affected_plan_ids if plan_id in plan_ids]
        if not bucket:
            continue
        partition.setdefault(item.discriminative_action, [])
        for plan_id in bucket:
            if plan_id in assigned:
                continue
            partition[item.discriminative_action].append(plan_id)
            assigned.add(plan_id)
    rest = [plan_id for plan_id in plan_ids if plan_id not in assigned]
    if rest:
        partition["unaffected"] = rest
    if len([bucket for bucket in partition.values() if bucket]) < 2:
        return None
    return partition
