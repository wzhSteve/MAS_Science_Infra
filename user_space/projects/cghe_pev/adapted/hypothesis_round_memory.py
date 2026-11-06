"""Round-scoped hypothesis memory for feedback-guided DAG regeneration.

A Planning DAG is temporary and versioned. Hypothesis evidence persists across
rounds. Execution failure alone never marks a hypothesis REFUTED.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence

from cghe_types import new_id, norm


class HypothesisStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNVERIFIED = "UNVERIFIED"


class SubgoalStatus(str, Enum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    UNKNOWN = "UNKNOWN"


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class PlanningImplication(str, Enum):
    KEEP = "KEEP"
    REVISE = "REVISE"
    AVOID = "AVOID"
    INVESTIGATE = "INVESTIGATE"


@dataclass
class VerifierFeedback:
    """Structured verifier output for HypothesisMemory updates (not a new DAG)."""

    subgoal_status: SubgoalStatus = SubgoalStatus.UNKNOWN
    execution_status: ExecutionStatus = ExecutionStatus.UNKNOWN
    evidence: str = ""
    failure_type: str = ""
    affected_hypothesis_ids: tuple[str, ...] = ()
    planning_implication: PlanningImplication = PlanningImplication.INVESTIGATE
    tool: str = ""
    argument: str = ""
    dag_version: str = ""
    round_id: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_status": self.subgoal_status.value,
            "execution_status": self.execution_status.value,
            "evidence": self.evidence,
            "failure_type": self.failure_type,
            "affected_hypothesis_ids": list(self.affected_hypothesis_ids),
            "planning_implication": self.planning_implication.value,
            "tool": self.tool,
            "argument": self.argument,
            "dag_version": self.dag_version,
            "round_id": self.round_id,
            "note": self.note,
        }


@dataclass
class PlanningHypothesis:
    """One planning assumption / Step-1 query hypothesis spanning DAG rounds."""

    hypothesis_id: str
    description: str
    preconditions: tuple[str, ...] = ()
    expected_outcome: str = ""
    verification_status: HypothesisStatus = HypothesisStatus.UNKNOWN
    execution_history: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    associated_dag_versions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    tool: str = ""
    argument: str = ""
    environment_fingerprint: str = ""
    semantic_key: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id:
            self.hypothesis_id = new_id("h")
        if not self.semantic_key:
            self.semantic_key = _semantic_key(self.tool, self.argument, self.description)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "description": self.description,
            "preconditions": list(self.preconditions),
            "expected_outcome": self.expected_outcome,
            "verification_status": self.verification_status.value,
            "execution_history": list(self.execution_history),
            "evidence": list(self.evidence),
            "associated_dag_versions": list(self.associated_dag_versions),
            "confidence": self.confidence,
            "tool": self.tool,
            "argument": self.argument,
            "environment_fingerprint": self.environment_fingerprint,
            "semantic_key": self.semantic_key,
        }


def _semantic_key(tool: str, argument: str, description: str = "") -> str:
    tool_n = norm(tool)
    arg_n = norm(argument) or norm(description)
    return f"{tool_n}|{arg_n}"


class HypothesisMemory:
    """Persistent across planning rounds; never mutates historical DAGs."""

    def __init__(self) -> None:
        self._by_id: dict[str, PlanningHypothesis] = {}
        self._by_key: dict[str, str] = {}
        self.round_index: int = 0
        self.last_feedback: Optional[VerifierFeedback] = None
        self.archived_dag_versions: list[str] = []
        self.events: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._by_id)

    def all(self) -> list[PlanningHypothesis]:
        return list(self._by_id.values())

    def get(self, hypothesis_id: str) -> Optional[PlanningHypothesis]:
        return self._by_id.get(hypothesis_id)

    def get_by_key(self, tool: str, argument: str, description: str = "") -> Optional[PlanningHypothesis]:
        key = _semantic_key(tool, argument, description)
        hid = self._by_key.get(key)
        return self._by_id.get(hid) if hid else None

    def upsert(
        self,
        *,
        description: str,
        tool: str = "",
        argument: str = "",
        expected_outcome: str = "",
        preconditions: Sequence[str] = (),
        dag_version: str = "",
        environment_fingerprint: str = "",
        confidence: float = 0.0,
    ) -> PlanningHypothesis:
        key = _semantic_key(tool, argument, description)
        existing_id = self._by_key.get(key)
        if existing_id and existing_id in self._by_id:
            hyp = self._by_id[existing_id]
            if dag_version and dag_version not in hyp.associated_dag_versions:
                hyp.associated_dag_versions.append(dag_version)
            if expected_outcome and not hyp.expected_outcome:
                hyp.expected_outcome = expected_outcome
            if environment_fingerprint:
                hyp.environment_fingerprint = environment_fingerprint
            return hyp
        hyp = PlanningHypothesis(
            hypothesis_id=new_id("h"),
            description=description or argument or tool,
            preconditions=tuple(preconditions or ()),
            expected_outcome=expected_outcome,
            tool=tool,
            argument=argument,
            environment_fingerprint=environment_fingerprint,
            confidence=float(confidence or 0.0),
            semantic_key=key,
        )
        if dag_version:
            hyp.associated_dag_versions.append(dag_version)
        self._by_id[hyp.hypothesis_id] = hyp
        self._by_key[key] = hyp.hypothesis_id
        self.events.append({"event": "hypothesis_added", "hypothesis_id": hyp.hypothesis_id, "key": key})
        return hyp

    def register_dag_hypotheses(self, graph: Any) -> list[PlanningHypothesis]:
        """Index Step-1 nodes of a newly generated DAG as UNKNOWN hypotheses."""
        created: list[PlanningHypothesis] = []
        version = str(getattr(graph, "version", "") or getattr(graph, "round_id", "") or "")
        for plan_id in list(getattr(graph, "active_plan_ids", lambda: ())()):
            sequence = list(graph.plan_node_ids(plan_id))
            if not sequence:
                continue
            node = graph.node(sequence[0])
            hyp = self.upsert(
                description=str(getattr(node, "subgoal", "") or getattr(node, "action", "") or ""),
                tool=str(getattr(node, "tool", "") or ""),
                argument=str(getattr(node, "argument", "") or getattr(node, "action", "") or ""),
                expected_outcome="SnippetsSufficient or subgoal satisfied",
                dag_version=version,
            )
            created.append(hyp)
        return created

    def update_from_feedback(
        self,
        feedback: VerifierFeedback,
        *,
        environment_fingerprint: str = "",
    ) -> list[PlanningHypothesis]:
        """Apply structured feedback. Timeouts/soft misses → UNVERIFIED, not REFUTED."""
        self.last_feedback = feedback
        updated: list[PlanningHypothesis] = []
        targets = list(feedback.affected_hypothesis_ids)
        if not targets and (feedback.tool or feedback.argument):
            hyp = self.get_by_key(feedback.tool, feedback.argument)
            if hyp is None:
                hyp = self.upsert(
                    description=feedback.argument or feedback.tool,
                    tool=feedback.tool,
                    argument=feedback.argument,
                    dag_version=feedback.dag_version,
                    environment_fingerprint=environment_fingerprint,
                )
            targets = [hyp.hypothesis_id]

        for hid in targets:
            hyp = self._by_id.get(hid)
            if hyp is None:
                continue
            if environment_fingerprint:
                # Material environment change: REFUTED under old fingerprint may be reopened.
                if (
                    hyp.verification_status == HypothesisStatus.REFUTED
                    and hyp.environment_fingerprint
                    and hyp.environment_fingerprint != environment_fingerprint
                ):
                    hyp.verification_status = HypothesisStatus.UNKNOWN
                hyp.environment_fingerprint = environment_fingerprint or hyp.environment_fingerprint

            status = self._status_from_feedback(feedback)
            # Never demote SUPPORTED to UNVERIFIED on a later soft miss of a different call.
            if status == HypothesisStatus.UNVERIFIED and hyp.verification_status == HypothesisStatus.SUPPORTED:
                pass
            elif status == HypothesisStatus.REFUTED or hyp.verification_status != HypothesisStatus.REFUTED:
                hyp.verification_status = status
            if feedback.evidence:
                hyp.evidence.append(feedback.evidence[:500])
            hyp.execution_history.append(feedback.to_dict())
            if feedback.dag_version and feedback.dag_version not in hyp.associated_dag_versions:
                hyp.associated_dag_versions.append(feedback.dag_version)
            if status == HypothesisStatus.REFUTED:
                hyp.confidence = 1.0
            elif status == HypothesisStatus.SUPPORTED:
                hyp.confidence = max(hyp.confidence, 0.8)
            updated.append(hyp)
            self.events.append({
                "event": "hypothesis_updated",
                "hypothesis_id": hyp.hypothesis_id,
                "status": hyp.verification_status.value,
                "implication": feedback.planning_implication.value,
                "failure_type": feedback.failure_type,
            })
        return updated

    def _status_from_feedback(self, feedback: VerifierFeedback) -> HypothesisStatus:
        if feedback.subgoal_status == SubgoalStatus.SATISFIED:
            return HypothesisStatus.SUPPORTED
        if feedback.planning_implication == PlanningImplication.AVOID and feedback.failure_type in {
            "SNIPPETS_INSUFFICIENT",
            "PARAM_ERROR",
            "PERMISSION_DENIED",
            "TOOL_NOT_FOUND",
            "SCHEMA_MISMATCH",
            "EXPLICIT_COUNTEREXAMPLE",
        }:
            return HypothesisStatus.REFUTED
        if feedback.subgoal_status == SubgoalStatus.NOT_SATISFIED:
            # Soft / inconclusive: timeout, empty-ish, verifier miss without hard CE.
            if feedback.failure_type in {
                "TRANSIENT",
                "TIMEOUT",
                "SUBGOAL_NOT_SATISFIED",
                "VERIFIER_UNCERTAIN",
                "EMPTY",
                "GENERIC",
                "",
            }:
                return HypothesisStatus.UNVERIFIED
            if feedback.execution_status == ExecutionStatus.FAILURE and feedback.failure_type in {
                "SNIPPETS_INSUFFICIENT",
                "PARAM_ERROR",
                "PERMISSION_DENIED",
                "TOOL_NOT_FOUND",
                "SCHEMA_MISMATCH",
                "EXPLICIT_COUNTEREXAMPLE",
            }:
                return HypothesisStatus.REFUTED
            return HypothesisStatus.UNVERIFIED
        return HypothesisStatus.UNKNOWN

    def refuted_keys(self, *, environment_fingerprint: str = "") -> set[str]:
        banned: set[str] = set()
        for hyp in self._by_id.values():
            if hyp.verification_status != HypothesisStatus.REFUTED:
                continue
            if (
                environment_fingerprint
                and hyp.environment_fingerprint
                and hyp.environment_fingerprint != environment_fingerprint
            ):
                continue
            banned.add(hyp.semantic_key)
        return banned

    def reusable(self) -> list[PlanningHypothesis]:
        return [
            hyp for hyp in self._by_id.values()
            if hyp.verification_status in {
                HypothesisStatus.UNKNOWN,
                HypothesisStatus.UNVERIFIED,
                HypothesisStatus.SUPPORTED,
            }
        ]

    def planner_guidance(self, *, environment_fingerprint: str = "") -> str:
        lines = [
            "HypothesisMemory guidance for a NEW Planning DAG:",
            "- Do NOT reuse REFUTED Step-1 queries under unchanged environment.",
            "- UNVERIFIED hypotheses are diagnostic only; they are NOT hard bans.",
            "- SUPPORTED / UNKNOWN hypotheses may be reused or reorganized.",
            "- Prefer NEW queries that address prior feedback.",
        ]
        if self.last_feedback is not None:
            fb = self.last_feedback
            lines.append(
                f"Previous feedback: subgoal={fb.subgoal_status.value} "
                f"exec={fb.execution_status.value} fail={fb.failure_type or 'none'} "
                f"implication={fb.planning_implication.value}"
            )
            if fb.evidence:
                lines.append(f"Evidence: {fb.evidence[:300]}")
        for hyp in self._by_id.values():
            if hyp.verification_status == HypothesisStatus.REFUTED:
                if (
                    environment_fingerprint
                    and hyp.environment_fingerprint
                    and hyp.environment_fingerprint != environment_fingerprint
                ):
                    continue
                lines.append(f"AVOID REFUTED: {hyp.tool} arg={hyp.argument or hyp.description}")
            elif hyp.verification_status == HypothesisStatus.UNVERIFIED:
                lines.append(f"HINT UNVERIFIED: {hyp.tool} arg={hyp.argument or hyp.description}")
            elif hyp.verification_status == HypothesisStatus.SUPPORTED:
                lines.append(f"KEEP SUPPORTED: {hyp.tool} arg={hyp.argument or hyp.description}")
        return "\n".join(lines)

    def filter_plans(self, plans: Sequence[Any], *, environment_fingerprint: str = "") -> list[Any]:
        """Drop plans whose Step-1 semantic key is REFUTED under this environment."""
        banned = self.refuted_keys(environment_fingerprint=environment_fingerprint)
        if not banned:
            return list(plans)
        kept = []
        for plan in plans:
            steps = list(getattr(plan, "steps", ()) or ())
            if not steps:
                continue
            step0 = steps[0]
            key = _semantic_key(
                str(getattr(step0, "tool", "") or ""),
                str(getattr(step0, "argument", "") or getattr(step0, "action", "") or ""),
                str(getattr(step0, "subgoal", "") or ""),
            )
            if key in banned:
                continue
            kept.append(plan)
        return kept

    def export_state(self) -> dict[str, Any]:
        return {
            "round_index": self.round_index,
            "hypotheses": [h.to_dict() for h in self._by_id.values()],
            "archived_dag_versions": list(self.archived_dag_versions),
            "last_feedback": self.last_feedback.to_dict() if self.last_feedback else None,
            "events": list(self.events[-200:]),
        }

    @classmethod
    def from_state(cls, raw: Optional[dict[str, Any]]) -> "HypothesisMemory":
        mem = cls()
        if not isinstance(raw, dict):
            return mem
        mem.round_index = int(raw.get("round_index") or 0)
        mem.archived_dag_versions = list(raw.get("archived_dag_versions") or [])
        for item in raw.get("hypotheses") or []:
            if not isinstance(item, dict):
                continue
            status = str(item.get("verification_status") or "UNKNOWN")
            try:
                vst = HypothesisStatus(status)
            except ValueError:
                vst = HypothesisStatus.UNKNOWN
            hyp = PlanningHypothesis(
                hypothesis_id=str(item.get("hypothesis_id") or new_id("h")),
                description=str(item.get("description") or ""),
                preconditions=tuple(item.get("preconditions") or ()),
                expected_outcome=str(item.get("expected_outcome") or ""),
                verification_status=vst,
                execution_history=list(item.get("execution_history") or []),
                evidence=list(item.get("evidence") or []),
                associated_dag_versions=list(item.get("associated_dag_versions") or []),
                confidence=float(item.get("confidence") or 0.0),
                tool=str(item.get("tool") or ""),
                argument=str(item.get("argument") or ""),
                environment_fingerprint=str(item.get("environment_fingerprint") or ""),
                semantic_key=str(item.get("semantic_key") or ""),
            )
            mem._by_id[hyp.hypothesis_id] = hyp
            mem._by_key[hyp.semantic_key or _semantic_key(hyp.tool, hyp.argument, hyp.description)] = hyp.hypothesis_id
        return mem


def build_verifier_feedback(
    *,
    satisfied: Optional[bool],
    label: str,
    analysis: Any = None,
    action: Optional[dict[str, Any]] = None,
    evidence_text: str = "",
    dag_version: str = "",
    round_id: int = 0,
    soft_continue: bool = False,
) -> VerifierFeedback:
    """Map FailureAnalyzer / verifier labels into structured feedback."""
    action = action or {}
    tool = str(action.get("tool_name") or action.get("tool") or "")
    argument = str(action.get("tool_input") or action.get("command") or "")
    failure_type = ""
    implication = PlanningImplication.INVESTIGATE
    if analysis is not None:
        failure_type = str(
            getattr(getattr(analysis, "evidence", None), "observation_code", "")
            or getattr(getattr(analysis, "constraint", None), "observation_code", "")
            or ""
        )
        if getattr(analysis, "verified_counterexample", False):
            implication = PlanningImplication.AVOID
            failure_type = failure_type or "EXPLICIT_COUNTEREXAMPLE"
        elif soft_continue:
            implication = PlanningImplication.KEEP
            failure_type = failure_type or "SUBGOAL_NOT_SATISFIED"
        else:
            implication = PlanningImplication.REVISE
            failure_type = failure_type or "SUBGOAL_NOT_SATISFIED"
    elif soft_continue:
        failure_type = "SUBGOAL_NOT_SATISFIED"
        implication = PlanningImplication.KEEP
    elif satisfied is False:
        failure_type = "SUBGOAL_NOT_SATISFIED"
        implication = PlanningImplication.REVISE

    if satisfied is True:
        subgoal = SubgoalStatus.SATISFIED
        execution = ExecutionStatus.SUCCESS
        implication = PlanningImplication.KEEP
    elif soft_continue:
        subgoal = SubgoalStatus.NOT_SATISFIED
        execution = ExecutionStatus.PARTIAL
    elif satisfied is False:
        subgoal = SubgoalStatus.NOT_SATISFIED
        execution = ExecutionStatus.FAILURE if implication == PlanningImplication.AVOID else ExecutionStatus.PARTIAL
    else:
        subgoal = SubgoalStatus.UNKNOWN
        execution = ExecutionStatus.UNKNOWN

    # Prefer bound search query when available.
    if analysis is not None:
        scope = getattr(getattr(analysis, "constraint", None), "scope", None)
        arg = str(getattr(scope, "argument", "") or "")
        if arg:
            argument = arg
        tool_scope = str(getattr(scope, "tool", "") or "")
        if tool_scope:
            tool = tool_scope

    return VerifierFeedback(
        subgoal_status=subgoal,
        execution_status=execution,
        evidence=str(evidence_text or label or "")[:800],
        failure_type=failure_type,
        planning_implication=implication,
        tool=tool,
        argument=argument,
        dag_version=dag_version,
        round_id=int(round_id or 0),
        note="soft_continue" if soft_continue else "",
    )


def freeze_dag_snapshot(graph: Any) -> dict[str, Any]:
    """Deep-copy structural metadata so Round t+1 cannot mutate Round t's DAG."""
    if graph is None:
        return {}
    nodes = []
    for node in graph.nodes():
        nodes.append({
            "node_id": node.node_id,
            "tool": getattr(node, "tool", ""),
            "action": getattr(node, "action", ""),
            "argument": getattr(node, "argument", ""),
            "subgoal": getattr(node, "subgoal", ""),
            "status": str(getattr(getattr(node, "status", None), "value", getattr(node, "status", ""))),
        })
    plans = []
    for plan in graph.plans():
        plans.append({
            "plan_id": plan.plan_id,
            "status": str(getattr(getattr(plan, "status", None), "value", getattr(plan, "status", ""))),
            "node_ids": list(graph.plan_node_ids(plan.plan_id)),
        })
    return {
        "version": str(getattr(graph, "version", "") or getattr(graph, "round_id", "") or ""),
        "round_id": str(getattr(graph, "round_id", "") or ""),
        "nodes": copy.deepcopy(nodes),
        "plans": copy.deepcopy(plans),
        "attempt_count": int(getattr(graph, "attempt_count", 0) or 0),
        "sealed": True,
    }
