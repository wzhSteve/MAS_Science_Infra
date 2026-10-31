"""Budgeted recovery over one frozen candidate universe at a time.

A failure enters the planning graph only when the verifier has explicitly
rejected the step and the analyzer produced an explorable, verified
counterexample. Otherwise the controller records the failure and leaves the
original PEV retry / replan path in place.

An empty universe asks for another round while recovery, replan, and tool-call
budget remain. It does not mark the task impossible. Python and Generalist
calls are never auto-replayed. Search and fetch may be retried once per
fingerprint, and every attempt still spends tool-call budget.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

from cghe_types import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    NEEDS_REPLAN_MESSAGE,
    Budget,
    CandidatePlan,
    ExecutionObservation,
    FailureType,
    GraphNode,
    PruneResult,
    RecoveryRound,
    Scope,
    candidate_plan_from_dict,
    new_id,
    norm,
    to_plain,
)
from evidence_memory import EvidenceMemory, EvidenceRecord, PlanningRecords
from failure_analyzer import FailureAnalysis, FailureAnalyzer
from hypothesis_pruner import HypothesisPruner
from planning_graph import PlanningGraph, UniverseFrozenError


EXPLORABLE_FAILURE_TYPES = frozenset({
    FailureType.TOOL_UNAVAILABLE,
    FailureType.PERMISSION_DENIED,
    FailureType.SCHEMA_MISMATCH,
    FailureType.PARAMETER_ERROR,
    FailureType.SNIPPETS_INSUFFICIENT,
})

LOW_CONFIDENCE_FAILURE_TYPES = frozenset({
    FailureType.TRANSIENT_ERROR,
    FailureType.GENERIC_FAILURE,
    FailureType.VERIFIER_UNCERTAIN,
    FailureType.MISSING_INFORMATION,
})

_EXPLICIT_FAILURE_LABELS = frozenset({
    "failure",
    "failed",
    "不满足",
    "unsatisfied",
    "not_satisfied",
    "subgoal_not_satisfied",
    "no",
})
_UNCERTAIN_LABELS = frozenset({"uncertain", "verifier_uncertain", "unknown"})
_SUCCESS_LABELS = frozenset({"success", "满足", "satisfied", "passed", "ok", "yes"})

PYTHON_TOOL = "Python_Code_Generator_Tool"
GENERALIST_TOOL = "Generalist_Solution_Generator_Tool"
BING_TOOL = "Bing_Search_Tool"
FETCH_TOOL = "Web_Fetch_Tool"

# One original call, plus a single retry for tools declared safe.
_SAFE_FINGERPRINT_CAP = 2
_UNSAFE_FINGERPRINT_CAP = 1

STOP_RECOVERY_ROUNDS = "recovery round budget exhausted"
STOP_REPLANS = "replan budget exhausted"
STOP_TOOL_CALLS = "tool-call budget exhausted"
CONTINUE_PEV_MESSAGE = "沿用原 PEV 重试/重规划"

_TOOL_ALIASES = {
    "python": PYTHON_TOOL,
    "pythoncodegeneratortool": PYTHON_TOOL,
    "pythoncodertool": PYTHON_TOOL,
    "python_code_generator_tool": PYTHON_TOOL,
    "python_coder_tool": PYTHON_TOOL,
    "generalist": GENERALIST_TOOL,
    "generalistsolutiongeneratortool": GENERALIST_TOOL,
    "basegenerator": GENERALIST_TOOL,
    "basegeneratortool": GENERALIST_TOOL,
    "base_generator_tool": GENERALIST_TOOL,
    "bing": BING_TOOL,
    "bingsearch": BING_TOOL,
    "bingsearchtool": BING_TOOL,
    "websearchtool": BING_TOOL,
    "webfetch": FETCH_TOOL,
    "webfetchtool": FETCH_TOOL,
    "web_fetch_tool": FETCH_TOOL,
}


class RecoveryDecision(str):
    """String decision so logs and exported state stay plain text."""

    CONTINUE_PEV = "continue_pev"
    CONTINUE_EXECUTION = "continue_execution"
    NEED_UNIVERSE = "need_universe"
    STOP_BUDGET = "stop_budget"


@dataclass(frozen=True)
class RetryPolicy:
    """Whether recovery may replay a tool without a new planned call."""

    tool_name: str
    safe_to_retry: bool
    reason: str


RETRY_POLICIES: dict[str, RetryPolicy] = {
    PYTHON_TOOL: RetryPolicy(
        PYTHON_TOOL,
        False,
        "Python execution is not proven idempotent and must not be replayed",
    ),
    GENERALIST_TOOL: RetryPolicy(
        GENERALIST_TOOL,
        False,
        "Generalist is not treated as a safe retry",
    ),
    BING_TOOL: RetryPolicy(
        BING_TOOL,
        True,
        "Bing search is safe to retry within the tool-call budget",
    ),
    FETCH_TOOL: RetryPolicy(
        FETCH_TOOL,
        True,
        "Web fetch is safe to retry within the tool-call budget",
    ),
}


@dataclass(frozen=True)
class TriggerDecision:
    intervene: bool
    reason: str
    failure_type: Optional[FailureType] = None


@dataclass(frozen=True)
class RecoveryLog:
    """One elimination attempt. Counts are the active plans around the prune."""

    active_before: int
    active_after: int
    reason: str
    reason_code: str
    evidence_source: str
    round_id: str = ""
    constraint_id: str = ""
    decision: str = ""
    admissible: bool = False


@dataclass(frozen=True)
class ToolAuthorization:
    allowed: bool
    replay: bool
    safe_to_retry: bool
    fingerprint: str
    reason: str
    tool_name: str
    budget_remaining: int


@dataclass
class RecoveryOutcome:
    """What the next PEV window should do. ``task_impossible`` stays false."""

    decision: str
    reason: str
    intervene: bool
    round_id: str = ""
    selected_node: Optional[GraphNode] = None
    prune_result: Optional[PruneResult] = None
    analysis: Optional[FailureAnalysis] = None
    budget: Optional[Budget] = None
    logs: tuple[RecoveryLog, ...] = ()
    message: str = ""
    inherited_constraint_ids: tuple[str, ...] = ()
    recovery_round: Optional[RecoveryRound] = None
    task_impossible: bool = False

    def __post_init__(self) -> None:
        self.logs = tuple(self.logs or ())
        self.inherited_constraint_ids = tuple(self.inherited_constraint_ids or ())
        self.task_impossible = False


def canonical_tool_name(tool: str) -> str:
    key = norm(tool).replace("-", "_")
    compact = key.replace("_", "")
    if compact in _TOOL_ALIASES:
        return _TOOL_ALIASES[compact]
    if key in _TOOL_ALIASES:
        return _TOOL_ALIASES[key]
    return str(tool or "").strip()


def retry_policy_for(tool: str) -> RetryPolicy:
    name = canonical_tool_name(tool)
    policy = RETRY_POLICIES.get(name)
    if policy is not None:
        return policy
    return RetryPolicy(name or "unknown", False, "tool has no safe-retry declaration")


def call_fingerprint(
    tool: str,
    *,
    command: str = "",
    parameters: Optional[Mapping[str, object]] = None,
    environment: str = "",
) -> str:
    """Stable id for one tool invocation inside a recovery."""
    payload = {
        "tool": canonical_tool_name(tool),
        "command": norm(command),
        "parameters": _normalize_parameters(parameters),
        "environment": norm(environment),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def verifier_explicitly_failed(
    verifier_label: str = "",
    verifier_satisfied: Optional[bool] = None,
) -> bool:
    """True only when the verifier itself marked the step failed or unsatisfied."""
    label = norm(verifier_label).replace(" ", "_")
    if label in _UNCERTAIN_LABELS:
        return False
    if verifier_satisfied is False:
        return True
    if label in _SUCCESS_LABELS and verifier_satisfied is not False:
        return False
    return label in _EXPLICIT_FAILURE_LABELS


def should_enter_recovery(
    *,
    verifier_label: str = "",
    verifier_satisfied: Optional[bool] = None,
    failure_type: Optional[FailureType | str] = None,
    confidence: Optional[float] = None,
    verified_counterexample: bool = False,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    analysis: Optional[FailureAnalysis] = None,
) -> TriggerDecision:
    """Gate for planning-graph recovery.

    Ordinary execution failures stay on the PEV path. Recovery starts only
    when the verifier explicitly rejects the step and the failure is an
    explorable, verified counterexample. Verifier uncertainty and
    low-confidence failures return ``intervene=False``.
    """
    if isinstance(failure_type, str):
        failure_type = FailureType(failure_type)
    if analysis is not None:
        if failure_type is None:
            failure_type = analysis.evidence.failure_type
        if confidence is None and analysis.constraint is not None:
            confidence = analysis.constraint.confidence
        verified_counterexample = verified_counterexample or analysis.verified_counterexample
    label = norm(verifier_label).replace(" ", "_")
    if failure_type == FailureType.VERIFIER_UNCERTAIN or label in _UNCERTAIN_LABELS:
        return TriggerDecision(
            False,
            "verifier uncertainty does not enter recovery",
            failure_type or FailureType.VERIFIER_UNCERTAIN,
        )
    if not verifier_explicitly_failed(verifier_label, verifier_satisfied):
        return TriggerDecision(
            False,
            "verifier did not explicitly mark the step unsatisfied",
            failure_type,
        )
    if failure_type == FailureType.SUBGOAL_NOT_SATISFIED:
        return TriggerDecision(
            False,
            "ordinary subgoal failure stays on the original PEV path",
            failure_type,
        )
    if failure_type in LOW_CONFIDENCE_FAILURE_TYPES or failure_type not in EXPLORABLE_FAILURE_TYPES:
        return TriggerDecision(
            False,
            "low-confidence failure does not enter recovery",
            failure_type,
        )
    if not verified_counterexample or confidence is None or float(confidence) < confidence_threshold:
        return TriggerDecision(
            False,
            "low-confidence failure does not enter recovery",
            failure_type,
        )
    return TriggerDecision(
        True,
        "verifier rejected an explorable verified counterexample",
        failure_type,
    )


class RecoveryController:
    """Analyze, admit, prune, remember, and pick the next shared node."""

    def __init__(
        self,
        memory: Optional[EvidenceMemory] = None,
        *,
        budget: Optional[Budget] = None,
        tool_calls: Optional[int] = None,
        replans: Optional[int] = None,
        recovery_rounds: Optional[int] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        environment: str = "",
        task_id: str = "",
        task_type: str = "",
        task_goal: str = "",
        db_path: Optional[str] = None,
        pruning: bool = True,
        shared_priority: bool = True,
        algorithm: str = "v1",
        v2_mode: str = "full_cghe_v2",
        lambda_cost: float = 0.0,
        mu_risk: float = 0.0,
    ) -> None:
        self.memory = memory if memory is not None else EvidenceMemory(
            db_path,
            confidence_threshold=confidence_threshold,
        )
        base = budget or Budget()
        self.budget = Budget(
            tool_calls=int(base.tool_calls if tool_calls is None else tool_calls),
            replans=int(base.replans if replans is None else replans),
            recovery_rounds=int(base.recovery_rounds if recovery_rounds is None else recovery_rounds),
            tool_calls_used=int(base.tool_calls_used),
            replans_used=int(base.replans_used),
            recovery_rounds_used=int(base.recovery_rounds_used),
        )
        self.confidence_threshold = float(confidence_threshold)
        self.environment = norm(environment)
        self.task_id = str(task_id or "")
        self.task_type = norm(task_type)
        self.task_goal = str(task_goal or "")
        self.analyzer = FailureAnalyzer(confidence_threshold=self.confidence_threshold)
        self.pruner = HypothesisPruner(confidence_threshold=self.confidence_threshold)
        self.pruning = bool(pruning)
        self.shared_priority = bool(shared_priority)
        self.algorithm = str(algorithm or "v1")
        self.v2_mode = str(v2_mode or "full_cghe_v2")
        self.lambda_cost = float(lambda_cost)
        self.mu_risk = float(mu_risk)
        self.v2_trace: list[dict] = []
        self.v2_candidates: list = []
        self.v2_hypotheses: list = []
        self._v2_evidence = None
        self.graph: Optional[PlanningGraph] = None
        self.current_round: Optional[RecoveryRound] = None
        self._rounds_opened = int(self.budget.recovery_rounds_used)
        self._round_exhausted = False
        self._stopped = False
        self._stop_reason = ""
        self._logs: list[RecoveryLog] = []
        self._fingerprint_counts: dict[str, int] = {}
        self._replayed_fingerprints: set[str] = set()
        self._call_log: list[dict[str, object]] = []

    def _v2_log(self):
        if self.algorithm != "v2":
            return None
        if self._v2_evidence is None:
            from pathlib import Path

            from v2.evidence_log import V2EvidenceLog

            root = Path(__file__).resolve().parents[4]
            self._v2_evidence = V2EvidenceLog(str(root / "experiments" / "cghe_v2" / "evidence.sqlite"))
        return self._v2_evidence

    def _audit_v2_prune(self, analysis: FailureAnalysis, result: Any) -> None:
        log = self._v2_log()
        constraint = analysis.constraint
        if log is None or constraint is None:
            return
        log.record_prune(
            task_id=self.task_id,
            environment_id=constraint.scope.environment,
            round_id=self.graph.round_id if self.graph is not None else "",
            constraint_id=constraint.constraint_id,
            tool=constraint.scope.tool,
            resource=constraint.scope.resource,
            identity=constraint.scope.identity,
            parameter=constraint.scope.parameter,
            argument=constraint.scope.argument,
            active_before=result.active_before,
            active_after=result.active_after,
            pruned_plan_ids=list(result.pruned_plan_ids),
            evidence_source=result.evidence_source,
        )

    def _record_v2(self, observation: ExecutionObservation, analysis: FailureAnalysis) -> None:
        """Store v2 candidates and hypotheses. They do not change v1 admission."""
        from v2.constraint_inference import infer
        from v2.counterfactual import generate_hypotheses

        observation_id = analysis.evidence.evidence_id or "observation"
        self.v2_candidates = infer(observation, observation_id=observation_id)
        self.v2_hypotheses = generate_hypotheses(
            tool=observation.tool_name,
            argument=observation.command,
            failure_type=analysis.evidence.failure_type.value if analysis.evidence.failure_type else "",
            plans=self.graph.active_plan_ids() if self.graph is not None else (),
        )
        self.v2_trace.append({
            "event": "inference",
            "candidates": [item.evidence_status for item in self.v2_candidates],
            "hypotheses": [item.hypothesis_id for item in self.v2_hypotheses],
        })
        log = self._v2_log()
        if log is None:
            return
        log.record_observation(
            observation_id=observation_id,
            task_id=self.task_id,
            environment_id=observation.environment or self.environment,
            round_id=self.graph.round_id if self.graph is not None else "",
            tool=observation.tool_name,
            resource=observation.resource,
            identity=observation.identity,
            parameter=getattr(observation, "parameter", ""),
            argument=observation.command,
            source=analysis.evidence.source,
        )
        for item in self.v2_hypotheses:
            log.record_hypothesis(
                hypothesis_id=item.hypothesis_id,
                task_id=self.task_id,
                environment_id=self.environment,
                tool=item.scope.get("tool", ""),
                argument=item.scope.get("argument", ""),
                status=item.status,
            )

    @property
    def logs(self) -> tuple[RecoveryLog, ...]:
        return tuple(self._logs)

    def should_enter_recovery(self, **kwargs: object) -> TriggerDecision:
        threshold = kwargs.pop("confidence_threshold", self.confidence_threshold)
        return should_enter_recovery(confidence_threshold=float(threshold), **kwargs)  # type: ignore[arg-type]

    def planning_records(
        self,
        *,
        environment: str = "",
        context: Optional[Scope] = None,
        task_type: str = "",
    ) -> PlanningRecords:
        """Verified facts and counterexamples for the next universe, plus diagnostic ``m_cf``."""
        return self.memory.records_for_planning(
            environment=environment or self.environment,
            context=context if context is not None else Scope(environment=self.environment, task_type=self.task_type),
            task_type=task_type or self.task_type,
        )

    def record_verified_success(
        self,
        observation: Optional[ExecutionObservation] = None,
        *,
        action: str = "",
        subgoal: str = "",
        source: str = "verifier",
        tool: str = "",
        note: str = "",
    ) -> EvidenceRecord:
        """Write ``m+`` for a step the verifier accepted."""
        return self.memory.record_success(
            task_id=self.task_id,
            task_type=self.task_type,
            task_goal=self.task_goal or subgoal,
            environment=self.environment or (observation.environment if observation else ""),
            action=action or subgoal,
            tool=tool or (observation.tool_name if observation else ""),
            command=observation.command if observation else "",
            observation=observation,
            source=source,
            note=note,
            round_id=self.graph.round_id if self.graph is not None else "",
            plan_id=observation.plan_id if observation else "",
            subgoal=subgoal,
        )

    def handle_failure(
        self,
        observation: ExecutionObservation,
        *,
        plans: Optional[Sequence[CandidatePlan]] = None,
        verifier_label: str = "",
        verifier_satisfied: Optional[bool] = None,
        action: str = "",
    ) -> RecoveryOutcome:
        """Analyze one failure and either prune, ask for a universe, or stay on PEV.

        ``plans`` is used only when this call has to open a round. Passing plans
        while a round is already open does not add candidates.
        """
        label = verifier_label or observation.verifier_label
        analysis = self.analyzer.analyze(observation)
        if self.algorithm == "v2":
            self._record_v2(observation, analysis)
        trigger = should_enter_recovery(
            verifier_label=label,
            verifier_satisfied=verifier_satisfied,
            analysis=analysis,
            confidence_threshold=self.confidence_threshold,
        )
        self._remember_failure(analysis, observation, action=action, promote=trigger.intervene)
        if not trigger.intervene:
            return self._continue_pev(trigger.reason, analysis)
        if self._stopped:
            return self._stopped_outcome()
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            return self._stop(STOP_TOOL_CALLS, analysis=analysis)

        environment = norm(observation.environment or self.environment)
        if self.environment and environment and self.environment != environment:
            return self._continue_pev(
                "constraint environment does not match the current environment",
                analysis,
            )
        if not self.environment and environment:
            self.environment = environment
        context = Scope(
            environment=environment,
            tool=observation.tool_name,
            resource=observation.resource,
            identity=observation.identity,
            task_type=self.task_type,
        )
        if not self.memory.constraint_applies(
            analysis.constraint,
            environment=environment,
            context=context,
            task_type=self.task_type,
            record_task_type=self.task_type,
        ):
            return self._continue_pev(
                "no verified constraint applies to the current environment",
                analysis,
            )

        if self.graph is not None and not self._round_exhausted:
            return self._prune_current(analysis, observation)
        if not plans:
            is_replan = self._rounds_opened > 0
            if not self._can_open_round(is_replan=is_replan):
                return self._stop(self._block_reason(is_replan=is_replan))
            return self._outcome(
                decision=RecoveryDecision.NEED_UNIVERSE,
                reason="verified constraint is ready for a fixed candidate universe",
                intervene=True,
                analysis=analysis,
                message=NEEDS_REPLAN_MESSAGE if is_replan else "",
            )
        opened = self.begin_round(plans)
        if opened.analysis is None:
            opened.analysis = analysis
        return opened

    def begin_round(
        self,
        plans: Sequence[CandidatePlan],
        *,
        round_id: str = "",
    ) -> RecoveryOutcome:
        """Open one frozen universe and apply verified memory before selection.

        The first call spends a recovery round. A later call, after the
        previous universe was exhausted, also spends a replan.
        """
        if self._stopped:
            return self._stopped_outcome()
        if self.graph is not None and not self._round_exhausted:
            raise UniverseFrozenError(
                f"round {self.graph.round_id} is frozen; exhaust it before opening another"
            )
        if not plans:
            raise ValueError("a recovery round requires at least one CandidatePlan")
        is_replan = self._rounds_opened > 0
        if not self._can_open_round(is_replan=is_replan):
            return self._stop(self._block_reason(is_replan=is_replan))
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            return self._stop(STOP_TOOL_CALLS)

        graph = PlanningGraph(plans, round_id=round_id or new_id("round"))
        self.budget.recovery_rounds_used += 1
        if is_replan:
            self.budget.replans_used += 1
        self._rounds_opened += 1
        self.graph = graph
        self._round_exhausted = False
        inherited = ()
        if self.pruning:
            inherited = self.memory.constraints_for_pruning(
                environment=self.environment,
                context=Scope(environment=self.environment, task_type=self.task_type),
                task_type=self.task_type,
            )
        inherited_ids = tuple(item.constraint_id for item in inherited)
        self.current_round = RecoveryRound(
            round_id=graph.round_id,
            candidate_plan_ids=graph.candidate_plan_ids,
            inherited_constraint_ids=inherited_ids,
        )
        call_logs: list[RecoveryLog] = []
        last: Optional[PruneResult] = None
        for constraint in inherited:
            last = self.pruner.prune(graph, constraint, apply=True)
            entry = self._log_prune(last, decision=RecoveryDecision.CONTINUE_EXECUTION)
            call_logs.append(entry)
            if last.needs_replan:
                return self._exhaust(last, call_logs, inherited_ids)
        removed_any = any(entry.active_before > entry.active_after for entry in call_logs)
        if self.pruning and not removed_any and not is_replan:
            self._abandon_open(consumed_replan=False)
            return self._continue_pev(
                "verified constraint did not remove a candidate",
                None,
                extra_logs=call_logs,
                inherited_ids=inherited_ids,
            )
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            return self._stop(STOP_TOOL_CALLS, extra_logs=call_logs, inherited_ids=inherited_ids)
        node = self._select_node()
        if node is None and not graph.active_plan_ids():
            return self._exhaust(last, call_logs, inherited_ids)
        if node is None:
            return self._continue_pev(
                "no executable shared node remains",
                analysis=None,
                extra_logs=call_logs,
                inherited_ids=inherited_ids,
            )
        return self._outcome(
            decision=RecoveryDecision.CONTINUE_EXECUTION,
            reason="shared node selected from the active universe",
            intervene=True,
            selected_node=node,
            prune_result=last,
            extra_logs=call_logs,
            inherited_ids=inherited_ids,
        )

    def authorize_tool_call(
        self,
        tool: str,
        *,
        command: str = "",
        parameters: Optional[Mapping[str, object]] = None,
        environment: str = "",
        replay: bool = False,
    ) -> ToolAuthorization:
        """Decide whether this call may run. Does not spend budget until commit."""
        policy = retry_policy_for(tool)
        fingerprint = call_fingerprint(
            tool,
            command=command,
            parameters=parameters,
            environment=environment or self.environment,
        )
        remaining = self.budget.tool_calls - self.budget.tool_calls_used
        used = self._fingerprint_counts.get(fingerprint, 0)
        cap = _SAFE_FINGERPRINT_CAP if policy.safe_to_retry else _UNSAFE_FINGERPRINT_CAP

        def deny(reason: str) -> ToolAuthorization:
            return ToolAuthorization(
                allowed=False,
                replay=replay,
                safe_to_retry=policy.safe_to_retry,
                fingerprint=fingerprint,
                reason=reason,
                tool_name=policy.tool_name,
                budget_remaining=max(remaining, 0),
            )

        if remaining <= 0:
            return deny(STOP_TOOL_CALLS)
        if replay and not policy.safe_to_retry:
            return deny(policy.reason)
        if replay and used <= 0:
            return deny("no prior call with this fingerprint to replay")
        if replay and fingerprint in self._replayed_fingerprints:
            return deny("same fingerprint must not be repeated again in this recovery")
        if used >= cap:
            return deny("same fingerprint must not be repeated again in this recovery")
        return ToolAuthorization(
            allowed=True,
            replay=replay,
            safe_to_retry=policy.safe_to_retry,
            fingerprint=fingerprint,
            reason=policy.reason if replay else "call authorized",
            tool_name=policy.tool_name,
            budget_remaining=remaining,
        )

    def may_auto_replay(
        self,
        tool: str,
        *,
        command: str = "",
        parameters: Optional[Mapping[str, object]] = None,
        environment: str = "",
    ) -> ToolAuthorization:
        """Replay only a call whose tool policy says it is safe."""
        return self.authorize_tool_call(
            tool,
            command=command,
            parameters=parameters,
            environment=environment,
            replay=True,
        )

    def commit_tool_call(self, authorization: ToolAuthorization) -> Budget:
        """Record a call that actually ran. Refused authorizations are not charged."""
        if not authorization.allowed:
            raise RuntimeError(authorization.reason)
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            raise RuntimeError(STOP_TOOL_CALLS)
        self.budget.tool_calls_used += 1
        count = self._fingerprint_counts.get(authorization.fingerprint, 0) + 1
        self._fingerprint_counts[authorization.fingerprint] = count
        if authorization.replay:
            self._replayed_fingerprints.add(authorization.fingerprint)
        self._call_log.append({
            "tool_name": authorization.tool_name,
            "fingerprint": authorization.fingerprint,
            "replay": authorization.replay,
            "safe_to_retry": authorization.safe_to_retry,
        })
        return self._budget_snapshot()

    def export_state(self) -> dict[str, object]:
        """Plain recovery state. The SQLite connection is not part of it."""
        plans = None
        node_status: dict[str, str] = {}
        round_id = ""
        if self.graph is not None:
            plans = [to_plain(plan) for plan in self.graph.plans()]
            node_status = {node.node_id: node.status.value for node in self.graph.nodes()}
            round_id = self.graph.round_id
        return {
            "db_path": self.memory.db_path,
            "budget": to_plain(self.budget),
            "confidence_threshold": self.confidence_threshold,
            "pruning": self.pruning,
            "shared_priority": self.shared_priority,
            "environment": self.environment,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_goal": self.task_goal,
            "round_id": round_id,
            "rounds_opened": self._rounds_opened,
            "round_exhausted": self._round_exhausted,
            "stopped": self._stopped,
            "stop_reason": self._stop_reason,
            "fingerprints": dict(self._fingerprint_counts),
            "replayed_fingerprints": sorted(self._replayed_fingerprints),
            "call_log": list(self._call_log),
            "logs": [to_plain(entry) for entry in self._logs],
            "graph_plans": plans,
            "node_status": node_status,
            "inherited_constraint_ids": (
                list(self.current_round.inherited_constraint_ids) if self.current_round else []
            ),
        }

    def restore_state(self, state: Mapping[str, object]) -> None:
        """Rebuild the controller from ``export_state``. The database is reopened by path."""
        self.memory.close()
        self.memory = EvidenceMemory(
            str(state.get("db_path") or self.memory.db_path),
            confidence_threshold=float(state.get("confidence_threshold") or self.confidence_threshold),
        )
        raw_budget = dict(state.get("budget") or {})
        self.budget = Budget(
            tool_calls=int(raw_budget.get("tool_calls", self.budget.tool_calls)),
            replans=int(raw_budget.get("replans", self.budget.replans)),
            recovery_rounds=int(raw_budget.get("recovery_rounds", self.budget.recovery_rounds)),
            tool_calls_used=int(raw_budget.get("tool_calls_used", 0)),
            replans_used=int(raw_budget.get("replans_used", 0)),
            recovery_rounds_used=int(raw_budget.get("recovery_rounds_used", 0)),
        )
        self.environment = norm(str(state.get("environment") or ""))
        self.task_id = str(state.get("task_id") or "")
        self.task_type = norm(str(state.get("task_type") or ""))
        self.task_goal = str(state.get("task_goal") or "")
        if "pruning" in state:
            self.pruning = bool(state.get("pruning"))
        if "shared_priority" in state:
            self.shared_priority = bool(state.get("shared_priority"))
        self._rounds_opened = int(state.get("rounds_opened") or self.budget.recovery_rounds_used)
        self._round_exhausted = bool(state.get("round_exhausted"))
        self._stopped = bool(state.get("stopped"))
        self._stop_reason = str(state.get("stop_reason") or "")
        self._fingerprint_counts = {str(key): int(value) for key, value in dict(state.get("fingerprints") or {}).items()}
        self._replayed_fingerprints = {str(item) for item in (state.get("replayed_fingerprints") or ())}
        self._call_log = [dict(item) for item in (state.get("call_log") or ())]
        self._logs = [_log_from_plain(item) for item in (state.get("logs") or ())]
        raw_plans = state.get("graph_plans")
        if raw_plans:
            plans = tuple(candidate_plan_from_dict(item) for item in raw_plans)
            self.graph = PlanningGraph(plans, round_id=str(state.get("round_id") or ""))
            for node_id, status in dict(state.get("node_status") or {}).items():
                if status == "satisfied":
                    self.graph.mark_satisfied(str(node_id))
                elif status == "failed":
                    self.graph.mark_failed(str(node_id))
            self.current_round = RecoveryRound(
                round_id=self.graph.round_id,
                candidate_plan_ids=self.graph.candidate_plan_ids,
                inherited_constraint_ids=tuple(state.get("inherited_constraint_ids") or ()),
            )
        else:
            self.graph = None
            self.current_round = None

    def _remember_failure(
        self,
        analysis: FailureAnalysis,
        observation: ExecutionObservation,
        *,
        action: str,
        promote: bool,
    ) -> EvidenceRecord:
        """Persist the failure. Only an admitted recovery writes ``m-``."""
        shared = dict(
            task_id=self.task_id,
            task_type=self.task_type,
            task_goal=self.task_goal,
            environment=observation.environment or self.environment,
            action=action or observation.command or observation.tool_name,
            command=observation.command,
            observation=observation,
            round_id=self.graph.round_id if self.graph is not None else "",
        )
        if promote:
            return self.memory.record_analysis(analysis, **shared)
        return self.memory.record_hypothesis(
            constraint=analysis.constraint,
            source=analysis.evidence.source,
            origin=analysis.evidence.origin,
            note=analysis.evidence.note,
            tool=analysis.evidence.tool,
            plan_id=analysis.evidence.plan_id,
            record_id=analysis.evidence.evidence_id,
            **shared,
        )

    def _prune_current(
        self,
        analysis: FailureAnalysis,
        observation: ExecutionObservation,
    ) -> RecoveryOutcome:
        assert self.graph is not None
        if not self.pruning:
            return self._advance_without_pruning(analysis, observation)
        assert analysis.constraint is not None
        result = self.pruner.prune(
            self.graph,
            analysis.constraint,
            evidence=analysis.evidence,
            executed_plan_id=observation.plan_id,
            apply=True,
        )
        self._audit_v2_prune(analysis, result)
        if observation.node_id and result.affected_plan_ids:
            self._mark_failed(observation.node_id)
        if not result.affected_plan_ids:
            entry = self._log_prune(result, decision=RecoveryDecision.CONTINUE_PEV)
            return self._continue_pev(
                result.reason or "verified constraint did not remove a candidate",
                analysis,
                prune_result=result,
                extra_logs=(entry,),
            )
        entry = self._log_prune(
            result,
            decision=RecoveryDecision.NEED_UNIVERSE if result.needs_replan else RecoveryDecision.CONTINUE_EXECUTION,
        )
        if result.needs_replan:
            return self._exhaust(result, (entry,), self.current_round.inherited_constraint_ids if self.current_round else (), analysis)
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            return self._stop(STOP_TOOL_CALLS, analysis=analysis, extra_logs=(entry,))
        node = self._select_node()
        if node is None:
            return self._continue_pev(
                "no executable shared node remains",
                analysis,
                prune_result=result,
                extra_logs=(entry,),
            )
        return self._outcome(
            decision=RecoveryDecision.CONTINUE_EXECUTION,
            reason="shared node selected after admissible elimination",
            intervene=True,
            analysis=analysis,
            selected_node=node,
            prune_result=result,
            extra_logs=(entry,),
        )

    def _select_node(self) -> Optional[GraphNode]:
        if self.graph is None:
            return None
        if self.algorithm == "v2":
            from v2.action_selector import choose_node

            chosen, trace = choose_node(
                self.graph,
                mode=self.v2_mode,
                lambda_cost=self.lambda_cost,
                mu_risk=self.mu_risk,
            )
            self.v2_trace.append(trace)
            log = self._v2_log()
            if log is not None:
                for row in trace.get("scores") or []:
                    log.record_feedback(
                        task_id=self.task_id,
                        round_id=self.graph.round_id,
                        action_id=row.get("action_id", ""),
                        epv=row.get("epv"),
                        cost=row.get("cost"),
                        risk=row.get("risk"),
                        score=row.get("score"),
                    )
            if chosen is not None:
                return chosen
        policy = "frontier" if self.shared_priority else "first_plan"
        return self.graph.select_node(policy=policy)

    def _advance_without_pruning(
        self,
        analysis: FailureAnalysis,
        observation: ExecutionObservation,
    ) -> RecoveryOutcome:
        """Drop only the plan that just failed, then take the next ordered node."""
        assert self.graph is not None
        if observation.plan_id:
            self.graph.retire_plans((observation.plan_id,))
        if observation.node_id:
            self._mark_failed(observation.node_id)
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            return self._stop(STOP_TOOL_CALLS, analysis=analysis)
        node = self._select_node()
        inherited = self.current_round.inherited_constraint_ids if self.current_round else ()
        if node is None and not self.graph.active_plan_ids():
            return self._exhaust(None, (), inherited, analysis)
        if node is None:
            return self._continue_pev(
                "no executable shared node remains",
                analysis,
                inherited_ids=inherited,
            )
        return self._outcome(
            decision=RecoveryDecision.CONTINUE_EXECUTION,
            reason="next plan selected without cross-plan pruning",
            intervene=True,
            analysis=analysis,
            selected_node=node,
            inherited_ids=inherited,
        )

    def _mark_failed(self, node_id: str) -> None:
        if self.graph is None or not node_id:
            return
        if any(node.node_id == node_id for node in self.graph.nodes()):
            self.graph.mark_failed(node_id)

    def _exhaust(
        self,
        result: Optional[PruneResult],
        call_logs: Sequence[RecoveryLog],
        inherited_ids: Sequence[str],
        analysis: Optional[FailureAnalysis] = None,
    ) -> RecoveryOutcome:
        self._round_exhausted = True
        if not self._can_open_round(is_replan=True):
            return self._stop(
                self._block_reason(is_replan=True),
                analysis=analysis,
                prune_result=result,
                extra_logs=call_logs,
                inherited_ids=inherited_ids,
            )
        return self._outcome(
            decision=RecoveryDecision.NEED_UNIVERSE,
            reason="current candidate universe is exhausted",
            intervene=True,
            analysis=analysis,
            prune_result=result,
            message=NEEDS_REPLAN_MESSAGE,
            extra_logs=call_logs,
            inherited_ids=inherited_ids,
        )

    def _continue_pev(
        self,
        reason: str,
        analysis: Optional[FailureAnalysis],
        *,
        prune_result: Optional[PruneResult] = None,
        extra_logs: Sequence[RecoveryLog] = (),
        inherited_ids: Sequence[str] = (),
    ) -> RecoveryOutcome:
        return self._outcome(
            decision=RecoveryDecision.CONTINUE_PEV,
            reason=reason,
            intervene=False,
            analysis=analysis,
            prune_result=prune_result,
            message=CONTINUE_PEV_MESSAGE,
            extra_logs=extra_logs,
            inherited_ids=inherited_ids,
        )

    def _stop(
        self,
        reason: str,
        *,
        analysis: Optional[FailureAnalysis] = None,
        prune_result: Optional[PruneResult] = None,
        extra_logs: Sequence[RecoveryLog] = (),
        inherited_ids: Sequence[str] = (),
    ) -> RecoveryOutcome:
        self._stopped = True
        self._stop_reason = reason
        return self._outcome(
            decision=RecoveryDecision.STOP_BUDGET,
            reason=reason,
            intervene=True,
            analysis=analysis,
            prune_result=prune_result,
            message=reason,
            extra_logs=extra_logs,
            inherited_ids=inherited_ids,
        )

    def _stopped_outcome(self) -> RecoveryOutcome:
        return self._outcome(
            decision=RecoveryDecision.STOP_BUDGET,
            reason=self._stop_reason or STOP_TOOL_CALLS,
            intervene=True,
            message=self._stop_reason or STOP_TOOL_CALLS,
        )

    def _outcome(
        self,
        *,
        decision: str,
        reason: str,
        intervene: bool,
        analysis: Optional[FailureAnalysis] = None,
        selected_node: Optional[GraphNode] = None,
        prune_result: Optional[PruneResult] = None,
        message: str = "",
        extra_logs: Sequence[RecoveryLog] = (),
        inherited_ids: Sequence[str] = (),
    ) -> RecoveryOutcome:
        if not extra_logs and decision == RecoveryDecision.CONTINUE_PEV:
            active = len(self.graph.active_plan_ids()) if self.graph is not None and not self._round_exhausted else 0
            source = ""
            if analysis is not None and analysis.constraint is not None:
                source = analysis.constraint.source
            elif analysis is not None:
                source = analysis.evidence.source
            extra_logs = (RecoveryLog(
                active_before=active,
                active_after=active,
                reason=reason,
                reason_code="not_entered",
                evidence_source=source,
                round_id=self.graph.round_id if self.graph is not None else "",
                decision=decision,
            ),)
            self._logs.append(extra_logs[0])
        return RecoveryOutcome(
            decision=decision,
            reason=reason,
            intervene=intervene,
            round_id=self.graph.round_id if self.graph is not None else "",
            selected_node=selected_node,
            prune_result=prune_result,
            analysis=analysis,
            budget=self._budget_snapshot(),
            logs=tuple(extra_logs),
            message=message,
            inherited_constraint_ids=tuple(inherited_ids),
            recovery_round=self.current_round,
            task_impossible=False,
        )

    def _log_prune(self, result: PruneResult, *, decision: str) -> RecoveryLog:
        entry = RecoveryLog(
            active_before=result.active_before,
            active_after=result.active_after,
            reason=result.reason,
            reason_code=result.reason_code,
            evidence_source=result.evidence_source,
            round_id=result.round_id,
            constraint_id=result.constraint_id,
            decision=decision,
            admissible=result.admissible,
        )
        self._logs.append(entry)
        return entry

    def _abandon_open(self, *, consumed_replan: bool) -> None:
        """Drop a round that did not eliminate any candidate and refund its budget."""
        self.graph = None
        self.current_round = None
        self._round_exhausted = False
        self.budget.recovery_rounds_used = max(0, self.budget.recovery_rounds_used - 1)
        self._rounds_opened = max(0, self._rounds_opened - 1)
        if consumed_replan:
            self.budget.replans_used = max(0, self.budget.replans_used - 1)

    def _can_open_round(self, *, is_replan: bool) -> bool:
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            return False
        if self.budget.recovery_rounds_used >= self.budget.recovery_rounds:
            return False
        if is_replan and self.budget.replans_used >= self.budget.replans:
            return False
        return True

    def _block_reason(self, *, is_replan: bool) -> str:
        if self.budget.tool_calls_used >= self.budget.tool_calls:
            return STOP_TOOL_CALLS
        if is_replan and self.budget.replans_used >= self.budget.replans:
            return STOP_REPLANS
        if self.budget.recovery_rounds_used >= self.budget.recovery_rounds:
            return STOP_RECOVERY_ROUNDS
        return STOP_RECOVERY_ROUNDS

    def _budget_snapshot(self) -> Budget:
        return replace(self.budget)


def _normalize_parameters(parameters: Optional[Mapping[str, object]]) -> list[list[str]]:
    if not parameters:
        return []
    items: list[list[str]] = []
    for key in sorted(parameters, key=lambda item: norm(item)):
        value = parameters[key]
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            rendered = str(value).strip()
        items.append([norm(key), rendered])
    return items


def _log_from_plain(raw: object) -> RecoveryLog:
    data = dict(raw or {})
    return RecoveryLog(
        active_before=int(data.get("active_before") or 0),
        active_after=int(data.get("active_after") or 0),
        reason=str(data.get("reason") or ""),
        reason_code=str(data.get("reason_code") or ""),
        evidence_source=str(data.get("evidence_source") or ""),
        round_id=str(data.get("round_id") or ""),
        constraint_id=str(data.get("constraint_id") or ""),
        decision=str(data.get("decision") or ""),
        admissible=bool(data.get("admissible")),
    )
