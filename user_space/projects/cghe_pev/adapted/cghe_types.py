"""Structured objects for one CGHE recovery round.

Sections 1–4 of the CGHE note define the objects. Public module names used by
later stages are ``PlanningGraph``, ``FailureAnalyzer`` and ``HypothesisPruner``.
This module only holds data. It does not schedule replanning or store evidence.

Two different "direct" bits show up on a constraint:

- ``origin`` distinguishes a real tool return or execution log from an LLM guess.
- ``direct`` is the paper's ``direct(c)`` bit. It is 1 only when a fixed
  observation→predicate row is fully bound. Timeouts, empty returns, generic
  exceptions and guesses stay at 0 even if they were really observed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence
from uuid import uuid4


DEFAULT_CONFIDENCE_THRESHOLD = 1.0
NEEDS_REPLAN_MESSAGE = "需要重新规划"

# Fixed Ψ rows. Keys are observation codes, values are premise predicates.
PSI_OBSERVATION_TO_PREDICATE = {
    "TOOL_NOT_FOUND": "ToolExists",
    "PERMISSION_DENIED": "CanAccess",
    "SCHEMA_MISMATCH": "SchemaOk",
    "PARAM_ERROR": "ValidID",
    "SNIPPETS_INSUFFICIENT": "SnippetsSufficient",
}

# Definition 12 removes only the executed plan when a failure is bound to this
# call/argument. Snippet gaps are query-instance local: a shared Step-1 search
# must not wipe sibling plans that use different queries. Tool absence and
# permission denials are not instance-local.
INSTANCE_BOUND_OBSERVATIONS = frozenset({
    "PARAM_ERROR",
    "SCHEMA_MISMATCH",
    "SNIPPETS_INSUFFICIENT",
})

_PREDICATE_ALIASES = {
    "toolexists": "ToolExists",
    "canaccess": "CanAccess",
    "schemaok": "SchemaOk",
    "validid": "ValidID",
    "snippetssufficient": "SnippetsSufficient",
}

_SPACE_RE = re.compile(r"\s+")


class FailureType(str, Enum):
    TOOL_UNAVAILABLE = "TOOL_UNAVAILABLE"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    MISSING_INFORMATION = "MISSING_INFORMATION"
    SUBGOAL_NOT_SATISFIED = "SUBGOAL_NOT_SATISFIED"
    TRANSIENT_ERROR = "TRANSIENT_ERROR"
    VERIFIER_UNCERTAIN = "VERIFIER_UNCERTAIN"
    PARAMETER_ERROR = "PARAMETER_ERROR"
    GENERIC_FAILURE = "GENERIC_FAILURE"
    SNIPPETS_INSUFFICIENT = "SNIPPETS_INSUFFICIENT"


class EvidenceStatus(str, Enum):
    VERIFIED_SUCCESS = "VERIFIED_SUCCESS"
    VERIFIED_COUNTEREXAMPLE = "VERIFIED_COUNTEREXAMPLE"
    UNVERIFIED_HYPOTHESIS = "UNVERIFIED_HYPOTHESIS"


class EvidenceOrigin(str, Enum):
    DIRECT_OBSERVATION = "direct_observation"
    INFERENCE = "inference"


class NodeStatus(str, Enum):
    PENDING = "pending"
    SATISFIED = "satisfied"
    FAILED = "failed"
    ORPHANED = "orphaned"


class PlanStatus(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PRUNED = "pruned"
    ABANDONED = "abandoned"
    # Soft-Bing chain reached a leaf (or soft-advance budget) without satisfaction.
    # Dropped from SELECT coverage without a hard SnippetsSufficient prune.
    SOFT_EXHAUSTED = "soft_exhausted"
    # Legacy logs only. New CGHE writes use PRUNED or ABANDONED.
    RETIRED = "retired"


class CausalRelation(str, Enum):
    REQUIRES = "REQUIRES"
    ENABLES = "ENABLES"
    OBSERVED_TO_CAUSE = "OBSERVED_TO_CAUSE"
    INVALIDATES = "INVALIDATES"
    ELIMINATES = "ELIMINATES"
    SUPPORTS = "SUPPORTS"


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def norm(value: Any) -> str:
    """Case-fold and collapse whitespace. Empty values stay empty."""
    text = str(value or "").strip().casefold()
    return _SPACE_RE.sub(" ", text)


def canonical_predicate(name: Any) -> str:
    key = norm(name).replace("_", "").replace(" ", "")
    if not key:
        return ""
    return _PREDICATE_ALIASES.get(key, str(name).strip())


def canonical_action(action: Any, tool: Any = "") -> tuple[str, str]:
    """Return ``(action, tool)`` used by the planning-graph merge key.

    A step that only names the tool and a step that only names the action
    refer to the same operation when the normalized strings are equal.
    Distinct action and tool strings stay distinct and do not merge.
    """
    action_key = norm(action)
    tool_key = norm(tool)
    if action_key and tool_key and action_key != tool_key:
        return action_key, tool_key
    name = tool_key or action_key
    return name, name


@dataclass(frozen=True)
class Scope:
    """Applicability domain ``D`` or a plan context.

    Empty fields do not constrain a match. A parameter-level scope must name
    ``parameter`` or ``argument`` and must not be read as "this tool is gone".
    """

    environment: str = ""
    tool: str = ""
    resource: str = ""
    identity: str = ""
    parameter: str = ""
    argument: str = ""
    schema_signature: str = ""
    task_type: str = ""
    call_instance: str = ""

    def normalized(self) -> "Scope":
        return Scope(
            environment=norm(self.environment),
            tool=norm(self.tool),
            resource=norm(self.resource),
            identity=norm(self.identity),
            parameter=norm(self.parameter),
            argument=norm(self.argument),
            schema_signature=norm(self.schema_signature),
            task_type=norm(self.task_type),
            call_instance=norm(self.call_instance),
        )

    def parameter_level(self) -> bool:
        scoped = self.normalized()
        return bool(scoped.parameter or scoped.argument)


@dataclass(frozen=True)
class AtomicPremise:
    """One observable atom that may appear in ``Dep(v)``."""

    predicate: str
    tool: str = ""
    resource: str = ""
    identity: str = ""
    parameter: str = ""
    argument: str = ""
    schema_signature: str = ""

    def normalized(self) -> "AtomicPremise":
        return AtomicPremise(
            predicate=canonical_predicate(self.predicate),
            tool=norm(self.tool),
            resource=norm(self.resource),
            identity=norm(self.identity),
            parameter=norm(self.parameter),
            argument=norm(self.argument),
            schema_signature=norm(self.schema_signature),
        )

    def key(self) -> str:
        item = self.normalized()
        return "|".join(
            (
                item.predicate,
                item.tool,
                item.resource,
                item.identity,
                item.parameter,
                item.argument,
                item.schema_signature,
            )
        )


def context_contained(context: Scope, domain: Scope) -> bool:
    """``MatchCertain``: every field set on ``domain`` equals ``context``.

    An empty context field against a set domain field is uncertain, so the
    path is not contained in ``D``.
    """
    ctx = context.normalized()
    dom = domain.normalized()
    for name in (
        "environment",
        "tool",
        "resource",
        "identity",
        "parameter",
        "argument",
        "schema_signature",
        "task_type",
        "call_instance",
    ):
        required = getattr(dom, name)
        if not required:
            continue
        actual = getattr(ctx, name)
        if not actual or actual != required:
            return False
    return True


def normalize_state(items: Optional[Sequence[Any]]) -> tuple[str, ...]:
    found = {norm(item) for item in (items or ())}
    found.discard("")
    return tuple(sorted(found))


def schema_signature(parameters: Optional[Mapping[str, Any]]) -> str:
    """Canonical call shape. The tool name is intentionally not included."""
    if not parameters:
        return ""
    parts = []
    for key in sorted(parameters):
        value = parameters[key]
        parts.append(f"{norm(key)}:{type(value).__name__.casefold()}")
    return ",".join(parts)


@dataclass
class PlanStep:
    """One already-structured step of a candidate plan."""

    subgoal: str
    action: str = ""
    tool: str = ""
    preconditions: tuple[AtomicPremise, ...] = ()
    required_state: tuple[str, ...] = ()
    estimated_cost: float = 1.0
    step_id: str = ""
    predecessor_ids: tuple[str, ...] = ()
    links_previous: bool = True
    resource: str = ""
    identity: str = ""
    parameter: str = ""
    argument: str = ""
    schema_signature: str = ""
    task_id: str = ""
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    instance_key: str = ""

    def __post_init__(self) -> None:
        self.preconditions = tuple(self.preconditions or ())
        self.required_state = normalize_state(self.required_state)
        self.predecessor_ids = tuple(self.predecessor_ids or ())
        self.consumes = normalize_state(self.consumes)
        self.produces = normalize_state(self.produces)
        self.estimated_cost = float(self.estimated_cost)


@dataclass
class CandidatePlan:
    """One path in the fixed candidate universe of a recovery round."""

    plan_id: str
    steps: tuple[PlanStep, ...] = ()
    status: PlanStatus = PlanStatus.ACTIVE
    context: Scope = field(default_factory=Scope)
    prune_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = PlanStatus(self.status)
        self.steps = tuple(self.steps or ())
        if not isinstance(self.context, Scope):
            self.context = Scope(**dict(self.context or {}))

    def explicit_premises(self) -> tuple[AtomicPremise, ...]:
        found: list[AtomicPremise] = []
        seen = set()
        for step in self.steps:
            for premise in step.preconditions:
                item = premise.normalized()
                if item.key() in seen:
                    continue
                seen.add(item.key())
                found.append(item)
        return tuple(found)


@dataclass
class GraphNode:
    """Shared DAG node. ``candidate_plan_ids`` stays structural after pruning.

    Active sharing is computed by ``PlanningGraph`` from plan status. Pruned
    plans remain listed here and stop counting toward the score.
    """

    node_id: str
    subgoal: str
    action: str
    tool: str
    preconditions: tuple[AtomicPremise, ...]
    dependencies: tuple[str, ...]
    candidate_plan_ids: tuple[str, ...]
    status: NodeStatus = NodeStatus.PENDING
    estimated_cost: float = 1.0
    required_state: tuple[str, ...] = ()
    resource: str = ""
    identity: str = ""
    parameter: str = ""
    argument: str = ""
    schema_signature: str = ""
    task_id: str = ""
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    instance_key: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.status, str):
            self.status = NodeStatus(self.status)
        self.preconditions = tuple(self.preconditions or ())
        self.dependencies = tuple(self.dependencies or ())
        self.candidate_plan_ids = tuple(self.candidate_plan_ids or ())
        self.required_state = normalize_state(self.required_state)
        self.consumes = normalize_state(self.consumes)
        self.produces = normalize_state(self.produces)
        self.estimated_cost = float(self.estimated_cost)

    def has_premise(self, premise: AtomicPremise) -> bool:
        target = premise.normalized().key()
        return any(item.normalized().key() == target for item in self.preconditions)


@dataclass
class AttemptRecord:
    """One real Step-1 Executor invocation under a single DAG."""

    attempt_id: str
    dag_id: str
    step1_node_id: str
    selected_plan_ids: tuple[str, ...] = ()
    coverage_at_selection: int = 0
    execution_status: str = "started"
    verifier_status: Optional[str] = None
    evidence_id: Optional[str] = None
    pruning_applied: bool = False
    pruned_plan_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.selected_plan_ids = tuple(self.selected_plan_ids or ())
        self.pruned_plan_ids = tuple(self.pruned_plan_ids or ())
        self.coverage_at_selection = int(self.coverage_at_selection)


@dataclass
class CausalEdge:
    source_id: str
    target_id: str
    relation_type: str
    evidence_id: str = ""
    scope: Optional[Scope] = None
    direct: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.relation_type, CausalRelation):
            self.relation_type = self.relation_type.value
        if isinstance(self.scope, Mapping):
            self.scope = Scope(**dict(self.scope))


@dataclass
class FailureEvidence:
    """One execution failure. ``direct`` is the counterexample bit, not merely "we saw text"."""

    evidence_id: str
    is_failure: bool
    failure_type: Optional[FailureType] = None
    observation_code: str = ""
    tool: str = ""
    resource: str = ""
    identity: str = ""
    parameter: str = ""
    argument: str = ""
    schema_signature: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    raw_tool_result: Any = None
    execution_log: str = ""
    llm_hypothesis: str = ""
    origin: EvidenceOrigin = EvidenceOrigin.DIRECT_OBSERVATION
    direct: bool = False
    bound_to_call: bool = False
    source: str = ""
    plan_id: str = ""
    node_id: str = ""
    call_instance: str = ""
    environment: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.failure_type, str):
            self.failure_type = FailureType(self.failure_type)
        if isinstance(self.origin, str):
            self.origin = EvidenceOrigin(self.origin)
        self.parameters = dict(self.parameters or {})


@dataclass
class Constraint:
    """A local execution constraint. Only a verified direct counterexample can prune."""

    constraint_id: str
    condition: AtomicPremise
    scope: Scope
    evidence_status: EvidenceStatus
    source: str
    confidence: float
    direct: bool = False
    origin: EvidenceOrigin = EvidenceOrigin.INFERENCE
    observation_code: str = ""
    failure_type: Optional[FailureType] = None
    parameter_level: bool = False
    disables_tool: bool = False
    evidence_id: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.condition, Mapping):
            self.condition = AtomicPremise(**dict(self.condition))
        self.condition = self.condition.normalized()
        if isinstance(self.scope, Mapping):
            self.scope = Scope(**dict(self.scope))
        self.scope = self.scope.normalized()
        if isinstance(self.evidence_status, str):
            self.evidence_status = EvidenceStatus(self.evidence_status)
        if isinstance(self.origin, str):
            self.origin = EvidenceOrigin(self.origin)
        if isinstance(self.failure_type, str):
            self.failure_type = FailureType(self.failure_type)
        self.confidence = float(self.confidence)
        if self.parameter_level:
            self.disables_tool = False


@dataclass
class PruneRecord:
    record_id: str
    active_before: int
    active_after: int
    reason: str
    reason_code: str
    evidence_source: str
    affected_plan_ids: tuple[str, ...]
    constraint_id: str = ""
    admissible: bool = False
    needs_replan: bool = False
    evidence_status: str = ""
    observation_code: str = ""

    def __post_init__(self) -> None:
        self.affected_plan_ids = tuple(self.affected_plan_ids or ())
        self.active_before = int(self.active_before)
        self.active_after = int(self.active_after)


@dataclass
class PruneResult:
    """Outcome of one elimination attempt.

    ``task_impossible`` is always false. An empty active set means the current
    universe is exhausted and a later controller should open another round.
    """

    active_before: int
    active_after: int
    pruned_plan_ids: tuple[str, ...]
    affected_plan_ids: tuple[str, ...]
    reason: str
    reason_code: str
    evidence_source: str
    needs_replan: bool
    admissible: bool
    dep_complete: bool
    message: str = ""
    instance_retired_plan_ids: tuple[str, ...] = ()
    records: tuple[PruneRecord, ...] = ()
    round_id: str = ""
    constraint_id: str = ""
    task_impossible: bool = False

    def __post_init__(self) -> None:
        self.pruned_plan_ids = tuple(self.pruned_plan_ids or ())
        self.affected_plan_ids = tuple(self.affected_plan_ids or ())
        self.instance_retired_plan_ids = tuple(self.instance_retired_plan_ids or ())
        self.records = tuple(self.records or ())
        self.active_before = int(self.active_before)
        self.active_after = int(self.active_after)
        self.task_impossible = False
        if self.needs_replan:
            self.message = NEEDS_REPLAN_MESSAGE
        elif not self.message:
            self.message = ""


@dataclass
class Budget:
    """Call, replan and recovery budgets. Enforcement belongs to a later controller."""

    tool_calls: int = 12
    replans: int = 1
    recovery_rounds: int = 2
    tool_calls_used: int = 0
    replans_used: int = 0
    recovery_rounds_used: int = 0


@dataclass
class RecoveryRound:
    """Identity of one fixed candidate universe. Scheduling a new round is out of scope."""

    round_id: str
    candidate_plan_ids: tuple[str, ...] = ()
    inherited_constraint_ids: tuple[str, ...] = ()
    frozen: bool = True
    needs_replan: bool = False

    def __post_init__(self) -> None:
        self.candidate_plan_ids = tuple(self.candidate_plan_ids or ())
        self.inherited_constraint_ids = tuple(self.inherited_constraint_ids or ())
        self.frozen = True


@dataclass
class ExecutionObservation:
    """Inputs the analyzer may trust. LLM text is never a verified source."""

    tool_name: str = ""
    resource: str = ""
    identity: str = ""
    environment: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    tool_result: Any = None
    execution_log: str = ""
    llm_hypothesis: str = ""
    verifier_label: str = ""
    plan_id: str = ""
    node_id: str = ""
    command: str = ""
    call_instance: str = ""

    def __post_init__(self) -> None:
        self.parameters = dict(self.parameters or {})


def to_plain(value: Any) -> Any:
    """JSON-ready form of dataclasses and enums."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {key: to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, tuple):
        return [to_plain(item) for item in value]
    if isinstance(value, list):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    return value


def _scope_from(data: Optional[Mapping[str, Any]]) -> Scope:
    raw = dict(data or {})
    allowed = {name: raw.get(name, "") for name in Scope.__dataclass_fields__}
    return Scope(**allowed)


def _premise_from(data: Optional[Mapping[str, Any]]) -> AtomicPremise:
    raw = dict(data or {})
    return AtomicPremise(
        predicate=str(raw.get("predicate") or ""),
        tool=str(raw.get("tool") or ""),
        resource=str(raw.get("resource") or ""),
        identity=str(raw.get("identity") or ""),
        parameter=str(raw.get("parameter") or ""),
        argument=str(raw.get("argument") or ""),
        schema_signature=str(raw.get("schema_signature") or ""),
    )


def plan_step_from_dict(data: Mapping[str, Any]) -> PlanStep:
    raw = dict(data)
    premises = tuple(_premise_from(item) for item in (raw.get("preconditions") or ()))
    return PlanStep(
        subgoal=str(raw.get("subgoal") or ""),
        action=str(raw.get("action") or ""),
        tool=str(raw.get("tool") or ""),
        preconditions=premises,
        required_state=tuple(raw.get("required_state") or ()),
        estimated_cost=float(raw.get("estimated_cost", 1.0)),
        step_id=str(raw.get("step_id") or ""),
        predecessor_ids=tuple(raw.get("predecessor_ids") or ()),
        links_previous=bool(raw.get("links_previous", True)),
        resource=str(raw.get("resource") or ""),
        identity=str(raw.get("identity") or ""),
        parameter=str(raw.get("parameter") or ""),
        argument=str(raw.get("argument") or ""),
        schema_signature=str(raw.get("schema_signature") or ""),
        task_id=str(raw.get("task_id") or ""),
        consumes=tuple(raw.get("consumes") or ()),
        produces=tuple(raw.get("produces") or ()),
        instance_key=str(raw.get("instance_key") or ""),
    )


def candidate_plan_from_dict(data: Mapping[str, Any]) -> CandidatePlan:
    raw = dict(data)
    steps = tuple(plan_step_from_dict(item) for item in (raw.get("steps") or ()))
    status = raw.get("status") or PlanStatus.ACTIVE
    return CandidatePlan(
        plan_id=str(raw["plan_id"]),
        steps=steps,
        status=PlanStatus(status),
        context=_scope_from(raw.get("context")),
    )


def constraint_from_dict(data: Mapping[str, Any]) -> Constraint:
    raw = dict(data)
    failure_type = raw.get("failure_type") or None
    return Constraint(
        constraint_id=str(raw.get("constraint_id") or new_id("c")),
        condition=_premise_from(raw.get("condition")),
        scope=_scope_from(raw.get("scope")),
        evidence_status=EvidenceStatus(raw["evidence_status"]),
        source=str(raw.get("source") or ""),
        confidence=float(raw.get("confidence") or 0.0),
        direct=bool(raw.get("direct", False)),
        origin=EvidenceOrigin(raw.get("origin") or EvidenceOrigin.INFERENCE),
        observation_code=str(raw.get("observation_code") or ""),
        failure_type=FailureType(failure_type) if failure_type else None,
        parameter_level=bool(raw.get("parameter_level", False)),
        disables_tool=bool(raw.get("disables_tool", False)),
        evidence_id=str(raw.get("evidence_id") or ""),
        note=str(raw.get("note") or ""),
    )


def observation_from_dict(data: Mapping[str, Any]) -> ExecutionObservation:
    raw = dict(data)
    return ExecutionObservation(
        tool_name=str(raw.get("tool_name") or ""),
        resource=str(raw.get("resource") or ""),
        identity=str(raw.get("identity") or ""),
        environment=str(raw.get("environment") or ""),
        parameters=dict(raw.get("parameters") or {}),
        tool_result=raw.get("tool_result"),
        execution_log=str(raw.get("execution_log") or ""),
        llm_hypothesis=str(raw.get("llm_hypothesis") or ""),
        verifier_label=str(raw.get("verifier_label") or ""),
        plan_id=str(raw.get("plan_id") or ""),
        node_id=str(raw.get("node_id") or ""),
        command=str(raw.get("command") or ""),
        call_instance=str(raw.get("call_instance") or ""),
    )
