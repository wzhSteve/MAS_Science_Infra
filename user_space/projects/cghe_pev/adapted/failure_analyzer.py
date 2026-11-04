"""Turn an execution failure into evidence without promoting an LLM guess.

``Ψ`` is fixed. Only four observation codes can become a verified
counterexample, and only when the binding is complete:

- tool missing → ``ToolExists`` is false for that tool in this environment
- permission denied → ``CanAccess`` is false for this resource and identity
- schema mismatch → ``SchemaOk`` is false for this call shape, not for the tool
- a parameter error bound to this call → ``ValidID`` is false for that argument
- Bing returned URLs and the verifier says those snippets miss the subgoal
  → ``SnippetsSufficient`` is false for that query only, not for the search tool

Timeouts, verifier uncertainty, generic exceptions, empty returns and text that
appears only in an LLM hypothesis are recorded as unverified. They do not
produce ``VERIFIED_COUNTEREXAMPLE``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from cghe_types import (
    AtomicPremise,
    Constraint,
    EvidenceOrigin,
    EvidenceStatus,
    ExecutionObservation,
    FailureEvidence,
    FailureType,
    PSI_OBSERVATION_TO_PREDICATE,
    Scope,
    new_id,
    norm,
    schema_signature,
)


_CODE_FIELDS = ("error_code", "code", "failure_type", "observation_code")
_MESSAGE_FIELDS = ("message", "error", "result", "output", "stderr", "reason")

_TEXT_RULES = (
    ("TOOL_NOT_FOUND", re.compile(
        r"tool[_\s-]*not[_\s-]*found|unknown tool|no such tool|"
        r"tool does not exist|tool is not installed|not installed",
        re.I,
    )),
    ("PERMISSION_DENIED", re.compile(
        r"permission[_\s-]*denied|access denied|not authorized|\bforbidden\b",
        re.I,
    )),
    ("SCHEMA_MISMATCH", re.compile(
        r"schema[_\s-]*mismatch|invalid schema|schema invalid",
        re.I,
    )),
    ("PARAM_ERROR", re.compile(
        r"param(?:eter)?[_\s-]*error|invalid[_\s-]*(?:id|argument|param(?:eter)?)|"
        r"bad parameter|missing required (?:id|argument|parameter)",
        re.I,
    )),
    ("TRANSIENT", re.compile(
        r"timeout|timed out|temporar(?:y|ily)|transient|http\s*429|http\s*503|rate limit",
        re.I,
    )),
    ("GENERIC", re.compile(r"\btraceback\b|\bexception\b|^error\b|\[\"error", re.I)),
)

_ALIASES = {
    "tool_not_found": "TOOL_NOT_FOUND",
    "toolnotfound": "TOOL_NOT_FOUND",
    "missing_tool": "TOOL_NOT_FOUND",
    "tool_unavailable": "TOOL_NOT_FOUND",
    "toolunavailable": "TOOL_NOT_FOUND",
    "permission_denied": "PERMISSION_DENIED",
    "permissiondenied": "PERMISSION_DENIED",
    "access_denied": "PERMISSION_DENIED",
    "schema_mismatch": "SCHEMA_MISMATCH",
    "schemamismatch": "SCHEMA_MISMATCH",
    "param_error": "PARAM_ERROR",
    "parameter_error": "PARAM_ERROR",
    "invalid_id": "PARAM_ERROR",
    "invalid_argument": "PARAM_ERROR",
    "transient": "TRANSIENT",
    "transient_error": "TRANSIENT",
    "timeout": "TRANSIENT",
    "generic": "GENERIC",
    "generic_failure": "GENERIC",
    "empty": "EMPTY",
    "subgoal_not_satisfied": "SUBGOAL_NOT_SATISFIED",
    "verifier_uncertain": "VERIFIER_UNCERTAIN",
}

_FAILURE_LABELS = {
    "failure",
    "failed",
    "不满足",
    "unsatisfied",
    "not_satisfied",
    "subgoal_not_satisfied",
}
_UNCERTAIN_LABELS = {"uncertain", "verifier_uncertain", "unknown"}
_SUCCESS_LABELS = {"success", "满足", "satisfied", "passed", "ok"}

_FAILURE_BY_CODE = {
    "TOOL_NOT_FOUND": FailureType.TOOL_UNAVAILABLE,
    "PERMISSION_DENIED": FailureType.PERMISSION_DENIED,
    "SCHEMA_MISMATCH": FailureType.SCHEMA_MISMATCH,
    "PARAM_ERROR": FailureType.PARAMETER_ERROR,
    "SNIPPETS_INSUFFICIENT": FailureType.SNIPPETS_INSUFFICIENT,
    "TRANSIENT": FailureType.TRANSIENT_ERROR,
    "GENERIC": FailureType.GENERIC_FAILURE,
    "EMPTY": FailureType.MISSING_INFORMATION,
    "SUBGOAL_NOT_SATISFIED": FailureType.SUBGOAL_NOT_SATISFIED,
    "VERIFIER_UNCERTAIN": FailureType.VERIFIER_UNCERTAIN,
}

_ARG_RE = re.compile(
    r"(?:invalid (?:id|argument|parameter)|parameter)\s*['\"]([^'\"]+)['\"]",
    re.I,
)


@dataclass(frozen=True)
class FailureAnalysis:
    """Analyzer output. ``verified_counterexample`` is local to ``Ψ`` and is not full admissibility."""

    evidence: FailureEvidence
    constraint: Optional[Constraint]
    mapped_predicate: str
    verified_counterexample: bool


class FailureAnalyzer:
    """Prefer ``tool_result``, then ``execution_log``. Never verify ``llm_hypothesis``."""

    def __init__(self, *, confidence_threshold: float = 1.0) -> None:
        self.confidence_threshold = float(confidence_threshold)

    def analyze(self, observation: ExecutionObservation) -> FailureAnalysis:
        tool_code, tool_payload = _extract_signal(observation.tool_result)
        log_code, log_payload = _extract_signal(observation.execution_log)
        if tool_code:
            code, payload, source = tool_code, tool_payload, "tool_result"
        elif log_code:
            code, payload, source = log_code, log_payload, "execution_log"
        else:
            code, payload, source = "", {}, ""

        llm_code, _llm_payload = _extract_signal(observation.llm_hypothesis)
        verifier = norm(observation.verifier_label).replace(" ", "_")

        if code in PSI_OBSERVATION_TO_PREDICATE:
            return self._mapped(observation, code, payload, source)
        if _snippet_gap(observation, verifier):
            return self._mapped(
                observation,
                "SNIPPETS_INSUFFICIENT",
                {
                    "parameter": "query",
                    "argument": _search_query(observation),
                    "tool": observation.tool_name,
                },
                "tool_result",
            )
        if code == "TRANSIENT":
            return self._unverified(
                observation,
                FailureType.TRANSIENT_ERROR,
                observation_code="TRANSIENT",
                source=source,
                origin=EvidenceOrigin.DIRECT_OBSERVATION,
                note="transient error is not a counterexample",
            )
        if code == "GENERIC":
            return self._unverified(
                observation,
                FailureType.GENERIC_FAILURE,
                observation_code="GENERIC",
                source=source,
                origin=EvidenceOrigin.DIRECT_OBSERVATION,
                note="generic exception does not enter Ψ",
            )
        if code == "EMPTY" or (
            not code
            and _is_empty_return(observation.tool_result)
            and not str(observation.execution_log or "").strip()
        ):
            return self._unverified(
                observation,
                FailureType.MISSING_INFORMATION,
                observation_code="EMPTY",
                source=source or "tool_result",
                origin=EvidenceOrigin.DIRECT_OBSERVATION,
                note="empty return does not enter Ψ",
            )
        if verifier in _UNCERTAIN_LABELS:
            return self._unverified(
                observation,
                FailureType.VERIFIER_UNCERTAIN,
                observation_code="VERIFIER_UNCERTAIN",
                source="verifier",
                origin=EvidenceOrigin.DIRECT_OBSERVATION,
                note="verifier uncertainty does not change the version space",
            )
        if verifier in _FAILURE_LABELS:
            return self._unverified(
                observation,
                FailureType.SUBGOAL_NOT_SATISFIED,
                observation_code="SUBGOAL_NOT_SATISFIED",
                source="verifier",
                origin=EvidenceOrigin.DIRECT_OBSERVATION,
                note="an unsatisfied subgoal does not name a false premise",
            )
        if llm_code:
            failure_type = _FAILURE_BY_CODE.get(llm_code, FailureType.GENERIC_FAILURE)
            return self._unverified(
                observation,
                failure_type,
                observation_code=llm_code,
                source="llm_hypothesis",
                origin=EvidenceOrigin.INFERENCE,
                note="LLM hypothesis is not a verified observation",
                hypothesized=True,
            )
        evidence = self._evidence(
            observation,
            is_failure=False,
            failure_type=None,
            observation_code="",
            source=source,
            origin=EvidenceOrigin.DIRECT_OBSERVATION,
            direct=False,
            bound_to_call=False,
            note="no failure signal" if verifier not in _SUCCESS_LABELS else "verifier reported success",
        )
        return FailureAnalysis(evidence, None, "", False)

    def _mapped(
        self,
        observation: ExecutionObservation,
        code: str,
        payload: dict[str, Any],
        source: str,
    ) -> FailureAnalysis:
        predicate_name = PSI_OBSERVATION_TO_PREDICATE[code]
        tool = norm(observation.tool_name or payload.get("tool") or "")
        resource = norm(observation.resource or payload.get("resource") or "")
        identity = norm(observation.identity or payload.get("identity") or "")
        environment = norm(observation.environment or payload.get("environment") or "")
        parameter, argument = _bind_parameter(observation, payload)
        signature = norm(
            payload.get("schema_signature")
            or observation_schema(observation, payload)
        )
        binding_ok, bind_note = _binding_complete(
            code,
            tool,
            resource,
            identity,
            environment,
            parameter,
            argument,
            signature,
        )
        premise = AtomicPremise(
            predicate=predicate_name,
            tool=tool,
            resource=resource if code == "PERMISSION_DENIED" else "",
            identity=identity if code == "PERMISSION_DENIED" else "",
            parameter=parameter if code in {"PARAM_ERROR", "SNIPPETS_INSUFFICIENT"} else "",
            argument=argument if code in {"PARAM_ERROR", "SNIPPETS_INSUFFICIENT"} else "",
            schema_signature=signature if code == "SCHEMA_MISMATCH" else "",
        )
        scope = _scope_for(code, tool, resource, identity, environment, parameter, argument, signature)
        if not binding_ok:
            analysis = self._unverified(
                observation,
                _FAILURE_BY_CODE[code],
                observation_code=code,
                source=source,
                origin=EvidenceOrigin.DIRECT_OBSERVATION,
                note=bind_note,
            )
            analysis.evidence.tool = tool
            analysis.evidence.resource = resource
            analysis.evidence.parameter = parameter
            analysis.evidence.argument = argument
            return analysis

        evidence = self._evidence(
            observation,
            is_failure=True,
            failure_type=_FAILURE_BY_CODE[code],
            observation_code=code,
            source=source,
            origin=EvidenceOrigin.DIRECT_OBSERVATION,
            direct=True,
            bound_to_call=code in {"PARAM_ERROR", "SCHEMA_MISMATCH", "SNIPPETS_INSUFFICIENT"},
            note="direct observation matches Ψ",
            tool=tool,
            resource=resource,
            identity=identity,
            parameter=parameter,
            argument=argument,
            schema_signature=signature if code == "SCHEMA_MISMATCH" else "",
            environment=environment,
        )
        constraint = Constraint(
            constraint_id=new_id("c"),
            condition=premise,
            scope=scope,
            evidence_status=EvidenceStatus.VERIFIED_COUNTEREXAMPLE,
            source=source,
            confidence=1.0,
            direct=True,
            origin=EvidenceOrigin.DIRECT_OBSERVATION,
            observation_code=code,
            failure_type=evidence.failure_type,
            parameter_level=code in {"PARAM_ERROR", "SNIPPETS_INSUFFICIENT"},
            disables_tool=False,
            evidence_id=evidence.evidence_id,
            note=evidence.note,
        )
        return FailureAnalysis(
            evidence=evidence,
            constraint=constraint,
            mapped_predicate=predicate_name,
            verified_counterexample=True,
        )

    def _unverified(
        self,
        observation: ExecutionObservation,
        failure_type: FailureType,
        *,
        observation_code: str,
        source: str,
        origin: EvidenceOrigin,
        note: str,
        hypothesized: bool = False,
    ) -> FailureAnalysis:
        evidence = self._evidence(
            observation,
            is_failure=True,
            failure_type=failure_type,
            observation_code=observation_code,
            source=source,
            origin=origin,
            direct=False,
            bound_to_call=False,
            note=note,
        )
        predicate = PSI_OBSERVATION_TO_PREDICATE.get(observation_code, "")
        if hypothesized and predicate:
            condition = AtomicPremise(predicate=predicate, tool=norm(observation.tool_name))
        else:
            condition = AtomicPremise(predicate="", tool=norm(observation.tool_name))
        constraint = Constraint(
            constraint_id=new_id("c"),
            condition=condition,
            scope=Scope(
                environment=norm(observation.environment),
                tool=norm(observation.tool_name),
                resource=norm(observation.resource),
                identity=norm(observation.identity),
            ),
            evidence_status=EvidenceStatus.UNVERIFIED_HYPOTHESIS,
            source=source,
            confidence=0.0,
            direct=False,
            origin=origin,
            observation_code=observation_code,
            failure_type=failure_type,
            parameter_level=observation_code == "PARAM_ERROR",
            disables_tool=False,
            evidence_id=evidence.evidence_id,
            note=note,
        )
        return FailureAnalysis(
            evidence=evidence,
            constraint=constraint,
            mapped_predicate=predicate if hypothesized else "",
            verified_counterexample=False,
        )

    def _evidence(self, observation: ExecutionObservation, **kwargs: Any) -> FailureEvidence:
        return FailureEvidence(
            evidence_id=new_id("e"),
            parameters=dict(observation.parameters),
            raw_tool_result=observation.tool_result,
            execution_log=str(observation.execution_log or ""),
            llm_hypothesis=str(observation.llm_hypothesis or ""),
            plan_id=observation.plan_id,
            node_id=observation.node_id,
            call_instance=observation.call_instance,
            tool=kwargs.pop("tool", norm(observation.tool_name)),
            resource=kwargs.pop("resource", norm(observation.resource)),
            identity=kwargs.pop("identity", norm(observation.identity)),
            environment=kwargs.pop("environment", norm(observation.environment)),
            **kwargs,
        )


def observation_schema(observation: ExecutionObservation, payload: dict[str, Any]) -> str:
    parameters = observation.parameters or payload.get("parameters") or payload.get("args")
    if isinstance(parameters, dict):
        return schema_signature(parameters)
    return ""


def _binding_complete(
    code: str,
    tool: str,
    resource: str,
    identity: str,
    environment: str,
    parameter: str,
    argument: str,
    signature: str,
) -> tuple[bool, str]:
    if not tool:
        return False, "observation does not name a tool"
    if not environment:
        return False, "observation does not name an environment"
    if code == "TOOL_NOT_FOUND":
        return True, ""
    if code == "PERMISSION_DENIED":
        if not resource or not identity:
            return False, "permission denial must name the current resource and identity"
        return True, ""
    if code == "SCHEMA_MISMATCH":
        if not signature:
            return False, "schema mismatch without a call shape is not scoped"
        return True, ""
    if code == "PARAM_ERROR":
        if not (parameter or argument):
            return False, "parameter error is not bound to an argument"
        return True, ""
    if code == "SNIPPETS_INSUFFICIENT":
        if parameter != "query" or not argument:
            return False, "snippet gap is not bound to the search query"
        return True, ""
    return False, "observation is outside Ψ"


def _scope_for(
    code: str,
    tool: str,
    resource: str,
    identity: str,
    environment: str,
    parameter: str,
    argument: str,
    signature: str,
) -> Scope:
    if code == "TOOL_NOT_FOUND":
        return Scope(environment=environment, tool=tool)
    if code == "PERMISSION_DENIED":
        return Scope(environment=environment, tool=tool, resource=resource, identity=identity)
    if code == "SCHEMA_MISMATCH":
        return Scope(environment=environment, tool=tool, schema_signature=signature)
    if code == "SNIPPETS_INSUFFICIENT":
        return Scope(environment=environment, tool=tool, parameter=parameter, argument=argument)
    return Scope(
        environment=environment,
        tool=tool,
        parameter=parameter,
        argument=argument,
    )


def _bind_parameter(observation: ExecutionObservation, payload: dict[str, Any]) -> tuple[str, str]:
    parameter = norm(payload.get("parameter") or payload.get("param") or "")
    argument = str(payload.get("argument") or payload.get("value") or payload.get("id") or "")
    argument = argument.strip()
    if not argument:
        match = _ARG_RE.search(str(payload.get("message") or ""))
        if match:
            argument = match.group(1).strip()
            if not parameter:
                parameter = "id"
    if (not parameter or not argument) and observation.parameters:
        named = ""
        message = str(payload.get("message") or "")
        for key in observation.parameters:
            if key and re.search(rf"\b{re.escape(str(key))}\b", message, re.I):
                named = str(key)
                break
        if named:
            parameter = parameter or norm(named)
            argument = argument or str(observation.parameters[named])
        elif len(observation.parameters) == 1 and (parameter or argument or payload):
            only_key, only_value = next(iter(observation.parameters.items()))
            parameter = parameter or norm(only_key)
            argument = argument or str(only_value)
    return norm(parameter), norm(argument)


def _extract_signal(value: Any) -> tuple[str, dict[str, Any]]:
    if _is_empty_return(value):
        return "", {}
    if isinstance(value, dict):
        payload = dict(value)
        for key in _CODE_FIELDS:
            if payload.get(key):
                raw = str(payload[key])
                code = _canonicalize(raw) or _classify_text(raw)
                if code:
                    payload.setdefault("message", raw)
                    return code, payload
        for key in _MESSAGE_FIELDS:
            if isinstance(payload.get(key), str):
                code = _classify_text(payload[key])
                if code:
                    payload["message"] = payload[key]
                    return code, payload
        return "", payload
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _extract_signal(value[0])
        return "", {}
    if isinstance(value, str):
        code = _classify_text(value)
        return code, {"message": value}
    return "", {}


def _classify_text(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return "EMPTY"
    if body.casefold() in {"null", "none", "empty", "no result", "no results"}:
        return "EMPTY"
    aliased = _canonicalize(body)
    if aliased and aliased not in {"GENERIC"}:
        # A bare token such as TOOL_NOT_FOUND should win over the generic "error" rule.
        if re.fullmatch(r"[A-Za-z0-9_\- ]+", body) and aliased in _FAILURE_BY_CODE:
            return aliased
    for code, pattern in _TEXT_RULES:
        if pattern.search(body):
            return code
    aliased = _canonicalize(body)
    return aliased


def _canonicalize(raw: str) -> str:
    token = norm(raw).replace("-", "_").replace(" ", "_")
    token = re.sub(r"_+", "_", token)
    if token in _ALIASES:
        return _ALIASES[token]
    upper = token.upper()
    if upper in PSI_OBSERVATION_TO_PREDICATE or upper in {"TRANSIENT", "GENERIC", "EMPTY"}:
        return upper
    return ""


def _is_empty_return(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, dict) and not value:
        return True
    return False


_URL_RE = re.compile(r"https?://[^\s\]\)\"'<>]+")
_QUERY_RE = re.compile(r"query\s*=\s*(['\"])(.+?)\1", re.I)
_HEADER_RE = re.compile(r"results for:\s*(.+)", re.I)


def _result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value or "")


def _search_query(observation: ExecutionObservation) -> str:
    match = _QUERY_RE.search(str(observation.command or ""))
    if match:
        return match.group(2).strip()
    header = _HEADER_RE.search(_result_text(observation.tool_result))
    if header:
        return header.group(1).split("\n", 1)[0].strip()
    return ""


def _snippet_gap(observation: ExecutionObservation, verifier: str) -> bool:
    """A thin Bing return that was judged insufficient. Rich snippets are not a counterexample."""
    if verifier not in _FAILURE_LABELS:
        return False
    if "bing" not in norm(observation.tool_name):
        return False
    text = _result_text(observation.tool_result)
    urls = _URL_RE.findall(text)
    if len(urls) < 1 or not _search_query(observation):
        return False
    if len(urls) >= 3 and len(text) >= 500:
        return False
    return True

