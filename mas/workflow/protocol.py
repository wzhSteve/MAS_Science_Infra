"""AgentMessage envelope + per-kind payload schemas (centralized MAS).

The explicit message envelope is the core of the new communication protocol.
Edges carry ``AgentMessage`` objects only — no ``tool_calls`` / ``ToolMessage``.
Each ``kind`` has a payload schema validated at window close; mismatches set
``ok=False`` and route back along feedback edges. ``trace_ref`` chains one
planner→router→tool→verifier loop into a causal trace.

See docs/MAS_AGENT_LANDING.md §2-3 for the contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from uuid import uuid4

from pydantic import BaseModel, Field

MSG_KINDS = Literal[
    "plan_step",
    "route_decision",
    "tool_invoke",
    "tool_result",
    "verify",
    "feedback",
    "final_answer",
    "error",
]

EVIDENCE_TYPES = Literal["DIRECT", "ABSENCE", "ERROR", "EMPTY"]
STEP_CONCLUSIONS = Literal["COMPLETE", "INCOMPLETE"]


class AgentMessage(BaseModel):
    """Envelope carried on every edge in the centralized topology."""

    msg_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str
    turn: int
    src: str
    dst: str
    kind: MSG_KINDS
    payload: Dict[str, Any] = Field(default_factory=dict)
    trace_ref: Optional[str] = None

    model_config = {"extra": "forbid"}

    def chain_to(self, child: "AgentMessage") -> "AgentMessage":
        """Set ``child.trace_ref`` to this message's id and return child."""
        child.trace_ref = self.msg_id
        return child

    def payload_digest(self) -> str:
        blob = json.dumps(self.payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


# --- per-kind payload schemas ------------------------------------------------

# Each entry: list of required keys + optional key-type expectations.
# Types are checked loosely (isinstance) when present; absence of a required
# key fails validation. Unknown extra keys are allowed (forward-compat).
PAYLOAD_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "plan_step": {
        "required": ["next", "args", "sub_goal", "done"],
        "types": {
            "next": (str, list),  # str (single) or list[str] (multi-router fan-out)
            "args": (dict, list),  # dict (single) or list[dict] (parallel)
            "sub_goal": str,
            "done": bool,
        },
    },
    "route_decision": {
        "required": ["selected", "args", "candidates", "strategy"],
        "types": {
            "selected": (str, list),
            "args": (dict, list),
            "candidates": list,
            "strategy": str,
        },
    },
    "tool_invoke": {
        "required": [],  # args schema is tool-agent-specific; checked separately
        "types": {},
    },
    "tool_result": {
        "required": ["output", "ok", "evidence_type"],
        "types": {
            "output": str,
            "ok": bool,
            "evidence_type": str,
        },
    },
    "verify": {
        "required": ["ok", "reason", "step_conclusion", "slot_updates"],
        "types": {
            "ok": bool,
            "reason": str,
            "step_conclusion": str,
            "slot_updates": list,
        },
    },
    "feedback": {
        "required": ["reason", "attributed_to"],
        "types": {
            "reason": str,
            "attributed_to": str,
        },
    },
    "final_answer": {
        "required": ["answer"],
        "types": {"answer": str},
    },
    "error": {
        "required": ["agent_id", "error"],
        "types": {"agent_id": str, "error": str},
    },
}

EVIDENCE_VALUES = {"DIRECT", "ABSENCE", "ERROR", "EMPTY"}
STEP_VALUES = {"COMPLETE", "INCOMPLETE"}


def _check_types(payload: Dict[str, Any], types: Dict[str, Any]) -> Optional[str]:
    for key, expected in types.items():
        if key not in payload:
            continue
        val = payload[key]
        if expected is str and not isinstance(val, str):
            return f"{key!r} must be str, got {type(val).__name__}"
        if expected is bool:
            if not isinstance(val, bool):
                return f"{key!r} must be bool, got {type(val).__name__}"
        elif isinstance(expected, tuple):
            if not isinstance(val, expected):
                return f"{key!r} must be one of {expected}, got {type(val).__name__}"
        elif isinstance(expected, type) and not isinstance(val, expected):
            return f"{key!r} must be {expected.__name__}, got {type(val).__name__}"
    return None


def validate_payload(kind: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a payload against its kind schema. Returns (ok, reason)."""
    schema = PAYLOAD_SCHEMAS.get(kind)
    if schema is None:
        return False, f"unknown message kind {kind!r}"
    if not isinstance(payload, dict):
        return False, f"payload must be a dict, got {type(payload).__name__}"
    for key in schema.get("required", []):
        if key not in payload:
            return False, f"missing required field {key!r} for kind {kind!r}"
    err = _check_types(payload, schema.get("types", {}))
    if err:
        return False, err
    # domain value checks
    if kind == "tool_result" and payload.get("evidence_type") not in EVIDENCE_VALUES:
        return False, f"evidence_type must be one of {sorted(EVIDENCE_VALUES)}"
    if kind == "verify" and payload.get("step_conclusion") not in STEP_VALUES:
        return False, f"step_conclusion must be one of {sorted(STEP_VALUES)}"
    if kind == "plan_step":
        nxt = payload.get("next")
        if isinstance(nxt, list) and not all(isinstance(x, str) for x in nxt):
            return False, "plan_step.next list items must be str"
        args = payload.get("args")
        if isinstance(args, list) and not all(isinstance(x, dict) for x in args):
            return False, "plan_step.args list items must be dict"
    return True, ""


# --- blank-agent JSON Schema validation --------------------------------------

DEFAULT_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}

DEFAULT_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"output": {"type": "string"}},
    "required": ["output"],
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True  # unknown types pass (forward-compat)


def validate_json_schema(value: Any, schema: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """Minimal JSON-Schema subset validator for blank-agent IO contracts.

    Supports ``type`` (single), ``properties``, ``required``, ``items``,
    and ``additionalProperties: false``. Defaults to ``{input:str}->{output:str}``
    when schema is None.
    """
    if schema is None:
        schema = DEFAULT_OUTPUT_SCHEMA if isinstance(value, dict) else {}
    if not isinstance(schema, dict):
        return False, "schema must be a dict"
    expected = schema.get("type")
    if expected and not _type_ok(value, expected):
        return False, f"value must be {expected}, got {type(value).__name__}"
    if expected == "object" or (expected is None and isinstance(value, dict)):
        if not isinstance(value, dict):
            return False, "value must be an object"
        for key in schema.get("required", []) or []:
            if key not in value:
                return False, f"missing required field {key!r}"
        props = schema.get("properties", {}) or {}
        for key, sub in props.items():
            if key not in value:
                continue
            ok, reason = validate_json_schema(value[key], dict(sub) if isinstance(sub, dict) else None)
            if not ok:
                return False, f"{key}: {reason}"
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                return False, f"unexpected fields: {sorted(extra)}"
    if expected == "array":
        if not isinstance(value, list):
            return False, "value must be an array"
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(value):
                ok, reason = validate_json_schema(item, dict(item_schema) if isinstance(item_schema, dict) else None)
                if not ok:
                    return False, f"[{i}]: {reason}"
    return True, ""


def make_message(
    *,
    task_id: str,
    turn: int,
    src: str,
    dst: str,
    kind: MSG_KINDS,
    payload: Optional[Dict[str, Any]] = None,
    trace_ref: Optional[str] = None,
) -> AgentMessage:
    """Construct + validate an AgentMessage. Raises ValueError on bad payload."""
    msg = AgentMessage(
        task_id=task_id,
        turn=turn,
        src=src,
        dst=dst,
        kind=kind,
        payload=dict(payload or {}),
        trace_ref=trace_ref,
    )
    ok, reason = validate_payload(kind, msg.payload)
    if not ok:
        raise ValueError(f"invalid {kind} payload: {reason}")
    return msg


def commit_to_fact(verify_payload: Dict[str, Any], tool_result_payload: Dict[str, Any]) -> bool:
    """HIVE-style commit gate: COMPLETE + DIRECT + slots filled => live fact.

    Phase 2 of the MAS refactor: delegates to
    ``workflow.memory.facts.FactLedger`` (the single classification
    implementation); the bool external signature is frozen by the Phase 0
    contract (docs/refactor/phase0_contract_freeze.md §2).
    """
    from .memory.facts import FactLedger

    return FactLedger.commits(FactLedger.classify(verify_payload, tool_result_payload))
