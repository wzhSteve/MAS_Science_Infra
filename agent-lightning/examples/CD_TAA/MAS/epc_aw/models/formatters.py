from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Planner: QueryAnalysis (Step-0)
#
# Proxy rejects Dict[str,str] inside response_format ("Extra required key ...").
# Store the outline as a JSON *string* (same pattern as ContextVerification's
# cautious typing). Planner parses the string into a dict. Legacy prompt keys
# Analysis / ExecutionOutline remain accepted by the planner parser.
class QueryAnalysis(BaseModel):
    analysis: str = Field(
        description="Precise interpretation of the user's intent and information gaps",
    )
    execution_outline: str = Field(
        description=(
            'JSON object string of ordered steps, e.g. '
            '{"1":"Target Information: ... Operation Details: ... Expected Output: ...",'
            '"2":"..."}. MUST contain at least one step.'
        ),
    )

    def __str__(self):
        return f"Analysis: {self.analysis}\nExecutionOutline: {self.execution_outline}"

# Planner: NextStep
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str

# Diagnoser: slot update from verifier
class SlotUpdate(BaseModel):
    slot: str = ""
    value: Optional[str] = None
    filled: bool = False


# Diagnoser: context verification (two-layer judgment)
class ContextVerification(BaseModel):
    analysis: str = Field(description="Evidence-grounded verification analysis")
    subgoal_conclusion: str = Field(
        description="SUBGOAL_COMPLETE or SUBGOAL_INCOMPLETE"
    )
    new_obtained_information_flag: bool = Field(
        description="True if new usable information was obtained this step"
    )
    new_obtained_information: Union[str, List[str], Dict[str, str]] = Field(
        default="",
        description="Newly obtained facts from this step only. Use a JSON string, "
                    "a list of fact strings, or a dict of key->fact. Avoid bare "
                    "list/dict types so the response_format schema stays "
                    "proxy-valid (every anyOf branch needs a `type` key).",
    )
    conclusion: Optional[str] = Field(
        default=None,
        description="CONTINUE or null — STOP is decided by SlotGate in code, not LLM"
    )
    slot_updates: List[SlotUpdate] = Field(
        default_factory=list,
        description="Per-slot fill status for required answer slots",
    )
    evidence_type: str = Field(
        default="DIRECT",
        description="DIRECT | ABSENCE | ERROR | EMPTY",
    )
    tool_appropriate: bool = Field(
        default=True,
        description="False when the tool class cannot satisfy this subgoal",
    )
    task_recommendation: str = Field(
        default="CONTINUE",
        description="Always CONTINUE — task stop is code-gated",
    )


# Diagnoser: outline update response
#
# Field is snake_case `execution_outline` (REQUIRED, no alias) because the API
# proxy validates response_format JSON schemas strictly: PascalCase property
# names (e.g. an `ExecutionOutline` alias) get normalized by the OpenAI SDK's
# strict-mode schema builder, producing a `required`/`properties` key mismatch
# and a permanent proxy 429 ("Extra required key ... supplied"). snake_case
# fields (like ContextVerification's) pass cleanly. The prompt is aligned to
# emit `execution_outline`; the parser still accepts the legacy `ExecutionOutline`
# key as a fallback. REQUIRED (no default) so the schema emits
# `required: ["execution_outline"]`, satisfying proxies that demand `required`
# list every property key.
class OutlineUpdateResponse(BaseModel):
    execution_outline: Dict[str, str] = Field(
        description="Remaining steps renumbered from 1, or empty if task complete"
    )


# Legacy alias kept for backward compatibility
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str


# Executor: multiple candidate commands from a single LLM call.
# Used by Level 2 Intervention (parameter candidate set) to obtain N
# diverse commands in ONE executor LLM invocation (cost-neutral vs the
# previous sequential retry which spent 1 LLM call per attempt).
class ToolCommandSet(BaseModel):
    candidates: List[ToolCommand] = Field(
        default_factory=list,
        description="Up to N mutually distinct candidate commands",
    )


class FinalAnswer(BaseModel):
    final_answer: str


class FailureAttributionOutput(BaseModel):
    attributed_to: str = Field(
        description="planner | actor | verifier | tool_environment",
    )
    reason: str = Field(description="Short failure reason")
    recovery_hint: str = Field(
        default="",
        description="Concrete next action hint for the attributed role",
    )


class FinalAuditDecision(BaseModel):
    verdict: str = Field(description="PASS or FAIL")
    reason: str = Field(description="Why the final answer is accepted or rejected")
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Answer fragments not supported by live verified facts",
    )