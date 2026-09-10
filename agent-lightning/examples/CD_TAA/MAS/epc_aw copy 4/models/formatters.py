from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    concise_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.concise_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

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