import json
import re
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field


class MilestoneSpecOutput(BaseModel):
    id: str
    description: str
    depends_on: List[str] = Field(default_factory=list)
    verification_description: str
    required_fact_names: List[str] = Field(default_factory=list)
    require_source: bool = True
    preferred_tools: List[str] = Field(default_factory=list)


class StateInitialization(BaseModel):
    analysis: str
    milestones: List[MilestoneSpecOutput]


class PlannerDecision(BaseModel):
    analysis: str
    milestone_id: str
    subgoal: str
    tool_name: str
    expected_evidence: str


class ToolAction(BaseModel):
    analysis: str = ""
    tool_name: str
    arguments: Dict[str, Any]


class VerifiedFactOutput(BaseModel):
    name: str
    value: Any


class VerificationDecision(BaseModel):
    analysis: str
    verdict: Literal["satisfied", "invalidated", "rejected"]
    reason: str
    evidence_summary: str = ""
    confidence: Literal["low", "medium", "high"] = "medium"
    facts: List[VerifiedFactOutput] = Field(default_factory=list)


class FailureAttributionOutput(BaseModel):
    attributed_to: Literal["planner", "actor", "verifier", "tool_environment"]
    reason: str
    recovery_hint: str


class FinalAnswerDraft(BaseModel):
    answer: str


class FinalAuditDecision(BaseModel):
    verdict: Literal["PASS", "FAIL"]
    reason: str
    unsupported_claims: List[str] = Field(default_factory=list)


# Backwards-compatible names used by engine wrappers and third-party imports.
class QueryAnalysis(BaseModel):
    concise_summary: str = ""
    required_skills: str = ""
    relevant_tools: str = ""
    additional_considerations: str = ""


class NextStep(BaseModel):
    justification: str = ""
    context: str = ""
    sub_goal: str
    tool_name: str


class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool = False


class ToolCommand(BaseModel):
    analysis: str = ""
    explanation: str = ""
    command: str = ""


class FinalAnswer(BaseModel):
    final_answer: str = ""


T = TypeVar("T", bound=BaseModel)


def parse_structured_response(response: Any, schema: Type[T]) -> T:
    """Parse JSON emitted by any supported engine into one Pydantic schema."""

    if isinstance(response, schema):
        return response
    if isinstance(response, BaseModel):
        payload = response.model_dump()
    elif isinstance(response, dict):
        payload = response
    else:
        text = str(response or "").strip()
        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if fenced:
            text = fenced.group(1).strip()
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            first, last = text.find("{"), text.rfind("}")
            if first < 0 or last <= first:
                raise ValueError(f"Expected JSON for {schema.__name__}: {text[:500]}")
            payload = json.loads(text[first : last + 1])
    if hasattr(schema, "model_validate"):
        return schema.model_validate(payload)
    return schema.parse_obj(payload)