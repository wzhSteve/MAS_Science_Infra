from typing import Any, Dict, Optional, Union

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

# Diagnoser: context verification (two-layer judgment)
class ContextVerification(BaseModel):
    analysis: str = Field(description="Evidence-grounded verification analysis")
    subgoal_conclusion: str = Field(
        description="SUBGOAL_COMPLETE or SUBGOAL_INCOMPLETE"
    )
    new_obtained_information_flag: bool = Field(
        description="True if new usable information was obtained this step"
    )
    new_obtained_information: Union[str, list, dict] = Field(
        default="",
        description="Newly obtained facts from this step only"
    )
    conclusion: Optional[str] = Field(
        default=None,
        description="STOP, CONTINUE, or null when subgoal is incomplete"
    )


# Diagnoser: outline update response
class OutlineUpdateResponse(BaseModel):
    execution_outline: Dict[str, str] = Field(
        alias="ExecutionOutline",
        default_factory=dict,
        description="Remaining steps renumbered from 1, or empty if task complete"
    )

    class Config:
        populate_by_name = True


# Legacy alias kept for backward compatibility
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str

class FinalAnswer(BaseModel):
    final_answer: str