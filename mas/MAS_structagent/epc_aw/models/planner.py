from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from MAS.epc_aw.engine.factory import create_llm_engine
from MAS.epc_aw.models.formatters import (
    PlannerDecision,
    StateInitialization,
    parse_structured_response,
)
from MAS.epc_aw.models.memory import PlannerMemory
from MAS.epc_aw.models.state import (
    AgentState,
    Milestone,
    VerificationSpec,
)


_PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts" / "planner"


def _load_prompt(name: str) -> str:
    return (_PROMPT_ROOT / name).read_text(encoding="utf-8")


class Planner:
    """State-conditioned subgoal selector.

    It may propose work but intentionally exposes no State mutation method.
    """

    def __init__(
        self,
        llm_engine_name: str,
        toolbox_metadata: Optional[dict] = None,
        available_tools: Optional[List[str]] = None,
        verbose: bool = False,
        is_multimodal: bool = False,
        check_model: bool = True,
        temperature: float = 0.0,
        n: int = 1,
        llm_engine: Any = None,
    ):
        del check_model
        self.llm_engine_name = llm_engine_name
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.verbose = verbose
        self.temperature = temperature
        self.n = n  # retained for API compatibility; StructAgent is single-plan.
        self.memory = PlannerMemory()
        self.llm_engine_fixed = llm_engine or create_llm_engine(
            model_string=llm_engine_name,
            is_multimodal=is_multimodal,
            temperature=temperature,
            n=1,
        )

    def initialize_state(
        self, question: str, image_path: Optional[str] = None
    ) -> AgentState:
        del image_path  # QA-native implementation is text/tool based.
        prompt = _load_prompt("state_initialization.txt").format(
            Question=question,
            Available_Tools=json.dumps(self.available_tools, ensure_ascii=False),
            Toolbox_Metadata=json.dumps(
                self.toolbox_metadata, ensure_ascii=False, default=str
            ),
        )
        response = self.llm_engine_fixed(
            prompt, response_format=StateInitialization, temperature=self.temperature, n=1
        )
        parsed = parse_structured_response(response, StateInitialization)
        milestones = [
            Milestone(
                id=item.id,
                description=item.description,
                depends_on=list(item.depends_on),
                verify=VerificationSpec(
                    description=item.verification_description,
                    required_fact_names=list(item.required_fact_names),
                    require_source=item.require_source,
                    preferred_tools=[
                        tool
                        for tool in item.preferred_tools
                        if tool in self.available_tools
                    ],
                ),
            )
            for item in parsed.milestones
        ]
        return AgentState(question=question, milestones=milestones)

    def select_subgoal(self, state: AgentState) -> PlannerDecision:
        reachable = state.reachable_milestones()
        if not reachable:
            raise RuntimeError("No reachable unverified milestone remains")
        prompt = _load_prompt("select_subgoal.txt").format(
            Question=state.question,
            State=json.dumps(state.compact_view(), ensure_ascii=False, default=str),
            Reachable_Milestones=json.dumps(
                [item.id for item in reachable], ensure_ascii=False
            ),
            Available_Tools=json.dumps(self.available_tools, ensure_ascii=False),
            Toolbox_Metadata=json.dumps(
                self.toolbox_metadata, ensure_ascii=False, default=str
            ),
        )
        response = self.llm_engine_fixed(
            prompt, response_format=PlannerDecision, temperature=self.temperature, n=1
        )
        decision = parse_structured_response(response, PlannerDecision)
        reachable_ids = {item.id for item in reachable}
        if decision.milestone_id not in reachable_ids:
            raise ValueError(
                f"Planner selected unreachable milestone {decision.milestone_id!r}; "
                f"reachable={sorted(reachable_ids)}"
            )
        if decision.tool_name not in self.available_tools:
            raise ValueError(f"Planner selected unavailable tool {decision.tool_name!r}")
        return decision

    # Compatibility helpers used by historical entry points.
    def generate_base_response(
        self, question: str, image: Optional[str], max_tokens: int = 2048
    ) -> str:
        del image
        return str(self.llm_engine_fixed(question, max_tokens=max_tokens))

    def analyze_query(self, question: str, image: Optional[str] = None):
        state = self.initialize_state(question, image)
        outline = {
            str(index + 1): milestone.description
            for index, milestone in enumerate(state.milestones)
        }
        return "Question decomposed into verifier-gated milestones.", outline

    def generate_next_step(self, *args, **kwargs):
        raise RuntimeError(
            "EPC-AW generate_next_step/BTS is disabled; use select_subgoal(state)."
        )
