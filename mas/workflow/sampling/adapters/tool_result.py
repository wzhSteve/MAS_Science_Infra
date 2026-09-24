"""Shared Tool Result opportunity projection for branch adapters."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from workflow.compiler import compile_spec
from workflow.sampling.contracts import (
    SamplingOpportunity,
    WindowKind,
    WindowSelector,
)


def tool_result_opportunities(
    workflow: Any,
    *,
    allowed_gates: Sequence[str],
    message: str,
) -> Iterable[SamplingOpportunity]:
    compiled = compile_spec(workflow)
    for agent_id, tools in compiled.tools_for.items():
        for tool_id in tools:
            yield SamplingOpportunity(
                selector=WindowSelector(
                    kind=WindowKind.TOOL_RESULT,
                    owner_agent_id=str(agent_id),
                    interaction={"tool_id": str(tool_id)},
                ),
                allowed_gates=list(allowed_gates),
                message=message,
            )
