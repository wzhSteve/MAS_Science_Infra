"""Standalone assistant package (chat engine, tools, skills, memory, HTTP routes).

Compatibility re-exports let legacy imports (``from ai_assistant import ...``)
keep working after science_infra/control/assistant.py was split into this
package.
"""

from ai_assistant.context import ApplyBody, ChatBody, build_workspace_snapshot, summarize_workflow
from ai_assistant.engine import (
    MAX_OUTPUT_TOKENS,
    MAX_TOOL_ROUNDS,
    WELCOME_MARKERS,
    ToolSession,
    _history_messages,
    _local_answer,
    iter_chat_events,
)
from ai_assistant.llm import AssistantLlm, resolve_assistant_llm
from ai_assistant.router import router  # re-export the APIRouter object (not the submodule)
from ai_assistant.tools import OPENAI_TOOLS, assistant_tools

__all__ = [
    "ApplyBody",
    "AssistantLlm",
    "ChatBody",
    "MAX_OUTPUT_TOKENS",
    "MAX_TOOL_ROUNDS",
    "OPENAI_TOOLS",
    "ToolSession",
    "WELCOME_MARKERS",
    "_history_messages",
    "_local_answer",
    "assistant_tools",
    "build_workspace_snapshot",
    "iter_chat_events",
    "resolve_assistant_llm",
    "router",
    "summarize_workflow",
]
