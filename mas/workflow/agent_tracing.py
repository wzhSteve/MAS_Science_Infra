"""One naming rule for LangGraph model nodes and training span selection."""

from __future__ import annotations

import re
from typing import Sequence
from urllib.parse import quote


def agent_node_name(agent_id: str) -> str:
    """Encode a business Agent ID without LangGraph's reserved ':' and '|'."""
    return f"agent__{quote(agent_id, safe='')}"


def agent_span_pattern(agent_ids: Sequence[str]) -> str | None:
    names = list(dict.fromkeys(agent_node_name(agent_id) for agent_id in agent_ids))
    return rf"^(?:{'|'.join(re.escape(name) for name in names)})$" if names else None
