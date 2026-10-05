"""Runner-owned execution details and a small, atomically published attempt index."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from .contracts import ExecutionTrace, RolloutTreeEdge, RolloutTreeNode
from .llm_diagnostics import redact_text, summarize_execution_error
from .rollout_tree import atomic_json, stable_id

logger = logging.getLogger(__name__)
MAX_DETAIL_CHARS = 128_000


def attempt_directory(root: Path, rollout_id: str, attempt_id: str) -> Path:
    return root / "executions" / stable_id(rollout_id, attempt_id)


def message_content(message: Any) -> dict[str, Any]:
    additional = getattr(message, "additional_kwargs", {}) or {}
    return {
        "role": getattr(message, "type", "unknown"),
        "content": getattr(message, "content", ""),
        "tool_calls": getattr(message, "tool_calls", None),
        "tool_call_id": getattr(message, "tool_call_id", None),
        "reasoning": additional.get("reasoning_content", additional.get("reasoning")),
    }


def bounded_detail(value: dict[str, Any]) -> dict[str, Any]:
    remaining = MAX_DETAIL_CHARS
    truncated = False

    def clean(item: Any) -> Any:
        nonlocal remaining, truncated
        if isinstance(item, str):
            text = redact_text(item)
            kept = text[:max(0, remaining)]
            remaining -= len(kept)
            truncated |= len(kept) < len(text)
            return kept
        if isinstance(item, dict):
            return {
                str(key): "[redacted]" if re.search(
                    r"(?i)^(api[_-]?key|authorization|password|access[_-]?token|secret)$", str(key)
                ) else clean(child)
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [clean(child) for child in item]
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return clean(str(item))

    result = clean(value)
    result["content_truncated"] = truncated
    return result


class ExecutionRecorder:
    def __init__(self, root: Path, rollout_id: str, attempt_id: str, *, resume: dict[str, Any] | None = None):
        self.directory = attempt_directory(root, rollout_id, attempt_id)
        self.trace = ExecutionTrace(rollout_id=rollout_id, attempt_id=attempt_id, resume=resume or {})
        self._details: dict[str, dict[str, Any]] = {}
        self._started: dict[str, float] = {}
        self._lock = RLock()
        self._disabled = False
        self._allowed_agents: set[str] | None = None
        self._publish()

    def allow_agents(self, agent_ids: list[str]) -> None:
        """Execution nodes may only use these workflow agent ids."""
        self._allowed_agents = {str(agent_id) for agent_id in agent_ids if str(agent_id)}

    def note_issue(self, message: str) -> None:
        with self._lock:
            if message and message not in self.trace.errors:
                self.trace.errors.append(message)
            self._publish()

    def note_tools(self, calls: list[dict[str, Any]]) -> None:
        """Attach inner tool calls to the latest agent node. Does not add a node."""
        if not calls:
            return
        with self._lock:
            agents = [node for node in self.trace.nodes if node.agent_kind == "agent"]
            if not agents:
                return
            detail = self._details.get(agents[-1].node_id)
            if detail is None:
                return
            detail["tool_calls"] = list(detail.get("tool_calls") or []) + list(calls)
            self._publish(agents[-1].node_id)

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> ExecutionRecorder | None:
        root = os.environ.get("TIR_ROLLOUT_TREE_DIR")
        rollout_id, attempt_id = task.get("_rollout_id"), task.get("_attempt_id")
        if not root or not rollout_id or not attempt_id or task.get("is_probe"):
            return None
        return cls(Path(root), str(rollout_id), str(attempt_id), resume={
            "source_rollout_id": task.get("resume_parent_id"),
            "source_attempt_id": task.get("source_attempt_id"),
            "fork_node_ids": task.get("fork_node_ids") or [],
            "window_id": task.get("window_id"),
            "snapshot_ref": task.get("snapshot_ref"),
        })

    def _publish(self, node_id: str | None = None) -> None:
        if self._disabled:
            return
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            if node_id is not None:
                atomic_json(self.directory / f"{node_id}.json", bounded_detail(self._details[node_id]))
            self.trace.revision += 1
            atomic_json(self.directory / "index.json", self.trace.model_dump(mode="json"))
        except (OSError, ValueError) as error:
            self._disabled = True
            logger.error("Execution recording failed rollout=%s attempt=%s: %s",
                         self.trace.rollout_id, self.trace.attempt_id, summarize_execution_error(error))

    def resume_applied(self, messages: Any) -> None:
        with self._lock:
            self.trace.resume_applied = True
            self.trace.resume["prefix_hash"] = stable_id(messages)
            self._publish()

    def begin(self, agent_id: str, agent_kind: Literal["agent", "tool"], turn: int,
              inputs: Any, *, parents: list[str] | None = None, model: str | None = None,
              input_changed: bool = False) -> str:
        with self._lock:
            allowed = self._allowed_agents
            if allowed is not None and (agent_kind != "agent" or agent_id not in allowed):
                message = f"execution agent_id {agent_id!r} is not a workflow agent"
                if message not in self.trace.errors:
                    self.trace.errors.append(message)
                self._publish()
                return ""
            node_id = uuid4().hex
            previous = list(self.trace.tail_ids if parents is None else parents)
            self.trace.nodes.append(RolloutTreeNode(
                node_id=node_id, kind="execution", rollout_id=self.trace.rollout_id,
                attempt_id=self.trace.attempt_id, agent_id=agent_id, agent_kind=agent_kind,
                turn=turn, status="running", started_at=time.time(),
                detail_ref=f"executions/{self.directory.name}/{node_id}.json",
            ))
            self.trace.edges.extend(RolloutTreeEdge(source_node_id=parent, target_node_id=node_id)
                                    for parent in previous)
            if not previous:
                self.trace.entry_ids.append(node_id)
            self.trace.tail_ids = [node_id]
            self._started[node_id] = time.monotonic()
            self._details[node_id] = {
                "node_id": node_id, "rollout_id": self.trace.rollout_id,
                "attempt_id": self.trace.attempt_id, "input": inputs, "model": model,
                "status": "running", "errors": [], "input_level": "application_messages",
                "input_changed": input_changed,
            }
            self._publish(node_id)
            return node_id

    def request_error(self, node_id: str, error: Exception, *, retry_input: Any = None) -> None:
        if not node_id:
            return
        with self._lock:
            detail = self._details.get(node_id)
            if detail is None:
                return
            detail["errors"].append(summarize_execution_error(error))
            if retry_input is not None:
                detail.setdefault("previous_inputs", []).append(detail["input"])
                detail["input"] = retry_input
                detail["input_changed"] = True
            self._publish(node_id)

    def finish(self, node_id: str, output: Any, *, reasoning: Any = None, tool_calls: Any = None,
               observation: Any = None, failed: bool = False, metrics: dict[str, Any] | None = None) -> None:
        if not node_id:
            return
        with self._lock:
            node = next((n for n in self.trace.nodes if n.node_id == node_id), None)
            if node is None:
                return
            node.status = "failed" if failed else "succeeded"
            node.ended_at = time.time()
            node.summary = redact_text(output if isinstance(output, str) else json.dumps(
                output, ensure_ascii=False, default=str))[:240]
            node.metrics = metrics or {}
            detail = self._details[node_id]
            detail.update(output=output, reasoning=reasoning, tool_calls=tool_calls,
                          observation=observation, status=node.status,
                          elapsed_seconds=time.monotonic() - self._started.pop(node_id),
                          output_origin="synthetic_fallback" if failed and node.agent_kind == "agent" else "execution")
            self._publish(node_id)

    def join(self, node_ids: list[str]) -> None:
        with self._lock:
            self.trace.tail_ids = list(node_ids)
            self._publish()

    def window(self, window_id: str, *, snapshot_ref: str | None = None, messages: Any = None) -> None:
        with self._lock:
            window = self.trace.windows.setdefault(window_id, {"fork_node_ids": list(self.trace.tail_ids)})
            if snapshot_ref:
                window["snapshot_ref"] = snapshot_ref
            if messages is not None:
                window["prefix_hash"] = stable_id(messages)
            self._publish()

    def sampling(self, window_id: str, decision: dict[str, Any]) -> None:
        with self._lock:
            window = self.trace.windows.get(window_id)
            if window:
                for node in self.trace.nodes:
                    if node.node_id in window["fork_node_ids"]:
                        node.decision = decision
                self._publish()

    def complete(self) -> None:
        with self._lock:
            self.trace.complete = True
            self._publish()
