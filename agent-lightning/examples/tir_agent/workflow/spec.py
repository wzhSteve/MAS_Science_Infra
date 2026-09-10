"""Static Framework spec (YAML / Pydantic). Not a Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

DEFAULT_SPEC_PATH = Path(__file__).resolve().parent.parent / "specs" / "hub_react.yaml"


class HubSpec(BaseModel):
    role: str = "orchestrator"
    skills: List[str] = Field(default_factory=lambda: ["react_loop"])
    verify: Optional[str] = None
    max_feedback_hops: int = 1
    system_prompt: str = ""

    model_config = {"extra": "forbid"}


class LLMBinding(BaseModel):
    kind: Literal["api", "local", "rl_endpoint"] = "api"
    model: str = ""
    base_url: Optional[str] = None

    model_config = {"extra": "forbid"}


class MemorySpec(BaseModel):
    agent: str = "messages"
    system: str = "none"

    model_config = {"extra": "forbid"}


class ArchiveSpec(BaseModel):
    window: str = "post_first_tool"

    model_config = {"extra": "forbid"}


class AgentNodeSpec(BaseModel):
    """Graph node (schema 0.2). Compiler executes planner/executor/verifier graphs."""

    id: str
    role: str = "agent"
    skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    memory_scope: str = "agent"
    system_prompt: str = ""
    model: str = "inherit"
    trainable: bool = True
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class EdgeSpec(BaseModel):
    """Communication edge between agents (UI / future compiler)."""

    source: str = Field(alias="from")
    target: str = Field(alias="to")
    kind: Literal["message", "tool_call", "feedback", "route"] = "message"
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid", "populate_by_name": True}


class MASSpec(BaseModel):
    schema_version: str = "0.1.0"
    topology: str = "hub_react"
    hub: HubSpec = Field(default_factory=HubSpec)
    tools: List[str] = Field(
        default_factory=lambda: ["web_search", "wikipedia_search", "execute_python"]
    )
    llm: LLMBinding = Field(default_factory=LLMBinding)
    memory: MemorySpec = Field(default_factory=MemorySpec)
    archive: ArchiveSpec = Field(default_factory=ArchiveSpec)
    # schema 0.2 graph extensions
    agents: List[AgentNodeSpec] = Field(default_factory=list)
    edges: List[EdgeSpec] = Field(default_factory=list)
    entry_agent: str = "hub"

    model_config = {"extra": "forbid"}

    def is_executable(self) -> tuple[bool, str]:
        """Hub ReAct always runs; graph topologies need a valid compile()."""
        if self.topology in ("hub_react", "single", ""):
            return True, "hub_react"
        if self.topology == "graph":
            from .compiler import compile_spec

            compiled = compile_spec(self)
            return compiled.ok, compiled.reason
        return False, f"unknown topology {self.topology}"


def load_spec(path: Optional[str] = None) -> MASSpec:
    p = Path(path) if path else DEFAULT_SPEC_PATH
    if not p.is_file():
        return MASSpec()
    text = p.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        raw = yaml.safe_load(text) or {}
    except Exception:
        raw = _parse_simple_yaml(text)
    if not isinstance(raw, dict):
        raise ValueError(f"spec must be a mapping: {p}")
    # Normalize edge keys from / from_ / source
    edges = raw.get("edges")
    if isinstance(edges, list):
        norm = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            item = dict(e)
            if "from" in item and "source" not in item:
                pass
            elif "source" in item and "from" not in item:
                item["from"] = item.pop("source")
            if "to" in item and "target" not in item:
                pass
            elif "target" in item and "to" not in item:
                item["to"] = item.pop("target")
            norm.append(item)
        raw["edges"] = norm
    return MASSpec.model_validate(raw)


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """Minimal fallback when PyYAML is absent (hub_react.yaml subset)."""
    data: Dict[str, Any] = {}
    current: Optional[str] = None
    nested: Dict[str, Any] = {}
    tools: List[str] = []
    skills: List[str] = []
    in_tools = False
    in_skills = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        s = line.strip()
        if indent == 0 and ":" in s and not s.startswith("-"):
            key, _, rest = s.partition(":")
            key = key.strip()
            rest = rest.strip().strip('"').strip("'")
            if rest:
                data[key] = rest
                continue
            if current == "hub":
                data["hub"] = nested
            elif current == "llm":
                data["llm"] = nested
            elif current == "memory":
                data["memory"] = nested
            elif current == "archive":
                data["archive"] = nested
            current = key
            nested = {}
            in_tools = key == "tools"
            in_skills = False
            if in_tools:
                tools = []
            continue
        if s.startswith("- "):
            item = s[2:].strip()
            if in_tools:
                tools.append(item)
            elif in_skills:
                skills.append(item)
            continue
        if indent > 0 and ":" in s:
            k, v = s.split(":", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k == "skills":
                in_skills = True
                skills = []
                if v.startswith("[") and v.endswith("]"):
                    inner = v[1:-1].strip()
                    skills = [x.strip() for x in inner.split(",") if x.strip()]
                    in_skills = False
                    nested[k] = skills
                continue
            nested[k] = v
    if current == "hub":
        if skills:
            nested["skills"] = skills
        data["hub"] = nested
    elif current == "llm":
        data["llm"] = nested
    elif current == "memory":
        data["memory"] = nested
    elif current == "archive":
        data["archive"] = nested
    if tools:
        data["tools"] = tools
    return data
