"""AgentRole / Skill sockets. Default: hub + react_loop + verifier; planner/executor roles registered."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SkillResult(BaseModel):
    """Skill.run return. route is an optional next agent_id (hub retry in v1.3)."""

    ok: bool = True
    output: Any = None
    route: Optional[str] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


@runtime_checkable
class Skill(Protocol):
    name: str

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Return SkillResult-like dict: {ok, output, route?}."""
        ...


@runtime_checkable
class AgentRole(Protocol):
    role_id: str
    level: int


class ReactLoopSkill:
    name = "react_loop"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "output": {"runtime": "react", "task_id": (ctx.get("task") or {}).get("id")}, "route": None}


class VerifierSkill:
    """Thin format/error check. Fail → route=hub. Not EPC-AW / not MAS_structagent."""

    name = "verifier"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        err = ctx.get("error")
        ans = ctx.get("final_answer")
        fmt = bool(ctx.get("format_ok"))
        if err or not str(ans or "").strip() or not fmt:
            return {
                "ok": False,
                "output": {"reason": "missing_or_error", "error": err},
                "route": "hub",
            }
        return {"ok": True, "output": {"reason": "pass"}, "route": None}


class HubRole:
    role_id = "hub"
    level = 0


class PlannerRole:
    role_id = "planner"
    level = 0


class ExecutorRole:
    role_id = "executor"
    level = 1


class VerifierRole:
    role_id = "verifier"
    level = 1


class PluginRegistry:
    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._roles: Dict[str, AgentRole] = {}

    def register_skill(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def register_role(self, role: AgentRole) -> None:
        self._roles[role.role_id] = role

    def get_skill(self, name: str) -> Skill:
        if name not in self._skills:
            raise KeyError(f"unknown skill {name!r}; registered={sorted(self._skills)}")
        return self._skills[name]

    def get_role(self, role_id: str) -> AgentRole:
        if role_id not in self._roles:
            raise KeyError(f"unknown role {role_id!r}; registered={sorted(self._roles)}")
        return self._roles[role_id]

    def list_skills(self) -> List[str]:
        return sorted(self._skills)

    def list_roles(self) -> List[str]:
        return sorted(self._roles)


def default_registry() -> PluginRegistry:
    reg = PluginRegistry()
    reg.register_skill(ReactLoopSkill())
    reg.register_skill(VerifierSkill())
    reg.register_role(HubRole())
    reg.register_role(VerifierRole())
    reg.register_role(PlannerRole())
    reg.register_role(ExecutorRole())
    return reg


REGISTRY = default_registry()


def _dump_skill_result(name: str, raw: Any) -> Dict[str, Any]:
    if isinstance(raw, SkillResult):
        dumped = raw.model_dump(mode="json")
    elif isinstance(raw, dict):
        dumped = SkillResult(
            ok=bool(raw.get("ok", True)),
            output=raw.get("output"),
            route=raw.get("route"),
            meta=dict(raw.get("meta") or {}),
        ).model_dump(mode="json")
    else:
        dumped = SkillResult(ok=True, output=raw).model_dump(mode="json")
    dumped["skill"] = name
    return dumped


def invoke_skill(name: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
    skill = REGISTRY.get_skill(name)
    dumped = _dump_skill_result(name, skill.run(ctx))
    if dumped.get("route"):
        ctx["route"] = dumped["route"]
    return dumped


def invoke_hub_skills(spec: Any, ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run every hub skill. Unknown names raise KeyError (no silent Solver import)."""
    REGISTRY.get_role("hub")
    skills = list(getattr(getattr(spec, "hub", None), "skills", None) or ["react_loop"])
    return [invoke_skill(name, ctx) for name in skills]
