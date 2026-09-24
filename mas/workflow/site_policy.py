"""Design-time Sampling capability projection backed by Adapter registry."""

from __future__ import annotations

from typing import Any

from workflow.contracts import BranchSite
from workflow.sampling.compat import selector_from_anchor, selector_key
from workflow.sampling.registry import sampling_adapters
from workflow.spec import MASSpec


def _opportunity_index(spec: MASSpec, strategy: str):
    adapter = sampling_adapters.resolve(strategy)
    return adapter, {
        selector_key(opportunity.selector): opportunity
        for opportunity in adapter.opportunities(spec)
    }


def site_capability(
    site: BranchSite,
    spec: MASSpec,
    *,
    strategy: str | None = None,
) -> tuple[str, str]:
    strategy = strategy or spec.sampling.mode
    _, opportunities = _opportunity_index(spec, strategy)
    selector = selector_from_anchor(site.anchor)
    opportunity = opportunities.get(selector_key(selector))
    if opportunity is None:
        if str(site.anchor.kind) == "after_agent_turn":
            return "warning", "旧 Agent Site 不属于当前策略的原生 Window；请迁移到明确的 Tool Result Site。"
        return "error", f"当前策略 {strategy} 不支持此采样窗口。"
    if site.fork.resume_mode != "messages":
        return "error", "当前执行器仅支持 messages Snapshot 续跑。"
    if site.gate.type not in opportunity.allowed_gates:
        return "error", f"当前策略不支持 Gate {site.gate.type}。"
    if site.when not in ("first", "every", "nth") or (
        site.when == "nth" and site.nth < 1
    ):
        return "error", "Apply on 必须是 first、every 或正整数 nth。"
    return "pass", opportunity.message


def _edge_id(spec: MASSpec, owner: str | None, tool_id: str | None) -> str | None:
    if not owner or not tool_id:
        return None
    for index, edge in enumerate(spec.edges):
        if edge.source == owner and edge.target == tool_id and edge.kind == "tool_call":
            return f"e-{edge.source}-{edge.target}-{index}"
    return None


def _canvas_node_id(selector) -> str:
    runtime_id = str(
        selector.interaction.get("tool_id") or selector.owner_agent_id or ""
    )
    return runtime_id.removeprefix("blank:")


def sampling_preview(workflow: dict[str, Any]) -> dict[str, Any]:
    spec = MASSpec.model_validate(workflow)
    sampling = spec.sampling
    adapter, supported = _opportunity_index(spec, sampling.mode)
    configured = {
        selector_key(selector_from_anchor(site.anchor)): site
        for site in sampling.sites
    }
    opportunities: list[dict[str, Any]] = []
    for key, opportunity in supported.items():
        selector = opportunity.selector
        tool_id = selector.interaction.get("tool_id")
        site = configured.get(key)
        opportunities.append(
            {
                "id": key,
                "node_id": _canvas_node_id(selector),
                "edge_id": _edge_id(spec, selector.owner_agent_id, str(tool_id or "")),
                "edge_source": selector.owner_agent_id,
                "edge_target": _canvas_node_id(selector),
                "selector": selector.model_dump(mode="json"),
                "anchor": {
                    "kind": "after_tool" if selector.kind.value == "tool_result" else selector.kind.value,
                    "agent_id": selector.owner_agent_id,
                    "tool_id": tool_id,
                    "edge_id": selector.interaction.get("edge_id"),
                },
                "label": (
                    f"{tool_id} 结果返回 {selector.owner_agent_id} 后"
                    if tool_id
                    else f"{selector.owner_agent_id} 窗口结束后"
                ),
                "support": opportunity.support,
                "message": opportunity.message,
                "runtime_event": selector.kind.value,
                "prefix": "messages" if opportunity.resumable else None,
                "allowed_gates": list(opportunity.allowed_gates),
                "configured": site is not None,
                "enabled": bool(site and site.enabled),
                "site_id": site.id if site else None,
            }
        )

    diagnostics: list[dict[str, Any]] = []
    legacy_sites: list[dict[str, Any]] = []
    for site in sampling.sites:
        selector = selector_from_anchor(site.anchor)
        key = selector_key(selector)
        status, message = site_capability(site, spec, strategy=sampling.mode)
        diagnostics.append(
            {
                "site_id": site.id,
                "status": status,
                "message": message,
                "opportunity_id": key if key in supported else None,
            }
        )
        if key not in supported:
            legacy_sites.append(
                {
                    "site_id": site.id,
                    "enabled": site.enabled,
                    "selector": selector.model_dump(mode="json"),
                    "message": message,
                }
            )

    group_n = max(1, int(sampling.group_n))
    initial = max(1, min(int(sampling.initial_rollouts), group_n))
    return {
        "strategy": {
            "id": adapter.id,
            "source_mode": sampling.mode,
        },
        "policy": {
            "mode": sampling.mode,
            "group_n": group_n,
            "initial_rollouts": initial,
            "remaining_budget": max(0, group_n - initial),
            "beam_size": max(1, int(sampling.beam_size)),
            "max_branch_depth": max(1, int(sampling.max_branch_depth)),
        },
        "opportunities": opportunities,
        "legacy_sites": legacy_sites,
        "diagnostics": diagnostics,
        "legacy": (
            {
                "barriers": list(sampling.barriers),
                "count": len(sampling.barriers),
                "message": "旧 barrier 尚未绑定明确 Tool Result Window；请转换为显式 Site。",
            }
            if not sampling.sites and sampling.barriers
            else None
        ),
    }
