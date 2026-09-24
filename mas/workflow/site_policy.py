"""Design-time projection of sampling opportunities onto a Workflow."""

from __future__ import annotations

from typing import Any

from workflow.compiler import compile_spec
from workflow.contracts import BranchAnchor, BranchSite
from workflow.spec import MASSpec

NATIVE_GATES = ["entropy_delta", "arpo", "always", "dual_entropy"]


def anchor_key(anchor: BranchAnchor) -> str:
    return ":".join(
        (
            str(anchor.kind or ""),
            str(anchor.agent_id or ""),
            str(anchor.tool_id or ""),
            str(anchor.edge_id or ""),
        )
    )


def site_capability(site: BranchSite, spec: MASSpec) -> tuple[str, str]:
    anchor = site.anchor
    agents = {agent.id: agent for agent in spec.agents}
    has_legacy_hub = not agents and spec.topology in ("hub_react", "single")
    tools = set(spec.tools) | {
        agent.id for agent in spec.agents if agent.kind == "tool"
    }
    if site.fork.resume_mode != "messages":
        return "error", "当前训练仅支持 messages 前缀续跑。"
    if site.gate.type not in ("entropy_delta", "arpo", "always", "dual_entropy"):
        return "error", f"Gate {site.gate.type} 尚无可靠的训练窗口信号。"
    if site.when not in ("first", "every", "nth") or (site.when == "nth" and site.nth < 1):
        return "error", "Apply on 必须是 first、every 或正整数 nth。"
    if anchor.kind == "after_tool":
        if anchor.tool_id and anchor.tool_id not in tools:
            return "error", f"Tool {anchor.tool_id} 不在当前 Workflow 中。"
        if anchor.agent_id and anchor.agent_id not in agents and not (
            has_legacy_hub and anchor.agent_id == "hub"
        ) and anchor.agent_id != anchor.tool_id:
            return "error", f"Agent {anchor.agent_id} 不在当前 Workflow 中。"
        return "pass", "Tool 返回后记录窗口并从该次前缀继续。"
    if anchor.kind == "after_agent_turn" and anchor.agent_id:
        agent = agents.get(anchor.agent_id)
        has_tools = (
            bool(agent.tools) or (agent.id == "hub" and bool(spec.tools))
            if agent else has_legacy_hub and anchor.agent_id == "hub" and bool(spec.tools)
        )
        if has_tools:
            return "warning", "当前兼容语义为该 Agent 内的 Tool 返回，不是整个 Agent 工作窗口结束。"
    return "error", f"{anchor.kind} 尚无对应的可恢复训练窗口。"


def sampling_preview(workflow: dict[str, Any]) -> dict[str, Any]:
    spec = MASSpec.model_validate(workflow)
    compiled = compile_spec(spec)
    sampling = spec.sampling
    configured = {anchor_key(site.anchor): site for site in sampling.sites}
    opportunities: dict[str, dict[str, Any]] = {}

    def add(
        anchor: BranchAnchor,
        *,
        node_id: str,
        label: str,
        support: str,
        message: str,
        runtime_event: str | None = None,
        allowed_gates: list[str] | None = None,
        edge_id: str | None = None,
    ) -> None:
        key = anchor_key(anchor)
        site = configured.get(key)
        opportunities[key] = {
            "id": key,
            "node_id": node_id,
            "edge_id": edge_id,
            "anchor": anchor.model_dump(mode="json"),
            "label": label,
            "support": support,
            "message": message,
            "runtime_event": runtime_event,
            "prefix": "messages" if support != "unavailable" else None,
            "allowed_gates": allowed_gates or [],
            "configured": site is not None,
            "enabled": bool(site and site.enabled),
            "site_id": site.id if site else None,
        }

    for agent_id, agent in compiled.agents.items():
        if agent.kind == "tool":
            continue
        tools = compiled.tools_for.get(agent_id) or []
        for tool_id in tools:
            display_tool = tool_id.removeprefix("blank:")
            add(
                BranchAnchor(kind="after_tool", agent_id=agent_id, tool_id=tool_id),
                node_id=display_tool,
                label=f"{display_tool} 执行完成后",
                support="native",
                message="训练会保留该次 Tool 返回后的消息上下文。",
                runtime_event="after_tool",
                allowed_gates=NATIVE_GATES,
            )
        if tools:
            add(
                BranchAnchor(kind="after_agent_turn", agent_id=agent_id),
                node_id=agent_id,
                label=f"{agent_id} 内的 Tool 返回后",
                support="compatibility",
                message="兼容旧配置；当前并不表示整个 Agent 工作窗口结束。",
                runtime_event="after_tool",
                allowed_gates=NATIVE_GATES,
            )
        elif agent.kind == "verifier":
            add(
                BranchAnchor(kind="after_verifier", agent_id=agent_id),
                node_id=agent_id,
                label=f"{agent_id} 验证完成后",
                support="unavailable",
                message="当前训练尚未保存 Verifier 边界的可恢复上下文。",
            )

    for router in spec.routers:
        add(
            BranchAnchor(kind="after_agent_turn", agent_id=router.id),
            node_id=router.id,
            label=f"{router.id} 完成路由后",
            support="unavailable",
            message="路由决策是动态的，当前尚无可恢复的 Router 窗口。",
        )

    for index, edge in enumerate(spec.edges):
        if edge.kind != "sample_barrier":
            continue
        edge_id = f"e-{edge.source}-{edge.target}-{index}"
        add(
            BranchAnchor(kind="on_edge", agent_id=edge.source, edge_id=edge_id),
            node_id=edge.target,
            edge_id=edge_id,
            label=f"{edge.source} → {edge.target} 通过时",
            support="unavailable",
            message="该边当前只是声明，运行时尚未产生可恢复窗口。",
        )

    diagnostics: list[dict[str, Any]] = []
    for site in sampling.sites:
        key = anchor_key(site.anchor)
        status, message = site_capability(site, spec)
        opportunity = opportunities.get(key)
        if opportunity is None:
            support = "compatibility" if status == "warning" else "native" if status == "pass" else "unavailable"
            opportunity = {
                "id": key,
                "node_id": site.anchor.tool_id or site.anchor.agent_id or "",
                "edge_id": site.anchor.edge_id,
                "anchor": site.anchor.model_dump(mode="json"),
                "label": site.anchor.tool_id or site.anchor.agent_id or site.id,
                "support": support,
                "message": message,
                "runtime_event": "after_tool" if support != "unavailable" else None,
                "prefix": "messages" if support != "unavailable" else None,
                "allowed_gates": NATIVE_GATES if support != "unavailable" else [],
                "configured": True,
                "enabled": site.enabled,
                "site_id": site.id,
            }
            opportunities[key] = opportunity
        elif status == "error":
            opportunity["support"] = "unavailable"
            opportunity["message"] = message
        elif status == "warning":
            opportunity["support"] = "compatibility"
            opportunity["message"] = message
        diagnostics.append(
            {
                "site_id": site.id,
                "status": status,
                "message": message,
                "opportunity_id": key,
            }
        )

    group_n = max(1, int(sampling.group_n))
    initial = max(1, min(int(sampling.initial_rollouts), group_n))
    return {
        "policy": {
            "mode": sampling.mode,
            "group_n": group_n,
            "initial_rollouts": initial,
            "remaining_budget": max(0, group_n - initial),
            "beam_size": max(1, int(sampling.beam_size)),
            "max_branch_depth": max(1, int(sampling.max_branch_depth)),
        },
        "opportunities": list(opportunities.values()),
        "diagnostics": diagnostics,
        "legacy": (
            {
                "barriers": list(sampling.barriers),
                "count": len(sampling.barriers),
                "message": "任意首次 Tool 返回后的兼容规则；转换为显式站点后可绑定具体位置。",
            }
            if not sampling.sites and sampling.barriers
            else None
        ),
    }
