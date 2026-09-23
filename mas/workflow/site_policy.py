"""Capabilities of the window boundaries currently executed by training."""

from __future__ import annotations

from workflow.contracts import BranchSite
from workflow.spec import MASSpec


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
