"""Compatibility mapping from legacy BranchSite/Event shapes into core contracts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from workflow.contracts import BranchAnchor, BranchGate, BranchSite
from workflow.sampling.contracts import SamplingWindow, WindowKind, WindowSelector


_ANCHOR_KINDS = {
    "after_tool": WindowKind.TOOL_RESULT,
    "after_agent_turn": WindowKind.AGENT_COMPLETE,
    "after_verifier": WindowKind.VERIFICATION_COMPLETE,
    "on_edge": WindowKind.EDGE,
    "on_token": WindowKind.TOKEN,
}

_EVENT_KINDS = {
    "after_tool": WindowKind.TOOL_RESULT,
    "tool_result": WindowKind.TOOL_RESULT,
    "after_agent_turn": WindowKind.AGENT_COMPLETE,
    "agent_message": WindowKind.AGENT_COMPLETE,
    "after_verifier": WindowKind.VERIFICATION_COMPLETE,
    "feedback": WindowKind.VERIFICATION_COMPLETE,
    "on_edge": WindowKind.EDGE,
    "sample_barrier": WindowKind.EDGE,
    "on_token": WindowKind.TOKEN,
}


def selector_from_anchor(anchor: BranchAnchor) -> WindowSelector:
    kind = _ANCHOR_KINDS.get(str(anchor.kind or "").lower())
    if kind is None:
        raise ValueError(f"Unknown sampling anchor kind: {anchor.kind}")
    interaction: Dict[str, Any] = {}
    if anchor.tool_id:
        interaction["tool_id"] = anchor.tool_id
    if anchor.edge_id:
        interaction["edge_id"] = anchor.edge_id
    if anchor.skill_id:
        interaction["skill_id"] = anchor.skill_id
    return WindowSelector(
        kind=kind,
        owner_agent_id=anchor.agent_id,
        interaction=interaction,
    )


def window_from_event(
    event: Dict[str, Any],
    *,
    fallback_owner: Optional[str] = None,
    fallback_snapshot: Optional[str] = None,
) -> SamplingWindow:
    raw_kind = str(event.get("kind") or "").lower()
    kind = _EVENT_KINDS.get(raw_kind)
    if kind is None:
        raise ValueError(f"Unknown sampling event kind: {raw_kind}")
    interaction: Dict[str, Any] = {}
    if event.get("tool_id"):
        interaction["tool_id"] = event["tool_id"]
    if event.get("edge_id"):
        interaction["edge_id"] = event["edge_id"]
    if event.get("skill_id"):
        interaction["skill_id"] = event["skill_id"]
    interaction["legacy_event_kind"] = raw_kind
    return SamplingWindow(
        window_id=str(event.get("event_id") or event.get("window_id") or ""),
        owner_agent_id=str(event.get("agent_id") or fallback_owner or ""),
        kind=kind,
        sequence=int(event.get("sequence", event.get("turn", 0)) or 0),
        snapshot_ref=str(event.get("snapshot_ref") or fallback_snapshot or ""),
        metrics=dict(event.get("metrics") or {}),
        interaction=interaction,
    )


def selector_matches_window(selector: WindowSelector, window: SamplingWindow) -> bool:
    if selector.kind != window.kind:
        return False
    if selector.owner_agent_id and selector.owner_agent_id != window.owner_agent_id:
        return False
    return all(window.interaction.get(key) == value for key, value in selector.interaction.items())


def selector_key(selector: WindowSelector) -> str:
    interaction = ",".join(
        f"{key}={selector.interaction[key]}" for key in sorted(selector.interaction)
    )
    return f"{selector.kind.value}:{selector.owner_agent_id or ''}:{interaction}"


def resolve_configured_sites(
    sites: Optional[Iterable[BranchSite]],
    config: Any,
) -> list[BranchSite]:
    if sites is None:
        sites = [
            BranchSite(
                id="default_after_tool",
                anchor=BranchAnchor(kind="after_tool"),
                gate=BranchGate(type="entropy_delta"),
            )
        ]
    resolved: list[BranchSite] = []
    for site in sites:
        if not site.enabled:
            continue
        params = dict(site.gate.params or {})
        if site.gate.type in ("entropy_delta", "arpo"):
            params = {
                "use_official_arpo_gate": config.use_official_arpo_gate,
                "branch_probability": config.branch_probability,
                "entropy_weight": config.entropy_weight,
                "entropy_threshold": config.entropy_threshold,
                **params,
            }
        elif site.gate.type == "dual_entropy":
            params = {"probe_k": config.probe_k, **params}
        resolved.append(
            site.model_copy(
                update={"gate": site.gate.model_copy(update={"params": params})}
            )
        )
    return resolved
