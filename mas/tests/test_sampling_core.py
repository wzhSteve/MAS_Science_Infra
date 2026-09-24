import pytest
from pydantic import ValidationError

from workflow.contracts import BranchAnchor, BranchGate, BranchSite
from workflow.sampling.adapters.configured import ConfiguredGateAdapter
from workflow.sampling.compat import selector_from_anchor, selector_matches_window
from workflow.sampling.contracts import SamplingWindow, WindowKind
from workflow.sampling.core import ResumableWindow, SamplingCore
from workflow.sampling.registry import sampling_adapters
from workflow.site_policy import sampling_preview


def tool_window(window_id: str, sequence: int = 1, snapshot: str = "archive:snap"):
    return SamplingWindow(
        window_id=window_id,
        owner_agent_id="hub",
        kind=WindowKind.TOOL_RESULT,
        sequence=sequence,
        snapshot_ref=snapshot,
        interaction={"tool_id": "execute_python"},
    )


def tool_site(*, when: str = "first", nth: int = 1):
    return BranchSite(
        id="python_result",
        anchor=BranchAnchor(
            kind="after_tool",
            agent_id="hub",
            tool_id="execute_python",
        ),
        when=when,
        nth=nth,
        gate=BranchGate(type="always"),
    )


def plan(core: SamplingCore, site: BranchSite, windows, remaining: int = 2):
    return core.plan(
        parent_rollout_id="parent",
        sites=[site],
        windows=windows,
        remaining=remaining,
        depth=0,
        max_depth=2,
        default_beam=2,
        context_factory=lambda _site, _window: {},
    )


def test_legacy_anchor_normalizes_to_strict_tool_window():
    selector = selector_from_anchor(tool_site().anchor)
    assert selector.kind == WindowKind.TOOL_RESULT
    assert selector_matches_window(selector, tool_window("w1"))
    assert not selector_matches_window(
        selector,
        SamplingWindow(
            window_id="w2",
            owner_agent_id="hub",
            kind=WindowKind.AGENT_COMPLETE,
            snapshot_ref="archive:snap",
        ),
    )


def test_core_requires_snapshot_and_resumable_messages():
    core = SamplingCore(ConfiguredGateAdapter())
    with pytest.raises(ValidationError):
        tool_window("w1", snapshot="")
    without_messages = ResumableWindow(
        window=tool_window("w2"),
        resume_messages=[],
    )
    assert plan(core, tool_site(), [without_messages]) == []


def test_nth_window_and_budget_are_applied_by_core():
    core = SamplingCore(ConfiguredGateAdapter())
    windows = [
        ResumableWindow(
            window=tool_window("w1", sequence=1),
            resume_messages=[{"role": "tool", "content": "1"}],
        ),
        ResumableWindow(
            window=tool_window("w2", sequence=2, snapshot="archive:snap2"),
            resume_messages=[
                {"role": "tool", "content": "1"},
                {"role": "tool", "content": "2"},
            ],
        ),
    ]
    planned = plan(core, tool_site(when="nth", nth=2), windows, remaining=1)
    assert len(planned) == 1
    assert planned[0].window.window_id == "w2"
    assert planned[0].plan.count == 1
    assert planned[0].plan.snapshot_ref == "archive:snap2"


def test_registry_separates_independent_and_branch_strategies():
    assert sampling_adapters.resolve("grpo").id == "independent"
    assert sampling_adapters.resolve("arpo").id == "arpo"


def test_adapter_rejects_window_kind_outside_its_capability():
    core = SamplingCore(ConfiguredGateAdapter())
    site = BranchSite(
        id="agent_end",
        anchor=BranchAnchor(kind="after_agent_turn", agent_id="hub"),
        gate=BranchGate(type="always"),
    )
    window = ResumableWindow(
        window=SamplingWindow(
            window_id="agent-w1",
            owner_agent_id="hub",
            kind=WindowKind.AGENT_COMPLETE,
            snapshot_ref="archive:agent",
        ),
        resume_messages=[{"role": "assistant", "content": "done"}],
    )
    assert plan(core, site, [window]) == []


def test_entropy_gate_does_not_fabricate_missing_metrics():
    core = SamplingCore(ConfiguredGateAdapter())
    site = tool_site()
    site.gate = BranchGate(type="entropy_delta")
    window = ResumableWindow(
        window=tool_window("w-no-entropy"),
        resume_messages=[{"role": "tool", "content": "42"}],
    )
    assert plan(core, site, [window]) == []


def test_dual_entropy_requires_probe_entropy():
    core = SamplingCore(ConfiguredGateAdapter())
    site = tool_site()
    site.gate = BranchGate(type="dual_entropy")
    window = ResumableWindow(
        window=tool_window("w-no-probe").model_copy(
            update={"metrics": {"h_root": 0.1, "h_tool": 0.6}}
        ),
        resume_messages=[{"role": "tool", "content": "42"}],
    )
    assert plan(core, site, [window]) == []


def test_preview_separates_legacy_sites_and_blank_canvas_identity():
    preview = sampling_preview({
        "schema_version": "0.3",
        "topology": "graph",
        "entry_agent": "planner",
        "hub": {},
        "tools": [],
        "agents": [
            {"id": "planner", "kind": "planner"},
            {"id": "expert", "kind": "blank"},
        ],
        "routers": [{
            "id": "route_main",
            "candidates": ["expert"],
            "strategy": "llm_choice",
        }],
        "edges": [{"from": "planner", "to": "route_main", "kind": "message"}],
        "sampling": {
            "mode": "arpo",
            "sites": [{
                "id": "legacy_planner",
                "anchor": {"kind": "after_agent_turn", "agent_id": "planner"},
                "gate": {"type": "always"},
            }],
        },
    })
    blank = next(
        item for item in preview["opportunities"]
        if item["selector"]["interaction"].get("tool_id") == "blank:expert"
    )
    assert blank["node_id"] == "expert"
    assert [item["site_id"] for item in preview["legacy_sites"]] == ["legacy_planner"]
