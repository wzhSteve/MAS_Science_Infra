from types import SimpleNamespace

from workflow.active_set import ActiveSetConfig, ActiveSetSession
from workflow.contracts import BranchAnchor, BranchGate, BranchSite
from workflow.sampling.adapters.arpo import ArpoSamplingAdapter
from workflow.sampling.contracts import SamplingWindow, WindowKind
from workflow.spec import MASSpec


class FixedRng:
    def random(self):
        return 0.0


def workflow():
    return MASSpec.model_validate({
        "schema_version": "0.3",
        "topology": "centralized",
        "entry_agent": "planner",
        "hub": {},
        "tools": ["python_coder"],
        "agents": [{
            "id": "planner",
            "kind": "planner",
            "tools": ["python_coder"],
        }],
        "sampling": {"mode": "arpo"},
    })


def site(gate: str = "entropy_delta", beam: int = 3):
    return BranchSite(
        id="python_result",
        anchor=BranchAnchor(
            kind="after_tool",
            agent_id="planner",
            tool_id="python_coder",
        ),
        gate=BranchGate(type=gate),
        fork={"beam_size": beam},
    )


def window(metrics=None):
    return SamplingWindow(
        window_id="window-1",
        owner_agent_id="planner",
        kind=WindowKind.TOOL_RESULT,
        sequence=1,
        snapshot_ref="archive:snapshot",
        interaction={"tool_id": "python_coder"},
        metrics=metrics or {},
    )


def test_centralized_template_sites_follow_agents_and_router():
    from workflow.site_policy import sampling_preview, site_capability
    from workflow.templates import load_template_workflow

    raw = load_template_workflow("centralized")
    raw["sampling"] = {"mode": "arpo"}
    spec = MASSpec.model_validate(raw)
    opportunities = list(ArpoSamplingAdapter().opportunities(spec))
    owners = {item.selector.owner_agent_id for item in opportunities}
    assert owners == {"planner", "verifier", "route_exec"}
    assert "wikipedia_search" not in owners
    kinds = {item.selector.owner_agent_id: item.selector.kind for item in opportunities}
    assert kinds["planner"] == WindowKind.AGENT_COMPLETE
    assert kinds["route_exec"] == WindowKind.AGENT_COMPLETE
    assert kinds["verifier"] == WindowKind.VERIFICATION_COMPLETE
    assert opportunities[0].allowed_gates == ["entropy_delta", "arpo", "always", "verifier_fail", "verifier_pass"]

    preview = sampling_preview(raw)
    by_owner = {item["node_id"]: item for item in preview["opportunities"]}
    assert by_owner["planner"]["edge_source"] == "planner"
    assert by_owner["planner"]["edge_target"] == "route_exec"
    assert by_owner["route_exec"]["edge_target"] == "verifier"
    assert by_owner["verifier"]["edge_target"] == "planner"
    assert by_owner["route_exec"]["label"].endswith("路由结束后")

    raw["sampling"]["sites"] = [{
        "id": "after_planner",
        "anchor": {"kind": "after_agent_turn", "agent_id": "planner"},
        "enabled": True,
    }]
    enabled = MASSpec.model_validate(raw)
    status, _message = site_capability(enabled.sampling.sites[0], enabled)
    assert status == "pass"


def test_fanout_template_lists_both_routers():
    from workflow.templates import load_template_workflow

    owners = {
        item.selector.owner_agent_id
        for item in ArpoSamplingAdapter().opportunities(load_template_workflow("fanout_parallel"))
    }
    assert {"planner", "verifier", "route_a", "route_b"} <= owners
    assert "python_coder" not in owners
    assert "think" not in owners


def test_arpo_entropy_decision_and_beam_allocation():
    adapter = ArpoSamplingAdapter()
    decision = adapter.decide(
        window({"h_root": 0.1, "h_tool": 0.8}),
        site(),
        {"rng": FixedRng()},
    )
    assert decision.passed
    assert decision.reason == "arpo_official"
    assert adapter.allocate(
        decision,
        remaining=5,
        site=site(),
        default_beam=2,
    ) == 2


def test_arpo_rejects_non_arpo_gate():
    decision = ArpoSamplingAdapter().decide(
        window({"h_root": 0.1, "h_tool": 0.8}),
        site("dual_entropy"),
        {},
    )
    assert not decision.passed
    assert decision.reason == "unsupported_arpo_gate"


def test_arpo_overlay_marks_effective_strategy():
    config = {"algorithm": {"tir": {"beam_size": 2}}}
    output = ArpoSamplingAdapter().training_overlay(config, workflow().sampling)
    assert output["algorithm"]["tir"]["sampling_strategy"] == "arpo"
    assert "sampling_strategy" not in config["algorithm"]["tir"]


def test_active_set_keeps_window_and_decision_identity():
    prefix = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"name": "python_coder"}]},
        {"role": "tool", "content": "42"},
    ]
    raw = SimpleNamespace(
        error=None,
        window_events=[{
            "event_id": "window-1",
            "snapshot_ref": "archive:snapshot",
            "agent_id": "planner",
            "kind": "after_agent_turn",
            "turn": 1,
            "metrics": {"h_root": 0.1, "h_tool": 0.8},
        }],
        window_snapshots=[{"event_id": "window-1", "messages": prefix}],
        branch_messages=prefix,
        h_root=0.1,
        h_tool=0.8,
        consecutive_high=0,
    )
    session = ActiveSetSession(
        ActiveSetConfig(
            strategy="arpo",
            beam_size=2,
            sites=[BranchSite(
                id="after_planner",
                anchor=BranchAnchor(kind="after_agent_turn", agent_id="planner"),
                gate=BranchGate(type="entropy_delta"),
                fork={"beam_size": 2},
            )],
            run_probes=False,
        ),
        rng=FixedRng(),
    )
    plans = session.plan_forks_from_raw(raw, parent_id="parent", remaining=1)
    assert len(plans) == 1
    assert plans[0].meta["window_id"] == "window-1"
    assert plans[0].meta["snapshot_ref"] == "archive:snapshot"
    assert plans[0].meta["decision"]["reason"] == "arpo_official"
