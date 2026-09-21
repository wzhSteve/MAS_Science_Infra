from workflow.active_set import ActiveSetConfig, ActiveSetSession
from workflow.contracts import BranchAnchor, BranchGate, BranchSite, SamplePolicy


def test_empty_barriers_and_sites_disable_all_branch_points():
    policy = SamplePolicy(mode="arpo", barriers=[], sites=[])
    assert policy.resolved_sites() == []


def test_default_policy_keeps_legacy_after_tool_behavior():
    policy = SamplePolicy()
    sites = policy.resolved_sites()
    assert len(sites) == 1
    assert sites[0].anchor.kind == "after_tool"


def test_explicit_entropy_gate_inherits_global_arpo_parameters():
    site = BranchSite(
        id="after_hub",
        anchor=BranchAnchor(kind="after_agent_turn", agent_id="hub"),
        gate=BranchGate(type="entropy_delta", params={"branch_probability": 0.7}),
    )
    session = ActiveSetSession(ActiveSetConfig(
        sites=[site],
        branch_probability=0.4,
        entropy_weight=0.8,
        entropy_threshold=0.25,
        use_official_arpo_gate=False,
    ))

    params = session._sites()[0].gate.params
    assert params == {
        "use_official_arpo_gate": False,
        "branch_probability": 0.7,
        "entropy_weight": 0.8,
        "entropy_threshold": 0.25,
    }
