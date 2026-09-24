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
        "topology": "hub_react",
        "hub": {},
        "tools": ["execute_python"],
        "agents": [{
            "id": "hub",
            "kind": "hub",
            "tools": ["execute_python"],
        }],
        "sampling": {"mode": "arpo"},
    })


def site(gate: str = "entropy_delta", beam: int = 3):
    return BranchSite(
        id="python_result",
        anchor=BranchAnchor(
            kind="after_tool",
            agent_id="hub",
            tool_id="execute_python",
        ),
        gate=BranchGate(type=gate),
        fork={"beam_size": beam},
    )


def window(metrics=None):
    return SamplingWindow(
        window_id="window-1",
        owner_agent_id="hub",
        kind=WindowKind.TOOL_RESULT,
        sequence=1,
        snapshot_ref="archive:snapshot",
        interaction={"tool_id": "execute_python"},
        metrics=metrics or {},
    )


def test_arpo_declares_only_tool_result_opportunities():
    opportunities = list(ArpoSamplingAdapter().opportunities(workflow()))
    assert len(opportunities) == 1
    assert opportunities[0].selector.kind == WindowKind.TOOL_RESULT
    assert opportunities[0].selector.owner_agent_id == "hub"
    assert opportunities[0].selector.interaction["tool_id"] == "execute_python"
    assert opportunities[0].allowed_gates == ["entropy_delta", "arpo", "always"]


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
        {"role": "assistant", "tool_calls": [{"name": "execute_python"}]},
        {"role": "tool", "content": "42"},
    ]
    raw = SimpleNamespace(
        error=None,
        window_events=[{
            "event_id": "window-1",
            "snapshot_ref": "archive:snapshot",
            "agent_id": "hub",
            "kind": "after_tool",
            "tool_id": "execute_python",
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
            sites=[site(beam=2)],
            run_probes=False,
        ),
        rng=FixedRng(),
    )
    plans = session.plan_forks_from_raw(raw, parent_id="parent", remaining=1)
    assert len(plans) == 1
    assert plans[0].meta["window_id"] == "window-1"
    assert plans[0].meta["snapshot_ref"] == "archive:snapshot"
    assert plans[0].meta["decision"]["reason"] == "arpo_official"
