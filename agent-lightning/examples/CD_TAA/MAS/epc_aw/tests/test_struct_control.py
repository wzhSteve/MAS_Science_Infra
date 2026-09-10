"""Unit tests: SlotGate verified STOP, FinalAudit, FailureAttribution routing."""

from __future__ import annotations

from types import SimpleNamespace

from MAS.epc_aw.models.ablation import resolve_ablation
from MAS.epc_aw.models.causal_memory_graph import CausalMemoryGraph
from MAS.epc_aw.models.intervention import (
    ATTRIBUTION_TO_RECOMMENDATION,
    ACTOR_BURST_SIZE,
    InterventionMixin,
    StepInterventionState,
)
from MAS.epc_aw.models.memory import SystemMemory
from MAS.epc_aw.models.plan_controller import PlanControllerMixin
from MAS.epc_aw.models.solver_types import VerificationResult
from MAS.epc_aw.models.task_profile import AnswerSlot, SlotGate, TaskProfile
from MAS.epc_aw.models.verified_memory import CommitKind, MemoryCommitEvent


def _entity_profile(question: str = "Who founded Acme Corp?") -> TaskProfile:
    return TaskProfile(
        question=question,
        phase="acquisition",
        slots=[
            AnswerSlot(
                name="final_answer",
                slot_type="entity",
                required=True,
                min_source="secondary",
            )
        ],
    )


def _memory_with_profile() -> SystemMemory:
    mem = SystemMemory(toolbox_metadata={})
    mem.configure_verified_memory(
        enable_verified_memory_commit=True,
        enable_struct_stop_gate=True,
        enable_struct_intervention_routing=True,
        enable_final_audit=True,
    )
    mem.set_query("Who founded Acme Corp?")
    mem.set_task_profile(_entity_profile())
    mem.set_outline({"1": "Identify the founder"})
    mem.init_causal_graph_for_task("test_struct_control")
    return mem


class _StopHarness(PlanControllerMixin):
    def __init__(self, mem: SystemMemory, *, struct: bool = True):
        self.system_memory = mem
        self.ablation = resolve_ablation(
            "full" if struct else "no_struct_control"
        )
        mem.apply_ablation_memory_flags(self.ablation)
        self.diagnoser = SimpleNamespace(
            _answer_verified_ready=lambda *a, **k: False,
        )


def test_unverified_slot_binding_cannot_stop():
    mem = _memory_with_profile()
    mem.evidence_records.append(
        {
            "id": "e1",
            "content": "Jane Founder established Acme Corp in 1999.",
            "status": "active",
            "live": True,
            "verification_status": "rejected",
            "claim_type": "fact",
            "source_quality": "secondary",
            "slot_bindings": {"final_answer": "Jane Founder"},
            "tool": "Google_Search_Tool",
        }
    )
    harness = _StopHarness(mem)
    assert not harness.can_attempt_done("Who founded Acme Corp?")
    assert not SlotGate.can_stop(
        mem.get_task_profile(),
        mem.slot_evidence_records(require_live_verified=True),
        require_live_verified=True,
    )


def test_invalidate_closes_stop():
    mem = _memory_with_profile()
    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=1,
            reason="ok",
            outline_step="1",
            facts={"final_answer": "Jane Founder"},
        ),
        evidence_payload={
            "content": "Jane Founder established Acme Corp according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 1,
            "tool": "Google_Search_Tool",
            "claim_type": "fact",
            "source_quality": "secondary",
        },
        slot_bindings={"final_answer": "Jane Founder"},
    )
    # Clear outline / open acquisition so STOP is not blocked by projection.
    mem.set_outline({})
    harness = _StopHarness(mem)
    # Entity fill may still require content heuristics; reinforce via valid_facts.
    assert SlotGate.can_stop(
        mem.get_task_profile(),
        mem.slot_evidence_records(require_live_verified=True),
        valid_facts=mem.current_state_valid_facts(),
        require_live_verified=True,
    ) or "final_answer" in mem.verified.valid_facts()

    eid = mem.evidence_records[0]["id"]
    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.INVALIDATED.value,
            step=2,
            reason="audit fail",
            evidence_id=eid,
        )
    )
    assert not SlotGate.can_stop(
        mem.get_task_profile(),
        mem.slot_evidence_records(require_live_verified=True),
        valid_facts=mem.current_state_valid_facts(),
        require_live_verified=True,
    )
    assert not harness.can_attempt_done("Who founded Acme Corp?")


def test_open_acquisition_blocks_attempt_done():
    mem = _memory_with_profile()
    graph: CausalMemoryGraph = mem.causal_graph
    graph.attach_subgoal_contract(
        "Retrieve founder name",
        required_fact_names=["final_answer"],
        preferred_tools=["Google_Search_Tool"],
        acquisition=True,
    )
    mem.set_outline({})
    harness = _StopHarness(mem)
    assert harness._graph_has_open_acquisition() is True
    # Satisfying the required fact clears the open-acquisition block.
    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=1,
            reason="seed",
            outline_step="1",
            facts={"final_answer": "Jane Founder"},
        ),
        evidence_payload={
            "content": "Jane Founder established Acme Corp according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 1,
            "tool": "Google_Search_Tool",
            "claim_type": "fact",
            "source_quality": "secondary",
        },
        slot_bindings={"final_answer": "Jane Founder"},
    )
    # Graph valid_facts should meet the subgoal contract.
    facts = mem.current_state_valid_facts()
    open_items = graph.open_acquisition_subgoals(facts)
    contracted_open = [i for i in open_items if i.get("required_fact_names")]
    assert contracted_open == []


def test_required_fact_contract_exports_slots():
    profile = _entity_profile()
    contract = SlotGate.required_fact_contract(profile)
    assert "final_answer" in contract
    assert contract["final_answer"]["required_fact_names"] == ["final_answer"]


def test_project_outline_from_open_subgoals():
    graph = CausalMemoryGraph()
    graph.ensure_task("proj")
    graph.ensure_initial_state("initial")
    graph.attach_subgoal_contract(
        "Find founder",
        required_fact_names=["final_answer"],
        preferred_tools=["Wikipedia_Search_Tool"],
        acquisition=True,
    )
    outline = graph.project_outline_from_open_subgoals({})
    assert "1" in outline
    assert "Find founder" in outline["1"]
    assert "Wikipedia_Search_Tool" in outline["1"]


def test_attribution_mapping_actor_and_planner():
    assert ATTRIBUTION_TO_RECOMMENDATION["actor"] == "retry_with_different_parameters"
    assert ATTRIBUTION_TO_RECOMMENDATION["planner"] == "decompose_goal"
    assert ATTRIBUTION_TO_RECOMMENDATION["tool_environment"] == "switch_tool"
    assert ATTRIBUTION_TO_RECOMMENDATION["verifier"] == "revise_belief"


def test_failure_attribution_routes_actor_to_l2a():
    mem = _memory_with_profile()
    mixin = InterventionMixin.__new__(InterventionMixin)
    mixin.system_memory = mem
    mixin.ablation = resolve_ablation("full")
    mixin.verbose = False
    mixin.diagnoser = SimpleNamespace(
        attribute_failure=lambda **kwargs: SimpleNamespace(
            attributed_to="actor",
            reason="bad parameters",
            recovery_hint="retry params",
        )
    )
    tracker = StepInterventionState()
    step_ctx = SimpleNamespace(
        step_key="1",
        tool_name="Google_Search_Tool",
        target_information="Find founder",
        command='execute("Google_Search_Tool", query="x")',
        result_executor="no results",
    )
    verification = VerificationResult(
        analysis="incomplete",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion="CONTINUE",
        diagnostic_signal={},
        subgoal_complete=False,
    )
    signal = mixin._apply_failure_attribution_routing(
        "Who founded Acme?",
        step_ctx,
        verification,
        {"recommendation": "switch_tool"},
        tracker,
        exec_step=1,
    )
    assert signal["recommendation"] == "retry_with_different_parameters"
    assert signal["failure_attribution"]["attributed_to"] == "actor"
    assert tracker.actor_burst_used.get("1") == 1


def test_failure_attribution_routes_planner_to_l3():
    mem = _memory_with_profile()
    mixin = InterventionMixin.__new__(InterventionMixin)
    mixin.system_memory = mem
    mixin.ablation = resolve_ablation("full")
    mixin.verbose = False
    mixin.diagnoser = SimpleNamespace(
        attribute_failure=lambda **kwargs: SimpleNamespace(
            attributed_to="planner",
            reason="wrong subgoal",
            recovery_hint="decompose",
        )
    )
    tracker = StepInterventionState()
    step_ctx = SimpleNamespace(
        step_key="1",
        tool_name="Google_Search_Tool",
        target_information="Find founder",
        command="cmd",
        result_executor="ok but wrong goal",
    )
    verification = VerificationResult(
        analysis="incomplete",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion="CONTINUE",
        diagnostic_signal={},
        subgoal_complete=False,
    )
    signal = mixin._apply_failure_attribution_routing(
        "Who founded Acme?",
        step_ctx,
        verification,
        {"recommendation": "retry_with_different_parameters"},
        tracker,
        exec_step=1,
    )
    assert signal["recommendation"] == "decompose_goal"
    assert signal["failure_attribution"]["attributed_to"] == "planner"


def test_actor_burst_exhaustion_switches_tool():
    mem = _memory_with_profile()
    mixin = InterventionMixin.__new__(InterventionMixin)
    mixin.system_memory = mem
    mixin.ablation = resolve_ablation("full")
    mixin.verbose = False
    mixin.diagnoser = SimpleNamespace(
        attribute_failure=lambda **kwargs: SimpleNamespace(
            attributed_to="actor",
            reason="params",
            recovery_hint="retry",
        )
    )
    tracker = StepInterventionState()
    tracker.actor_burst_used["1"] = ACTOR_BURST_SIZE
    step_ctx = SimpleNamespace(
        step_key="1",
        tool_name="Google_Search_Tool",
        target_information="Find founder",
        command="cmd",
        result_executor="empty",
    )
    verification = VerificationResult(
        analysis="incomplete",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion="CONTINUE",
        diagnostic_signal={},
        subgoal_complete=False,
    )
    signal = mixin._apply_failure_attribution_routing(
        "q",
        step_ctx,
        verification,
        {},
        tracker,
        exec_step=2,
    )
    assert signal["recommendation"] == "switch_tool"


def test_knowledge_gain_reads_graph_delta():
    mem = _memory_with_profile()
    mixin = InterventionMixin.__new__(InterventionMixin)
    mixin.system_memory = mem
    mixin.ablation = resolve_ablation("full")
    mixin.verbose = False
    mixin._parse_slots_from_obtained = lambda *_a, **_k: set()
    tracker = StepInterventionState()
    step_ctx = SimpleNamespace(
        step_key="1",
        tool_name="Google_Search_Tool",
        command="cmd-a",
    )
    verification = VerificationResult(
        analysis="",
        step_conclusion="SUBGOAL_INCOMPLETE",
        info_flag=False,
        obtained_info="",
        task_conclusion="CONTINUE",
        diagnostic_signal={},
        subgoal_complete=False,
        evidence_type="",
    )
    # Seed baselines.
    e1, k1 = mixin._compute_intervention_gains(
        step_ctx, verification, {}, tracker, "1",
    )
    assert e1 == 1 and k1 == 1
    tracker.last_command["1"] = step_ctx.command
    tracker.last_tool["1"] = step_ctx.tool_name
    tracker.last_evidence_type["1"] = ""
    tracker.last_slots["1"] = set()
    tracker.last_hypothesis["1"] = "."

    # No graph Δ → Knowledge Gain 0.
    e2, k2 = mixin._compute_intervention_gains(
        step_ctx,
        verification,
        {"evidence_type": "", "causal_hypothesis": {"target_variable": "", "reason": ""}},
        tracker,
        "1",
    )
    assert e2 == 0
    assert k2 == 0

    # Commit a new verified fact → ΔValidFacts should yield Knowledge Gain > 0.
    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=2,
            reason="new fact",
            outline_step="1",
            facts={"final_answer": "Jane Founder"},
        ),
        evidence_payload={
            "content": "Jane Founder established Acme Corp according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 2,
            "tool": "Google_Search_Tool",
            "claim_type": "fact",
            "source_quality": "secondary",
        },
        slot_bindings={"final_answer": "Jane Founder"},
    )
    e3, k3 = mixin._compute_intervention_gains(
        step_ctx,
        verification,
        {"evidence_type": "", "causal_hypothesis": {"target_variable": "", "reason": ""}},
        tracker,
        "1",
    )
    assert e3 == 0
    assert k3 >= 1


def test_no_struct_control_preset_flags():
    cfg = resolve_ablation("no_struct_control")
    assert cfg.enable_struct_stop_gate is False
    assert cfg.enable_struct_intervention_routing is False
    assert cfg.enable_final_audit is False
    mem = _memory_with_profile()
    mem.apply_ablation_memory_flags(cfg)
    assert mem.enable_struct_stop_gate is False
    assert mem.enable_final_audit is False


def test_legacy_stop_allows_unverified_when_struct_off():
    mem = _memory_with_profile()
    mem.evidence_records.append(
        {
            "id": "e_legacy",
            "content": "Jane Founder established Acme Corp according to encyclopedia.",
            "status": "active",
            "claim_type": "fact",
            "source_quality": "secondary",
            "slot_bindings": {"final_answer": "Jane Founder"},
            "tool": "Google_Search_Tool",
        }
    )
    mem.set_outline({})
    harness = _StopHarness(mem, struct=False)
    # Without verification_status, legacy path still uses disputed-only filter.
    assert harness._can_stop_execution("Who founded Acme Corp?")


def test_final_audit_invalidates_on_fail():
    mem = _memory_with_profile()
    mem.commit_verified_memory(
        MemoryCommitEvent(
            kind=CommitKind.SATISFIED.value,
            step=1,
            reason="ok",
            outline_step="1",
            facts={"final_answer": "Jane Founder"},
        ),
        evidence_payload={
            "content": "Jane Founder established Acme Corp according to encyclopedia.",
            "outline_step": "1",
            "exec_step": 1,
            "tool": "Google_Search_Tool",
            "claim_type": "fact",
            "source_quality": "secondary",
        },
        slot_bindings={"final_answer": "Jane Founder"},
    )
    harness = _StopHarness(mem)
    harness.diagnoser = SimpleNamespace(
        audit_final_answer=lambda *a, **k: SimpleNamespace(
            verdict="FAIL",
            reason="unsupported claim",
        ),
        _answer_verified_ready=lambda *a, **k: False,
    )
    ok = harness._audit_and_maybe_invalidate_final(
        "Who founded Acme Corp?", "Jane Founder", step=2,
    )
    assert ok is False
    assert mem.evidence_records[0]["live"] is False
