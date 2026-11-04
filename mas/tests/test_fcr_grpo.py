"""Numerical, gate, schema, and frozen-replay checks for FCR-GRPO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import numpy as np


def _frozen_rows():
    path = Path(__file__).parent / "fixtures" / "fcr_replay.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_binary_brier_has_gradient_in_correct_direction():
    from rl.fcr import binary_brier_from_scores

    satisfy = torch.tensor(-0.4, requires_grad=True)
    unsatisfy = torch.tensor(-1.2, requires_grad=True)
    loss, probability = binary_brier_from_scores(satisfy, unsatisfy, True)
    loss.backward()
    assert 0.5 < probability.item() < 1.0
    assert satisfy.grad.item() < 0
    assert unsatisfy.grad.item() > 0


def test_correct_pattern_scales_by_batch_max():
    from rl.hooks.advantage import correct_pattern_rewards

    # Hits need prev (p>=τ)≠Y then cur p>τ and Y=1 (not matched fail→success).
    rewards, valid, n_max = correct_pattern_rewards(
        rollout_ids=["a", "a", "a", "a", "b", "b"],
        planner_steps=[1, 2, 3, 4, 1, 2],
        p_satisfy=[0.8, 0.9, 0.2, 0.85, 0.7, 0.9],
        actual_satisfied=[False, True, True, True, False, True],
        threshold=0.5,
    )
    assert n_max == 2
    # Trajectory scalar on the last row: N_a=2 → 1, N_b=1 → 0.5.
    assert rewards == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0, 0.5])
    assert valid == [False, False, False, True, False, True]


def test_correct_pattern_from_event_chain_on_one_row():
    from rl.hooks.advantage import correct_pattern_rewards

    events = [
        {"planner_step": 1, "p_satisfy": 0.8, "actual_satisfied": False},
        {"planner_step": 2, "p_satisfy": 0.9, "actual_satisfied": True},
    ]
    rewards, valid, n_max = correct_pattern_rewards(
        rollout_ids=["only"],
        planner_steps=[2],
        p_satisfy=[0.9],
        actual_satisfied=[True],
        prediction_events=[events],
        threshold=0.5,
    )
    assert n_max == 1
    assert rewards == pytest.approx([1.0])
    assert valid == [True]


def test_synthetic_batch_injects_correct_pattern_plus_answer():
    from tensordict import TensorDict
    from verl import DataProto
    from rl.hooks.advantage import inject_failure_reallocation_rewards

    tensor_batch = TensorDict(
        {
            "token_level_rewards": torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "response_mask": torch.ones((3, 3)),
        },
        batch_size=[3],
    )
    batch = DataProto(batch=tensor_batch)
    fields = {
        "rollout_id_list": ["r", "r", "r"],
        "planner_step_list": [1, 2, 3],
        "p_satisfy_list": [0.8, 0.9, 0.2],
        "actual_satisfied_list": [False, True, False],
    }
    for key, values in fields.items():
        array = np.empty(len(values), dtype=object)
        array[:] = values
        batch.non_tensor_batch[key] = array
    output = inject_failure_reallocation_rewards(batch, {"fcr_threshold": 0.5})
    assert output.batch["token_level_rewards"][0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert output.batch["token_level_rewards"][1].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert output.batch["token_level_rewards"][2].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert float(output.batch["token_level_rewards"][2].sum()) == pytest.approx(1.0)
    assert output.meta_info["fcr_correction_hits"] == 1
    assert output.meta_info["fcr_correct_pattern_n_max"] == 1


def test_false_correction_counts_success_then_success():
    from rl.hooks.advantage import false_correction_stats

    events = [[
        {"planner_step": 1, "p_satisfy": 0.2, "actual_satisfied": True},
        {"planner_step": 2, "p_satisfy": 0.8, "actual_satisfied": True},
    ]]
    stats = false_correction_stats(events)
    assert stats["fcr/pattern_hit_count"] == 1.0
    assert stats["fcr/false_correction_rate"] == 1.0


def test_advantage_broadcast_std_is_zero():
    from tensordict import TensorDict
    from verl import DataProto
    from rl.hooks.advantage import record_advantage_locality

    batch = DataProto(batch=TensorDict({
        "advantages": torch.tensor([[0.2, 0.2, 0.0], [1.0, 1.0, 1.0]]),
        "response_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        "token_level_rewards": torch.tensor([[0.0, 0.5, 0.0], [1.0, 0.0, 0.0]]),
    }, batch_size=[2]))
    output = record_advantage_locality(batch)
    assert output.meta_info["fcr/adv_within_response_std"] == pytest.approx(0.0)
    assert output.meta_info["fcr/adv_within_response_max_std"] == pytest.approx(0.0)


def test_quadrants_and_new_failure_gates():
    from workflow.contracts import BranchGate
    from workflow.gates import GateContext, evaluate_gate
    from rl.hooks.daemon import prediction_quadrants

    rows = [
        {"p_satisfy": 0.2, "actual_satisfied": False, "prediction_event_id": "a", "branch_triggered": True},
        {"p_satisfy": 0.8, "actual_satisfied": False, "prediction_event_id": "b", "branch_triggered": False},
        {"p_satisfy": 0.2, "actual_satisfied": True, "prediction_event_id": "c", "branch_triggered": False},
        {"p_satisfy": 0.9, "actual_satisfied": True, "prediction_event_id": "d", "branch_triggered": False},
    ]
    stats = prediction_quadrants(rows)
    assert stats["fcr/quad_pred_fail"] == 1
    assert stats["fcr/quad_unexpected"] == 1
    assert stats["fcr/gate_precision"] == 1.0
    assert stats["fcr/gate_recall"] == 0.5
    fail = GateContext(signals={"actual_satisfied": False, "p_satisfy": 0.8, "tool_fault": False})
    assert evaluate_gate(BranchGate(type="all_fail"), fail).passed is True
    assert evaluate_gate(BranchGate(type="unexpected_fail"), fail).passed is True
    assert evaluate_gate(
        BranchGate(type="unexpected_fail"),
        GateContext(signals={"actual_satisfied": False, "p_satisfy": 0.2, "tool_fault": False}),
    ).passed is False
    assert evaluate_gate(
        BranchGate(type="all_fail"),
        GateContext(signals={"actual_satisfied": False, "p_satisfy": 0.1, "tool_fault": True}),
    ).passed is False


def test_hard_beam_cap_truncates_only_when_asked():
    from rl.hooks.daemon import cap_expansion_plans

    plans = [{"plan_id": i} for i in range(4)]
    assert cap_expansion_plans(plans, tir_algo="fcr_grpo", hard_cap=1) == plans[:1]
    assert cap_expansion_plans(plans, tir_algo="fcr_grpo") == plans


def test_matched_failure_is_not_correct_pattern():
    """p-low and Y=0 is calibrated failure, not p-vs-Y mismatch → no Correct hit."""
    from rl.hooks.advantage import correct_pattern_rewards

    rewards, valid, n_max = correct_pattern_rewards(
        rollout_ids=["a", "a"],
        planner_steps=[1, 2],
        p_satisfy=[0.2, 0.8],
        actual_satisfied=[False, True],
        threshold=0.5,
        require_prev_fail=True,
    )
    assert n_max == 0
    assert rewards == [0.0, 0.0]
    assert valid == [False, False]


def test_failure_confirmed_gate_is_not_plain_verifier_failure():
    from workflow.contracts import BranchGate
    from workflow.gates import GateContext, evaluate_gate

    gate = BranchGate(type="failure_confirmed")
    assert evaluate_gate(gate, GateContext(verifier_ok=False, signals={})).passed is False
    decision = evaluate_gate(
        gate,
        GateContext(
            verifier_ok=True,
            signals={"failure_confirmed": True, "p_satisfy": 0.2, "actual_satisfied": False},
        ),
    )
    assert decision.passed is True
    assert decision.meta["p_satisfy"] == pytest.approx(0.2)


def test_verifier_satisfaction_is_independent_of_stop_continue():
    import sys

    adapted = Path(__file__).parents[2] / "user_space" / "projects" / "simple_pev" / "adapted"
    sys.path.insert(0, str(adapted))
    try:
        from sft_prompts import parse_verifier_text

        stopped_failure = parse_verifier_text(
            "Conclusion: STOP\nSatisfied: NO\nReason: budget\nObtained: NONE"
        )
        continuing_success = parse_verifier_text(
            "Conclusion: CONTINUE\nSatisfied: YES\nReason: subgoal done\nObtained: fact"
        )
    finally:
        sys.path.remove(str(adapted))
    assert stopped_failure["conclusion"] == "STOP"
    assert stopped_failure["satisfied"] == "NO"
    assert continuing_success["conclusion"] == "CONTINUE"
    assert continuing_success["satisfied"] == "YES"


def test_prediction_before_execution_is_not_leakage():
    from rl.hooks.daemon import binary_prediction_metrics

    rows = [{
        "prediction_event_id": "e1",
        "p_satisfy": 0.2,
        "actual_satisfied": False,
        "prediction_mismatch": False,
        "predicted_at": 1.0,
        "executed_at": 2.0,
    }]
    metrics = binary_prediction_metrics(rows)
    assert metrics["prediction_coverage"] == 1.0
    assert metrics["label_coverage"] == 1.0


def test_missing_pair_fields_are_noop():
    from tensordict import TensorDict
    from verl import DataProto
    from rl.hooks.advantage import inject_failure_reallocation_rewards, reset_pipeline_noop_counter

    reset_pipeline_noop_counter()
    batch = DataProto(batch=TensorDict({"token_level_rewards": torch.zeros((1, 2))}, batch_size=[1]))
    output = inject_failure_reallocation_rewards(batch, {"fcr_lambda": 0.1})
    assert torch.equal(output.batch["token_level_rewards"], torch.zeros((1, 2)))
    from rl.hooks.advantage import pipeline_noop_count

    assert pipeline_noop_count() >= 1


def test_fcr_preset_and_overlay_register_failure_confirmed_site():
    from rl.hooks.overlay import apply_algo_overlay
    from rl.loss import expand_preset

    spec = expand_preset("fcr_grpo")
    assert spec.base == "grpo"
    assert [op.id for op in spec.operators] == ["failure_reallocation"]
    cfg = apply_algo_overlay(
        {"actor_rollout_ref": {"rollout": {"n": 4}, "actor": {}}, "algorithm": {}, "trainer": {}},
        "fcr_grpo",
    )
    sites = cfg["algorithm"]["tir"]["sites"]
    assert sites
    assert sites[0]["gate"]["type"] == "failure_confirmed"
    assert cfg["algorithm"]["tir"]["fcr_k"] == 2
    assert cfg["algorithm"]["tir"]["max_branch_depth"] == 2
    assert cfg["algorithm"]["tir"]["initial_rollouts"] == 2
    assert cfg["algorithm"]["tir"]["beam_size"] == 2


def test_prediction_metrics_frozen_replay():
    from rl.hooks.daemon import binary_prediction_metrics

    rows = [
        {
            "prediction_event_id": row["event_id"],
            "p_satisfy": row["p_satisfy"],
            "actual_satisfied": row["actual_satisfied"],
            "prediction_mismatch": False,
        }
        for row in _frozen_rows()
    ]
    metrics = binary_prediction_metrics(rows)
    assert metrics["prediction_coverage"] == 1.0
    assert metrics["label_coverage"] == 1.0
    assert 0.0 <= metrics["brier"] <= 1.0


def test_fcr_does_not_cap_plans_beyond_arpo_fill():
    from rl.hooks.daemon import cap_expansion_plans

    plans = [{"plan_id": i} for i in range(5)]
    assert cap_expansion_plans(plans, tir_algo="fcr_grpo", fcr_k=2) == plans
    assert cap_expansion_plans(plans, tir_algo="arpo", fcr_k=2) == plans


def test_reward_density_uses_pattern_tokens():
    from tensordict import TensorDict
    from verl import DataProto
    from rl.hooks.advantage import inject_failure_reallocation_rewards, record_reward_density

    tensor_batch = TensorDict(
        {
            "token_level_rewards": torch.zeros((3, 3)),
            "response_mask": torch.ones((3, 3)),
        },
        batch_size=[3],
    )
    batch = DataProto(batch=tensor_batch)
    fields = {
        "rollout_id_list": ["r", "r", "r"],
        "rollout_role": ["parent", "parent", "parent"],
        "planner_step_list": [1, 2, 3],
        "p_satisfy_list": [0.8, 0.9, 0.2],
        "actual_satisfied_list": [False, True, False],
        "data_id_list": ["g", "g", "g"],
    }
    for key, values in fields.items():
        array = np.empty(len(values), dtype=object)
        array[:] = values
        batch.non_tensor_batch[key] = array
    output = record_reward_density(
        inject_failure_reallocation_rewards(batch, {"fcr_threshold": 0.5})
    )
    assert output.meta_info["fcr/nonzero_token_reward_frac"] == pytest.approx(1.0 / 9.0)
    assert float(output.batch["token_level_rewards"].sum()) == pytest.approx(1.0)
    assert output.meta_info["fcr/n_corr_hits"] == 1.0


def test_qwen3_yes_no_are_single_tokens():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B",
        trust_remote_code=True,
    )
    yes_ids = tok.encode("YES", add_special_tokens=False)
    no_ids = tok.encode("NO", add_special_tokens=False)
    satisfy_ids = tok.encode("满足", add_special_tokens=False)
    unsatisfy_ids = tok.encode("不满足", add_special_tokens=False)
    assert len(yes_ids) == 1
    assert len(no_ids) == 1
    assert len(yes_ids) / len(no_ids) == 1.0
    assert len(satisfy_ids) == 1
    assert len(unsatisfy_ids) == 2


def test_bce_recovers_harder_than_brier_at_extreme_p():
    from rl.fcr import binary_bce_from_scores, binary_brier_from_scores

    # Craft scores so p ≈ sigmoid(z) is tiny with Y=1.
    satisfy = torch.tensor(-8.0, requires_grad=True)
    unsatisfy = torch.tensor(0.0, requires_grad=True)
    brier, p_b = binary_brier_from_scores(satisfy, unsatisfy, True)
    brier.backward()
    brier_g = abs(satisfy.grad.item())
    satisfy.grad = None
    unsatisfy.grad = None
    satisfy2 = torch.tensor(-8.0, requires_grad=True)
    unsatisfy2 = torch.tensor(0.0, requires_grad=True)
    bce, p_c = binary_bce_from_scores(satisfy2, unsatisfy2, True)
    bce.backward()
    bce_g = abs(satisfy2.grad.item())
    assert p_b.item() < 0.01
    assert bce_g > 10 * brier_g


def test_persistent_replay_samples_both_classes_when_batch_all_neg():
    from rl.fcr import PersistentPredReplay, PredReplayItem

    buf = PersistentPredReplay(capacity=32)
    for i in range(10):
        buf.add(PredReplayItem([i], [1], [2], label=False))
    buf.add(PredReplayItem([99], [1], [2], label=True))
    sample = buf.sample_balanced(k_pos=2, k_neg=2)
    assert any(item.label for item in sample)
    assert any(not item.label for item in sample)


def test_budget_violation_counts_unique_rollouts_not_turns():
    """Simulate the fixed aggregation: many turns, one child rollout → 0 violation."""
    roles = ["parent", "child", "child", "child"]  # 3 turn rows, same child rid
    parent_ids = ["", "p1", "p1", "p1"]
    rollout_ids = ["p1", "c1", "c1", "c1"]
    children_by_parent = {}
    for role, parent, rid in zip(roles, parent_ids, rollout_ids):
        if role in ("child", "probe", "branch") and parent:
            children_by_parent.setdefault(parent, set()).add(rid)
    beam = 2
    per_parent_cap = max(1, beam - 1)
    violation = sum(max(0, len(ids) - per_parent_cap) for ids in children_by_parent.values())
    assert violation == 0
    # Two distinct child rollouts → violation 1
    children_by_parent["p1"].add("c2")
    violation = sum(max(0, len(ids) - per_parent_cap) for ids in children_by_parent.values())
    assert violation == 1


def test_branch_repair_rewards_parent_fail_child_success():
    from rl.hooks.advantage import branch_repair_rewards

    credits, hits = branch_repair_rewards(
        rollout_ids=["p", "c"],
        roles=["parent", "child"],
        parent_ids=["", "p"],
        actual_satisfied=[False, True],
    )
    assert hits == 1
    assert credits == pytest.approx([0.0, 1.0])


def test_failure_or_uncertainty_gate():
    from workflow.contracts import BranchGate
    from workflow.gates import GateContext, evaluate_gate

    gate = BranchGate(type="failure_or_uncertainty", params={"unc_lo": 0.2, "unc_hi": 0.8})
    assert evaluate_gate(
        gate,
        GateContext(signals={"failure_confirmed": False, "p_satisfy": 0.5, "actual_satisfied": True}),
    ).passed is True
    assert evaluate_gate(
        gate,
        GateContext(signals={"failure_confirmed": False, "p_satisfy": 0.05, "actual_satisfied": True}),
    ).passed is False
    assert evaluate_gate(
        gate,
        GateContext(signals={"failure_confirmed": True, "p_satisfy": 0.05, "actual_satisfied": False}),
    ).passed is True


def test_correct_reward_normalizes_inside_question_not_batch():
    from rl.hooks.advantage import correct_pattern_rewards

    rewards, valid, n_max = correct_pattern_rewards(
        rollout_ids=["a1", "a1", "b1", "b1"],
        planner_steps=[1, 2, 1, 2],
        p_satisfy=[0.8, 0.9, 0.7, 0.9],
        actual_satisfied=[False, True, False, True],
        group_ids=["q1", "q1", "q2", "q2"],
        threshold=0.5,
        require_prev_fail=True,
    )
    assert n_max == 1
    assert rewards == pytest.approx([0.0, 1.0, 0.0, 1.0])
    assert valid == [False, True, False, True]


def test_require_prev_fail_flag_does_not_change_correct_pattern():
    from rl.hooks.advantage import correct_pattern_rewards

    # False-alarm then success is a design Correct hit either way.
    kwargs = dict(
        rollout_ids=["a", "a"],
        planner_steps=[1, 2],
        p_satisfy=[0.2, 0.8],
        actual_satisfied=[True, True],
        threshold=0.5,
    )
    weak, _, n_weak = correct_pattern_rewards(**kwargs, require_prev_fail=False)
    strict, _, n_strict = correct_pattern_rewards(**kwargs, require_prev_fail=True)
    assert n_weak == 1 and weak[1] == pytest.approx(1.0)
    assert n_strict == 1 and strict[1] == pytest.approx(1.0)


def test_failure_confirmed_does_not_fire_when_prediction_is_positive():
    from workflow.contracts import BranchGate
    from workflow.gates import GateContext, evaluate_gate

    high = GateContext(signals={
        "failure_confirmed": False, "p_satisfy": 0.8, "actual_satisfied": False, "tool_fault": False,
    })
    assert evaluate_gate(BranchGate(type="failure_confirmed"), high).passed is False
    assert evaluate_gate(BranchGate(type="all_fail"), high).passed is True


def test_random_gate_is_execution_failure_pool_and_cap_samples():
    from workflow.contracts import BranchGate
    from workflow.gates import GateContext, evaluate_gate
    from rl.hooks.daemon import cap_expansion_plans

    gate = BranchGate(type="random")
    fail = GateContext(signals={"actual_satisfied": False, "tool_fault": False, "p_satisfy": 0.9})
    ok = GateContext(signals={"actual_satisfied": True, "tool_fault": False, "p_satisfy": 0.1})
    fault = GateContext(signals={"actual_satisfied": False, "tool_fault": True, "p_satisfy": 0.1})
    assert evaluate_gate(gate, fail).passed is True
    assert evaluate_gate(gate, ok).passed is False
    assert evaluate_gate(gate, fault).passed is False
    plans = [{"plan_id": i} for i in range(4)]
    picked = cap_expansion_plans(plans, tir_algo="fcr_grpo", hard_cap=1, sample="random", seed=1)
    assert len(picked) == 1
    assert picked[0]["plan_id"] in {0, 1, 2, 3}
    assert cap_expansion_plans(plans, tir_algo="fcr_grpo", hard_cap=1, sample="prefix") == plans[:1]


def test_event_audit_separates_recovery_sources():
    from scripts.audit_fcr_events import audit_records

    rows = [{
        "events": [
            {"recovery_chain_id": "c", "planner_step": 1, "p_satisfy": 0.2, "actual_satisfied": False, "command": "a"},
            {"recovery_chain_id": "c", "planner_step": 2, "p_satisfy": 0.8, "actual_satisfied": True, "command": "b"},
        ],
    }]
    stats = audit_records(rows)
    assert stats["strict_adjacent"]["num"] == 1
    assert stats["execution_recovery"]["num"] == 1
    assert stats["prediction_recovery"]["num"] == 1
    assert stats["lineage"]["count"] == 1
    assert stats["sparsity_source"] == "events_present"


def test_bss_in_binary_prediction_metrics():
    from rl.hooks.daemon import binary_prediction_metrics

    rows = [
        {"prediction_event_id": "a", "p_satisfy": 0.0, "actual_satisfied": False},
        {"prediction_event_id": "b", "p_satisfy": 0.0, "actual_satisfied": False},
        {"prediction_event_id": "c", "p_satisfy": 0.0, "actual_satisfied": True},
    ]
    m = binary_prediction_metrics(rows)
    assert "bss" in m and "bs_constant" in m and "mean_p_given_pos" in m
    assert m["accuracy"] > 0.5
    assert m["bss"] <= 0.0 + 1e-6  # constant-NO is not better than base-rate constant
