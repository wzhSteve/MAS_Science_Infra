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

    rewards, valid, n_max = correct_pattern_rewards(
        rollout_ids=["a", "a", "a", "a", "b", "b"],
        planner_steps=[1, 2, 3, 4, 1, 2],
        p_satisfy=[0.2, 0.8, 0.2, 0.9, 0.1, 0.9],
        actual_satisfied=[False, True, False, True, False, True],
        threshold=0.5,
    )
    assert n_max == 2
    assert rewards == pytest.approx([0.0, 0.5, 0.0, 0.5, 0.0, 0.5])
    assert valid == [False, True, False, True, False, True]


def test_correct_pattern_from_event_chain_on_one_row():
    from rl.hooks.advantage import correct_pattern_rewards

    events = [
        {"planner_step": 1, "p_satisfy": 0.2, "actual_satisfied": False},
        {"planner_step": 2, "p_satisfy": 0.8, "actual_satisfied": True},
    ]
    rewards, valid, n_max = correct_pattern_rewards(
        rollout_ids=["only"],
        planner_steps=[2],
        p_satisfy=[0.8],
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
        "p_satisfy_list": [0.2, 0.8, 0.2],
        "actual_satisfied_list": [False, True, False],
    }
    for key, values in fields.items():
        array = np.empty(len(values), dtype=object)
        array[:] = values
        batch.non_tensor_batch[key] = array
    output = inject_failure_reallocation_rewards(batch, {"fcr_threshold": 0.5})
    assert output.batch["token_level_rewards"][0].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert output.batch["token_level_rewards"][1].tolist() == pytest.approx([1.0, 1.0, 1.0])
    assert output.batch["token_level_rewards"][2].tolist() == pytest.approx([0.0, 0.0, 0.0])
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


def test_strict_pattern_requires_previous_failure():
    from rl.hooks.advantage import correct_pattern_rewards

    rewards, valid, n_max = correct_pattern_rewards(
        rollout_ids=["a", "a"],
        planner_steps=[1, 2],
        p_satisfy=[0.2, 0.8],
        actual_satisfied=[True, True],
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
        "p_satisfy_list": [0.2, 0.8, 0.2],
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
    assert output.meta_info["fcr/nonzero_token_reward_frac"] == pytest.approx(1.0 / 3.0)
    assert output.meta_info["fcr/mean_abs_score_nonfinal"] > 0
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
