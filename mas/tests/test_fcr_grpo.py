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


def test_fixed_k_pair_reward_and_cross_chain_rejection():
    from rl.hooks.advantage import failure_reallocation_rewards

    rows = _frozen_rows()
    rewards, valid = failure_reallocation_rewards(
        rollout_ids=[row["rollout_id"] for row in rows],
        roles=[row["role"] for row in rows],
        resume_parent_ids=[row["resume_parent_id"] for row in rows],
        event_ids=[row["event_id"] for row in rows],
        parent_event_ids=[row["parent_event_id"] for row in rows],
        recovery_chain_ids=[row["chain_id"] for row in rows],
        p_satisfy=[row["p_satisfy"] for row in rows],
        actual_satisfied=[row["actual_satisfied"] for row in rows],
        failure_confirmed=[row["failure_confirmed"] for row in rows],
        k=2,
        threshold=0.5,
    )
    assert rewards == [0.0, 0.5, 0.0, 0.0]
    assert valid == [False, True, True, False]
    both_hit = json.loads(json.dumps(rows))
    both_hit[2]["p_satisfy"] = 0.7
    both_hit[2]["actual_satisfied"] = True
    full_reward, _ = failure_reallocation_rewards(
        rollout_ids=[row["rollout_id"] for row in both_hit],
        roles=[row["role"] for row in both_hit],
        resume_parent_ids=[row["resume_parent_id"] for row in both_hit],
        event_ids=[row["event_id"] for row in both_hit],
        parent_event_ids=[row["parent_event_id"] for row in both_hit],
        recovery_chain_ids=[row["chain_id"] for row in both_hit],
        p_satisfy=[row["p_satisfy"] for row in both_hit],
        actual_satisfied=[row["actual_satisfied"] for row in both_hit],
        failure_confirmed=[row["failure_confirmed"] for row in both_hit],
        k=2,
        threshold=0.5,
    )
    assert sum(full_reward) == 1.0


def test_synthetic_batch_injects_only_valid_paired_correction():
    from tensordict import TensorDict
    from verl import DataProto
    from rl.hooks.advantage import inject_failure_reallocation_rewards

    rows = _frozen_rows()
    tensor_batch = TensorDict(
        {
            "token_level_rewards": torch.zeros((4, 3)),
            "response_mask": torch.ones((4, 3)),
        },
        batch_size=[4],
    )
    batch = DataProto(batch=tensor_batch)
    fields = {
        "rollout_id_list": [row["rollout_id"] for row in rows],
        "rollout_role": [row["role"] for row in rows],
        "resume_parent_id": [row["resume_parent_id"] for row in rows],
        "prediction_event_id_list": [row["event_id"] for row in rows],
        "parent_prediction_event_id_list": [row["parent_event_id"] for row in rows],
        "recovery_chain_id_list": [row["chain_id"] for row in rows],
        "p_satisfy_list": [row["p_satisfy"] for row in rows],
        "actual_satisfied_list": [row["actual_satisfied"] for row in rows],
        "failure_confirmed_list": [row["failure_confirmed"] for row in rows],
    }
    for key, values in fields.items():
        array = np.empty(len(values), dtype=object)
        array[:] = values
        batch.non_tensor_batch[key] = array
    output = inject_failure_reallocation_rewards(
        batch, {"fcr_k": 2, "fcr_threshold": 0.5, "fcr_lambda": 0.1}
    )
    assert output.batch["token_level_rewards"][1].tolist() == pytest.approx([0.05, 0.05, 0.05])
    assert torch.count_nonzero(output.batch["token_level_rewards"]).item() == 3
    assert output.meta_info["fcr_valid_pairs"] == 2


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
    assert cfg["algorithm"]["tir"]["fcr_lambda"] == 0.1
    assert cfg["algorithm"]["tir"]["fcr_brier_beta"] == 0.1


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


def test_fcr_caps_child_plans_at_k():
    from rl.hooks.daemon import cap_expansion_plans

    plans = [{"plan_id": i} for i in range(5)]
    assert cap_expansion_plans(plans, tir_algo="fcr_grpo", fcr_k=2) == plans[:2]
    assert cap_expansion_plans(plans, tir_algo="grpo", fcr_k=2) == plans


def test_reward_density_uses_child_response_tokens():
    from tensordict import TensorDict
    from verl import DataProto
    from rl.hooks.advantage import inject_failure_reallocation_rewards, record_reward_density

    rows = _frozen_rows()
    tensor_batch = TensorDict(
        {
            "token_level_rewards": torch.zeros((4, 3)),
            "response_mask": torch.ones((4, 3)),
        },
        batch_size=[4],
    )
    batch = DataProto(batch=tensor_batch)
    fields = {
        "rollout_id_list": [row["rollout_id"] for row in rows],
        "rollout_role": [row["role"] for row in rows],
        "resume_parent_id": [row["resume_parent_id"] for row in rows],
        "prediction_event_id_list": [row["event_id"] for row in rows],
        "parent_prediction_event_id_list": [row["parent_event_id"] for row in rows],
        "recovery_chain_id_list": [row["chain_id"] for row in rows],
        "p_satisfy_list": [row["p_satisfy"] for row in rows],
        "actual_satisfied_list": [row["actual_satisfied"] for row in rows],
        "failure_confirmed_list": [row["failure_confirmed"] for row in rows],
    }
    for key, values in fields.items():
        array = np.empty(len(values), dtype=object)
        array[:] = values
        batch.non_tensor_batch[key] = array
    output = record_reward_density(
        inject_failure_reallocation_rewards(
            batch, {"fcr_k": 2, "fcr_threshold": 0.5, "fcr_lambda": 0.1}
        )
    )
    assert output.meta_info["fcr/nonzero_token_reward_frac"] == pytest.approx(0.25)
    assert output.meta_info["fcr/mean_abs_score_nonfinal"] > 0
    assert output.meta_info["fcr/n_corr_hits"] == 1.0
