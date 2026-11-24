from __future__ import annotations

import importlib.util
from pathlib import Path


def _watch():
    path = Path(__file__).resolve().parents[2] / "scripts" / "watch_fcr_metrics.py"
    spec = importlib.util.spec_from_file_location("watch_fcr_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_watch_gates_stop_warn_pass():
    watch = _watch()
    stop, reasons, *_ = watch.evaluate_step({"fcr/leakage_count": 1, "step": 1})
    assert stop == "STOP"
    assert "leakage" in reasons[0]
    warn, warn_reasons, misses, _ = watch.evaluate_step(
        {"fcr/failure_confirmed_rate": 0.0, "fcr/prediction_coverage": 1.0, "step": 2}
    )
    assert warn == "WARN"
    assert misses == 0
    assert any("failure_confirmed_rate=0" in item for item in warn_reasons)
    first_miss, _, misses, _ = watch.evaluate_step({"fcr/prediction_coverage": 0.5, "step": 3})
    assert first_miss == "WARN"
    stop_cov, _, misses, _ = watch.evaluate_step(
        {"fcr/prediction_coverage": 0.5, "step": 4}, coverage_misses=misses
    )
    assert stop_cov == "STOP"
    passed, reasons, *_ = watch.evaluate_step(
        {
            "step": 5,
            "fcr/prediction_coverage": 1.0,
            "fcr/label_coverage": 1.0,
            "fcr/leakage_count": 0,
            "fcr/cross_chain_pair_count": 0,
            "fcr/failure_confirmed_rate": 0.2,
            "fcr/pair_valid_rate": 1.0,
            "critic/score/mean": 0.4,
        }
    )
    assert passed == "PASS"
    assert reasons == []
    budget_warn, budget_reasons, *_ = watch.evaluate_step(
        {
            "step": 6,
            "fcr/prediction_coverage": 1.0,
            "fcr/label_coverage": 1.0,
            "fcr/leakage_count": 0,
            "fcr/cross_chain_pair_count": 0,
            "fcr/budget_violation_count": 13,
            "fcr/failure_confirmed_rate": 0.2,
            "fcr/pair_valid_rate": 1.0,
            "critic/score/mean": 0.4,
        }
    )
    assert budget_warn == "STOP"
    assert any("budget_violation" in item for item in budget_reasons)
    infra_stop, infra_reasons, *_ = watch.evaluate_step(
        {"step": 7, "fcr/infra_drop_rate": 0.25, "critic/score/mean": 0.4}
    )
    assert infra_stop == "STOP"
    assert any("infra_drop_rate" in item for item in infra_reasons)


def test_watch_label_collapse_and_false_correction_gates():
    watch = _watch()
    base = {
        "fcr/prediction_coverage": 1.0,
        "fcr/leakage_count": 0,
        "fcr/cross_chain_pair_count": 0,
        "fcr/failure_confirmed_rate": 0.2,
        "fcr/pair_valid_rate": 1.0,
        "critic/score/mean": 0.4,
    }
    # <0.6 stops even with --relax-coverage (substrate death, run 9a45bbd456ad).
    collapse, reasons, *_ = watch.evaluate_step(
        {**base, "step": 6, "fcr/label_coverage": 0.37}, relax_coverage=True
    )
    assert collapse == "STOP"
    assert any("label_collapse" in item for item in reasons)
    # 0.6..0.8 under relax: WARN only on the 2nd consecutive low step.
    first, _, low_streak, _ = watch.evaluate_step(
        {**base, "step": 7, "fcr/label_coverage": 0.70}, relax_coverage=True
    )
    assert first == "WARN"
    assert low_streak == 1
    second, second_reasons, _, _ = watch.evaluate_step(
        {**base, "step": 8, "fcr/label_coverage": 0.70},
        coverage_misses=low_streak,
        relax_coverage=True,
    )
    assert second == "WARN"
    assert any("label_coverage_low_streak" in item for item in second_reasons)
    # Full recovery resets the streak (no coverage miss at all).
    recovered, _, low_streak, _ = watch.evaluate_step(
        {**base, "step": 9, "fcr/label_coverage": 1.0},
        coverage_misses=low_streak,
        relax_coverage=True,
    )
    assert recovered == "PASS"
    assert low_streak == 0
    # false_correction_rate > 0.3 warns; <= 0.3 passes.
    false_corr, false_reasons, *_ = watch.evaluate_step(
        {**base, "step": 10, "fcr/label_coverage": 1.0, "fcr/false_correction_rate": 0.44}
    )
    assert false_corr == "WARN"
    assert any("false_correction_rate" in item for item in false_reasons)
    # verifier_unlabeled_rate > 0.05 warns.
    unlabeled, unlabeled_reasons, *_ = watch.evaluate_step(
        {**base, "step": 11, "fcr/label_coverage": 0.96, "fcr/verifier_unlabeled_rate": 0.08}
    )
    assert unlabeled == "WARN"
    assert any("verifier_unlabeled_rate" in item for item in unlabeled_reasons)


def test_watch_pred_parse_and_identity_gates():
    watch = _watch()
    base = {
        "fcr/prediction_coverage": 1.0,
        "fcr/label_coverage": 1.0,
        "fcr/leakage_count": 0,
        "fcr/cross_chain_pair_count": 0,
        "fcr/failure_confirmed_rate": 0.2,
        "fcr/pair_valid_rate": 1.0,
        "critic/score/mean": 0.4,
    }
    # pred_parse_rate < 0.9 warns (leading indicator before label_coverage).
    parse_warn, parse_reasons, *_ = watch.evaluate_step(
        {**base, "step": 12, "fcr/pred_parse_rate": 0.85}
    )
    assert parse_warn == "WARN"
    assert any("pred_parse_rate" in item for item in parse_reasons)
    # parse rate exactly 0.9 or above passes.
    parse_ok, parse_ok_reasons, *_ = watch.evaluate_step(
        {**base, "step": 13, "fcr/pred_parse_rate": 0.95}
    )
    assert parse_ok == "PASS"
    assert parse_ok_reasons == []
    # score_drop_rate > 0.05 warns.
    drop_warn, drop_reasons, *_ = watch.evaluate_step(
        {**base, "step": 14, "fcr/score_drop_rate": 0.09}
    )
    assert drop_warn == "WARN"
    assert any("score_drop_rate" in item for item in drop_reasons)
    # reward_identity_gap > 1e-6 STOPs even when every other gate is clean.
    gap_stop, gap_reasons, *_ = watch.evaluate_step(
        {**base, "step": 15, "fcr/reward_identity_gap": 2e-3}
    )
    assert gap_stop == "STOP"
    assert any("reward_identity_gap" in item for item in gap_reasons)
    # A gap at or below 1e-6 passes.
    gap_ok, *_ = watch.evaluate_step(
        {**base, "step": 16, "fcr/reward_identity_gap": 1e-9}
    )
    assert gap_ok == "PASS"

