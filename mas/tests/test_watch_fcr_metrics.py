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
    assert budget_warn == "WARN"
    assert any("budget_violation" in item for item in budget_reasons)
    infra_stop, infra_reasons, *_ = watch.evaluate_step(
        {"step": 7, "fcr/infra_drop_rate": 0.25, "critic/score/mean": 0.4}
    )
    assert infra_stop == "STOP"
    assert any("infra_drop_rate" in item for item in infra_reasons)
