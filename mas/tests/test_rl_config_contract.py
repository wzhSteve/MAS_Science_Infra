from __future__ import annotations

from science_infra.control.experiments import apply_gpu_selection


def test_gpu_selection_preserves_device_and_trainer_extensions():
    config = {
        "profile": "a800_2gpu",
        "rollout_per_gpu": 4,
        "devices": {"ids": [3], "placement": "exclusive"},
        "trainer": {"n_gpus_per_node": 1, "logger": ["console"]},
        "actor_rollout_ref": {"rollout": {"n": 2, "name": "vllm"}},
        "extension": {"keep": True},
    }

    updated = apply_gpu_selection(config, [0, 1])

    assert updated["devices"] == {"ids": [0, 1], "placement": "exclusive"}
    assert updated["trainer"] == {"n_gpus_per_node": 2, "logger": ["console"]}
    assert updated["actor_rollout_ref"]["rollout"] == {"n": 4, "name": "vllm"}
    assert updated["extension"] == {"keep": True}


def test_single_gpu_downgrades_two_gpu_profile_without_mutating_input():
    config = {
        "profile": "a800_2gpu",
        "devices": {"ids": [0, 1]},
        "trainer": {"n_gpus_per_node": 2},
    }

    updated = apply_gpu_selection(config, [1])

    assert updated["profile"] == "fast"
    assert updated["_profile_downgraded"] == "a800_2gpu→fast (need 2 GPUs)"
    assert updated["devices"]["ids"] == [1]
    assert config["profile"] == "a800_2gpu"
    assert config["devices"]["ids"] == [0, 1]


def test_ensure_trainer_horizon_raises_epochs_to_cover_steps():
    from rl.hooks.overlay import ensure_trainer_horizon

    cfg = ensure_trainer_horizon({"trainer": {"total_epochs": 1, "total_training_steps": 3}})
    assert cfg["trainer"]["total_epochs"] == 3
    assert cfg["trainer"]["total_training_steps"] == 3
    untouched = ensure_trainer_horizon({"trainer": {"total_epochs": 4, "total_training_steps": 3}})
    assert untouched["trainer"]["total_epochs"] == 4
