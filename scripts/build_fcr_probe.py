#!/usr/bin/env python3
"""Slice the shared 64/32 prompt files and write the FCR probe yamls."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "rl_datasets" / "train_10k.parquet"
VALID = ROOT / "data" / "rl_datasets" / "valid.parquet"
OUT = ROOT / "data" / "rl_datasets"
SPEC = ROOT / "experiments" / "fcr_short"
CKPT = ROOT / "mas" / "checkpoints" / "AgentLightning"


def _slice(src: Path, dest: Path, n: int) -> None:
    frame = pd.read_parquet(src)
    take = min(n, len(frame))
    frame.sample(n=take, random_state=42).reset_index(drop=True).to_parquet(dest, index=False)
    print(f"wrote {dest} rows={take}", flush=True)


def _site(gate: str) -> list:
    return [{
        "id": "fcr_gate",
        "enabled": True,
        "anchor": {"kind": "after_verifier"},
        "when": "every",
        "gate": {"type": gate, "params": {}},
        "fork": {
            "beam_size": 2,
            "share_observation": True,
            "resume_mode": "messages",
            "probe_max_tokens": 128,
        },
        "reward": {"scheme": "scalar_grpo", "p_plus": 0.8, "k_min": 2, "epsilon_f": 0.001},
        "priority": 100,
    }]


def _yaml(
    name: str,
    *,
    algo: str,
    steps: int,
    train: Path,
    gate: str | None,
    pattern: bool,
    brier: bool,
    mode: str = "fixed",
    beta: float = 0.1,
    require_prev_fail: bool = False,
    hard_cap: bool = False,
    env: dict | None = None,
    val: Path | None = None,
    test_freq: int = 0,
) -> None:
    tir = {
        "fcr_k": 2,
        "fcr_threshold": 0.5,
        "fcr_lambda": 0.0,
        "fcr_pattern_enabled": pattern,
        "fcr_pattern_require_prev_fail": require_prev_fail,
        "fcr_brier_enabled": brier,
        "fcr_brier_beta": beta if brier else 0.0,
        "fcr_brier_mode": mode,
        "fcr_hard_beam_cap": hard_cap,
        "initial_rollouts": 2,
        "max_branch_depth": 2,
        "beam_size": 2,
        "ready_batch": True,
        "sampling_strategy": algo,
    }
    if gate:
        tir["sites"] = _site(gate)
    elif algo == "fcr_grpo":
        tir["sites"] = []
    doc = {
        "profile": "a800_2gpu",
        "algo": algo,
        "n_runners": 2,
        "rollout_per_gpu": 2,
        "devices": {"ids": [0, 1]},
        "model_path": str(ROOT / "LLM" / "Qwen3-4B"),
        "algorithm": {
            "adv_estimator": "grpo",
            "use_kl_in_reward": False,
            "tir_algo": algo,
            "tir": tir,
        },
        "data": {
            "train_files": str(train),
            "val_files": str(val or (OUT / "fcr_probe_val32.parquet")),
            "shuffle": True,
            "seed": 42,
            "train_batch_size": 2,
            "max_prompt_length": 6144,
            "max_response_length": 4096,
            "truncation": "left",
        },
        "actor_rollout_ref": {
            "rollout": {
                "tensor_model_parallel_size": 1,
                "n": 4,
                "log_prob_micro_batch_size_per_gpu": 1,
                "name": "vllm",
                "gpu_memory_utilization": 0.35,
                "engine_kwargs": {"vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                    "max_model_len": 10240,
                    "served_model_name": "LLM/Qwen3-4B",
                }},
            },
            "actor": {
                "ppo_mini_batch_size": 1,
                "ppo_micro_batch_size_per_gpu": 1,
                "optim": {"lr": 1.0e-6},
                "entropy_coeff": 0,
                "fsdp_config": {"param_offload": True, "optimizer_offload": True},
            },
            "ref": {
                "log_prob_micro_batch_size_per_gpu": 1,
                "fsdp_config": {"param_offload": True},
            },
            "model": {
                "path": str(ROOT / "LLM" / "Qwen3-4B"),
                "use_remove_padding": True,
                "enable_gradient_checkpointing": True,
            },
        },
        "trainer": {
            "n_gpus_per_node": 2,
            "val_before_train": False,
            "critic_warmup": 0,
            "logger": ["console", "tensorboard"],
            "project_name": "AgentLightning",
            "experiment_name": name,
            "default_local_dir": str(CKPT / name),
            "nnodes": 1,
            "test_freq": test_freq,
            "total_epochs": 1,
            "total_training_steps": steps,
            "save_freq": 1000,
            "max_actor_ckpt_to_keep": 1,
            "resume_mode": "disable",
        },
    }
    if env:
        doc["env"] = env
    path = SPEC / f"{name}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def main() -> None:
    train64 = OUT / "fcr_probe_train64.parquet"
    train8 = OUT / "fcr_probe_train8.parquet"
    val32 = OUT / "fcr_probe_val32.parquet"
    val64 = OUT / "fcr_probe_val64.parquet"
    _slice(TRAIN, train64, 64)
    _slice(TRAIN, train8, 8)
    _slice(VALID, val32, 32)
    _slice(VALID, val64, 64)
    snap = SPEC / "snapshots" / "fcr_p1_pred_fail"
    _yaml(
        "fcr_p0_diag", algo="fcr_grpo", steps=1, train=train8, gate="failure_confirmed",
        pattern=True, brier=False,
    )
    for name, algo, gate in (
        ("fcr_p1_none", "grpo", None),
        ("fcr_p1_all_fail", "fcr_grpo", "all_fail"),
        ("fcr_p1_pred_fail", "fcr_grpo", "failure_confirmed"),
        ("fcr_p1_unexpected", "fcr_grpo", "unexpected_fail"),
    ):
        env = {"FCR_SNAPSHOT_DIR": str(snap)} if name == "fcr_p1_pred_fail" else None
        _yaml(name, algo=algo, steps=8, train=train64, gate=gate, pattern=False, brier=False, env=env)
    arms = {
        "fcr_p2_grpo": dict(algo="grpo", gate=None, pattern=False, brier=False),
        "fcr_p2_B": dict(algo="fcr_grpo", gate="failure_confirmed", pattern=False, brier=False),
        "fcr_p2_R": dict(algo="fcr_grpo", gate=None, pattern=True, brier=False),
        "fcr_p2_P": dict(algo="fcr_grpo", gate=None, pattern=False, brier=True),
        "fcr_p2_BR": dict(algo="fcr_grpo", gate="failure_confirmed", pattern=True, brier=False),
        "fcr_p2_BP": dict(algo="fcr_grpo", gate="failure_confirmed", pattern=False, brier=True),
        "fcr_p2_full": dict(algo="fcr_grpo", gate="failure_confirmed", pattern=True, brier=True),
        "fcr_p2_bp_adapt": dict(
            algo="fcr_grpo", gate="failure_confirmed", pattern=False, brier=True, mode="magnitude",
        ),
        "fcr_p2_full_strict": dict(
            algo="fcr_grpo", gate="failure_confirmed", pattern=True, brier=True, require_prev_fail=True,
        ),
        "fcr_p2_hardcap": dict(
            algo="fcr_grpo", gate="failure_confirmed", pattern=False, brier=False, hard_cap=True,
        ),
    }
    for name, spec in arms.items():
        env = {"FCR_GRAD_DIAGNOSTICS": "0", "FCR_BRIER_MAX_ROWS": "1"} if spec.get("brier") else None
        if name == "fcr_p2_bp_adapt":
            env = {"FCR_GRAD_DIAGNOSTICS": "1", "FCR_BRIER_MAX_ROWS": "1"}
        _yaml(name, steps=8, train=train64, env=env, **spec)


if __name__ == "__main__":
    main()
