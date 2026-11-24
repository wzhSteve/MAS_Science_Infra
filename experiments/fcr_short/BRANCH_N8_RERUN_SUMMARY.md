# Branch+Correct UI resume to 10 steps (batch=32, n_runners=8)

- Status: **cancelled** via Control UI
- experiment_id: `fcr-branch-corr`
- run_id (resume redo step5–10 @timeout60): `9a45bbd456ad`
- prior run_id: `3f009e81c379` (steps 1–4); hung `8dba971df314`; interrupted mid-val `2ab54e2d70aa` (old timeout=10)
- Metrics: `mas/checkpoints/AgentLightning/fcr_branch_corr/metrics.jsonl`
- Tree dir: `experiments/fcr-branch-corr/artifacts/runs/9a45bbd456ad/rollout-trees`
- resume_mode: auto from global_step_4; total_training_steps=10; watch `--relax-coverage`; Python timeout=60s

## Checkpoints
- kept: ['global_step_4', 'global_step_7']
- latest_iteration: 7
- best_val_checkpoint: `{"step": 4, "val/reward": 0.30482571639829703, "path": "/root/autodl-tmp/MAS_Science_Infra/mas/checkpoints/AgentLightning/fcr_branch_corr/global_step_4"}`

## Per-step metrics (Ra=answer, Rc=correct)
| step | train_r | val_r | cov | turn | branch | infra | Ra_val | Rc_val | Ra_train | Rc_train | n_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1944 | 0.2673 | 1.000 | 8.61 | 57.0 | 0.0086 | 0.3000 | 0.1222 | 1.0000 | 0.1181 | 30.0 |
| 2 | 0.2230 | 0.2713 | 1.000 | 8.14 | 50.0 | 0.0231 | 0.3222 | 0.1500 | 1.0000 | 0.1793 | 45.0 |
| 3 | 0.1235 | 0.2904 | 1.000 | 8.24 | 57.0 | 0.0044 | 0.3333 | 0.1167 | 1.0000 | 0.0745 | 20.0 |
| 4 | 0.2250 | 0.3048 | 1.000 | 7.94 | 62.0 | 0.0046 | 0.3556 | 0.1167 | 1.0000 | 0.0922 | 24.0 |
| 5 | 0.2746 | 0.2825 | 1.000 | 7.05 | 55.0 | 0.0666 | 0.3333 | 0.0722 | 1.0000 | 0.0506 | 13.0 |
| 6 | 0.1790 | 0.2612 | 1.000 | 7.41 | 60.0 | 0.0209 | 0.3278 | 0.0111 | 1.0000 | 0.0476 | 12.0 |
| 7 | 0.1399 | 0.2353 | 1.000 | 6.75 | 49.0 | 0.0130 | 0.3111 | 0.0889 | 1.0000 | 0.0119 | 3.0 |

## vs GRPO+Correct
| | GRPO best (step2) | Branch best |
|---|---|---|
| step | 2 | 4 |
| val/reward | **0.285** | **0.3048** |
| Ra (val/fcr/R_answer_mean) | N/A | **0.3556** |
| Rc (val/fcr/R_correct_mean) | 0.011 | **0.1167** |
| coverage | 1.0 | 1.0 |
| turn | ≈9.4 | 7.94 |
| infra | 0.015 | 0.0046 |
| branch_count | 0 | 62.0 |
