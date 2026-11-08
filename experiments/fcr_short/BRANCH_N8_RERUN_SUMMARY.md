# Branch+Correct UI resume to 10 steps (batch=32, n_runners=8)

- Status: **RUNNING** (resumed from step 4 → target 10)
- experiment_id: `fcr-branch-corr`
- run_id (resume): `2ab54e2d70aa`
- prior run_id: `3f009e81c379` (steps 1–4)
- hung attempt: `8dba971df314` (cancelled — stalled ~2h after train 256/256 with GPU util 0%)
- Metrics: `mas/checkpoints/AgentLightning/fcr_branch_corr/metrics.jsonl`
- Tree dir: `experiments/fcr-branch-corr/artifacts/runs/2ab54e2d70aa/rollout-trees`
- resume_mode: auto from `global_step_4`; total_training_steps=10; watch `--relax-coverage`

## Checkpoints (interim)
- kept: `global_step_4`
- latest_iteration: 4
- best_val_checkpoint: step 4, val/reward≈0.3048

## Per-step metrics so far (Ra=answer, Rc=correct)
| step | train_r | val_r | cov | turn | branch | infra | Ra_val | Rc_val | Ra_train | Rc_train | n_corr |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.1944 | 0.2673 | 1.0 | 8.61 | 57 | 0.0086 | 0.3000 | 0.1222 | 1.0000 | 0.1181 | 30 |
| 2 | 0.2230 | 0.2713 | 1.0 | 8.14 | 50 | 0.0231 | 0.3222 | 0.1500 | 1.0000 | 0.1793 | 45 |
| 3 | 0.1235 | 0.2904 | 1.0 | 8.24 | 57 | 0.0044 | 0.3333 | 0.1167 | 1.0000 | 0.0745 | 20 |
| 4 | 0.2250 | **0.3048** | 1.0 | 7.94 | 62 | 0.0046 | **0.3556** | 0.1167 | 1.0000 | 0.0922 | 24 |

## vs GRPO+Correct (interim best=step4)
| | GRPO best (step2) | Branch best |
|---|---|---|
| val/reward | **0.285** | **0.3048** |
| Ra (val/fcr/R_answer_mean) | N/A | **0.3556** |
| Rc (val/fcr/R_correct_mean) | 0.011 | **0.1167** |
| coverage | 1.0 | 1.0 |
| turn | ≈9.4 | 7.94 |
| infra | 0.015 | 0.0046 |
| branch_count | 0 | 62 |
