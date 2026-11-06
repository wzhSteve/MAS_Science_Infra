# GRPO+Correct vs Branch+Correct (5 steps)

## Setup

- `rollout.n=8` / `group_n=8`（每样本 8 条轨迹，一步 `train_batch_size=2` → 16 条）
- 训练 `data/rl_datasets/train_10k.parquet`；每步验证完整 `data/rl_datasets/valid.parquet`（180 题）
- **Correct pattern（奖励）**：相邻步 prev `(p≥τ)≠Y`（预测 vs 执行失配），cur `p>τ` 且 `Y=1`；同 query 内 `R_correct=N_i/N_max`，再加终局 answer 0/1
- **Branch 门控**（仅 `fcr_branch_corr`）：`failure_confirmed`（`p<0.5` 且 `Y=0`，非 tool_fault）
- 下表七个恢复率为 **事件链诊断**（`audit_fcr_events`）。其中 `strict_adjacent` 仍是旧基线（`p`低∧`Y`失败 → 双成功），**不等于** Correct 奖励命中；奖励命中看 `R_correct_mean` / `strict_event_count`（现与新 pattern 对齐）

## fcr_grpo_corr
### step 1
- train reward (combined): 0.15208333333333335
- train R_answer / R_correct / R_total: 1.0 / 0.0 / 1.0
- val reward (outcome-only from trainer): 0.2368121693121693
- val R_answer / R_correct / R_total: 0.2722222222222222 / 0.15 / 0.42222222222222217
- train Correct pattern hits (new design): 5.0

| rate | train | val |
|---|---|---|
| prediction_failure_rate | 0.9455 (780/825) | 0.7498 (1127/1503) |
| execution_failure_rate | 0.9939 (820/825) | 0.9441 (1419/1503) |
| prediction_recovery_rate | 0.0643 (45/700) | 0.0821 (83/1011) |
| execution_recovery_rate | 0.0068 (5/740) | 0.0350 (45/1286) |
| strict_adjacent_rate | 0.0071 (5/700) | 0.0080 (8/1004) |
| lineage_recovery_rate | 0.0000 (0/820) | 0.0275 (39/1419) |
| false_correction_rate | 0.0000 (0/5) | 0.2000 (2/10) |

- frac_Nmax_zero: 1.0 | budget_violation: 0.0 | val correct+ frac: 0.15

## fcr_branch_corr
### step 1
- train reward (combined): 0.34375
- train R_answer / R_correct / R_total: 1.0 / 0.08333333333333333 / 1.0833333333333333
- val reward (outcome-only from trainer): 0.2185434282493106
- val R_answer / R_correct / R_total: 0.25 / 0.07222222222222222 / 0.3222222222222222
- train Correct pattern hits (new design): 24.0

| rate | train | val |
|---|---|---|
| prediction_failure_rate | 0.7461 (805/1079) | 0.8102 (1332/1644) |
| execution_failure_rate | 0.9453 (1020/1079) | 0.9568 (1573/1644) |
| prediction_recovery_rate | 0.1446 (108/747) | 0.0438 (52/1188) |
| execution_recovery_rate | 0.0633 (59/932) | 0.0197 (28/1419) |
| strict_adjacent_rate | 0.0328 (24/732) | 0.0042 (5/1180) |
| lineage_recovery_rate | 0.0000 (0/1020) | 0.0114 (18/1573) |
| false_correction_rate | 0.0000 (0/24) | 0.0000 (0/5) |

- frac_Nmax_zero: 0.5 | budget_violation: 0.0 | val correct+ frac: 0.07222222222222222

### step 2
- train reward (combined): 0.0
- train R_answer / R_correct / R_total: 1.0 / 0.1111111111111111 / 1.1111111111111112
- val reward (outcome-only from trainer): 0.23245030357387336
- val R_answer / R_correct / R_total: 0.26666666666666666 / 0.08888888888888889 / 0.35555555555555557
- train Correct pattern hits (new design): 0.0
- 该步训练集 Correct pattern 命中为 0，Correct 奖励没有进入更新。

| rate | train | val |
|---|---|---|
| prediction_failure_rate | 0.9752 (786/806) | 0.8179 (1334/1631) |
| execution_failure_rate | 0.9938 (801/806) | 0.9454 (1542/1631) |
| prediction_recovery_rate | 0.0212 (15/707) | 0.0559 (67/1199) |
| execution_recovery_rate | 0.0069 (5/722) | 0.0272 (38/1395) |
| strict_adjacent_rate | 0.0000 (0/707) | 0.0109 (13/1194) |
| lineage_recovery_rate | 0.0062 (5/801) | 0.0259 (40/1542) |
| false_correction_rate | 0.0000 (0/0) | 0.0714 (1/14) |

- frac_Nmax_zero: 0.5 | budget_violation: 0.0 | val correct+ frac: 0.08888888888888889

### step 3
- train reward (combined): 0.0
- train R_answer / R_correct / R_total: 1.0 / 0.3333333333333333 / 1.3333333333333333
- val reward (outcome-only from trainer): 0.22306788527376764
- val R_answer / R_correct / R_total: 0.25555555555555554 / 0.08333333333333333 / 0.33888888888888885
- train Correct pattern hits (new design): 3.0

| rate | train | val |
|---|---|---|
| prediction_failure_rate | 0.9154 (725/792) | 0.8080 (1313/1625) |
| execution_failure_rate | 0.9823 (778/792) | 0.9532 (1549/1625) |
| prediction_recovery_rate | 0.0201 (13/647) | 0.0390 (46/1180) |
| execution_recovery_rate | 0.0186 (13/700) | 0.0236 (33/1399) |
| strict_adjacent_rate | 0.0046 (3/647) | 0.0068 (8/1176) |
| lineage_recovery_rate | 0.0167 (13/778) | 0.0174 (27/1549) |
| false_correction_rate | 0.0000 (0/3) | 0.0000 (0/8) |

- frac_Nmax_zero: 0.5 | budget_violation: 0.0 | val correct+ frac: 0.08333333333333333

### step 4
- train reward (combined): 0.03125
- train R_answer / R_correct / R_total: 1.0 / 0.0 / 1.0
- val reward (outcome-only from trainer): 0.25681741182068085
- val R_answer / R_correct / R_total: 0.3 / 0.14444444444444443 / 0.4444444444444444
- train Correct pattern hits (new design): 0.0
- 该步训练集 Correct pattern 命中为 0，Correct 奖励没有进入更新。

| rate | train | val |
|---|---|---|
| prediction_failure_rate | 0.8456 (1260/1490) | 0.7619 (1168/1533) |
| execution_failure_rate | 1.0000 (1490/1490) | 0.9295 (1425/1533) |
| prediction_recovery_rate | 0.0178 (20/1126) | 0.0826 (87/1053) |
| execution_recovery_rate | 0.0000 (0/1336) | 0.0332 (43/1294) |
| strict_adjacent_rate | 0.0000 (0/1126) | 0.0134 (14/1048) |
| lineage_recovery_rate | 0.0000 (0/1490) | 0.0295 (42/1425) |
| false_correction_rate | 0.0000 (0/0) | 0.1250 (2/16) |

- frac_Nmax_zero: 1.0 | budget_violation: 0.0 | val correct+ frac: 0.14444444444444443

### step 5
- train reward (combined): 0.48125000000000007
- train R_answer / R_correct / R_total: 1.0 / 0.06666666666666667 / 1.0666666666666667
- val reward (outcome-only from trainer): 0.23396494708994708
- val R_answer / R_correct / R_total: 0.2722222222222222 / 0.18888888888888888 / 0.4611111111111111
- train Correct pattern hits (new design): 0.0
- 该步训练集 Correct pattern 命中为 0，Correct 奖励没有进入更新。

| rate | train | val |
|---|---|---|
| prediction_failure_rate | 0.7654 (1067/1394) | 0.7106 (1063/1496) |
| execution_failure_rate | 0.9713 (1354/1394) | 0.9251 (1384/1496) |
| prediction_recovery_rate | 0.0318 (30/943) | 0.0807 (77/954) |
| execution_recovery_rate | 0.0082 (10/1220) | 0.0398 (50/1256) |
| strict_adjacent_rate | 0.0000 (0/943) | 0.0063 (6/946) |
| lineage_recovery_rate | 0.0000 (0/1354) | 0.0354 (49/1384) |
| false_correction_rate | 0.0000 (0/0) | 0.3333 (3/9) |

- frac_Nmax_zero: 0.5 | budget_violation: 0.0 | val correct+ frac: 0.18888888888888888
