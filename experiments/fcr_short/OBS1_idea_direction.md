# 观察 1：沿 idea 优化是否有效

范围：seed=42 / 2×A800 / **各 5 个 actor step** / GRPO vs FCR-full。短程 on-policy，不宣称 held-out 认知纠正。

- 判定：**5-step 未出现正向机制信号（仍不够支持继续）**
- GPU 训练已结束（GRPO exit=0；FCR-full 第一次在 step3 actor OOM，降 util / 关 Brier 额外前向 / `n_runners=1` 后重跑 exit=0）
- 完整性：两边 `prediction_coverage=1`、`label_coverage=1`、`leakage=0`、`cross_chain=0`
- 三条用户关心的信号：

## 逐步曲线

GRPO（`training/reward` / accuracy / brier / repair / n_corr_hits / infra）

- step1：reward=0.50 acc=0.69 brier=0.28 repair=0 hits=0 infra=0.13
- step2：reward=0.00 acc=0.67 brier=0.27 repair=0 hits=0 infra=0.40
- step3：reward=0.00 acc=0.51 brier=0.40 repair=0 hits=0 infra=0.13
- step4：reward=0.00 acc=0.26 brier=0.71 repair=0 hits=0 infra=0.13
- step5：reward=0.50 acc=0.18 brier=0.79 repair=0 hits=0 infra=0.13

FCR-full

- step1：reward=0.56 acc=0.62 brier=0.37 repair=0 hits=0 pair_valid=1.00 infra=0.14 budget_viol=15
- step2：reward=0.11 acc=0.69 brier=0.26 repair=0 hits=0 pair_valid=0.81 infra=0.44 budget_viol=10
- step3：reward=0.00 acc=0.17 brier=0.83 repair=0 hits=0 pair_valid=0.36 infra=0.28 budget_viol=16
- step4：reward=0.00 acc=0.12 brier=0.86 repair=0 hits=0 pair_valid=1.00 infra=0.00 budget_viol=6
- step5：reward=0.84 acc=0.11 brier=0.86 repair=0 hits=0 pair_valid=0.00 infra=0.36 budget_viol=0

## 对照判定

1. **错误修正率**：`error_step_repair_rate` / `recovery_rate` / `correction_rate` 全程为 0。FCR 有分支（`pair_valid_rate` 多步 >0），但 `n_corr_hits=0`：子轨迹从未同时满足 p≥0.5 且 Y=1，R_corr 没加上去。
2. **预测精准度**：两边 accuracy 都下降、Brier/ECE 都上升。FCR 没有相对 GRPO 的校准改善。
3. **局部 reward 更密**：`fcr/nonzero_token_reward_frac` 与 `fcr/mean_abs_score_nonfinal` **未进 metrics.jsonl**（advantage 后 meta 未稳定带回 logger）。用可观测代理：`n_corr_hits=0`，局部校正 bonus 未发生；不能声称比 GRPO 更密。

step5 FCR `training/reward=0.84` vs GRPO `0.50` 是 8 条 rollout 噪声，且同 step 修正率仍为 0、accuracy 更差，**不算机制正向**。

FCR 仍有 `budget_violation>0`（K 截断未按 triplet parent 计满）和间歇 `infra_drop>0.2`。观察 2 / 旋钮短跑按计划不在本轮默认展开。
