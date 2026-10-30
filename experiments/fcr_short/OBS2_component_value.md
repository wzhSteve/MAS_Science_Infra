# 观察 2：branch / R_corr / Brier 是否有正面价值

正面价值：打开该组件后对应机制指标改善，且任务 reward 不劣于关闭时。仅 correction_rate 升而任务不升记为过程奖励错位。

- **branch（纯再分配）**: 未见正面价值（pilot）; reward Δ=0.0375; fcr/error_step_repair_rate Δ=0; steps=1
- **R_corr（配对奖励）**: 未见正面价值（pilot）; reward Δ=0.0375; fcr/correction_rate Δ=0; steps=1
- **Brier（预测监督）**: 未见正面价值（pilot）; reward Δ=-0.2186; fcr/brier Δ=0.231; steps=1
- **完整方法**: 有正面价值（pilot）; reward Δ=0.0375; critic/score/mean Δ=0.0375; steps=1

说明：均为 **pilot，单 seed，1 个健康 step**（vLLM 在第 2 step 会 EngineDeadError，故 Wave2 收成 1-step）。branch/corr 与 full 的任务 reward 同为 0.5375，相对 GRPO 的 +0.0375 在单 step 噪声内；repair/correction 均为 0。branch 与 corr 都出现 `budget_violation_count` 12–15。brier_only 任务 reward 下降且 Brier 变差，且 `actor/fcr_brier_loss≈0`、`fcr_grad_norm_ratio=0`，预测监督未真正反传到 GRPO。full 的「正面价值」只来自任务 reward 点估计，不构成机制证据。

一旋钮（β 0.1→0.05，`rl_fcr_beta05.yaml`，1-step）：reward=0.6875，brier=0.335，repair=0.024，correction=0.038，infra_drop=0。相对 FCR-full：任务与 Brier、repair 点估计都更好，但 `fcr_grad_norm_ratio` 仍为 0，且 `budget_violation_count=18` 未消失。单 step 不能当成 β 有效；下一步应先修分支预算，而不是继续拧 β。

短程单 seed，不能替代 5-seed 等预算终局。
