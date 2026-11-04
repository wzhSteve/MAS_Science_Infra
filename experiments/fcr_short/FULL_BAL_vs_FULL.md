# Full_bal vs Full：防坍塌重测

设置：同为 B+R+P（`failure_confirmed`），8 actor steps，seed=42，`fcr_probe_train64`。  
`fcr_p2_full_bal` 相对旧 `fcr_p2_full`：**类别平衡 Brier 采样 + `pos_weight` + `FCR_BRIER_MAX_ROWS=2` + `fcr_brier_beta=0.05`**。作业 exit=0；权重 `global_step_8` 已删，仅留 `metrics.jsonl`。

## 总判

| 标准 | 结果 |
|---|---|
| 坍塌门控（连续 2 step `acc≥0.99` 且 `unexpected+pred_success=0`） | **仍触发**（bal 在 step5；旧 Full 在 step3） |
| `unexpected_share` 全程不塌到 0 | **否**（bal 从 step3 起为 0；旧 Full 从 step2） |
| `n_corr_hits` 或 `pair_valid` | bal **合计 hits=2**（仅 step1）；旧 Full=0 |
| `training/reward` ≥ 旧 Full 0.341 | **是**（bal **0.445**；亦略高于旧 BP 0.425） |

结论：**平衡采样推迟了坍塌约 1–2 step，并让 R 在 step1 短暂开火、任务 reward 点估计上升；但未解除「恒预测失败」终态。** 防坍塌未达标，Full 机制闭环仍未证实。

## 逐步对照

### fcr_p2_full_bal

| step | reward | brier | acc | unexpected | pred_success | hits | mean_p | collapsed |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.900 | 0.556 | 0.390 | 36 | 8 | **2** | 0.439 | no |
| 2 | 0.413 | 0.164 | 0.789 | 16 | 0 | 0 | 0.034 | no |
| 3 | 0.326 | 0.070 | 0.930 | 0 | 0 | 0 | ≈0 | no* |
| 4 | 0.138 | ≈0 | 1.000 | 0 | 0 | 0 | ≈0 | **yes** |
| 5 | 0.933 | ≈0 | 1.000 | 0 | 0 | 0 | ≈0 | **yes**（门控成立） |
| 6 | 0.000 | ≈0 | 1.000 | 0 | 0 | 0 | ≈0 | yes |
| 7 | 0.138 | ≈0 | 1.000 | 0 | 0 | 0 | ≈0 | yes |
| 8 | 0.713 | ≈0 | 1.000 | 0 | 0 | 0 | ≈0 | yes |

\*step3：`acc=0.93<0.99`，门控未计；但 `unexpected+success` 已为 0，分布已单峰失败预测。

### 旧 fcr_p2_full（摘要）

| step | reward | acc | unexpected | hits | collapsed |
|---|---|---|---|---|---|
| 1 | 0.500 | 0.431 | 33 | 0 | no |
| 2 | 0.675 | 1.000 | 0 | 0 | **yes** |
| 3–8 | … | ≈1 | 0 | 0 | yes（门控在 step3） |

均值 reward：旧 Full **0.341** / BP **0.425** / Full_bal **0.445**。

## 机制解读

1. **推迟有效、根治无效**：少数类优先 + 双行采样让 step1 仍有多样预测（`mean_p≈0.44`、hits=2）；step2 起 `mean_p` 骤降，step4 起完全塌缩。
2. **`pos_weight` 日志多为 1.0**：本 micro-batch 候选上未稳定拉高正类权重，或正负候选计数接近；不足以对抗失败占优先验。
3. **R 仍几乎死掉**：仅 step1 有 hits；坍塌后 \(p_t>0.5\) 消失，与旧 Full 一样无法维持 correct pattern。
4. **Reward 改善不宜过度解读**：8-step、单 seed；后半已塌缩却仍有高 reward 步（step5/8），噪声大。

## 下一步（若继续）

- 更强防塌：显式惩罚 `P(pred_fail)→1` / 保留 `unexpected` 质量地板；或课程制延后开启 Brier。
- 查清 `pos_weight` 为何常为 1（候选集是否过小、`MAX_ROWS=2` 是否仍采不到 Y=1）。
- 在坍塌解除前不要上 10k。
