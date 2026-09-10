# Week 3 实验：Reward 消融（面试项目增量）

目标：用**可复现的小实验**讲清 reward 设计，而不是再堆一个更大模型。

## 假设

1. **binary（默认）**：只优化最终对错，最干净，但不管格式。
2. **shaped**：正确 +0.1 格式奖励 − 过长惩罚 → 鼓励 `### N ###`，抑制废话。
3. **format_only**：只看有没有严格格式 → **故意展示 hacking**：模型可以输出漂亮的错误答案拿高分。

## 代码入口


| 组件                                                   | 路径                                                                             |
| ---------------------------------------------------- | ------------------------------------------------------------------------------ |
| `compute_reward(..., mode=)`                         | `[math_agent.py](../../math_agent.py)`                                         |
| `LitMathAgent(reward_mode=...)` / `MATH_REWARD_MODE` | 同上                                                                             |
| 离线 dry-run（无 GPU）                                    | `[scripts/reward_ablation_dryrun.py](../../scripts/reward_ablation_dryrun.py)` |


```bash
cd /root/autodl-tmp/agent-lightning/examples/math_gsm
python scripts/reward_ablation_dryrun.py
```

训练时切换（正式对比建议短跑 + 固定 seed/步数）：

```bash
MATH_REWARD_MODE=binary   python train_math_agent.py fast
MATH_REWARD_MODE=shaped   python train_math_agent.py fast
MATH_REWARD_MODE=format_only python train_math_agent.py fast   # 仅作 hacking 演示
```



## 建议记录的指标


| 指标                    | 为什么            |
| --------------------- | -------------- |
| Val 准确率（binary 口径）    | 主结论：是否真会做题     |
| 严格格式率（含 `### N ###`）  | shaped 是否学到格式  |
| 平均 assistant 字符数 / 轮数 | 是否在刷长度或空转 tool |
| 解析失败率                 | reward 噪声来源    |
| 组内 reward 方差          | GRPO 有没有区分度    |


**重要**：即使用 `shaped` / `format_only` 训练，汇报「能力」时仍应用 **binary 准确率** 做公平对比，否则 format_only 会「刷分成功、考试失败」。

## 预期 takeaway（面试三句话）

1. **Reward 定义了学习目标**：`format_only` 会学「像答案」而不是「算对」。
2. **Shaping 是双刃剑**：格式奖励改善可解析性；长度惩罚对抗 verbosity；权重调不好会扭曲主目标。
3. **评测口径要与优化目标解耦**：优化可用 shaped，验收必须用正确性（和/或执行）。



## 可选加分实验（有算力再做）

- 同一 `a800` 短训：`n ∈ {2,4,8}`，画 val 曲线（对应超参词典）
- 开/关 `use_kl_loss`，观察熵与格式崩
- 对照 spider：执行匹配奖励 vs 字符串匹配（见 `docs/SPIDER_WALKTHROUGH.md`）



## Dry-run 结果归档

运行脚本后会写入 `[reward_ablation_dryrun.json](reward_ablation_dryrun.json)`。把该表贴进简历/面试附录即可证明「我做过 reward 设计消融」。