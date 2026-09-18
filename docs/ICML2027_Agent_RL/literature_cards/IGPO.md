# IGPO — Information Gain-based Policy Optimization

## 1. 出处
- Venue / arXiv / 日期：ICLR 2026；arXiv:2510.14967
- 代码：https://github.com/GuoqingWang1/IGPO

## 2. Abstract 要点
- 多轮搜索 agent 的结局奖励稀疏 → advantage collapse、信用难分、样本效率差。
- 每 turn 奖励 = 模型对正确答案概率（长度归一 logprob）的边际增量。
- 与 outcome 结合成稠密轨迹；无需外部 PRM 或昂贵 MC。
- 域内/域外多跳搜索基准持续优于强基线。

## 3. Intro 诊断的社区盲区
稀疏结局无法告诉 agent“哪一步真正获取了关于答案的信息”。

## 4. 方法三层定位
| 层 | 内容 |
| --- | --- |
| Rollout | 标准多轮 |
| Advantage | $r^{IG}_t=\Delta\log\pi(y^*\mid C_t)$ + outcome（常有权重） |
| Loss | GRPO 类 + 常含 KL |
| 失败用法 | 负 IG / 低 outcome |

## 5. 实验设定线索
- 设定：multi-turn search agents；需要 ground-truth $y^*$
- 与 Search-R1 类 QA 兼容；Deep Research 栈已扩展到长 turn

## 6. 对 RAE 四问
| Q | 判定 | 说明 |
| --- | --- | --- |
| Q1 | ○ | 不负责选点 |
| Q2 | ○ | |
| Q3 | ○ | IG 与 outcome 常线性组合 |
| Q4 | ○ | 信内部信念增益 |

## 7. 与 RAE：同 / 异 / 优 / 劣
- **相似**：多轮搜索、过程级信号。
- **不同**：信号是**内部** $P(y^*)$——正中 G2；错误自信同样涨。
- **对方优势**：搜索域默认过程基线；实现相对清晰。
- **对方劣势**：Dark Room 等工作指出密预测奖励进 GRPO channel 有结构性风险；RAE 明确拒绝未经 $Y$ 校验的信念增益。

## 8. Baseline 决策
- **Yes（B5）**
- 理由：证伪“稠密答案 IG 就够”；仅在有 $y^*$ 的 QA 上跑。
