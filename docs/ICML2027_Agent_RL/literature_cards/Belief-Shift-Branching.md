# Belief-Shift Branching（增量 P1）

## 1. 出处
- Venue / arXiv / 日期：arXiv:2609.11061（2026-09-10）——相对 baselines 文档（09-06）为新文献

## 2. Abstract 要点
- 树 RL 预算下 fork 位置决定步级信用质量。
- 主流用结构位或 next-token 熵；本文用答案信念转移找 value curve pivots。
- 三实例：黑盒探针 / logit-lens / 学到的激活方向；信号只用于放置 fork。
- 数学与代码域 RL 上优于熵与结构基线。

## 3. Intro 诊断的社区盲区
fork 放在结局已定处几乎无对比信号；熵放置不可靠。

## 4. 方法三层定位
| 层 | 内容 |
| --- | --- |
| Rollout | 信念转移选 fork |
| Advantage | 兄弟结局差 → 步价值 |
| Loss | 树结构 RLVR |
| 失败用法 | 标量/对比 |

## 6. 对 RAE 四问
| Q | 判定 | 说明 |
| --- | --- | --- |
| Q1 | ◐ | 第二因子是内部答案信念，非外部探针 $Y$ 熵 |
| Q2–Q4 | ○ | 无支持集消除 / 恢复单元 |

## 8. Baseline 决策
- **Cite-only / 讨论近邻**
- 理由：强化“2026-09 仍在用内部信念修选点”；主表已有 APPO，不重复实现。
