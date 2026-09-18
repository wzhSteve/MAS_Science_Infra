# Fission-GRPO

## 1. 出处
- Venue / arXiv / 日期：ACL 2026；arXiv:2601.15625
- 角色：失败→恢复采样的 RL 近邻

## 2. Abstract 要点（基于既有核对 + 社区报道）
- 诊断与 RAE G3 同句级：标准 RL 把丰富失败压成稀疏负奖励。
- 失败轨迹 + Error Simulator 诊断 → 裂变成 $G'$ 条 on-policy 恢复 rollout。
- 报告 **error recovery rate**；工具调用/BFCL 等设定常见。

## 3. Intro 诊断的社区盲区
失败信息被压成标量负奖励，学不会从错误恢复。

## 4. 方法三层定位
| 层 | 内容 |
| --- | --- |
| Rollout | 失败后额外恢复采样（裂变） |
| Advantage | 恢复成功后当正样本进 GRPO |
| Loss | GRPO |
| 失败用法 | 再采恢复；依赖诊断器与成功恢复 |

## 5. 实验设定线索
- 更贴代码/工具报错环境；搜索域需移植诊断器
- 指标：recovery rate（与 RAE 必须对齐）

## 6. 对 RAE 四问
| Q | 判定 | 说明 |
| --- | --- | --- |
| Q1 | ○ | |
| Q2 | ◐ | 用失败，但要采到成功 |
| Q3 | ●/◐ | |
| Q4 | ● | 显式优化 RR |

## 7. 与 RAE：同 / 异 / 优 / 劣
- **相似**：同诊断；同关心 recovery rate。
- **不同**：处方相反——裂变再试 vs KL 投影消除；无成功恢复则退化。
- **对方优势**：RR 指标已成审稿预期。
- **对方劣势**：实现重（Error Simulator）；搜索域非原生。

## 8. Baseline 决策
- **Appendix（B6 备选）**
- 理由：机制近邻极强；主文用 Agent-R 控成本，附录或讨论对齐 RR 定义。
