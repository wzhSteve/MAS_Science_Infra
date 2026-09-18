# AEPO — Agentic Entropy-Balanced Policy Optimization

## 1. 出处
- Venue / arXiv / 日期：WWW 2026；arXiv:2510.14545
- 代码：同 ARPO 仓库 `AEPO/` 子目录

## 2. Abstract 要点
- 过度依赖熵信号会导致 high-entropy rollout collapse 与训练不稳。
- AEPO：熵预监控分配全局/分支预算 + 连续高熵分支惩罚。
- 更新侧：高熵 clip 项 stop-gradient；熵感知 advantage。
- 14 数据集上优于 7 个主流 RL；Qwen3-14B + 1K RL 样本：GAIA Pass@1 47.6%、Pass@5 65.0%。

## 3. Intro 诊断的社区盲区
熵引导探索必要，但熵不平衡会耗尽预算并剪掉高熵 token 梯度。

## 4. 方法三层定位
| 层 | 内容 |
| --- | --- |
| Rollout | 仍熵驱动；加预算与连续高熵惩罚 |
| Advantage | outcome + 熵感知加权 |
| Loss | clip + stop-grad 保护高熵 |
| 失败用法 | 标量 |

## 5. 实验设定线索
- 模型：Qwen3-14B 等 web agent 设定
- Benchmark：GAIA、HLE、WebWalkerQA 等 14 集
- Tool：web 搜索 / 浏览（与 ARPO deep search 同源栈）
- 预算：1K RL samples 叙事

## 6. 对 RAE 四问
| Q | 判定 | 说明 |
| --- | --- | --- |
| Q1 | ◐ | 修熵动力学，不引入外部结局熵 |
| Q2 | ○ | |
| Q3 | ●/◐ | 无 IG-λ，但有熵感知加权 |
| Q4 | ○ | |

## 7. 与 RAE：同 / 异 / 优 / 劣
- **相似**：承认纯熵会崩；同属工具后分支族。
- **不同**：熵平衡 ≠ 结局可分辨；仍无 validate/invalidate。
- **对方优势**：GAIA 公开强数字；直接打“你们只是没把熵修好”。
- **对方劣势**：不覆盖低熵灾难自信；失败语义不变。

## 8. Baseline 决策
- **Yes（B3）**
- 理由：开放工具/搜索主场的熵修对照；实现可复用 ARPO 代码。
