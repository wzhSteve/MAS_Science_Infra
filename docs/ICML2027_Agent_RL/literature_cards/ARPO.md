# ARPO — Agentic Reinforced Policy Optimization

## 1. 出处
- Venue / arXiv / 日期：ICLR 2026；arXiv:2507.19849（2025-07）
- 代码：https://github.com/RUC-NLPIR/ARPO（含 AEPO）

## 2. Abstract 要点
- 现有 agentic RL 在多轮工具交互上仍偏轨迹级采样，难对齐逐步 tool-use。
- 发现工具返回后前 10–50 token 熵显著升高。
- ARPO：高熵 tool-call 轮做 adaptive partial rollout，平衡全局与步级采样。
- Advantage Attribution（hard/soft）区分共享前缀与分支 token。
- 13 benchmark 上优于轨迹级 RL；约一半 tool-use 预算达到更好效果。

## 3. Intro 诊断的社区盲区
轨迹级 RL 忽视工具反馈引入的步级不确定性与可探索行为。

## 4. 方法三层定位
| 层 | 内容 |
| --- | --- |
| Rollout | $U^{\mathrm{ARPO}}=H^\pi$；工具后熵升超阈 → partial branch |
| Advantage | 组相对 outcome + 共享/分支归因 |
| Loss | clip 类 GRPO/PPO 管线 |
| 失败用法 | 负 advantage / 低回报；无支持集约束 |

## 5. 实验设定线索
- 模型：Qwen2.5 / Qwen3 / Llama3.1 系列（含 7B–14B）
- Benchmark：计算推理（AIME/MATH/GSM8K）、知识多跳（HotpotQA, 2Wiki, MuSiQue, Bamboogle）、Deep Search（GAIA, WebWalker 等）共 13
- Tool：（1）Search Engine（2）Web Browser Agent（3）Python 解释器
- 预算：强调 tool-call 次数；半预算叙事

## 6. 对 RAE 四问
| Q | 判定 | 说明 |
| --- | --- | --- |
| Q1 | ◐ | 仅策略熵；全成/全败仍可能狂分叉；低熵自信永不分支 |
| Q2 | ○ | 失败仍是标量 |
| Q3 | ● | 无显式 $\lambda$-IG；但目标仍是 outcome RL |
| Q4 | ○ | 不报 recovery segment / RR 为主指标 |

## 7. 与 RAE：同 / 异 / 优 / 劣
- **相似**：工具后熵尖峰是合法选点启发式；同预算效率叙事。
- **不同**：无 $\bar H^B$；无失败触发探针；无 Renorm 消除；无单一 KL 到 $\pi^{tgt}$。
- **对方优势**：公开代码成熟、13-bench 可比、半预算数字强、社区认知度最高。
- **对方劣势**：G1 未解（高熵 ≠ 可分辨）；灾难性自信盲区。

## 8. Baseline 决策
- **Yes（B2）**
- 理由：搜索主场默认熵分支对照；RAE 的 A1 消融直接退化为 ARPO。
