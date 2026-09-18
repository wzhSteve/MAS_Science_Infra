# Search-R1

## 1. 出处
- Venue / arXiv / 日期：arXiv:2503.09516（多版本）；代码持续更新
- 代码：https://github.com/PeterGriffinJin/Search-R1（基于 veRL）

## 2. Abstract 要点
- 仅靠 prompting 让 LLM 与搜索引擎多轮交互往往次优。
- Search-R1：RL 训练交错推理与多次 search；检索 token mask；简单 outcome 奖励。
- 七个 QA 集上相对 RAG 基线大幅提升（报道约 +20–26% 量级，视模型而定）。
- 作为 DeepSeek-R1 Zero 在检索增强场景的开源延伸。

## 3. Intro 诊断的社区盲区
模型需要**学会**何时/如何调用搜索，而不是只在推理时被提示去搜。

## 4. 方法三层定位
| 层 | 内容 |
| --- | --- |
| Rollout | 标准多轮完整轨迹（PPO/GRPO 等） |
| Advantage | 结局可验证奖励（EM） |
| Loss | PPO / GRPO / REINFORCE 等 |
| 失败用法 | $R=0$ 标量 |

## 5. 实验设定线索（RAE 主协议锚点）
- 训练：NQ + HotpotQA 混合常见
- 评测：NQ, TriviaQA, PopQA；多跳 HotpotQA, 2Wiki, MuSiQue, Bamboogle
- 模型：Qwen2.5-3B/7B（base/instruct）、Llama 变体
- Tool：本地稀疏/稠密检索器；也可接在线搜索
- 惯例：retrieved token loss masking

## 6. 对 RAE 四问
| Q | 判定 | 说明 |
| --- | --- | --- |
| Q1 | ○ | 无熵/结局熵选点 |
| Q2 | ○ | |
| Q3 | ● | 刻意简单 outcome |
| Q4 | ○ | |

## 7. 与 RAE：同 / 异 / 优 / 劣
- **相似**：单 agent 搜索推理 + 可验证结局；veRL 生态。
- **不同**：纯结局 RL；无分支、无消除。
- **对方优势**：社区默认 Stage-1；数据与检索栈可直接复用。
- **对方劣势**：长轨迹稀疏奖励与恢复能力弱——正是 RAE 动机。

## 8. Baseline 决策
- **Yes（B0 = Search-R1/GRPO）**
- 理由：所有方法的共同底盘与公平对比起点。
