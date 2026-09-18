# RAIL — Recoverability-Aware Intervention Learning

## 1. 出处
- Venue / arXiv / 日期：arXiv:2608.05080（2026-08）
- 全称：*Optimizing What Policies Learn From: Recoverability-aware Rollout Intervention Learning*

## 2. Abstract 要点
- Critic-free 组 RL 常均匀分配 rollout，尽管学习价值高度状态依赖。
- 两个缺口：non-stationary（启发式不随策略演化）、non-scalar（只调数量不调 where/how）。
- RAIL：把干预建成 contextual bandit；用 **realized recoverability gains**（干预带来的奖励分布/对比增益）训控制器。
- Shadow-to-live 收集干预轨迹后 online gating；预算受限下更强、更少冗余信号。

## 3. Intro 诊断的社区盲区
“采什么”不应靠固定熵/难度启发式，而应按可实现的 recoverability 增益学一个控制器。

## 4. 方法三层定位
| 层 | 内容 |
| --- | --- |
| Rollout | 可学习控制器决定 where/how（预算×解码体制等）；recoverability ≈ 组奖励方差/对比增益 $\Delta$ |
| Advantage | **仍交回 GRPO 标量组相对** |
| Loss | GRPO |
| 失败用法 | 间接：避开不可恢复态；不消除动作支持集 |

## 5. 实验设定线索
- 评测轴：effectiveness / adaptivity / expressiveness / efficiency
- 公开图：AgentBench OS/DB；叙事含相对 ARPO/Tree-GRPO；WebShop 等 agent 设定在相关叙述中出现
- 注意：主表不完全等同 ARPO 的 13 搜索 bench——**搜索主协议优先 APPO**

## 6. 对 RAE 四问
| Q | 判定 | 说明 |
| --- | --- | --- |
| Q1 | ◐ | 第二因子与二值奖励下的 $\bar H^B$ 单调相关（$\mathrm{Var}(Y)=p(1-p)$） |
| Q2 | ○ | 不写 $\mathrm{Renorm}(\pi_{old}\setminus a_f)$ |
| Q3 | ●/◐ | 控制器另训；策略更新侧仍 GRPO |
| Q4 | ◐ | 关心可恢复态的干预，但不以恢复片段为学习单元 |

## 7. 与 RAE：同 / 异 / 优 / 劣
- **相似**：标题含 Recoverability；批判纯熵启发式；用结局对比度量“值不值得干预”。
- **不同**：只优化“采什么”；采完后仍 GRPO；无 validate/invalidate；无单一 KL 到重构目标；无失败触发覆盖低熵自信。
- **对方优势**：名字碰撞 + recoverability 词汇；审稿人第一反应会以为 RAE 已被做完。
- **对方劣势**：停在 Rollout 层；对二值 $Y$，$\Delta$ 与 $\bar H^B$ 几乎同标量不同包装，但后续学习语义完全不同。

## 8. Baseline 决策
- **Appendix（主文强制区分段；若 APPO 实现失败则升主表）**
- 理由：novelty 威胁最高，但搜索管线对齐成本高于 APPO。

### 强制区分句（可进论文）
> RAIL 同样把 rollout 干预表述为 recoverability，并用组内奖励对比作为干预收益。对二值可验证奖励，该对比与本文 $\bar H^B$ 在 $p=0.5$ 处同时取最大，因此 **RAIL 是 Rollout 层最强近邻**。分歧在干预之后：RAIL 把更可分辨的 rollout 组交回 GRPO；本文把可分辨的成败写成对动作支持集的 validate/invalidate，并以 KL 投影重构 $\pi^{tgt}$。
