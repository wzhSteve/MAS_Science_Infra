# RAE：Related Work 与 Baseline 调研

> 配套 `ICML2027_Agent_RL/idea.md`（RAE: Recovery-Aware Exploration with Counterfactual Policy Elimination）。  
> 检索范围：**ICLR 2026、ICML 2026**、以及截至 **2026-09-14** 的 arXiv。  
> 用途：related work 写作、novelty 边界、实验基线分层。不含实验实现。  
> `idea.md` 初稿文献截止约 2026-08；v1.0（2026-09-06）补全 **RAIL、APPO、IGRPO、Fission-GRPO、BranPO、BPO**。  
> **v1.1（2026-09-14）**：深读卡片见 `[literature_cards/](literature_cards/00_INDEX.md)`；**搜索/单 agent 推理主场定稿**见 §0.5 与 `[experiment_protocol_search.md](experiment_protocol_search.md)`；增量文献 **Belief-Shift Branching (2609.11061)、GACA (2609.12424)**。

---

## 0.5 搜索推理主场定稿（2026-09-14）

> 本节把全谱系调研**收窄**为 ICML 主实验可执行集合。具身/代码仍见 §4–§5，但**不进主表**。

### 最优 Baseline 集合（主表 ≤ 7 + RFT）

| ID | 方法 | 谱系 | 证伪对象 | 实现优先级 |
| --- | --- | --- | --- | --- |
| B0 | Search-R1 / GRPO | 默认结局 RL | 标量失败学不会恢复 | P0 |
| B1 | DAPO | Loss 稳定 | “丢掉全错组 / 训练不稳” | P0 |
| B2 | ARPO | 纯熵分支 | G1：高熵 ≠ 值得探索 | P0 |
| B3 | AEPO | 熵平衡 | “熵修好就够” | P0 |
| B4 | **APPO** | 内部第二因子 $H\cdot\Omega$ | $\Omega$ ≠ $\bar H^B$ | P0 |
| B5 | IGPO | 答案 IG | G2 | P0（仅有 $y^*$ 的 QA） |
| B6 | Agent-R（轻量拼接近似） | 失败→正样本 | G3：无成功侧分层 | P1 |
| B7 | RFT / STaR | 只模仿成功 | 隔离成功侧 δ | P1 |

**附录升主表条件**：APPO 实现失败 → **RAIL** 升 B4；需要原生 RR 对齐 → 附录跑 Fission-GRPO。  
**附录固定**：RAIL、BranPO、NGRPO、VinePPO、IGRPO、Belief-Shift Branching（cite）。  
**Cite-only**：ASearcher、AgentFlow、Dark Room、GACA、InfoReasoner、RAPO、SIGHT。

### Agent × Benchmark × Tool（摘要）

| 维度 | 锁定选择 |
| --- | --- |
| Agent | 单 agent ReAct；动作=`search` /（GAIA）`browse` / `answer`；无 multi-agent |
| Backbone | 主：Qwen2.5-7B-Instruct；泛化：Llama-3.1-8B-Instruct；校验：Qwen3-8B（单独小节） |
| 训练栈 | veRL（与 Search-R1 / ARPO / APPO 同生态） |
| QA Bench | HotpotQA、2WikiMultiHopQA、Bamboogle（主）；MuSiQue（附录） |
| 长时程 | GAIA 文本/搜索子集（主报 pass@1 + tool/token 成本） |
| Tool | 本地 Wikipedia 检索（Search-R1 风格）+ ARPO 同源 browser（仅 GAIA）；Python 不作搜索主表必选项 |
| 预算 $M$ | 生成 token + tool-call **双重记账**；探针/分支全部计入（禁止免费探针） |
| 机制表 | RR、熵生命周期、Invalidation Precision、重复错误率、无成功侧有效更新比例、探针占比–收益 |

### RAIL / APPO 校准（基于 2026-09-14 深读）

- **APPO**：显式写“token entropy alone does not reliably reflect impact on final outcomes”；$\mathrm{BS}=z(H)\cdot z(\Omega)$，与 $U=\bar H^\pi\bar H^B$ **形式同构、机制不同**（内部似然增益 vs 外部探针结局熵）。与 ARPO **同 13-bench / 同 tool cache** → 搜索主表第一对照。
- **RAIL**：recoverability = 干预的 **realized reward-contrast / 组方差增益**；控制器 bandit；**Loss 仍 GRPO**。对二值 $Y$，$\mathrm{Var}(Y)=p(1-p)$ 与 Bernoulli 熵单调相关 → Rollout 层最强名字近邻；必须在 related work 第二段区分（见 §3）。
- **增量**：Belief-Shift Branching（2609.11061）继续用**内部答案信念**放 fork，加强“第二因子仍常内部化”叙事，不进主表。

完整复现协议 → `[experiment_protocol_search.md](experiment_protocol_search.md)`。论文卡片 → `[literature_cards/](literature_cards/00_INDEX.md)`。

---

## 0. 本稿主张（对照坐标系）

`idea.md` 的核心不是“更好的熵启发式”或“更密的过程奖励”，而是把 agentic RL 从**轨迹结局优化**改写成**反事实策略精化**：

1. **诊断（G1–G4）**：现有方法共享“失败 = 标量惩罚”。长时程 Agent 真正要学的是从被证伪的状态恢复。
2. **Rollout**：探索价值 $U=\bar H^\pi\cdot\bar H^B$——只有“模型不确定 **且** 环境结局可分辨”才值得完整分支；失败触发探针覆盖低熵灾难性自信。
3. **Reward**：结局做 validate / invalidate 二值裁决，失败侧 $\pi^{tgt}=\mathrm{Renorm}(\pi_{old}\setminus a_f)$（KL 投影 / 贝叶斯条件化），不是负 advantage。
4. **Loss**：单一 $D_{\mathrm{KL}}(\pi^{tgt}\Vert\pi_\theta)$，**无任何 $\lambda$**；熵生命周期是结果不是目标。
5. **学习单元**：恢复片段 $\sigma$（错 → 探索熵升 → 再承诺熵降），不是整条轨迹。

因此文献对照必须回答四个问题，而不是只问“谁也用了熵 / 分支 / 信息增益 / 从失败学习”：


| 编号 | 判别问题 | 本稿要求 |
| --- | --- | --- |
| Q1 | 选点信号是什么？ | **外部结局熵** $\bar H^B$（探针成功率的 Bernoulli 熵），乘以策略熵；不是纯 $H^\pi$，也不是 $\Delta P(y^*)$ 或策略似然比 $\Omega$ |
| Q2 | 失败如何进入学习？ | **支持集约束**（删除 $a_f$ 后按 $\pi_{old}$ 重归一），不是标量负梯度、也不是“拼一条成功后缀” |
| Q3 | 损失有没有 $\lambda$？ | 单一 KL 到重构目标；拒绝 $R_{out}+\lambda_1 R_{IG}+\lambda_2 R_{ent}$ |
| Q4 | 学的是什么能力？ | **错→对恢复**（RR、熵生命周期、invalidation precision），不是只涨成功率 |


**一句话总判断（给 related work 用）：**  
2026 年 Agentic RL 已经把“在哪里分支”做成显学（ARPO/AEPO/APPO/RAIL/IGRPO/BPO/BranPO），也把“从失败恢复”做成显学（Agent-R / Fission-GRPO / ReGRPO / CLEANER / CausalFlow / ERL / DPR）。**尚未同时出现的是：**（i）用**外部环境结局熵**（而非内部信念/似然比）与策略熵做乘积选点；（ii）把失败定义为对旧策略的 **KL 投影消除**，失败侧目标在无成功样本时仍良定义；（iii）用**无 $\lambda$ 的单一 KL** 同时承载 exploit 与 explore。这三条合取，才是 RAE 的贡献位置。

**命名警告：** 2026-08 出现 **RAIL**（Recoverability-Aware Intervention Learning, arXiv:2608.05080）。审稿人第一反应会认为本稿已被做完。必须在 related work 开篇用一段把 RAIL 与 RAE 拆开（见 §3 P0）。

---

## 1. 文献地图：按本稿三层坐标系放置

`idea.md` §3 的三层是组织 related work 的主轴。2026 社区几乎全部工作仍停在“修其中一层、失败仍是标量”。


| 层 | 本稿定义 | 2026 社区实际在做什么 | 代表 |
| --- | --- | --- | --- |
| **Rollout（采什么）** | 双熵 $U=\bar H^\pi\bar H^B$ + 失败触发探针 | 高熵分支、学一个 recoverability 控制器、用 $P(y^*)$ / $\Omega$ 过滤假高熵 | ARPO、AEPO、**RAIL**、**APPO**、**IGRPO**、BPO、Tree-GRPO、SIGHT |
| **Advantage / Reward（什么算好）** | validate / invalidate → 重构 $\pi^{tgt}$ | 组相对 outcome、答案 IG、锚点相对、对比分支、过程 PRM | GRPO、DAPO、IGPO、InfoReasoner、GiGPO、VinePPO、BranPO、SSVPO、NGRPO |
| **Loss（如何更新）** | 单一 KL$(\pi^{tgt}\Vert\pi_\theta)$，可 clip | clip / clip-higher / 熵加权 / 非对称 clip；几乎都是 $R$ 的线性组合 | DAPO、AEPO stop-grad、GTPO、GAPO、G²RPO、XRPO |
| **失败的用法（横切）** | 失败 = 支持集约束 | 造正样本 / 回合内恢复 / 负 advantage / 擦除再生 | Agent-R、Fission-GRPO、ReGRPO、CLEANER、CausalFlow、ERL、DPR、NGRPO |


下面按主题展开。每条标注：**venue / arXiv、做了什么、相对本稿差在哪里**。

---

## 2. Related Work（按主题）

### 2.1 Loss 层：组相对优化的稳定性（本稿可复用、不声称新颖）

这类工作回答“如何稳定地更新”，不回答“失败告诉了我们什么”。RAE 的 $A^{\mathrm{RAE}}$ 可直接跑在 GRPO/DAPO clip 管线上（`idea.md` §6.6）。


| 工作 | 出处 | 要点 | 与本稿 |
| --- | --- | --- | --- |
| **PPO** | Schulman et al., arXiv:1707.06347 | clip + GAE | 需要 critic；长时程状态难估 |
| **GRPO** | Shao et al., arXiv:2402.03300 | 组内标准化，免 critic | 全对/全错零梯度；失败=负 advantage |
| **DAPO** | Yu et al., arXiv:2503.14476 | clip-higher、动态采样滤零方差组、token 级损失 | 修动力学，丢掉全错组——与本稿“失败侧仍良定义”相反 |
| **LLD / GRPO collapse** | arXiv:2512.04220，**ICLR 2026** | Search-R1 式 GRPO 因 Lazy Likelihood-Displacement 坍缩；LLDS 只正则似然下降的 token | 优化器修补；仍是结局 RL |
| **Stratified GRPO** | arXiv:2510.06214，**ICML 2026** | 按工具调用结构分层归一，消 cross-stratum bias | 信用分配公平性，不改失败语义 |
| **XRPO** | arXiv:2510.06672，**ICML 2026** | 按不确定度分配 prompt 级 rollout；新颖正确轨迹优势放大 | 任务级探索-利用，不是状态级反事实消除 |
| **G²RPO** | **ICML 2026** | GRPO 在单纯形上 winner-take-all；给低频正确 mode 加 granularity bonus | 多样性崩溃的几何修补；成功侧多样性，不是失败侧支持集 |
| **GTPO / GRPO-S** | **ICML 2026** | 用 token 熵把同一轨迹奖励重分配 | 熵作权重，不是选点，更不是消除 |
| **GAPO** | arXiv:2609.00444，EMNLP 2026 | 按 advantage 自适应 clip 边界 | Loss 层插件 |
| **ATR-GRPO** | arXiv:2602.05494 | KL3 作 trust region | 约束形式，目标仍是 outcome |
| **SIGNBALANCE** | arXiv:2609.04063（2026-09-03） | GRPO 会把“蒙对”当成高优势（spurious advantage） | **支持 G2/侥幸成功**：RAE 的 $p_+$ 阈值与此同病 |
| **NGRPO** | arXiv:2509.18851 | 全错组注入虚拟 $r_{\max}$ 得到非零负优势 + 非对称 clip | **失败仍是标量负优势**；不约束质量流向（对照定理 1） |
| **SAPO** | arXiv:2608.19842 | 单 rollout actor-critic，共享自回归骨干 | 换采样拓扑，不换失败语义 |


**对照句：** DAPO 丢掉全错组；NGRPO 给全错组一个负标量。二者都承认“失败有信息”，但信息被压成一个数。RAE 要的是失败的**结构**：哪一个动作被证伪。

### 2.2 Rollout 层：在哪里分支（G1 的主战场，2026 增长最快）

#### A. 纯熵 / 工具后熵尖峰（`idea.md` 已覆盖）


| 工作 | 出处 | 要点 | 与本稿 |
| --- | --- | --- | --- |
| **ARPO** | arXiv:2507.19849，**ICLR 2026** | 工具返回后 token 熵升高 → 高熵处 partial rollout；半预算 | $U^{\mathrm{ARPO}}=H^\pi$；全成/全败仍狂分叉；低熵灾难性自信永不分支 |
| **AEPO** | arXiv:2510.14545，**WWW 2026** | 修 ARPO：熵预监控预算、连续高熵惩罚、stop-grad 保护高熵 token；GAIA 47.6% (Qwen3-14B, 1K) | 熵平衡 ≠ 结局可分辨；G1 未解 |
| **BPO** | arXiv:2607.14171 | 沙箱可快照：主干 + 高熵处分叉；**兄弟回报基线**（无 critic）；ALFWorld/WebShop/SWE-bench | 选点仍是熵；advantage 是 $V(s)$ 的 MC，不是 $\bar H^B$ 选点 + validate/invalidate |
| **Tree-GRPO** | arXiv:2509.21240 | 以 Thought-Action-Observation 为节点做树采样；树内/树间组相对；同预算约 1.5× 样本 | 随机/结构分叉，无双熵门控 |


**对照句：** 熵只回答“哪里不确定”。RAE 的 $\bar H^B$ 回答“这里的不确定能否被环境分辨”。BPO 的兄弟回报与 VinePPO 同源：估价值期望；RAE 的探针估结局熵，价值由后续裁决承接。

#### B. “熵不够”已被 2026 年三篇独立工作说出来（P0）

这是 related work 里**必须正面区分**的一组。它们都承认 G1 的前半句（高熵 ≠ 值得探索），但第二因子不是外部环境结局熵。


| 工作 | 出处 | 第二因子是什么 | 失败怎么用 | 与本稿的关键差 |
| --- | --- | --- | --- | --- |
| **APPO** | arXiv:2606.12384 | $\mathrm{BS}=z(H)\cdot z(\Omega)$，$\Omega$ 是后续 token 相对 $\pi_{old}$ 的衰减重要性比（**内部似然增益**） | 组相对标量 + procedure-level advantage scaling | 实证上直接支持 G1（高熵 token 的分支准确率并不更高）。但 $\Omega$ 是**策略内部**的“未来被当前策略更喜欢”，错误自信同样抬高 $\Omega$。RAE 的第二因子是探针的 **$Y\in\{0,1\}$ 熵** |
| **RAIL** | arXiv:2608.05080 | 可学习的 recoverability：干预带来的**组内奖励方差增益** $\Delta=I(Y_b)-I(Y_\emptyset)$；contextual bandit 控制器决定 where/how（预算×温度） | 仍用 GRPO 标量优势 | **名字与问题陈述最像。** 对二值奖励，$\mathrm{Var}(Y)=p(1-p)$ 与 Bernoulli 熵单调相关，故 RAIL 的 recoverability 与 RAE 的 $\bar H^B$ **几乎是同一标量的不同包装**。差别在后续：RAIL 只改 Rollout 分配，Loss 仍是 GRPO；RAE 把可分辨的分支变成 validate/invalidate 与 $\pi^{tgt}$。RAIL 无失败触发探针、无支持集消除、无单一 KL |
| **IGRPO** | arXiv:2607.06223 | 节点 informativeness = $\Delta \pi_\theta(a^*\mid h)$（**需要 GT 答案**）；按 softmax(val) 分配树扩展预算；诱导 teacher 分布 $\mu\propto\pi\exp(\gamma V)$ | 对 teacher 采样做 GRPO | 选点用内部 $P(y^*)$，正是 G2。全败节点 IG 低会被抑制（好），但抑制标准是“不像正确答案”，不是“环境证伪了哪个动作” |
| **SIGHT** | arXiv:2602.11551 | 工具结果的 posterior–prior likelihood 差；高 IG 分叉，低 IG 当 noise trap 并注入反思 prompt | GRPO + SES 奖励 | 内部答案确定性；干预是改 prompt，不是重构策略 |


**对照句（可直接进论文）：**  
APPO / RAIL / IGRPO 把 G1 从“社区没意识到”推进到“社区用三种不同的第二因子修补熵”。三种第二因子分别是：**内部续写似然比**、**组奖励方差（可学习）**、**对 GT 的信念增益**。RAE 的第二因子是 **对外部环境可验证结局的熵**，并且该量不只用于选点，还驱动动作级消除。RAIL 必须单列一段（见 §3）。

#### C. 对比分支与非单调正确性


| 工作 | 出处 | 要点 | 与本稿 |
| --- | --- | --- | --- |
| **BranPO** | arXiv:2602.03719 | 截断前缀、重采样续写，构造**同前缀、不同结局**的对比分支；提出 non-monotonic correctness（早错可被后步救、看似好的中间态可因后续失败） | **最接近“结局可分辨才有学习价值”。** 但学到的是对比优势 / 偏好，不是 $\mathrm{Renorm}(\pi_{old}\setminus a_f)$。需要结局分叉，失败侧单独不构成目标 |
| **VinePPO** | arXiv:2410.01679 | 中间状态 MC 续采估 $V(s)$ | 估价值，不是估 $\bar H^B$ |
| **SSVPO** | **ICLR 2026** | Sequential Shapley 逐步边际贡献；零贡献步可删 | 数学推理信用分配；需要重排步骤，开放工具环境难 |


### 2.3 Advantage 层：过程奖励与信息增益（G2）

社区对稀疏结局的标准回应是把中间步打分。RAE 明确拒绝任何未经外部结局校验的内部信念变化。


| 工作 | 出处 | 信号 | 与本稿 |
| --- | --- | --- | --- |
| **IGPO** | arXiv:2510.14967，**ICLR 2026** | $r^{IG}_t=\Delta\log\pi(y^*\mid C_t)$ | **G2 的点名对象。** 需要 GT；错误自信同样涨。Deep Research 训练栈（DR-Venus）已把它用到 200+ turn |
| **InfoReasoner** | arXiv:2602.00845，**ICML 2026** | 语义聚类后的信念熵降；检索奖励 $r_t=\log p(c^*\mid C_t)-\log p(c^*\mid B)$ | 仍是答案信念；明确把 IG 与 outcome **加权**（有 $\lambda$，正中 RAE 的无 $\lambda$ 主张） |
| **GiGPO** | arXiv:2505.10978，**NeurIPS 2025** | 轨迹组 + 可哈希锚点状态的第二级相对优势；ALFWorld +12%、WebShop +9% vs GRPO | 开放工具观测几乎不重复；锚点脆弱 |
| **HiPER** | arXiv:2602.16165，**ICML 2026** | 显式 Plan–Execute + Hierarchical Advantage Estimation；ALFWorld 97.4%、WebShop 83.3% | 分层信用分配，子目标仍服务任务成功；失败仍是低回报 |
| **AgentPRM** | arXiv:2511.08325，WWW 2026 | promise/progress 重定义 agent PRM | 过程正确性 ≠ 被证伪动作的支持集 |
| **WebArbiter** | arXiv:2601.21872，**ICLR 2026** | 生成式原则引导 WebPRM | 更好的过程监督员 |
| **EAPO** | arXiv:2605.08978，**ICML 2026** | 探索 vs 完成分组；变分推断奖励“探索对未来决策的价值” | 会在该探索时探索；不确定的是任务上下文，失败不是支持集约束 |
| **PTA-GRPO** | **ICML 2026** | 先高层 plan 再 CoT；复合奖励含 plan quality | 推理域；监督计划能否导向正确答案 |
| **AgentFlow / Flow-GRPO** | arXiv:2510.05592，**ICLR 2026 Oral** | 把轨迹级成败广播到每一步 Planner | 局部决策被全局成败染色；Retry→Success 与恢复无法区分 |
| **ASearcher** | **ICLR 2026** | 纯 RL 搜索 agent，单次最多 128 步；GAIA 58.1 | 更长的搜索轨迹，不是恢复 |
| **Dark Room** | arXiv:2607.21273 | **密预测奖励进 GRPO 的 std 归一会把策略训进 dark room**；channel 比 content 更决定成败；点名 IGPO 类信号 | **对无 $\lambda$ 主张的独立证据**：不要把 $\bar H^B$ / IG / 熵做成另一路 dense GRPO 奖励。RAE 走 KL 到 $\pi^{tgt}$，正是避开 reward channel |


**对照句：** Process reward 和 IG 把“哪一步像是对的”变密了。模型对错误答案越来越自信时，同样表现为熵降 / 正 IG。RAE 只在 $Y=1$ 之后才把内部自信转化成 validated commitment。

### 2.4 失败的用法：从“造正样本”到“在线恢复”（G3/G4）

2026 年“从失败学习”已成为显学。主流操作仍是 **repair the trajectory**，不是 **revise the support of $\pi$**。

#### A. 训练时把失败改写成成功轨迹 / 恢复轨迹（需要成功侧）


| 工作 | 出处 | 失败怎么用 | 与本稿 |
| --- | --- | --- | --- |
| **Reflexion** | NeurIPS 2023 | 语言反思进上下文，不改参数 | 无训练对照 |
| **Agent Q** | arXiv:2408.07199 | MCTS 成败分支 → DPO | 需要成功分支 |
| **Agent-R** | arXiv:2501.11425 | MCTS 在首错点拼接失败前缀+成功后缀，SFT 及时反思 | **失败利用的经典近邻**；本质是模仿正确后缀。难题上成功后缀稀缺则退化 |
| **CLEANER** | arXiv:2601.15141 | 同轨迹内自纠正后，SAAR 回滚把错误段替换成纠正，训“净化轨迹” | 把失败从数据里**擦掉**；RAE 把失败留作消除证据 |
| **Fission-GRPO** | arXiv:2601.15625，**ACL 2026** | **诊断与本稿 G3 同句：** “standard RL collapses rich failure into sparse negative rewards”。失败轨迹 + Error Simulator 诊断反馈，裂变成 $G'$ 条 on-policy 恢复 rollout；报告 **error recovery rate** | **恢复指标与 G3 诊断的最强近邻。** 仍是“采到恢复成功再当正样本”；依赖诊断器；失败本身不是 $\pi^{tgt}$ 的支持集约束。无成功恢复时目标退化 |
| **ReGRPO** | arXiv:2606.31392 | 近失配动作 → 真实失败观测 → (ErrorType, Evidence, FixPlan) + 纠正动作；GRPO 联合优反思 token；$R=\lambda_{\mathrm{exec}}\mathbf{1}_{\mathrm{succ}}-\eta C+\lambda_{\mathrm{val}}V$（**有 $\lambda$**） | 把反思做成可训动作；奖励仍是加权和。RAE 无反思 token 超参 |
| **CausalFlow** | arXiv:2605.25338 | 逐步反事实干预算 CRS，生成最小修复，作 DPO/奖励数据 | 失败→对比对；需要能把结局翻成成功的修复 |
| **ERL / ESearch** | **ICLR 2026** | 识别错误步、擦除、原地再生 | 修推理链，不是支持集投影 |
| **StructRefl** | arXiv:2509.18847 | 反思-修复显式成可训练动作 | 恢复动作有了，消除目标没有 |
| **WebRL** | arXiv:2411.02337，**ICLR 2025** | 从失败任务自举新任务课程 | 课程生成 |


**对照句：** Fission-GRPO 与本稿共享 G3 的问题陈述，处方相反：它把失败**裂变**成更多恢复尝试（仍要成功）；RAE 把失败**投影**成 $\pi^{tgt}(a_f)=0$（不要成功）。Agent-R / CLEANER / CausalFlow / ERL 都需要一条正确的另一侧。

#### B. 测试时 / 回合内恢复（无参数或弱参数）


| 工作 | 出处 | 要点 | 与本稿 |
| --- | --- | --- | --- |
| **DPR** | **ICML 2026** | 快过程 TextGrad 精炼 + 慢过程 Reflexion 诊断；低进度门控；ALFWorld Qwen3-8B 35%→75% | 回合内恢复，无参数更新 |
| **State-Aware GRPO** | **ICML 2026** | 科学工具 agent：重放工具调用重建状态 | 状态一致性，不是动作消除 |


#### C. 负梯度 / Unlikelihood（失败侧的理论近亲，质量流向不受控）


| 工作 | 出处 | 要点 | 与本稿 |
| --- | --- | --- | --- |
| **Unlikelihood** | Welleck et al., ICLR 2020 | 压低负 token | 不保持剩余动作相对比例 |
| **DPO / KTO** | NeurIPS 2023 / arXiv:2402.01306 | 偏好对或效用 → 隐式负梯度 | 同上 |
| **NGRPO / PSR-NSR** | arXiv:2509.18851 | 全错组负优势 | 标量；质量可流向词表任意处 |
| **Action Elimination** | Even-Dar et al., JMLR 2006 | 置信区间永久移除次优臂 | RAE 是其在 LLM 策略上的**软化、非平稳**版本：消除仅相对当前 $\pi_{old}$ |

**定理 1 的写作策略：** 把 Even-Dar 当理论祖先，把 Unlikelihood/DPO/NGRPO 当“压低但不 Renorm”的对照。贡献句：据我们所知，agentic RL 中尚无方法把环境证伪写成 $\min_{\pi:\,\pi(a_f)=0}D_{\mathrm{KL}}(\pi\Vert\pi_{old})$ 的投影。

### 2.5 探索感知、任务级分配、其它 Rollout 干预


| 工作 | 出处 | 要点 | 与本稿 |
| --- | --- | --- | --- |
| **EAPO (Learning to Explore)** | arXiv:2605.08978，**ICML 2026** | 显式探索模式 + 探索/完成分组 GRPO | 该探索时探索 ≠ 证伪动作 |
| **VIP / TAMPO** | RAIL 文中基线（Nguyen 2026 / Dang 2026） | 任务级预算或温度 | 标量预算，不是状态级消除 |
| **RAPO** | arXiv:2603.03078，**KDD 2026** | 检索 off-policy step；检索后熵降当奖励 | 熵降可来自错误自信（G2） |
| **Search-R1** | arXiv:2503.09516 | 多轮搜索 + 结局奖励 | Stage 1 默认 |


### 2.6 环境与系统（评测资源，不是方法竞品）

`idea.md` §7.1 的三类环境仍然成立。补充 2026 年会在这些环境上被问到的 SOTA 数字，避免审稿人用过时基线打。


| 环境 | 2026 强结果（供“不低于最强基线”的心理锚） | 对 RAE 的含义 |
| --- | --- | --- |
| **ALFWorld** | HiPER 97.4%；DPR 把 Qwen3-8B 从 35% 拉到 75%（无训练）；GiGPO +12% vs GRPO | 成功率可能饱和；必须报 **RR / 熵生命周期**，否则 HiPER 会赢主表 |
| **WebShop** | HiPER 83.3%；RAIL 在 L2 难分层上大幅超过 ARPO | 难分层是 RAE 的主场叙事 |
| **GAIA / 搜索** | AEPO 47.6% (14B, 1K)；ASearcher 58.1；AgentFlow 搜索平均 +14.9% | Stage 1 主场；RAE 不要只用 pass@1 开战 |
| **SWE / 工具调用** | BPO 在 SWE-bench Verified 上相对 GRPO +3.6–6.1；Fission-GRPO 在 BFCL 上报 recovery rate | 代码环境天然有 $c_t$ 与二值 $Y$，适合失败触发探针 |

---

## 3. 最近邻工作：必须写进论文的“不是 X”列表

按对 novelty 的威胁从高到低。每条给出 **一句话区分**。


| 优先级 | 工作 | 看起来像本稿的地方 | 不是本稿的地方 |
| --- | --- | --- | --- |
| **P0** | **RAIL** (arXiv:2608.05080) | 标题含 Recoverability；批判熵启发式；用奖励对比/方差决定是否干预；对标 ARPO/AEPO | **只改 Rollout 分配**（学 bandit 控制器），Loss 仍是 GRPO 标量；无 validate/invalidate、无 Renorm 消除、无单一 KL、无失败触发覆盖低熵自信。Recoverability 定义为干预带来的组方差，不重构 $\pi^{tgt}$ |
| **P0** | **APPO** (arXiv:2606.12384) | 明确写“token entropy 不能可靠反映对结局的影响”；$\mathrm{BS}=z(H)\cdot z(\Omega)$ 与 $U=\bar H^\pi\bar H^B$ **同为乘积门控** | $\Omega$ 是内部续写似然比，不是探针结局熵；更新仍是组相对标量 + 加权超参 $b$ |
| **P0** | **IGRPO** (arXiv:2607.06223) | 不在低价值节点浪费分支预算；有明确 teacher 分布当优化目标 | Teacher 由 $P(y^*)$ 倾斜；需要 GT；失败不是支持集约束 |
| **P0** | **Fission-GRPO** (ACL 2026) | G3 原话级诊断；优化 **error recovery rate** | 失败 → 诊断上下文 → 再采样恢复（要成功侧）；有 Error Simulator |
| **P0** | **BranPO** | 同前缀不同结局才构成有效监督；承认非单调正确性（与本稿 C2 归因同源） | 对比优势，不是 KL 投影消除；失败单独不构成目标 |
| **P1** | **ARPO / AEPO / BPO** | 高熵处分支、同预算效率叙事 | 纯熵；BPO 估 $V(s)$ |
| **P1** | **Agent-R / CLEANER / CausalFlow / ERL / ReGRPO** | 失败有用、会恢复 | 造正样本 / 擦除 / 偏好对 / 反思 token；依赖成功后缀或 $\lambda$ |
| **P1** | **IGPO / InfoReasoner / SIGHT / RAPO** | 信息增益 / 熵降进 RL | 内部信念；G2；InfoReasoner 显式 $\lambda$ |
| **P1** | **NGRPO** | 全错组也要学 | 虚拟 $r_{\max}$ → 负标量，质量流向不受控 |
| **P1** | **GiGPO / VinePPO / HiPER / SSVPO** | 过程级信用 | 估相对优势或 $V(s)$ 或 Shapley，不重构支持集 |
| **P1** | **Dark Room** | 支持“不要把密信号塞进 GRPO 奖励” | 诊断论文，不是方法竞品 |
| **P2** | **DPR / Reflexion** | 恢复 | 无参数或纯上下文 |
| **P2** | **AgentFlow / ASearcher / DAPO / Stratified / XRPO / G²RPO** | Agentic RL SOTA | 修稳定性、深度或多样性 |


**Novelty 的诚实表述（建议替换 `idea.md` §2.7 末句）：**

> 2026 年已有工作分别触及：熵不够所以加第二因子（APPO 的 $\Omega$、RAIL 的可学习奖励方差、IGRPO 的 $P(y^*)$）、从失败恢复（Fission-GRPO、ReGRPO、Agent-R）、以及对比分支（BranPO）。**尚未被同时满足的是：**（i）第二因子是**外部环境可验证结局的熵**，并与失败触发一起覆盖低熵灾难性自信；（ii）失败将目标策略约束为 $\mathrm{Renorm}(\pi_{old}\setminus a_f)$，该目标在无成功分支时仍良定义；（iii）探索与利用由**同一无 $\lambda$ 的 KL** 承担，熵生命周期作为可证伪的涌现预言而非奖励。这三条合取，才是 RAE 的贡献位置。

**关于 RAIL 的强制段落（建议 related work 第二段就出现）：**

> RAIL (Zhang et al., 2026) 同样把 rollout 干预表述为 recoverability，并用组内奖励对比作为干预收益。对二值可验证奖励，该对比与本文 $\bar H^B$ 在 $p=0.5$ 处同时取最大，因此 **RAIL 是 Rollout 层最强近邻**。本文与 RAIL 的分歧在干预之后：RAIL 把更可分辨的 rollout 组交回 GRPO；本文把可分辨的成败写成对动作支持集的 validate/invalidate，并以 KL 投影重构 $\pi^{tgt}$。换言之，RAIL 优化“采什么”，本文同时优化“采什么、什么算被证伪、以及证伪之后概率质量去哪”。

---

## 4. Baseline 建议（实验设计）

原则与 `idea.md` §7 一致：**同一环境、同一骨干、同一采样预算 $M$**（生成 token 或环境步；探针/分支/树搜索全部计入）。不允许“免费探针”。

### 4.1 Tier A：必须实现的算法基线（论文主表）

覆盖“结局 RL → 熵分支 → 第二因子选点 → 过程/IG → 失败利用 → 恢复 RL”整条谱系。缺 RAIL/APPO/Fission 中至少两个，审稿人会问。


| # | 基线 | 代表谱系 | 实现要点 | 用来证伪什么 |
| --- | --- | --- | --- | --- |
| A1 | **GRPO** | 默认结局 RL | 轨迹级 outcome | 标量失败学不会恢复 |
| A2 | **DAPO** | Loss 稳定版 | clip-higher + 滤零方差组 | “只是训练不稳 / 丢掉全错组更好”的替代解释 |
| A3 | **ARPO** | 纯熵分支 | 工具后高熵 partial rollout | G1：高熵 ≠ 值得探索 |
| A4 | **AEPO** 或 **BPO** | 熵平衡 / 兄弟基线 | 开放工具用 AEPO；可快照具身用 BPO | “熵平衡或更好的 $V(s)$ 估计就够了” |
| A5 | **APPO** 或 **RAIL** | 第二因子选点（P0） | 至少实现一个；资源够则两个都做 | 内部 $\Omega$ / 可学习方差 ≠ 外部 $\bar H^B$ + 消除 |
| A6 | **IGPO** | 答案 IG | 仅搜索/QA（有 $y^*$） | G2：信念增益 ≠ 正确 |
| A7 | **GiGPO** 或 **HiPER** | 过程信用 / 分层 | 具身：二者择一；开放工具可不跑 | 锚点/分层仍是标量优势 |
| A8 | **Agent-R** 或 **Fission-GRPO** | 失败→正样本/恢复采样 | 报告成功后缀稀缺分层；Fission 若实现成本高，主文 Agent-R、附录 Fission | G3：无成功侧时本稿应仍成立 |
| A9 | **RFT / STaR** | 只模仿成功 | 隔离“成功侧 $\delta$ 目标”的贡献 | 成功侧不是新颖性来源（`idea.md` 已承认） |
| A10 | **Reflexion** 或 **DPR** | 无训练恢复 | 上下文 vs 参数化 | 恢复能力不是提示技巧 |


环境约束导致的裁剪：

- **搜索 / HotpotQA / GAIA**：A6 必做；A4 用 AEPO；A5 优先 APPO（与 ARPO 同设定、13 benchmark）；A7 可省 HiPER。
- **ALFWorld / WebShop**：A4 用 BPO 或 GiGPO；A5 优先 RAIL（它的主表含 WebShop）；IGPO 需改写成目标状态匹配的 logprob，否则标 N/A。
- **SWE / 代码**：A8 用 Fission-GRPO 更自然（工具报错 $c_t$）；Agent-R 的 MCTS 成本高。
- **不可重置真实网页**：所有 MC 探针/分支（含 RAE、ARPO、RAIL、BPO）都困难；改用可重置仿真。

### 4.2 Tier B：附录或消融近邻


| 基线 | 何时加 |
| --- | --- |
| **RAIL**（若主文已用 APPO） | 附录专段：方差 recoverability vs $\bar H^B$ |
| **IGRPO** | 搜索 QA；证明“用 $P(y^*)$ 分配预算”仍中 G2 |
| **BranPO** | 证明对比分支 ≠ 支持集消除 |
| **NGRPO** | 消融 A2 的加强版：虚拟 $r_{\max}$ vs Renorm |
| **Tree-GRPO** | 树采样效率对照 |
| **ReGRPO / CLEANER** | 代码/工具环境，打“反思 token / 净化轨迹” |
| **CausalFlow / ERL** | 若强调反事实归因（C2） |
| **InfoReasoner** | 与 IGPO 并列的 $\lambda$-IG 对照 |
| **VinePPO** | MC 价值 vs MC 结局熵（`idea.md` 已列） |
| **Dark Room 设定** | 不必重跑；在讨论里引用，解释为何不把 $U$ 当 dense reward |

### 4.3 Tier C：只引用、不重跑

Search-R1、ASearcher、AgentFlow、Tongyi DeepResearch、WebRL、OpenHands、XRPO、G²RPO、GTPO、GAPO、SAPO、SIGNBALANCE、LLD、Stratified GRPO、WebArbiter、AgentPRM、EAPO、PTA-GRPO、SIGHT、RAPO。它们或域不同、或不可复现、或只修 Loss/系统结构。

### 4.4 本稿必须报告的机制指标

对齐 `idea.md` §7.4。若只有成功率，HiPER / AEPO / ASearcher / RAIL 可能持平或更好。


| 指标 | 预言 | 打哪些替代解释 |
| --- | --- | --- |
| **成功率 / pass@k / token 与 tool 成本** | 同预算不低于最强基线；难题分层（按首错位置分桶）显著更好 | 端到端没掉点 |
| **Recovery Rate（RR）** | RR 提升幅度 > 成功率提升幅度 | 不是靠“少犯错”，是靠“会恢复” |
| **熵生命周期**（以证伪事件对齐） | RAE：low→high→low；GRPO/ARPO：单调或无结构 | §6.4 可证伪预言 |
| **Invalidation Precision** | 被 eliminate 的动作在更大 $K$ 探针下确属不可恢复 | C2 归因风险 |
| **重复错误率** | 显著低于负 advantage 基线 | 标量惩罚易反弹 |
| **无成功分支时的有效更新比例** | Agent-R / Fission 在该分层崩溃；RAE 仍更新 | G3 核心 |
| **探针占比–收益曲线** | 少量探针换选点质量 > 等量盲目分支 | vs ARPO/BPO 效率叙事 |

### 4.5 与 `idea.md` §7.5 消融的衔接（建议微调）

原 A1–A7 仍然正确。建议**新增两条**，专门打 2026 新近邻：

| # | 消融 | 回应的主张 |
| --- | --- | --- |
| A8 | $\bar H^B$ → 组内奖励方差（RAIL 式） / → $\Omega$（APPO 式） | 外部结局熵 vs 内部第二因子 |
| A9 | 失败侧 Renorm → Fission 式“再采恢复” / → 只做对比分支（BranPO 式） | 支持集约束 vs 造正样本 vs 对比优势 |

原 A3（单一 KL vs $R_{out}+\lambda_1 R_{IG}+\lambda_2 R_{ent}$）应在讨论中引用 Dark Room：即使网格搜到好的 $\lambda$，把密信号放进 GRPO reward channel 也有结构性风险。

---

## 5. 评测环境建议（相对 idea.md §7.1 的补丁）

三类环境划分保持不变。补充实验叙事：


| 类别 | 环境 | 2026 注意点 | 主对比 |
| --- | --- | --- | --- |
| 具身 / 可哈希 | ALFWorld、WebShop | HiPER 成功率可能接近天花板 → 主报 RR 与难分层；RAIL 已在 WebShop 发过主表 | GiGPO / HiPER / BPO / **RAIL** |
| 开放工具 / 搜索 | HotpotQA、2Wiki、Bamboogle、GAIA 文本子集 | APPO 与 ARPO 同 13 benchmark；IGPO 是该域默认过程基线 | ARPO / AEPO / APPO / IGPO / IGRPO |
| 代码 / 可验证执行 | SWE-bench-lite、LiveCodeBench-agent、BFCL | $c_t$ 天然；Fission-GRPO 已报 recovery rate，必须对齐该指标 | GRPO / DAPO / Fission-GRPO / Agent-R / BPO |

模型：`idea.md` 的 Qwen2.5-7B-Instruct + Llama-3.1-8B 仍合理。注意 2026 实验多用 Qwen3-4B/8B/14B；若只报 2.5-7B，应用一段说明“为与 ARPO/GiGPO 原论文对齐”，并至少一组 Qwen3 验证。

**实验叙事建议：** 主文一张成功率表（证明没掉点）+ 一张机制表（RR / 熵曲线 / 无成功侧分层）。机制表才是 novelty 载体。不要只用 GAIA pass@1 与 ASearcher 开战。

---

## 6. 对照矩阵（G1–G4 × 代表工作）

符号：● 直接处理；◐ 部分触及；○ 未处理。


| 工作 | G1 选点（不确定且可分辨） | G2 外部校验 | G3 失败=支持集 | G4 恢复片段 | 无 $\lambda$ |
| --- | --- | --- | --- | --- | --- |
| GRPO / DAPO | ○ | ○ | ○ 负优势 / 丢全错组 | ○ | ●（但目标错） |
| ARPO / AEPO / BPO | ◐ 仅 $H^\pi$ | ○ | ○ | ○ | ● |
| **APPO** | ◐ $H\cdot\Omega$（内部） | ○ | ○ | ○ | ○（有 $b$） |
| **RAIL** | ◐ 可学习方差 | ◐ 奖励对比 | ○ 仍 GRPO | ◐ 干预避免不可恢复态 | ●（控制器另训） |
| **IGRPO / IGPO / InfoReasoner / SIGHT** | ◐ 用 $P(y^*)$ 或 IG | ○ 内部信念 | ○ | ○ | ○ 常有加权 |
| VinePPO / GiGPO / HiPER / BranPO | ◐ 分支/锚点/对比 | ◐ 结局 | ○ 相对优势 | ○ | ● |
| Agent-R / CLEANER / CausalFlow / ERL | ○ | ◐ | ○ 造正样本 | ● | SFT/DPO |
| **Fission-GRPO / ReGRPO** | ○ | ◐ 诊断器 | ○ 再采恢复 | ● 报 RR | ○ ReGRPO 有 $\lambda$ |
| NGRPO | ○ | ○ | ◐ 负标量 | ○ | ● |
| DPR / Reflexion | ○ | ○ | ○ | ● 回合内 | 无训练 |
| **RAE（本稿）** | ● $H^\pi\cdot H^B$ + 失败触发 | ● 仅 $Y$ | ● Renorm 消除 | ● $\sigma$ + 熵生命周期 | ● 单一 KL |

**搜索主场精简矩阵（只保留主表 + 附录近邻）**

| 工作 | G1 | G2 | G3 | G4 | 无λ | 主表？ |
| --- | --- | --- | --- | --- | --- | --- |
| Search-R1/GRPO | ○ | ○ | ○ | ○ | ● | B0 |
| DAPO | ○ | ○ | ○ 丢全错 | ○ | ● | B1 |
| ARPO | ◐ $H^\pi$ | ○ | ○ | ○ | ● | B2 |
| AEPO | ◐ 熵平衡 | ○ | ○ | ○ | ◐ | B3 |
| APPO | ◐ $H\cdot\Omega$ 内部 | ○ | ○ | ○ | ○ | B4 |
| RAIL | ◐ 可学习方差 | ◐ | ○ GRPO | ◐ | ◐ | 附录 |
| IGPO | ○ | ○ 内部 $P(y^*)$ | ○ | ○ | ○ | B5 |
| Agent-R | ○ | ◐ | ○ 造正样本 | ● | ● SFT | B6 |
| BranPO | ◐ 对比分支 | ◐ | ○ | ○ | ◐ | 附录 A9 |
| Belief-Shift (增量) | ◐ 信念转移 | ○ 内部 | ○ | ○ | ● | Cite |
| **RAE** | ● | ● | ● | ● | ● | 本稿 |

---

## 7. 对 `idea.md` 文本的修订建议

1. **§2.7 定位表过时。** 必须加入 RAIL、APPO、IGRPO、Fission-GRPO、BranPO、BPO、NGRPO、ERL。原句“尚无已发表方法把失败定义为删除被证伪动作后重归一化”**可以保留**，但前面要承认“选点侧已被 RAIL/APPO 逼近”。
2. **RAIL 必须升级为 P0，且 related work 不能放在附录。** 名字碰撞 + recoverability + 奖励方差 ≈ $\bar H^B$。不写清楚会被 desk-level 误读为 incremental。
3. **APPO 的乘积门控**与 $U=\bar H^\pi\bar H^B$ 形式同构。正文应写：乘积是“任一因子为零则价值为零”的最小组合；差别在第二因子的**生成机制**（外部探针结局 vs 内部 $\Omega$）。
4. **Fission-GRPO 与 G3 同诊断、反处方。** 应作为失败利用的第一对照，Agent-R 降为经典代表。
5. **Dark Room（2026-07）** 给无 $\lambda$ 主张提供独立证据：密 IG/预测奖励进 GRPO 会崩。方法叙述应强调 $\bar H^B$ **只用于选点，不进入 reward channel**。
6. **§7.2 基线列表**按本稿 §4.1 替换；机制指标保持 §7.4，并加上“无成功分支分层”。
7. **命名。** 考虑是否给方法一个不会与 RAIL 撞车的缩写强调（例如在摘要首次出现时写全称 Recovery-Aware Exploration with counterfactual policy **Elimination**，突出 Elimination 而非 Recoverability）。不强制改名，但审稿回应要准备好。

---

## 8. 参考文献（按主题，均已交叉核对）

标注：★ 为 ICLR 2026 或 ICML 2026；† 为 2026-06 及以后的 arXiv（相对 idea 初稿为新文献）。检索日 **2026-09-14**。

### 8.1 优化器与 Stage 1 Agentic RL

- Schulman et al. *PPO*. arXiv:1707.06347.
- Shao et al. *DeepSeekMath* (GRPO). arXiv:2402.03300.
- Yu et al. *DAPO*. arXiv:2503.14476.
- Jin et al. *Search-R1*. arXiv:2503.09516.
- ★ Dong et al. *ARPO*. arXiv:2507.19849. **ICLR 2026**.
- *AEPO*. arXiv:2510.14545. **WWW 2026**.
- ★ Li et al. *AgentFlow / Flow-GRPO*. arXiv:2510.05592. **ICLR 2026 Oral**.
- ★ Wang et al. *IGPO*. arXiv:2510.14967. **ICLR 2026**.
- Feng et al. *GiGPO*. arXiv:2505.10978. **NeurIPS 2025**.
- ★ *ASearcher*. **ICLR 2026**.
- ★ *On GRPO Collapse / LLD*. arXiv:2512.04220. **ICLR 2026**.
- ★ *Stratified GRPO*. arXiv:2510.06214. **ICML 2026**.
- ★ *XRPO*. arXiv:2510.06672. **ICML 2026**.
- ★ *G²RPO*. **ICML 2026**.
- ★ *GTPO / GRPO-S*. **ICML 2026**.
- ★ *HiPER*. arXiv:2602.16165. **ICML 2026**.
- ★ *PTA-GRPO*. **ICML 2026**.
- *RAPO*. arXiv:2603.03078. **KDD 2026**.
- † *GAPO*. arXiv:2609.00444. EMNLP 2026.
- † *SIGNBALANCE*. arXiv:2609.04063.
- *NGRPO*. arXiv:2509.18851.
- *ATR-GRPO*. arXiv:2602.05494.
- † *SAPO*. arXiv:2608.19842.
- *The Landscape of Agentic RL for LLMs: A Survey*. arXiv:2509.02547.

### 8.2 熵、分支、第二因子选点（G1 主文献）

- † *APPO*. arXiv:2606.12384.
- † *RAIL* (Recoverability-Aware Intervention Learning). arXiv:2608.05080.
- † *IGRPO*. arXiv:2607.06223.
- † *BPO* (Branching Policy Optimization). arXiv:2607.14171.
- *Tree-GRPO*. arXiv:2509.21240.
- *BranPO*. arXiv:2602.03719.
- *SIGHT*. arXiv:2602.11551.
- ★ Hua et al. *EAPO / Learning to Explore*. arXiv:2605.08978. **ICML 2026**.
- † *Belief-Shift Branching*. arXiv:2609.11061.（2026-09-10；内部信念转移选 fork）

### 8.3 过程奖励、IG、信用分配（G2）

- ★ *InfoReasoner*. arXiv:2602.00845. **ICML 2026**.
- Kazemnejad et al. *VinePPO*. arXiv:2410.01679.
- ★ *SSVPO*. **ICLR 2026**.
- Xi et al. *AgentPRM*. arXiv:2511.08325. WWW 2026.
- ★ *WebArbiter*. arXiv:2601.21872. **ICLR 2026**.
- *From Reasoning to Agentic: Credit Assignment survey*. arXiv:2604.09459.
- † *Dark Room in the Reward Channel*. arXiv:2607.21273.
- † *GACA*. arXiv:2609.12424.（2026-09-11；具身信用粒度；搜索主表不重跑）

### 8.4 失败、恢复、消除（G3/G4）

- Shinn et al. *Reflexion*. NeurIPS 2023.
- Putta et al. *Agent Q*. arXiv:2408.07199.
- Yuan et al. *Agent-R*. arXiv:2501.11425.
- Qi et al. *WebRL*. arXiv:2411.02337. **ICLR 2025**.
- *StructRefl*. arXiv:2509.18847.
- *CLEANER*. arXiv:2601.15141.
- Zhang et al. *Fission-GRPO*. arXiv:2601.15625. **ACL 2026**.
- † *ReGRPO*. arXiv:2606.31392.
- *CausalFlow*. arXiv:2605.25338.
- ★ *ERL / ESearch*. **ICLR 2026**.
- ★ *DPR* (within-episode recovery). **ICML 2026**.
- ★ *State-Aware GRPO*. **ICML 2026**.
- Welleck et al. *Unlikelihood Training*. ICLR 2020.
- Rafailov et al. *DPO*. NeurIPS 2023.
- Ethayarajh et al. *KTO*. arXiv:2402.01306.
- Even-Dar et al. *Action Elimination …*. JMLR 2006.
- Zelikman et al. *STaR*. NeurIPS 2022.
- Yuan et al. *RFT*. arXiv:2308.01825.
- Dong et al. *RAFT*. TMLR 2023.

---

## 9. 建议的 related work 章节提纲（可直接用于成文）

1. **Group-relative policy optimization.** GRPO / DAPO / LLD / Stratified / NGRPO。收束：稳定更新 ≠ 理解失败。
2. **Entropy-guided branching is necessary but insufficient.** ARPO / AEPO / BPO / Tree-GRPO。收束：G1 前半。
3. **Second factors for “where to branch”.** APPO（$\Omega$）、RAIL（可学习方差）、IGRPO（$P(y^*)$）、SIGHT、**Belief-Shift**。收束：第二因子几乎都是内部量；RAIL 最像，但停在 Rollout。
4. **Denser credit is still a scalar.** IGPO / InfoReasoner / GiGPO / HiPER / BranPO / VinePPO。收束：G2；BranPO 的对比仍不是支持集。
5. **Failures are used to repair trajectories, not to constrain policies.** Agent-R、Fission-GRPO、ReGRPO、CLEANER、CausalFlow、ERL、DPR。收束：G3/G4；Fission 同诊断、反处方。
6. **Negative gradients without renormalization.** Unlikelihood / DPO / NGRPO / Even-Dar。收束：定理 1 的位置。
7. **Dense rewards in the GRPO channel are dangerous.** Dark Room。收束：无 $\lambda$ 与“$U$ 不进奖励”的理由。
8. **This work.** 三条合取（§3 框）+ 与 RAIL 的强制区分段。

---

*文档版本：v1.1 | 2026-09-14 | 对应 idea.md v0.1；搜索定稿见 §0.5 与 experiment_protocol_search.md*
