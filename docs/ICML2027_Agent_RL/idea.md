# RAE: Recovery-Aware Exploration with Counterfactual Policy Elimination for Agentic Reinforcement Learning

**RAE：面向 Agent 强化学习的恢复感知探索与反事实策略消除**

> 论文初稿 v0.1（method-only draft）。仅包含动机、相关工作、问题形式化、方法与实验设计；不含实验结果与代码。
> 引用均已核实存在（截至 2026-08，见文末参考文献；ARPO 已被 ICLR 2026 接收，RAPO 已被 KDD 2026 接收）。
> 仓库落地设计见 [BRANCH_SITE_DESIGN.md](../BRANCH_SITE_DESIGN.md)（UI 站点、gates、RAE R0/R1）。

---

## 摘要（Abstract）

基于可验证奖励的强化学习（RLVR）正在从单轮推理扩展到多轮工具交互的 Agent 场景（agentic RL）。近期工作分别修补了训练管线的三个层面：DAPO 等修复了组相对优化的**损失稳定性**；ARPO / AEPO 用 token 熵决定**在哪里分支采样**；IGPO / GiGPO 通过信息增益或状态锚点改进**过程级信用分配**。然而，这些方法共享同一个未被质询的假设：**失败只是一个标量惩罚**（$R=0$ 或负 advantage）。我们指出，长时程 Agent 的核心能力不是复现一条静态的成功路径，而是**从错误中恢复**——在被环境证伪的状态下重新打开备选、修正承诺并重新收敛。据此我们提出 RAE（Recovery-Aware Exploration with counterfactual policy Elimination），它做出三个结构性改变：（1）**Rollout 层**：用策略熵 $\bar H^\pi$ 与分支结局熵 $\bar H^B$ 的乘积 $U=\bar H^\pi\cdot\bar H^B$ 选择反事实扩展点，只有"模型不确定**且**环境结局可分辨"的状态才值得探索，并以失败触发探针覆盖低熵的灾难性自信状态；（2）**Reward 层**：环境结局不再折算为标量，而是对动作做 validate / invalidate 二值裁决，并据此重构目标策略——被验证的动作集合按旧策略条件化，被证伪的动作被**删除后重归一化**（我们证明该目标等价于旧策略在"该动作不可行"证据下的贝叶斯后验，也是支持集约束下对旧策略的 KL 投影）；（3）**Loss 层**：单一 KL 目标同时承载探索与利用，**不引入任何熵系数或奖励加权超参 $\lambda$**，熵的升降（low→high→low 的恢复生命周期）是策略精化的结果而非训练目标。我们给出与 GRPO/PPO 管线兼容的优势函数构造、面向开放动作空间的 token 层实现方案、采样预算记账，以及覆盖具身/网页/搜索三类环境、含五组消融与四项机制性指标的完整实验设计。

**关键词**：Agentic RL；错误恢复；反事实推理；策略消除；探索-利用

---

## 1 引言（Introduction）

### 1.1 背景与动机

以 GRPO 为代表的组相对策略优化使 RLVR 成为训练推理 LLM 的主流范式 [GRPO]。当模型进一步作为 Agent 与环境交互——调用工具、浏览网页、执行代码——RL 的对象从"一段静态文本"变成"一条动态轨迹"，催生了 agentic RL 这一快速增长的方向 [Survey]。

LLM RL 与 agentic RL 的本质差异并非动作空间大小，而在于**动作会主动改变环境状态，并使未来信息变得可观测**。这带来两个后果：

1. **正确路径不可复用。** 数学题的正确推理链可以被反复强化；Agent 的正确路径依赖具体的环境反馈，observation 换一次就可能失效。
2. **动作具有双重价值。** 除执行价值外，动作还有信息获取价值——一次工具调用可能不改变任务进度，却改变了 Agent 的信念状态。

因此，长时程 Agent 真正需要学习的不是"哪条完整路径最终成功"，而是：

> **在已经发生错误或不确定的状态下，哪个动作能最快、最可靠地把 Agent 带回正确决策区域？**

### 1.2 现有工作的共同盲区

近期 agentic RL 的进展可以在一个统一的三层坐标系中定位（§3）：**Rollout（采什么）→ Advantage（什么算好）→ Loss（好坏如何变成参数更新）**。DAPO [DAPO] 修 Loss 层稳定性；ARPO [ARPO]、AEPO [AEPO]、RAPO [RAPO] 修 Rollout 层的探索位置与来源；IGPO [IGPO]、GiGPO [GiGPO]、VinePPO [VinePPO] 修 Advantage 层的信用分配。这条"修漏洞"谱系卓有成效，但四个系统性缺口从未被同时回答：

- **G1（不确定 ≠ 值得探索）**：ARPO/AEPO 在高熵处分支，但高熵可能是 productive uncertainty（结局尚未定），也可能是 unproductive uncertainty（怎么走都失败/都成功）。熵只回答"哪里不确定"，不回答"这里的不确定是否可以被环境分辨"。
- **G2（信念增益 ≠ 正确）**：IGPO 用 $\Delta P_\theta(y^*)$ 做稠密奖励，RAPO 奖励检索后的熵降。二者度量的都是**内部信念的边际变化**——模型对错误答案越来越自信时，同样表现为熵降。内部置信度的上升必须经外部结局校验后才可信。
- **G3（失败 ≠ 标量惩罚）**：所有上述方法把失败折算为 $R=0$ 或负 advantage，即"整体压低这段行为的概率"。但环境证伪一个动作时给出的信息是**结构化的**：该动作不可行，而其余动作的相对可信度不应被牵连。标量惩罚丢失了这一结构。
- **G4（学习单元错位）**：基本 rollout 单元仍是完整轨迹，而长时程 Agent 最有学习价值的片段是"错误执行 → 检测 → 修正 → 正确执行"这一**局部恢复过程**。

### 1.3 我们的方案与贡献

我们提出 RAE，把 agentic RL 从"基于结局的轨迹优化"重写为"**反事实策略精化**"（counterfactual policy refinement）：熵定位 Agent 应当重新考虑决策的位置；成功与失败的分支分别 validate 和 invalidate 策略承诺。贡献如下：

1. **双熵探索准则（Rollout 层，回应 G1）**：定义探索价值 $U(s_t)=\bar H_t^\pi\cdot\bar H_t^B$，其中 $\bar H^B$ 是少量反事实探针的**结局熵**。仅当策略不确定且结局可分辨时才做完整分支扩展；配合失败触发探针与矛盾信号，覆盖"低熵灾难性自信"这一 reward hacking 高发区。
2. **反事实消除目标（Reward 层，回应 G2/G3）**：成功分支把目标策略条件化到被验证的动作集合上；失败分支构造 $\pi^{tgt}=\mathrm{Renorm}(\pi_{old}\setminus a_f)$。我们证明该目标是旧策略在证伪证据下的贝叶斯条件化，等价于支持集约束下的 KL 投影（定理 1、命题 1），并给出面向开放 token 动作空间的可执行近似（§6.5）。
3. **无 $\lambda$ 的统一损失（Loss 层，回应 G3）**：单一 $D_{\mathrm{KL}}(\pi^{tgt}\Vert\pi_\theta)$ 同时承载 exploit（$\delta$ 型目标）与 explore（消除后熵自然升高），不含任何熵系数、信息增益权重或奖励加权项；并给出与 GRPO/PPO clip 管线兼容的优势构造 $A^{\mathrm{RAE}}$。
4. **恢复导向的评测协议（回应 G4）**：以"错 → 对"恢复片段为学习与评测单元，提出 recovery rate、熵生命周期、invalidation precision 等机制性指标与五组消融的完整实验设计（§7），每一项都对应一个可证伪的机制性预言。

---

## 2 相关工作（现有工作）

我们按三层坐标系与"失败的用法"两条线组织相关工作。所有文献均已核实（arXiv 编号见参考文献）。

### 2.1 组相对策略优化：Loss 层的稳定性

**PPO** [PPO] 以 clip 代理目标与 GAE 成为 RLHF 标准，但需要额外的 value model；长时程 Agent 的状态是完整对话加工具返回，critic 既贵又难准。**GRPO** [GRPO] 用同一 prompt 的一组完成做组内标准化，免去 critic，但整条轨迹共享同一个 outcome advantage，且组内全对/全错时优势坍缩为零梯度。**DAPO** [DAPO] 用非对称 clip（clip-higher）、动态采样过滤零方差组、token 级损失归一与超长惩罚修复 GRPO 的训练动力学。

*与本文关系*：这些方法回答"如何稳定地更新"，不回答"失败告诉了我们什么"。RAE 的底层优化器可以完全复用该家族的稳定化机制（§6.6）。

### 2.2 Rollout 层：在哪里、从哪里探索

**ARPO** [ARPO] 观察到工具返回后 LLM 的 token 熵显著升高（论文报告集中于前 10–50 个 token），据此在高熵处做 partial rollout 分支，同预算下减少约一半工具调用。**AEPO** [AEPO] 修复 ARPO 的两个失败模式——连续高熵步骤耗尽分支预算（high-entropy rollout collapse）与高熵 token 梯度被 clip 剪除——引入熵预监控的预算分配与 stop-gradient 保护。**RAPO** [RAPO] 从外部 off-policy 轨迹库做 step 级检索注入，扩大探索来源，并以"高熵处检索后熵降"构造检索奖励。

*与本文关系*：三者共享"熵 = 探索信号"的假设。RAE 指出高熵必须与**分支结局的可分辨性**联合判断（$U=\bar H^\pi\bar H^B$），否则"全成/全败仍狂分叉"（G1）；且 RAPO 的熵降奖励与 IGPO 的信念增益同样面临"错误自信也表现为熵降"的风险（G2）。

### 2.3 Advantage 层：过程级信用分配

**IGPO** [IGPO] 把每一 turn 的奖励定义为模型对 ground-truth 答案概率的边际增量 $\Delta P_\theta(y^*)$，缓解稀疏奖励与优势坍缩。**GiGPO** [GiGPO] 在已生成的轨迹组内寻找可哈希的重复"锚点状态"，对同一状态下的动作做第二级组内相对优势，无需额外 rollout 或 critic；但依赖状态可精确复现，开放工具环境中观测文本几乎不会重复。**VinePPO** [VinePPO] 利用语言环境可从任意中间状态重置续采的特性，用蒙特卡洛 rollout 替代 value network 估计中间状态价值，显著改善数学推理的信用分配。

*与本文关系*：RAE 的反事实探针在机制上与 VinePPO 的 MC 估计同源，但估计目标不同——VinePPO 估**价值的期望**并直接作为 advantage；RAE 估**结局的熵**并仅用于选点（价值高低由后续的 validate/invalidate 承接）。对 IGPO，RAE 不接受任何未经外部结局校验的内部信念变化作为训练信号。

### 2.4 从失败中学习

**Reflexion** [Reflexion] 用语言化的自我反思改进后续尝试，但不更新参数，恢复能力停留在上下文技巧。**Agent Q** [AgentQ] 用 MCTS 生成探索树，以成败分支构成偏好对做 off-policy DPO，是"用失败分支训练"的早期代表。**Agent-R** [AgentR] 与本文动机最接近：用 MCTS 在首个错误点拼接失败前缀与成功后缀，构造"修正轨迹"做迭代自训练（SFT），让模型学会及时反思与修复。**WebRL** [WebRL] 从失败任务自举生成新任务，构成自进化课程。**结构化反思** [StructRefl] 把"反思-修复"显式化为可训练动作。

*与本文关系*：这条线确认了"错误恢复"是社区正在收敛的问题，但现有方法的失败利用方式是**构造更好的正样本**（拼接/偏好对/新任务），本质仍是模仿或偏好学习。RAE 的差异在于：失败**本身**即训练信号——无需找到正确后缀，也无需偏好对，被证伪的动作直接变成对目标策略支持集的约束（消除 + 重归一化）。当环境中难以采到成功分支时（Agent-R 与 Agent Q 均需成功侧存在），RAE 的失败侧目标仍然良定义。

### 2.5 拒绝采样微调与偏好/负梯度方法

**STaR** [STaR]、**RFT** [RFT]、**RAFT** [RAFT] 采样-过滤-模仿成功轨迹，与 RAE 成功侧的 $\delta$ 型目标同构——我们明确承认这一点：RAE 的成功侧不是新贡献，新颖性集中在失败侧与选点侧。**Unlikelihood training** [UL] 直接压低负 token 的概率；**DPO / KTO** [DPO][KTO] 通过偏好对或效用函数隐式产生负梯度。这些方法压低坏行为时**不约束剩余概率质量的去向**，可能把质量推向未经验证的任意方向；RAE 的消除目标显式保持其余动作的相对比例（KL 最小意义下对旧策略的最小修改，定理 1）。

### 2.6 动作消除的经典理论

Bandit 与 RL 理论中的 **action elimination** [EvenDar] 在置信区间意义下永久移除次优动作并保证样本复杂度。RAE 可视为其在 LLM 策略空间的软化、非平稳版本：消除仅相对 $\pi_{old}$ 在当前迭代内生效（§6.4），以适应能力随训练增长的非平稳性。

### 2.7 小结：定位

| 方法 | Rollout | Advantage/Reward | Loss | 失败的用法 |
|---|---|---|---|---|
| PPO | 独立轨迹 | GAE + critic | clip | 负 advantage |
| GRPO / DAPO | 组内独立完整采样（DAPO 过滤零方差组） | 组内标准化 outcome | clip（DAPO 非对称+token级） | 负 advantage |
| ARPO / AEPO | 高熵处分支（AEPO 加预算控制） | outcome + 归因/熵加权 | clip（AEPO 加 stop-grad） | 负 advantage |
| RAPO | 检索注入 off-policy step | outcome × 检索熵降 | clip + 比率整形 | 负 advantage |
| IGPO | 标准多轮 | $\Delta P(y^*)$ + outcome | clip + KL | 负信念增益 |
| GiGPO | 标准组采样 + 事后锚点 | 双层相对优势 | clip | 负 advantage |
| VinePPO | 中间状态 MC 续采 | MC value | clip | 低 value |
| Agent Q / Agent-R | MCTS 树 | 偏好对 / 修正轨迹 | DPO / SFT | 造正样本 |
| **RAE（本文）** | **双熵选点 + 失败触发探针** | **validate / invalidate 二值裁决** | **单一 KL 到重构目标** | **失败＝支持集约束** |

据我们所知，在 agentic RL 语境下，尚无已发表方法把失败定义为"删除被证伪动作后对旧策略重归一化的目标分布"。

---

## 3 预备知识：三层坐标系（Preliminaries）

Agent 任务形式化为 POMDP $(\mathcal{S},\mathcal{A},P,R,\Omega)$。策略 $\pi_\theta$ 在状态 $s_t$（完整交互历史）生成动作 $a_t$（一次工具调用或一段决策文本），环境返回观测 $o_t$，任务结束给出可验证结局 $Y\in\{0,1\}$。任何 agentic RL 算法都可分解为三层：

$$\text{Layer 1: Rollout（采什么）}\rightarrow\text{Layer 2: Advantage（什么算好）}\rightarrow\text{Layer 3: Loss（如何更新 }\pi_\theta)$$

**符号表**

| 符号 | 含义 |
|---|---|
| $H_t^\pi$ / $\bar H_t^\pi$ | 状态 $s_t$ 的策略熵 / 归一化策略熵 |
| $\hat p_t$, $H_t^B$ / $\bar H_t^B$ | 探针成功率、分支结局熵 / 归一化结局熵 |
| $U(s_t)$ | 探索价值 $\bar H_t^\pi\cdot\bar H_t^B$ |
| $Y_k\in\{0,1\}$ | 第 $k$ 条分支的可验证结局 |
| $c_t\in\{0,1\}$ | 外部矛盾信号（§6.1 操作化） |
| $a^+$ / $a_f$ | 被验证 / 被证伪的动作 |
| $\pi^{tgt}$ | 重构的目标策略 |
| $K$, $B$ | 探针数、完整扩展分支数 |
| $\tilde{\mathcal{A}}(s)$ | 采样支持集（开放动作空间的有限近似） |

在此坐标系下，GRPO 一族的更新可统一写作 clip 代理目标下的组相对优势回传；各方法的差异仅在于三层各自的选择（表 1，§2.7）。

## 4 问题：从轨迹优化到错误恢复（新的问题）

### 4.1 问题重述

传统 agentic RL 求解 $\max_\theta\ \mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]$，以完整轨迹为基本单元。我们主张对长时程 Agent，真正决定策略质量的是一个不同的对象：

> **错误恢复能力**：在动作已被环境证伪或信念发生冲突的状态 $s_t$ 下，策略能否（i）撤回已失效的承诺，（ii）在剩余备选中高效重新探索，（iii）收敛到被环境验证的新承诺。

对应的学习单元不再是 $\tau$ 整条，而是**恢复片段**（recovery segment）：

$$\sigma = (s_e, a_f, \underbrace{\text{explore}}_{\text{熵}\uparrow}, a^+, \underbrace{\text{commit}}_{\text{熵}\downarrow}),$$

其中 $s_e$ 是错误暴露状态。理想策略在 $\sigma$ 上呈现熵生命周期 $H:\ \text{low}\to\text{high}\to\text{low}$：承诺（低熵）→ 检测到证伪、重新打开备选（熵升）→ 恢复并再承诺（熵降）。

### 4.2 为什么现有信号学不到它（不足之处的形式化）

**（i）标量失败丢失结构。** 设 $\pi_{old}(\cdot|s)=[0.8,0.1,0.05,0.05]$，$a_1$ 被环境证伪。负 advantage 的更新方向是"压低 $\log\pi(a_1)$"，但概率质量流向不受控：可能流向 $a_2$，也可能流向词表中任何未验证的序列。环境证据实际上是一个**支持集约束**——$a_1\notin\mathrm{supp}(\pi^{tgt})$——而非一个数值惩罚。

**（ii）熵信号双向失灵。** 在错误循环（错→错→错）中熵可以持续稳定（无信号）；在灾难性自信中熵极低（ARPO/AEPO 永不分支）；在信念增益类奖励下，对错误答案的自信上升同样呈现熵降/正增益（奖励错误方向）。四象限分析（$H^\pi$ 高/低 × 可恢复性升/不升）中，现有熵基方法只能感知"高熵"两个象限，且无法区分其中的 productive / unproductive。

**（iii）reward hacking 的具体形态。** 高熵、稳定错误的轨迹偶然蒙对时，outcome RL 给整条正信号，中间的无效试错被一并强化。约束"从错误到正确的路径"（而非只看终点）才能要求中间犯错是有效的。

## 5 挑战（Challenges）

把"错误恢复"变成可训练目标，需要同时解决四个技术挑战：

- **C1 选点**：在一条长轨迹中，哪些状态值得付出反事实采样的代价？纯熵准则在全成/全败处浪费预算，在灾难性自信处漏检。
- **C2 裁决与归因**：一条从 $(s_t,a_t)$ 出发的续采失败，错误可能出在 $a_{t+5}$ 而非 $a_t$——证伪信号应打在哪个动作上？成功侧对称地存在"侥幸成功"的假阳性。
- **C3 目标构造**：如何把 validate / invalidate 写成一个不引入加权超参 $\lambda$、且对旧策略修改最小的训练目标？失败目标在 $\pi_{old}(a_f)\to 1$ 时如何保持数值稳定？
- **C4 开放动作空间**：LLM 的动作是开放 token 序列，$|\mathcal{A}|$ 无限——熵归一化、消除-重归一化、KL 目标全部需要有限支持集上的可执行近似。

§6 逐一给出解法：C1→§6.1–6.2，C2→§6.3，C3→§6.4/6.6，C4→§6.5。

## 6 方法：RAE（提出的方法）

RAE 的三层分工可以概括为：

> **熵决定在哪里重新采样；分支结局决定哪些行为被保留/删除；目标策略重构给出最终损失。**

### 6.1 Rollout 层：双熵选点与失败触发探针（解 C1）

**候选点检测。** 沿正常 rollout $\tau=(s_0,a_0,o_0,\ldots)$ 记录归一化策略熵 $\bar H_t^\pi = H_t^\pi/\log|\tilde{\mathcal{A}}(s_t)|$（支持集见 §6.5）。候选集由三路触发并集构成：

1. 高熵态：$\bar H_t^\pi$ 位于轨迹内 top 分位（继承 ARPO 的观察：工具返回后熵尖峰）；
2. **矛盾态**：$c_t=1$。$c_t$ 操作化为可插拔检测器的析取：工具报错/环境显式拒绝、代码执行或单元测试失败、检索结果与当前结论的轻量 verifier 判定冲突、self-consistency 投票分歧超阈；
3. **失败触发**：对最终 $Y=0$ 的轨迹，无论熵高低，其最后一个承诺点（最后一次不可逆动作前的决策态）强制进入候选。

路 2、3 专门覆盖低熵的**灾难性自信**象限——这是纯熵方法（ARPO/AEPO）的结构性盲区：错误承诺越坚定，熵越低，越不会被熵准则选中。

**反事实探针。** 对每个候选 $s_t$，从 $\pi_{old}$ 续采 $K$ 条轻量探针得结局 $\{Y_k\}$：

$$\hat p_t=\frac{1}{K}\sum_k Y_k,\qquad \bar H_t^B=\frac{-\hat p_t\log\hat p_t-(1-\hat p_t)\log(1-\hat p_t)}{\log 2}.$$

小 $K$ 下以 Beta 后验的期望熵替代点估计以降低方差：$\bar H_t^B=\mathbb{E}_{p\sim\mathrm{Beta}(1+\sum Y_k,\,1+K-\sum Y_k)}[H(p)]/\log 2$。连续奖励环境以 $R>\theta_R$ 二值化（消融备选：branch return 方差）。

**探索价值与扩展。**

$$U(s_t)=\bar H_t^\pi\cdot\bar H_t^B,\qquad \mathcal{S}^\star=\text{top-}m\ \arg\max_t U(s_t),$$

对 $\mathcal{S}^\star$ 中状态做 $B$ 条完整分支扩展。三种典型情形说明与 ARPO 准则（$U^{\mathrm{ARPO}}=H^\pi$）的差别：

| 情形 | $\bar H^\pi$ | $\bar H^B$ | $U$ | RAE 行为 |
|---|---|---|---|---|
| 高熵、全部探针成功 | 高 | $\approx 0$ | $\approx 0$ | 不扩展（怎么走都对，无学习价值） |
| 高熵、成败参半 | 高 | 高 | 高 | 完整扩展（最值得的反事实点） |
| 高熵、全部探针失败 | 高 | $\approx 0$ | $\approx 0$ | 不扩展，但触发 **dead-end 回传**（§6.3） |
| 低熵、$c_t=1$ 或失败触发 | 低 | — | — | 经路 2/3 进入候选，直接探针 |

**与四象限门控的对应。** 早期设想中的 recoverability $C_t$ 在此被操作化为 $\hat p_t$（可恢复性的 MC 估计）：象限 A/B（高熵、可恢复性升/不升）由 $U$ 区分；象限 C（低熵高可恢复）不触发、自然保留；象限 D（低熵低可恢复，灾难性自信）由路 2/3 捕获。

**信息论解释。** $\bar H^\pi$ 度量"模型未定"，$\bar H^B$ 度量"环境可分辨"。二者乘积是 BALD 型关于最优动作的互信息 $I(Y;a\mid s_t)$ 的代理上界思路：只有当模型的分歧能映射为结局的分歧时，一次分支扩展的期望信息增益才非零。严格的界与乘积形式的最优性作为附录推导目标（此处承认：乘积是满足"任一因子为零则价值为零"的最简单组合，替代组合形式进入消融 A5）。

**冷启动回退。** 训练初期难题上处处 $\hat p\approx 0$ 会使 $U\equiv 0$。此时按 $\bar H^\pi$ 回退为 ARPO 式选点，并将该题移入课程队列，待策略在邻近难度获得非零 $\hat p$ 后再启用双熵准则。

### 6.2 Reward 层：validate / invalidate，而非标量（解 C3 前半）

RAE 不把结局折算为奖励标量，也不与任何内部信号加权求和：

$$Y\ \longrightarrow\ \begin{cases}\text{validate}(a) & \text{分支经外部环境确认成功}\\ \text{invalidate}(a) & \text{分支被外部环境证伪}\end{cases}$$

明确排除两类被 G2 否定的信号：内部信念增益 $\Delta P_\theta(y^*)$（IGPO）与检索/步骤熵降（RAPO）。**信念的自信 ≠ 信念的正确**；任何内部量必须在 $Y=1$ 之后才转化为 validated commitment。

### 6.3 裁决规则与归因（解 C2）

在扩展点 $s^\star$ 上得到分支集合 $\{(a_i, Y_i)\}_{i=1}^{B}$，同一动作（语义去重后）的多条续采汇聚为动作级成功率 $\hat p(a_i)$。裁决采用**条件证伪/条件验证**，避免把下游错误归罪于当前动作：

- **invalidate**：$\hat p(a_i)=0$ 且经过 $a_i$ 的续采数 $\ge K_{\min}$（所有出路皆死，才裁决动作本身死）；
- **validate**：$\hat p(a_i)\ge p_{+}$（多数续采成功，抑制侥幸成功的假阳性；$p_+$ 消融）；
- **abstain**：介于两者之间的动作不进入目标重构，只作为 Renorm 的保留质量。

**Dead-end 上游回传。** 若 $s^\star$ 处所有动作全部被证伪（$\hat p_{s^\star}=0$，$K\ge K_{\min}$），则证据上移一步：对进入 $s^\star$ 的动作 $a_{t-1}$ 施加 invalidate。这实现了非对称原则——**无效探索惩罚初始承诺，有效探索奖励完整恢复过程**：整组皆死说明错误发生在更早的分叉承诺，而不应把整段探索胡乱负向平均。回传深度默认 1（递归回传的代价-收益进入 future work）。

### 6.4 目标策略重构（解 C3 后半）

**失败侧（核心）。** 被证伪的 $a_f$ 从支持集中删除，其余动作按 $\pi_{old}$ 重归一化：

$$\pi^{tgt}(a\mid s)=\begin{cases}0, & a=a_f\\[4pt] \dfrac{\pi_{old}(a\mid s)}{1-\pi_{old}(a_f\mid s)}, & a\neq a_f.\end{cases}$$

**定理 1（KL 投影）.** $\pi^{tgt}=\mathrm{Renorm}(\pi_{old}\setminus a_f)$ 是约束优化问题 $\min_{\pi:\,\pi(a_f)=0}D_{\mathrm{KL}}(\pi\Vert\pi_{old})$ 的唯一解。*证明概要*：在支持集约束下 KL 对 $\pi$ 的拉格朗日驻点即条件分布，凸性给出唯一性。∎

**命题 1（贝叶斯条件化）.** $\pi^{tgt}(a)=\pi_{old}(a\mid a\neq a_f)$，即把环境证伪当作证据事件的后验更新。∎

两个结果共同回答"为什么保持剩余比例不变"：这是**与证据一致的分布中对旧策略修改最小的一个**——失败教会 Agent 的只有"这条路不通"，不应臆造其余动作的新排序。

**成功侧（与失败侧对偶）。** 设被验证动作集合 $\mathcal{V}=\{a:\text{validate}(a)\}$：

$$\pi^{tgt}(\cdot\mid s)=\mathrm{Renorm}\big(\pi_{old}\restriction_{\mathcal{V}}\big),$$

即条件化到 $\mathcal{V}$ 上。$|\mathcal{V}|=1$ 时退化为 $\delta_{a^+}$（与 RFT/STaR 的成功侧同构，我们不声称此处新颖）；$|\mathcal{V}|>1$ 时天然消解多成功分支的目标冲突，并保留成功动作间的多样性以缓解 $\delta$ 目标的熵坍缩。进一步以温度 $T_{tgt}\ge 1$ 软化目标（等价 label smoothing）作为熵保护，$T_{tgt}$ 进入消融而非正文超参。

**数值稳定。** $\pi_{old}(a_f)\to 1$ 时 Renorm 放大系数 $1/(1-\pi_{old}(a_f))$ 爆炸，且剩余排序未必可信。为 $a_f$ 保留 floor 概率 $\epsilon_f$：$\pi^{tgt}(a_f)=\epsilon_f$，其余按比例分摊——等价于把"硬消除"软化为"$\pi(a_f)\le\epsilon_f$ 的约束投影"（定理 1 的约束松弛版）。极端自信的错误通过多轮 sequential elimination 逐步剥离，而非单步剧烈重构。

**熵是结果，不是目标。** 例：$\pi_{old}=[0.8,0.1,0.05,0.05]$，$a_1$ 被证伪 → $\pi^{tgt}=[0,0.5,0.25,0.25]$，熵自然升高；随后 $a_2$ 被验证 → 目标向 $\delta_{a_2}$ 收缩，熵降。恢复片段上的 low→high→low 生命周期由 validate/invalidate 序列诱导，无需任何熵奖励。**非平稳性声明**：$\pi^{tgt}$ 是 $\pi_{old}$ 的函数，消除仅在当前迭代内生效；能力增长后动作可重新进入支持集——区别于经典 action elimination 的永久删除。

### 6.5 开放动作空间的 token 层实现（解 C4）

LLM 动作是变长 token 序列，以下近似使 §6.1–6.4 的全部量可计算：

1. **采样支持集**：$\tilde{\mathcal{A}}(s)=\mathrm{dedup}\{a^{(j)}\sim\pi_{old}(\cdot\mid s)\}_{j=1}^{n}$（语义去重：规范化工具名+参数，或嵌入聚类）。$\pi_{old}(a\mid s)$ 以序列对数似然在 $\tilde{\mathcal{A}}$ 内 softmax 重归一（长度归一化对数似然，消融备选）。熵、Renorm、KL 全部在 $\tilde{\mathcal{A}}$ 上计算；$n$ 的截断偏差随 $n$ 单调减小，$n\in\{4,8,16\}$ 进入消融。**探针复用**：§6.1 的 $K$ 条探针就是 $\tilde{\mathcal{A}}$ 的采样来源，选点与裁决共享同一批样本，无额外开销。
2. **KL 的序列级实现**：$D_{\mathrm{KL}}(\pi^{tgt}\Vert\pi_\theta)$ 在 $\tilde{\mathcal{A}}$ 上展开为加权序列 NLL：$\mathcal{L}(s)=-\sum_{a\in\tilde{\mathcal{A}}}\pi^{tgt}(a)\log\pi_\theta(a\mid s)+\text{const}$，其中 $\log\pi_\theta(a\mid s)$ 是动作 token 的对数似然和。只对决策/动作 token 反传；观测与工具返回 token 一律 mask（同 IGPO 惯例）。
3. **首分歧退化版（低成本变体）**：仅在 $a_f$ 与其余候选的首个分歧 token 处施加消除-重归一化，退化为带比例保持的 token 级 unlikelihood；作为效率-精度权衡进入消融 A6。

### 6.6 Loss 层：单一 KL 与 GRPO/PPO 兼容

**主目标（无 $\lambda$）：**

$$\mathcal{L}_{\mathrm{RAE}}(\theta)=\mathbb{E}_{s\in\mathcal{S}^\star}\,D_{\mathrm{KL}}\big(\pi^{tgt}(\cdot\mid s)\,\Vert\,\pi_\theta(\cdot\mid s)\big).$$

成功侧展开为 $-\log\pi_\theta(a^+\mid s)$ 型模仿项，失败侧展开为比例保持的重分布项——**一个 KL 同时承载 exploit 与 explore，不存在熵系数、IG 权重或任何 $R_1+\lambda R_2$ 结构**。我们显式拒绝 $R=R_{\mathrm{outcome}}+\lambda_1 R_{\mathrm{IG}}+\lambda_2 R_{\mathrm{entropy}}$：线性加权中 $\lambda$ 没有理论地位，且熵作为奖励会被 G2 的错误自信劫持。

**与 clip 管线的兼容（可直接跑在 GRPO/verl 型框架上）。** 定义目标诱导优势

$$A^{\mathrm{RAE}}(s,a)=\log\pi^{tgt}(a\mid s)-\log\pi_{old}(a\mid s)$$

（validate 动作为正；invalidate 动作取 $\log\epsilon_f$ 截断的大负值；abstain 动作为小正数 $-\log(1-\pi_{old}(a_f))$，即 Renorm 的对数放大系数）。**命题 2**：以 $A^{\mathrm{RAE}}$ 回传的策略梯度与 $\nabla_\theta\mathcal{L}_{\mathrm{RAE}}$ 在 $\pi_\theta=\pi_{old}$ 处一阶一致（证明目标：两者都是 KL 在旧策略处的线性化）。于是标准管线

$$\mathcal{L}_{\mathrm{RAE\text{-}clip}}=-\mathbb{E}\Big[\min\big(r_t(\theta)A_t^{\mathrm{RAE}},\ \mathrm{clip}(r_t,1{-}\varepsilon,1{+}\varepsilon)A_t^{\mathrm{RAE}}\big)\Big]$$

可原样复用 DAPO 的 clip-higher 与 token 级归一。组相对结构保留为**裁决置信开关**：分支组成功率 $\bar Y$ 决定该组以哪侧目标为主（$\bar Y$ 高→成功侧模仿为主；$\bar Y$ 低→消除为主；$\bar Y\approx 0.5$→两侧信息都最丰富），仍无加权超参。

### 6.7 采样预算记账

设每题总预算 $M$（以生成 token 或环境步数计）。RAE 分配 $M = N\cdot\bar\ell + \underbrace{|\mathcal{C}|\cdot K\cdot\bar\ell_{probe}}_{\text{探针}} + \underbrace{m\cdot B\cdot\bar\ell_{branch}}_{\text{扩展}}$，其中探针为轻量续采（截断长度 $\bar\ell_{probe}\ll\bar\ell$）、且被 §6.5 复用为支持集与裁决样本。与 ARPO 的 $M=N+\text{branch}$ 对齐同一 $M$ 进行所有对比（§7），保证"效率主张"在同预算下成立。

### 6.8 训练流程（九步）

1. 初始 rollout：$\tau_1..\tau_N\sim\pi_{old}$，记录 $\bar H_t^\pi$；
2. 候选检测：高熵 ∪ 矛盾 $c_t{=}1$ ∪ 失败触发（§6.1）；
3. 反事实探针：每候选 $K$ 条轻量续采 → $\{Y_k\}$；
4. 结局熵：$\hat p_t$、$\bar H_t^B$（Beta 平滑）；
5. 选点：$\mathcal{S}^\star=\text{top-}m\,U(s_t)$；冷启动回退（§6.1）；
6. 完整扩展：$s^\star$ 上 $B$ 条分支（探针复用，增量采样）；
7. 裁决：validate / invalidate / abstain + dead-end 上游回传（§6.3）;
8. 目标重构：成功侧条件化到 $\mathcal{V}$，失败侧 floor-Renorm（§6.4）；
9. 更新：$\mathcal{L}_{\mathrm{RAE}}$ 或 $\mathcal{L}_{\mathrm{RAE\text{-}clip}}$，仅决策 token 反传。

---

## 7 实验设计（Experimental Design）

> 本节为设计方案，无实验结果。设计原则：每个实验回应正文的一个可证伪主张；所有对比在**同一采样预算 $M$**（§6.7）下进行。

### 7.1 环境与任务

| 类别 | 环境 | 选择理由 | 主要对比 |
|---|---|---|---|
| 具身/状态可复现 | ALFWorld、WebShop | 状态可哈希，GiGPO 的主场；检验 RAE 在锚点可得环境是否仍占优 | GiGPO、GRPO |
| 开放工具/搜索 | HotpotQA、2WikiMultiHopQA、Bamboogle、GAIA（文本子集） | 观测不可复现，工具后熵尖峰显著；ARPO/AEPO/IGPO 的主场 | ARPO、AEPO、IGPO |
| 代码/可验证执行 | SWE-bench-lite 或 LiveCodeBench-agent 子集 | 单元测试提供天然 $c_t$ 与二值 $Y$；错误恢复行为最丰富 | GRPO、DAPO、Agent-R |

模型：Qwen2.5-7B-Instruct（主）、Llama-3.1-8B-Instruct（泛化性）；14B 级一组验证规模趋势。

### 7.2 基线

- **Loss 层**：GRPO、DAPO；
- **Rollout 层**：ARPO、AEPO（开放工具环境）；
- **Advantage 层**：IGPO（搜索环境）、GiGPO（具身环境）、VinePPO（作为 MC 估计的公允对照）；
- **失败利用**：Agent-R（MCTS 拼接修正轨迹 + SFT）、RFT/STaR 型 rejection-sampling SFT（隔离"成功侧模仿"的贡献）；
- **无训练对照**：Reflexion（上下文恢复 vs 参数化恢复）。

预算对齐规则：所有含分支/探针/树搜索的方法按生成 token 总数对齐 $M$；RAE 的探针开销计入 $M$（不允许"免费探针"）。

### 7.3 主实验（H1：端到端性能）

**主张**：同预算下 RAE 在三类环境的任务成功率不低于最强基线，且在含长时程错误的难题分层（按首错位置分桶）上显著更优。
**报告**：成功率、pass@1/pass@4、每成功一次的 token 成本与 tool-call 成本（对标 ARPO 的效率叙事）、训练曲线（含熵曲线）。

### 7.4 机制性指标（H2：RAE 学到的是"恢复"）

比 success rate 更能证明机制的四个指标：

1. **Recovery Rate（RR）**：出现首个被证伪动作后仍最终成功的轨迹比例——直接度量"错→对"能力；预言：RAE 的 RR 提升幅度 > 成功率提升幅度（成功率可由"少犯错"提升，RR 只能由"会恢复"提升）；
2. **熵生命周期曲线**：以证伪事件为原点对齐，绘制事件前后各 $k$ 步的 $\bar H^\pi$ 均值带——预言 low→high→low 形态随训练涌现，而 GRPO/ARPO 组呈单调或无结构（§6.4 的可证伪预言，正文主图）；
3. **Invalidation Precision**：被 invalidate 的动作在 held-out 重放（更大 $K$ 的高置信探针）下确属不可恢复的比例——回应归因风险（C2）；
4. **重复错误率**：同一状态语义等价错误动作的复发频率——预言 RAE 显著低于负 advantage 基线（标量惩罚易反弹，支持集消除不易）。

### 7.5 消融（H3：每个组件都必要）

| # | 消融 | 回应的主张 |
|---|---|---|
| A1 | $U=\bar H^\pi\bar H^B$ → 仅 $\bar H^\pi$（ARPO 式）/ 仅 $\bar H^B$ | 双熵选点必要性（G1，§6.1） |
| A2 | Renorm 失败目标 → 均匀负目标 / 普通负 advantage | 比例保持的价值（G3，定理 1） |
| A3 | 单一 KL → $R_{out}+\lambda_1 R_{IG}+\lambda_2 R_{ent}$（网格搜 $\lambda$） | 无 $\lambda$ 主张：RAE 应不劣于调参后的最优加权组合 |
| A4 | 去掉失败触发/矛盾触发（仅高熵候选） | 象限 D 覆盖的必要性（§6.1 路 2/3） |
| A5 | $U$ 组合形式：乘积 → min / 加权和 | 乘积形式的经验合理性（§6.1） |
| A6 | 全序列 KL → 首分歧 token 退化版 | 实现精度-成本权衡（§6.5） |
| A7 | 去掉 dead-end 上游回传 | 非对称规则的贡献（§6.3） |

### 7.6 敏感性与稳健性（H4）

- $K\in\{2,4,8,16\}$、$B\in\{4,8\}$、$n\in\{4,8,16\}$、$p_+\in\{0.6,0.8,1.0\}$、$\epsilon_f\in\{0,10^{-3},10^{-2}\}$；
- 冷启动实验：从弱模型（1.5B/3B）起训，验证回退机制（§6.1）避免 $U\equiv 0$ 死锁；
- 非平稳性检查：追踪早期被消除动作在后期重新进入支持集的频率与正确性；
- Reward hacking 压力测试：构造"侥幸成功"陷阱任务（错误路径以小概率蒙对），对比各方法强化错误中间步骤的程度。

### 7.7 分析实验

- **探针预算-收益曲线**：固定 $M$，扫探针占比，验证"少量探针换来的选点质量优于等量盲目分支"（vs ARPO 的核心效率论点）；
- **与 Agent-R 的对照分析**：在成功后缀稀缺（难题）分层上对比——预言 Agent-R（依赖拼接成功路径）退化快于 RAE（失败侧目标不依赖成功样本存在）；
- **案例研究**：可视化恢复片段的 $\pi_{old}\to\pi^{tgt}\to\pi_\theta$ 演化与熵轨迹。

### 7.8 风险与预案

| 风险 | 预案 |
|---|---|
| 语义去重失败导致支持集碎片化 | 退回工具名+规范化参数的粗粒度键；A6 退化版兜底 |
| 探针截断长度过短导致 $\hat p$ 偏差 | $\bar\ell_{probe}$ 敏感性扫描；对截断未终止分支按保守规则计 $Y{=}0$ 并报告两种口径 |
| 具身环境成功率过高使候选稀疏 | 按任务难度分层采样，保证证伪事件密度 |
| 成功侧熵坍缩 | 监控承诺点熵；启用 $T_{tgt}$ 软化与 $|\mathcal{V}|>1$ 保留 |

## 8 局限与未来工作

（i）dead-end 回传深度固定为 1，多步信用回传的代价-收益未刻画；（ii）$U$ 与期望信息增益的严格界仍是推导目标而非已证结果；（iii）裁决依赖可验证结局 $Y$，对无 verifier 的开放任务需引入外部评审模型，可能重新引入 G2 风险；（iv）$c_t$ 检测器的召回率直接决定象限 D 覆盖，其错误率对训练的影响需要量化；（v）探针在不可重置的真实环境（真实网页、有副作用的 API）中不可行，需要世界模型或沙箱替代。

## 9 结论

RAE 把 agentic RL 的失败信号从标量惩罚升级为结构化的支持集约束：熵定位需要重新考虑的决策，反事实分支的外部结局裁决动作的存废，单一 KL 目标让探索与利用共享一个无超参的损失。我们期望这一"反事实策略精化"视角把社区的注意力从"找到成功路径"转向"学会从失败恢复"。

---

## 参考文献（均已核实，检索时间 2026-08）

- [PPO] Schulman et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347.
- [GRPO] Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
- [DAPO] Yu et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476.
- [ARPO] Dong et al. Agentic Reinforced Policy Optimization. arXiv:2507.19849. ICLR 2026.
- [AEPO] Agentic Entropy-Balanced Policy Optimization. arXiv:2510.14545.
- [IGPO] Wang et al. Information Gain-based Policy Optimization: A Simple and Effective Approach for Multi-Turn LLM Agents. arXiv:2510.14967.
- [GiGPO] Feng et al. Group-in-Group Policy Optimization for LLM Agent Training. arXiv:2505.10978.
- [RAPO] RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization. arXiv:2603.03078. KDD 2026.
- [VinePPO] Kazemnejad et al. VinePPO: Unlocking RL Potential for LLM Reasoning Through Refined Credit Assignment. arXiv:2410.01679.
- [AgentQ] Putta et al. Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents. arXiv:2408.07199.
- [AgentR] Yuan et al. Agent-R: Training Language Model Agents to Reflect via Iterative Self-Training. arXiv:2501.11425.
- [WebRL] Qi et al. WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning. ICLR 2025.
- [StructRefl] Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection. arXiv:2509.18847.
- [Reflexion] Shinn et al. Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
- [STaR] Zelikman et al. STaR: Bootstrapping Reasoning with Reasoning. NeurIPS 2022.
- [RFT] Yuan et al. Scaling Relationship on Learning Mathematical Reasoning with LLMs (RFT). arXiv:2308.01825.
- [RAFT] Dong et al. RAFT: Reward rAnked FineTuning for Generative Foundation Model Alignment. TMLR 2023.
- [UL] Welleck et al. Neural Text Generation with Unlikelihood Training. ICLR 2020.
- [DPO] Rafailov et al. Direct Preference Optimization. NeurIPS 2023.
- [KTO] Ethayarajh et al. KTO: Model Alignment as Prospect Theoretic Optimization. arXiv:2402.01306.
- [EvenDar] Even-Dar et al. Action Elimination and Stopping Conditions for the Multi-Armed Bandit and Reinforcement Learning Problems. JMLR 2006.
- [Survey] The Landscape of Agentic Reinforcement Learning for LLMs: A Survey. arXiv:2509.02547.

---

### 附：与源文档的映射（写作备忘，投稿前删除）

| 论文章节 | 来源 |
|---|---|
| §1.2 四缺口 G1–G4 | 体系说明 §2 + 分析.md §四 |
| §2.4/2.5/2.6 近邻对比 | 分析.md §4.9（Agent-R 为新核实补充，最重要竞品） |
| §4 恢复片段/熵生命周期 | 体系说明 §3.3/§15.4 |
| §6.1 路 2/3 失败触发 | 分析.md §4.2 象限 D 补丁 |
| §6.3 条件证伪 + dead-end 回传 | 分析.md §4.3/§4.4 |
| §6.4 定理 1/命题 1、floor、成功侧对偶 | 分析.md §4.5/§4.7/§4.8 |
| §6.5 token 层实现 | 分析.md §4.1（P0 断层的解法） |
| §6.6 $A^{\mathrm{RAE}}$ | 分析.md §4.7 |
| §7 全部 | 分析.md §五 扩展 |