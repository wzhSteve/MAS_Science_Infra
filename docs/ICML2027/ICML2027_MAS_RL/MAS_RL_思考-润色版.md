# Epistemic Planning: From Reward-Driven Policies to Experience-Driven World Models in Agentic RL

**认知规划：Agentic RL 从"用奖励优化策略"到"用经验优化内部信念"的范式转变**

> 思考整理稿 v1.3（position / framework draft）。整理自 MAS_RL 讨论记录；包含问题诊断、范式主张、核心机制、双输出世界模型与结构选型、形式化、研究路线图与实验底座。不含实验实现。
> v1.3（2026-10-10）：依 [世界模型驱动MAS-文献调研.md](世界模型驱动MAS-文献调研.md)（25 篇，两轮检索）修订——收紧全部 novelty 声明（三项合取）、修正信息论表述与 Reward Hacking 结论的 overclaim、§3.4 重写、新增 §3.5（与世界/价值模型的边界）、§5.4 扩充为四组近邻划界、§7 增补对照臂与指标。
> 一句话主线：**Current Agentic RL optimizes policies with rewards, whereas future Agentic RL should optimize internal beliefs with experience.**

---

## 摘要（Abstract）

当前几乎所有 Agentic RL 工作（DeepResearch、AgentFlow、OpenHands RL、WebArena RL、GAIA RL、GRPO-Agent 等）都建立在一个未被质询的假设之上：**Task Success = Good Planner**——只要奖励任务成功，Planner 就会变好。本文指出这个假设是错的，并给出一条完整的替代逻辑链：（1）**诊断**：Reward Hacking 不是奖励设计缺陷，而是"奖励只监督结果、不监督认知"这一学习目标错位的表象；由此可系统性地识别出现有范式的七个认知盲区（认知奖励缺口、系统知识缺失、预测无监督、能力边界无建模、失败被当作负信号、恢复能力无奖励、反事实无学习）。（2）**范式主张**：Planner 不应被当作追求外部回报最大化的策略网络（Policy），而应被当作追求认知误差最小化的认知系统（Cognitive System）；RL 的优化对象应从 Task Policy 转向 Internal World Model（能力模型、工具模型、环境模型、恢复模型），成功与失败不再是奖励的正负两端，而是两种证据——成功提供**确认性证据**，失败提供**修正性证据**。（3）**核心机制**：预测不止用于评估，而应**主动引导探索**（prediction-guided epistemic planning）——当 Planner 预测某计划会失败时，它应当设计"信息价值最大的失败"（诊断性试错 / 主动实验），使每一次失败都缩小假设空间并沉淀为可迁移的系统认知。在此框架下，无信息的 Retry 因信息增益趋零而在目标定义上不再获得正的内部信号——这为 Reward Hacking 提供了一个釜底抽薪式的**机制性回应**；但其经验有效性（hacking 行为是否确实减少、内部信号是否与外部行为对齐）是本文以实验检验的假设，而非由架构自动保证的结论。最后给出三阶段路线图（Outcome-driven RL → Epistemic RL → World Model RL）、失败类型学（无信息 / 诊断性 / 建构性）与待解决的核心技术问题。

**关键词**：Agentic RL；认知奖励；世界模型；信息增益；失败驱动探索；主动实验；Reward Hacking

---

## 1 问题：Task Success = Good Planner 这个假设错在哪里

### 1.1 一个所有人都默认的假设

当前 Agentic RL 的通用管线是：

$$\text{Task}\ \rightarrow\ \text{Plan}\ \rightarrow\ \text{Execute}\ \rightarrow\ \text{Success?}\ \rightarrow\ \text{Reward}\ \rightarrow\ \text{Policy Update}$$

它默认：只要最终成功获得奖励，Planner 的规划能力就会随之提升。但任务成功只是**外部结局**，它并不区分"Planner 理解了系统所以成功"与"Planner 碰巧成功"。

### 1.2 Reward Hacking 是表象，不是病根

典型现象：Planner 发现只要不断 Retry 最终也能成功，于是 RL 学会 Retry → Retry → Retry → Success，Reward 很高，但 Planner 什么都没学。这是**执行层优化**（execution-level optimization）冒充了**规划层优化**（planning-level optimization）。

社区对此的主流回应是"设计更好的奖励"（process reward、reward verifier、约束 Retry 次数等）。但真正的病根在于：

> **Reward 只奖励"成功"，从不奖励"为什么成功"。奖励信号没有监督 Planner 的内部认知，RL 只能学到捷径。**

我们把这个缺口称为**认知奖励缺口（Cognitive Reward Gap / Epistemic Reward Missing）**。Reward Hacking 只是它在行为层的投影：修补奖励函数是在治标；只要学习目标仍是"最大化外部回报"，策略网络就永远有动机寻找下一个捷径。

### 1.3 更深一层：这个范式在否定错误

现有范式还有一个隐蔽的副作用——它把失败一律折算为惩罚，等于**否定了错误的价值**。然而不管成功还是失败，实则都携带信息；尤其是"失败后如何探索并最终找到成功路径"这一过程，恰恰包含了最多能教会智能体"为什么失败、为什么成功"的信号。这一观察是后文全部主张的种子（§4–§6）。

---

## 2 诊断：现有 Agentic RL 的七个认知盲区

沿着 §1 的病根往下推，可以系统性地识别出七个相互关联的盲区。它们共同刻画了"奖励只监督结果"所遗漏的全部学习信号。

### 盲区一：没有学习系统知识（System Knowledge）

目前 Planner 学习的映射是 `Task → Plan → Success`；它实际应该学习的是整个多智能体系统的结构：

$$\text{Environment}\ \rightarrow\ \text{Executor}\ \rightarrow\ \text{Tool}\ \rightarrow\ \text{Capability}\ \rightarrow\ \text{Plan}$$

即 RL 最终应得到 **System Knowledge**（可迁移），而不是 **Task Policy**（任务绑定）。AgentFlow 学习 Action Trajectory，但仍不学习"Executor 为什么失败"，因此 Planner 无法迁移。

### 盲区二：预测没有被监督（Calibration）

Planner 输出 Plan A 并自信 $p=0.95$，Executor 失败，当前 RL 只给 $R=-1$ 便结束。但真正重要的信息是：**Planner 预测错了**。Prediction Error 才是学习信号（这正是 Predictive Processing 的核心主张）。奖励至少应包含校准项：

$$R = R_{\text{task}} - \lambda\,\lvert p - y\rvert,$$

其中 $p$ 是 Planner 的成功率预测、$y$ 是真实结局——这样 RL 学的是**预测能力**，不是运气。

### 盲区三：没有学习能力边界（Capability Boundary）

Executor 不会 OCR，Planner 让它读 PDF，失败，正常 RL 给 Penalty 便结束。真正该学的是一条可复用的知识：**"Executor 没有 OCR 能力"**。这类知识累积成 Capability Graph（OCR / Vision / Table Parsing / SQL……），Planner 此后所有任务都能受益。

需要划界（v1.3）：能力自评（competence / metacognition）本身已有人做——MUSE（arXiv:2411.13537）预测计划成功概率、MAGELLAN（arXiv:2502.07709）预测自身胜任度与学习进度、CSA（arXiv:2606.00251）用 RL 教 LLM 知道自己不会什么。本框架与它们的差别在于粒度与对象：它们是**标量/路由级**的「行不行」判断（单 agent、自身）；本框架维护的是**结构化的执行链信念**——「Executor 在这类子目标上如何落地参数、这个 Tool 在这种输入下如何失败」，预判内容本身就是被监督的对象（三层残差的参数层与工具层）。Capability Model 与 Task Policy 的区分不是空位，**空位是「能力认知的可执行内容」而不仅是其标量投影**。

### 盲区四：失败被当作负信号，而非知识增益（Knowledge Gain）

传统流程是 `Failure → Negative Reward`；应当变为：

$$\text{Failure}\ \rightarrow\ \text{Identify Cause}\ \rightarrow\ \text{Repair}\ \rightarrow\ \text{Knowledge Update}\ \rightarrow\ \text{Positive Signal}$$

例：第一次失败发现 Google Tool 超时，Planner 学到 `Google Tool Reliability ≈ 0.3`，此后全部任务受益——**这个失败是值钱的**。奖励应写作

$$R = R_{\text{task}} + \lambda\,\text{InformationGain},$$

其中信息增益可来自能力更新、工具可靠性更新、环境认知更新、世界模型更新。

> **表述精确性（v1.3）**：信息论上并非「失败恒比成功信息量大」——高置信预测下的意外成功同样携带大信息量。严格说法是：**传统的失败信号只保留了「失败」这一个 bit，不作结构归因就丢弃了绝大部分互信息**；本框架三层残差（§5.1）的职责就是把这部分被丢弃的信息交还给学习信号。此修正同样适用于 §3.1。

### 盲区五：恢复能力没有被奖励（Recovery）

现有 Reflection 流程 `Failure → Repair → Success` 中，RL 仍只奖励最终 Success，"迅速定位并修复"这一恢复能力本身无信号。未来奖励应纳入 Recovery Efficiency：修复代价、修复步数、修复成功概率。

### 盲区六：没有学习反事实（Counterfactual）

Planner 用 Google Search 成功了，但它不知道 Wikipedia 也能成功。RL 应学习 Alternative Plan Evaluation，使 Planner 形成 **Plan Space** 的结构认知，而不是记住 **One Trajectory**。

### 盲区七：内部模型的准确性没有被奖励（Internal Model）

以上六个盲区的汇聚点：Planner 应维护 Capability Model、Tool Model、Environment Model、Executor Model，而奖励应该衡量的是**这些内部模型是否随经验越来越准确**。RL 真正的优化对象应是 Internal World Model，而不是 Task Return。

### 小结：统一根源与双目标形式化

七个盲区共享一个根源：**当前 Agentic RL 把 Planner 当成了策略网络（Policy），而不是认知系统（Cognitive System）**。策略网络天然追求最大化外部回报，因此利用一切捷径；认知系统追求的是最小化对世界的认知误差（Epistemic Error）。这意味着优化目标需要从单一的 Maximize Return 转向双目标：

$$\max_\pi\ \ \mathbb{E}[R_{\text{task}}]\ -\ \lambda\,\mathcal{E}_{\text{epistemic}},$$

其中第二项衡量 Planner 内部认知与真实系统的偏差，可分解为：

| 认知误差分量 | 含义 |
|---|---|
| 能力预测误差 | 是否准确预测 Executor/Tool 能否完成当前步骤 |
| 环境预测误差 | 对环境状态、工具可靠性的估计是否准确 |
| 恢复预测误差 | 是否准确判断失败可否恢复、恢复代价多大 |
| 世界模型误差 | 能力图、工具图、因果图是否随经验逼近真实系统 |

一句话概括这次转向：**Planner 学习的对象应当是 MAS 本身，而不是 Task；Agentic RL 学习的应当是 System Cognition，而不是 Action Policy。**

---

## 3 范式主张：用经验优化信念，而非用奖励优化策略

### 3.1 为什么失败比成功更值钱（信息论论证）

范式转变的第一块基石来自一个信息论观察：

- **成功且被高置信预测**：Prediction = Success，Reality = Success——符合预期，信息增益低，Planner 只是确认了已有认知；
- **失败且被高置信预测**：Planner 自信 $0.95$，结果失败——信息增益大，Planner 必须回答"为什么？是 Executor、Tool、Environment 还是 Plan 出了问题？"

严格表述（v1.3）：决定信息量的是**预测与现实的偏差（surprise = 预测分布与真实结果的互信息）**，不是失败/成功这个二值标签本身。高置信预测下的意外成功同样是高信息事件。工程语境里失败之所以在**实践中**更常是信息富矿，是因为（a）多数系统对成功不作归因（成功不告诉你为什么成功），（b）失败的陈列面更宽：参数、工具动力学、结局三层都有暴露口。因此更准确的口号是：**被高置信预测错的执行才是认知更新的主要来源**；本框架对其做结构化归因（三层残差），而传统奖励把它压成一个 −1。

因此真正推动世界模型成长的是 **Prediction Error**，而不是 Reward。这与主动学习、贝叶斯信息价值、Predictive Processing 的思想一脉相承。

### 3.2 成功与失败是两种证据，不是奖励的正负两端

在开放世界、多智能体、长时程任务中，真正有价值的不是"这一次是否成功"，而是**每一次交互后智能体对系统的认识是否更准确**。据此重新定义交互产物：

- 成功提供**确认性证据**（Confirmatory Evidence）：确认先验（Confirm Prior），强化当前认知；
- 失败提供**修正性证据**（Corrective Evidence）：修正先验（Update Prior），暴露认知偏差，驱动能力/工具/环境/恢复模型更新。

由此甚至可以主张：这个信号不该再叫 Reward——Reward 天然蕴含 Good/Bad 的二元评价，而 Planner 每次交互真正获得的是 **Experience**。相应的范式可以命名为 **Experience-centric RL** 或 **Knowledge-driven RL**。

### 3.3 优化对象的迁移：Policy 成为世界模型的副产品

传统 RL 优化 $\max_\pi \mathbb{E}[R]$。新范式中 Planner 维护内部模型 $M_t$（注意：$M_t$ 不是 Memory，而是 Internal Model——包含 Capability、Environment、Tool Reliability、Recovery 等分量），每次交互更新的不是 Policy 而是模型：

$$M_{t+1} = \mathrm{Update}\big(M_t,\ \mathrm{Observation}(a_t)\big),\qquad \pi(a\mid M_t).$$

两条范式链的对照：

- **当前范式**：Experience → Reward → Policy Update；
- **新范式**：Experience（成功 + 失败）→ Belief Update → World Model Improvement → **Policy Emerges**。

关键区别在于：新范式里 **Policy 不再是直接优化的对象，而是世界模型不断完善后的自然产物**。这已经从传统强化学习走向了认知学习（Cognitive Learning）。

### 3.4 新颖性的诚实定位

需要预先声明："只用经验、不用奖励"并非全新观点——神经科学中的 Predictive Processing / Active Inference 以及部分世界模型工作早已强调预测误差与经验更新。真正的新颖性不在"失败也有价值"这一直觉（它已有理论基础），而在于回答：

> **如何把"失败带来的认知更新"形式化为一种适用于 LLM Agent 与 Multi-Agent Planner 的学习机制？**

就目前的文献检索而言，v1.2 原稿的表述「尚无工作把 Capability Belief、Epistemic Calibration、Failure-driven World Model Update 作为 RL 的核心优化对象」已不成立——RWML 把预测误差当 RL 信号、WorldEvolver 用失败驱动记忆信念修订、DyMo 双输出动作与执行后状态、ReasonPlanner 对比预测与真实状态触发重规划。**正确的定位是一个三项合取，每项的半边都被占住、合取仍然空着：**

1. **预判对象是 MAS 自身执行链**（Executor 参数落地、Tool 搜集返回、失败落在哪层），而非环境状态 / 网页快照（DyMo、FOREAGENT、SiRA、WebDreamer 已占环境侧）、亦非标量自评（MUSE、MAGELLAN、CSA）或他者心智（BayesBeliefAgent、MetaMind、CoBel-World）；
2. **残差三层结构化、按层写入持久信念**（Cap / ToolRel / Env / Rec）——DyMo 残差只用于当轮打分、ReasonPlanner 反思是 prompt 级不训练不持久化、RWML 残差是 embedding 空间标量，均无信念体系；
3. **残差只有改写后续 plan 才计入信息 \(I\)，且预测用于主动设计诊断实验**（argmax IG、失败类型学）——WorldEvolver 是被动修订、FOREAGENT 只用预测剪枝。

同时，"利用失败"本身也不足以构成 novelty（Curriculum、Self-play 都在用失败；WorldEvolver / BayesBeliefAgent 也已把失败当证据）。在三项合取的限定下，本文的差异化主张是：

> **Failure 不是负样本，而是认知更新（Belief Revision）的主要来源——且只有改写后续决策的信念更新才计入信息。**

$$\text{现有：Failure}\rightarrow\text{Penalty}\qquad\Longrightarrow\qquad \text{主张：Failure}\rightarrow\text{Evidence}\rightarrow\text{Belief Update}\rightarrow\text{Capability Update}\rightarrow\text{World Model Update}$$

### 3.5 与价值/评论模型（Critic / ORM / PRM）的边界

一个必答的质疑：**为什么不把 Planner 的第二输出做成一个「预测结果」的标量头（outcome reward model），而必须是结构化的世界模型？** 三点区分（数学上三者的形式分别是：动力学 $f_\phi:(s,a)\mapsto s'$、奖励 $r_\phi:(s,a,s')\mapsto\mathbb{R}$、价值 $V_\psi(s)=\mathbb{E}[\sum\gamma^t r_t]$）：

1. **信息单向包含**：世界模型 + 奖励 + 策略 ⇒ 可推出任何 $V^\pi$（$V$ 是「世界模型 ∘ 策略」对轨迹的期望投影）；反向不成立——标量 $0.7$ 无法区分「Executor 参数落错」与「Tool 返回空结果」。critic 是 world model 的**有损投影**，把三层残差压成一个数字后，参数层与工具层的信念就**在数学上不可分更新**。
2. **Critic 的错误是隐性的**：分数自信地错，失败后你不知道哪层断了——正是 §4.5 的 Type I 无信息失败；世界模型的错误是**可观测的**：残差自带失败定位（failure locus）。
3. **IG 只能定义在观测预测上**：$\mathrm{IG}(H,a)$ 需要观测层面的预测分布 $p_{\phi}(o\mid a)$ 才能计算「哪个探针最能降低哪个信念的熵」；对标量 $\hat v$ 至多定义成功概率的方差，回答不了「探哪里能改写 plan」。本文的认知闭环（§4.2）在 critic 形态下第一步就断。

与 LLM-RL 现有术语对齐：ORM ≈ episode 级 critic，PRM ≈ step 级 critic，GRPO 组内基线是无模型的方差缩减——它们都在**评估**；DyMo / RWML / 本框架在**模拟**。**评估输出分数，模拟输出内容；内容可推出分数，分数推不出内容。** Critic 不是被抛弃：参照 Dreamer，value head 可以作为世界模型之上的 readout 服务于第二阶段门控 RL 的 advantage 估计——被替换的是「认知对象」，不是「价值估计」这一工程组件。对应的 value-only ablation 见 §7.3。

---

## 4 核心机制：预测引导的认知规划（Prediction-Guided Epistemic Planning）

§3 若止步于"预测误差 → 更新世界模型"，仍未抓住要害。关键的修正是：

> **Prediction is not merely for evaluation; prediction is used to actively steer planning toward informative exploration.**
> 预测不止用于评估计划的成败，而应主动控制 Plan 的方向：当预测到 Executor 会失败时，问题变成——**如何设计这次试错，使失败本身产生最大信息，并最终导向成功路径？**

这一步把框架从 predictive planning 推进到 **epistemic planning / failure-guided exploration**，也是本方向区别于现有 Agentic RL 的真正核心。

### 4.1 从"预测成功"到"利用预测控制探索"

**传统 Planner**（仍是 outcome-driven）：

```
Task → Plan A → Prediction: Success=30% → 换成功率更高的 Plan B → Execute
```

目标：$\max P(\text{Success}\mid \text{Plan})$。

**认知 Planner**：

```
Task → 生成 Plan A → 预测：Executor 很可能失败
     → 为什么会失败？这个失败能告诉我什么？
     → 设计一个"有信息价值"的试错 Plan → Execute → 观察 Failure
     → 更新 Executor/Environment Model → 重新规划 → Success
```

目标升级为：

$$\text{Expected Utility} = \text{Task Utility} + \lambda\,\text{Information Gain}.$$

**关键约束**：Information Gain 不能独立存在，它必须反过来改变下一步 Plan——否则会退化成"我收集了很多信息"，而不是"我通过有目的的试错改变了搜索方向"。

### 4.2 核心闭环

```
        ┌───────────────────────┐
        │     Internal Model    │
        │  Executor Capability  │
        │  Tool Reliability     │
        │  Environment Dynamics │
        │  Recovery Knowledge   │
        └───────────┬───────────┘
                    │
                    ▼
             ┌──────────────┐
             │    PLANNER   │
             │ (world model)│
             └──────┬───────┘
                    │  dual output
          ┌─────────┴──────────┐
          ▼                    ▼
     PLAN HEAD           IMAGINED EXEC
  subgoal, tool,       executor args,
  hypothesis           tool dynamics,
                       predicted obs,
                       failure locus
          │                    │
          └─────────┬──────────┘
                    ▼
         score inside imagination
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
       Exploit             Explore
   (likely success)   (predicted failure)
          │                   │
          │                   ▼
          │          Design informative
          │             experiment
          │                   │
          └─────────┬─────────┘
                    ▼
          EXECUTOR (selected only)
                    │
                    ▼
       real args / tool return
              / verdict
                    │
                    ▼
         three-layer residual
                    │
                    ▼
              MODEL UPDATE
                    │
                    ▼
              RE-PLANNING
```

即 **Dual Prediction（Plan + Imagined Execution）→ Planning → Exploration → Real Observation → Residual Update → Re-planning** 的自演化循环。PREDICTION 不再是一个成功率标量，而是 Planner 作为世界模型同时给出的两个头；完整输出契约、三层残差与结构选型见 §5。这比单纯的 Predictive Processing 更接近主动推断（Active Inference）。形式化地：

$$M_t\ \rightarrow\ \text{Prediction}\ \rightarrow\ \text{Action}\ \rightarrow\ \text{Observation}\ \rightarrow\ M_{t+1},\qquad a_t=\arg\max_a\Big[\text{TaskUtility}(a)+\lambda\,\text{InformationGain}(a)\Big].$$

Planner 不仅学习 World Model，还学习**如何通过行动主动改善 World Model**——这两者差别巨大。

### 4.3 失败本身应当被"设计"：诊断性规划

这是整个方向最有潜力的一点。传统 Agent 的立场是"我希望不要失败"；认知 Agent 的立场是：**"如果失败不可避免，我希望失败得有价值。"**

例：Executor 有三个未知能力——能否处理 PDF（A）、能否 OCR（B）、能否解析复杂表格（C）。直接执行完整任务 `Complex PDF → OCR → Table extraction → Reasoning` 失败后，你不知道到底哪一环失败了——这是**模糊失败（Ambiguous Failure）**，信息价值极低。

Planner 可以主动分解实验：

| 实验 | 设计 | 结果 | 学到的知识 |
|---|---|---|---|
| Experiment 1 | 只让 Executor 读 PDF 第一页 | 成功 | PDF parsing ✓ |
| Experiment 2 | 让 Executor 识别扫描文字 | 失败 | OCR ✗ |

于是最终 Plan 可确定为 `PDF → External OCR → Executor → Table reasoning`。这就是 **Failure-guided Planning / Diagnostic Planning**：用试错定位能力边界，再绕开边界组装成功路径。

### 4.4 与传统 Exploration 的本质区别：主动实验

传统 RL 的探索（$\epsilon$-greedy、熵最大化）本质是**随机尝试未尝试过的动作**。这里的主张不同：**基于当前模型的不确定性，主动设计最能区分不同假设的 Plan**。

设 Planner 有两个竞争假设：$H_1$：Executor 不会 OCR；$H_2$：Executor 只是处理不了大 PDF。它应选择实验

$$e^* = \arg\max_e\ IG(e;\,H),$$

即最大程度区分 $H_1$ 与 $H_2$ 的行动。这不是 exploration，而是 **Active Experimentation**（最优实验设计的规划版）。

### 4.5 失败的类型学

失败按信息价值分为三类，可作为理论框架的骨架：

| 类型 | 模式 | 信息价值 |
|---|---|---|
| **Type I：Uninformative Failure** | Plan A → Failure → 不知道为什么 | 低 |
| **Type II：Diagnostic Failure** | Plan A → Failure → 明确定位缺失能力（如 OCR ✗） | 高 |
| **Type III：Constructive Failure** | Plan A 失败发现限制 → Plan B 失败发现另一限制 → Plan C 成功 | 最高：每次失败都缩小搜索空间，失败序列在**构造**成功路径 |

### 4.6 失败甚至可以获得正信号

由此得到一个反直觉但自洽的结论：传统上 $R(\text{Failure})<0$，而本框架中失败的信号不必为负，更合理的形式是

$$R_t = U_{\text{task}} + \lambda\, I(\text{Failure}) - C(\text{Experiment}),$$

其中 $I(\text{Failure})$ 是该失败产生的有用信息量、$C$ 是实验代价。对照不同行为：

| 行为 | 最终结果 | 信息增量 | 评价 |
|---|---|---|---|
| 随机 Retry 10 次 | 成功 | 很低 | 差 |
| 一次诊断性测试 | 失败 | 很高 | 好 |
| 盲目换 Tool | 成功 | 很低 | 一般 |
| 针对能力边界试错 | 失败 | 很高 | 好 |
| 发现失败原因并修复 | 成功 | 很高 | 最好 |

> **一个有信息的失败，可能比一个没有信息的成功更有价值。**——这句话可以作为整个研究的核心 intuition。

### 4.7 双层规划：Plan to Act 与 Plan to Learn 的统一

认知 Planner 实际在做两层规划：

- **Object-level Planning**：如何完成 Task？（How can I succeed?）
- **Meta-level Planning**：如何通过行动获得信息，从而知道如何完成 Task？（If I cannot succeed now, where should I probe the system to reduce uncertainty? What failure would be most informative for reaching success?）

即 $\text{Plan to Act}$ 与 $\text{Plan to Learn}$ 的统一。

### 4.8 对 Reward Hacking 的釜底抽薪

回到 §1 的病例，本框架给出的不是补丁而是消解：

- **疯狂 Retry**：Retry → 没有新信息 → Belief 无变化 → $IG\approx 0$——即便最终 Success、传统 Reward 给 $R=1$，它也不是好的探索策略；
- **尝试新 Tool 后失败**：发现 Tool 不支持 OCR → Capability Model 更新——即便 $\text{Task}=\text{Failure}$，也有 $IG\gg 0$，是高价值经验。

于是无需人为规定"Retry 不得超过 5 次"：Agent 自己会发现无信息增量的 Retry 没有价值。这比一切基于奖励修补的 Reward Hacking 防御都更根本。

**结论强度的诚实校准（v1.3）**：以上是机制论证，不是经验断言。"被釜底抽薪"应理解为**在该框架的目标定义下，无信息 Retry 不获得正的内部信号**——这是一个设计性质（by construction）；它不能自动推出「部署中 Reward Hacking 行为会消失」：策略可能仍因 \(U_{\text{task}}\) 项或优化缺陷而走捷径，两项近似（\(\lambda\) 的选择、\(I\) 的估计误差）也可能被利用而衍生新的 hacking（如伪造高信息残差）。因此，「实验上是否显著降低 hacking 行为并将内部信号与外部行为对齐」必须作为 §7 的待验证假设，\(\hat{e}\) 的表示精度与残差对比噪声等实现差距也需在实验中显式衡量，而不是由架构自动保证。

---

## 5 可执行形态：Planner 作为认知世界模型

§4 把预测从评估提升为探索的控制器，但预测的对象仍接近一个标量：这条 plan 会不会成功。§2 盲区二的校准项 $R=R_{\text{task}}-\lambda\lvert p-y\rvert$ 因此只监督结局层。本节把该标量降级为三层残差中的一层，并规定 Planner 的输出契约、真实调用的门控，以及哪类结构能承载这个训练目标。问题诊断（§1–§3）不变；这里只把机制写成可执行形态。

### 5.1 现稿缺的那一层：可比较的想象轨迹

§4.2 的内部模型已有四块信念：Executor Capability、Tool Reliability、Environment Dynamics、Recovery。预测仍停在“这条 plan 会不会成功”。执行细节被外包给真实 Executor 和真实 Tool，Planner 只在结局回来之后才知道自己错了。

要补的是一条可比较的想象轨迹。Planner 在调用任何外部工具之前，先在内部滚一遍：

```
Task, Belief M_t
    → Plan          子目标、选哪个 tool、这条行动在检验哪个假设
    → Imagined Exec  Executor 会怎么填参数
                    Tool 会怎么检索 / 计算 / 返回什么形态的证据
                    预期观测、预期失败点、这次失败能区分哪些假设
    → 只对选中的那一条做真实执行
    → 逐层对比想象轨迹 vs 真实轨迹
    → 更新 M_{t+1}，再改 plan
```

这不是把 $\lvert p-y\rvert$ 写得更细，而是把优化对象从 Task Policy 换成对整条 MAS 因果链的内部模型。三层误差对应三块认知，不能压成一个标量：

| 层 | 预判 | 对照的真实量 | 错了说明什么 |
|---|---|---|---|
| 参数层 | query / code / URL / 参数 | Executor 实际发出的调用 | Planner 不懂 Executor 怎么落地，不是不懂任务 |
| 工具动力学层 | 会搜到哪类页面、代码会打印什么、证据是否可核验 | 工具真实返回 | 更新 Tool Model：可靠性、覆盖范围、超时、空结果 |
| 结局层 | 子目标是否被证据满足 | Diagnoser / Verifier 的裁决 | 才是 Capability Boundary 或 Environment 的更新 |

§2 盲区二的 $\lvert p-y\rvert$ 在此降级为结局层的一个分量。主监督改为三层轨迹残差。只有结局层有监督时，一次 Google 超时和一次“搜到了但证据不对”会被涂成同一个失败，正是 §4.5 的 Type I 无信息失败。三层对比把失败变成可定位的修正性证据。

学习信号保持 §4.6 的原则，不把失败当成负奖励：

$$R_t = U_{\text{task}} + \lambda\, I(\hat{e}_t, e_t) - \beta\, C(\text{real call}).$$

$I(\hat{e}_t, e_t)$ 是想象轨迹与真实轨迹的差异里、真正改变了下一步 plan 的那部分信息。无信息 Retry 的 $I\approx 0$，即便最后成功也不该被当成好探索。真实工具调用是实验，不是搜索过程本身。

### 5.2 为什么 AgentFlow / EPC-AW 的调用环不适合这个目标

两者是同一条工具链上的两种用法，不是两种环境。

**AgentFlow**（arXiv:2510.05592，ICLR 2026 Oral）四模块：Planner $\pi_\theta$、Executor、Verifier、Generator，共享 memory $M$ 和工具集 $K$。每回合：

$$a^t \sim \pi_\theta(a^t\mid q,K,M^t),\quad e^t\sim\mathcal{E}(e^t\mid a^t,K),\quad v^t\sim\mathcal{V}.$$

只有 Planner 用 Flow-GRPO 训练。轨迹级成败被广播到每一步。Planner 学的是“选哪个工具能让最终答案对”，不学 Executor 为什么失败。

五件工具（论文实验设定与附录 E.2）：

- **Base Generator**：不调外部工具，Qwen2.5-7B 直接推理。
- **Python Coder**：生成并执行受限算术代码，无第三方库。
- **Google Search**：实时网页检索，返回 Top-K 摘要。
- **Wikipedia Search**：Wikipedia API 拿 URL，再切块、embedding、RAG 摘要。
- **Web Search**：给定 URL 抓页面，同样切块 RAG。论文建议它跟在 Google / Wikipedia 后面用。

Flow-GRPO 之后，2Wiki 上 Google Search 用量上升 42%；MedQA 上 Google 从 66.2% 降到 10.9%，Wikipedia 升到 59.8%。模型学会的是换搜索源，不是建立对工具动力学的可迁移认知。

**EPC-AW**（arXiv:2605.23414）三角色：Planner、Executor、Diagnoser，记忆故意不共享。Planner 出 $(g,a)$，Executor 把 $a$ 实例化成带参数的工具调用，Diagnoser 判断结果是否支持计划。IPS 在一轮内对 $n=9$ 个候选计划做跨 agent 一致性选择；CESR 把跨轮不一致写成约束写回 Planner 记忆。评测集是 Bamboogle、2Wiki、HotpotQA、Musique、GAIA、MedQA。

工具集明确写着 following AgentFlow：同一五件。他们把官方 Google API 换成了本机 Chrome 抓 Top-10，避开 Gemini 掺进检索结果。延迟表（论文附录）是这条环不适合认知规划的直接证据：

- Google Search 约 80 s（多跳抓页 + 浏览）；
- Wikipedia Search 约 40 s；
- 30B 一次 LLM 推理约 3 s。

时间复杂度被写成 $\mathcal{O}\big(T(\tau_{\text{llm}}+\tau_{\text{tool}})\big)$，作者自己说总延迟由外部工具主导。认知规划要的是多次诊断性试探；在这条链上每一次试探都是一次 40–80 秒的真搜索。IPS 的 9 个候选只是多了 LLM 打分，选中的那条仍然要真搜。

对世界模型更致命的是：Google / Wikipedia / Web Search 的返回都再经过 LLM 摘要。预测误差会混进三件事——工具没搜到、摘要模型丢了证据、计划本身错了。这正好制造 Type I 模糊失败。网页也不可重置、非平稳，不适合当世界模型的训练动力学。

### 5.3 StructAgent：像在三角色，不像在认知算法

StructAgent（arXiv:2607.11388）是长程计算机使用上的 planner–actor–verifier。共享的不是原始历史，而是一份只有 verifier 能提交的任务状态（里程碑 + 已绑定事实 + 证据）。Planner 读状态、选下一个子目标，不能自己标完成。Actor 做一小段动作。Verifier 用证据决定 verified / invalidated / 不变。失败归因再路由回 planner、actor、verifier 或环境。

它和 AgentFlow / EPC-AW 的相似点只有三角色闭环。环境完全不同，所以慢搜索消失了。

Actor 侧可改状态的动作（论文附录）：

- GUI / pyautogui：`click`、`left_double`、`right_single`、`drag`、`type`、`hotkey`、`scroll`、`wait`（坐标在 0–1000 网格上）；
- 宿主：`cli_run`、`edit_json`、`extract_info`、`open_app`、`navigate`；
- 应用：`calc_*`、`writer_*`、`impress_*`（LibreOffice）。

Verifier 侧只读探针，不能当执行工具：`file_grep`、`url_match`、`a11y_match`、`calc_verify`、`impress_verify`、`writer_verify`，以及白名单只读 `shell_command`（`which`、`pgrep`、`test`、`gsettings get`、`dpkg -l`、`snap list`，禁止管道和重定向）。域不同，暴露的探针不同：Chrome 给 a11y / URL / file；Calc 给表格核验；OS 给 file 和 shell。

评测是 OSWorld-Verified、Mind2Web，以及 Minecraft 实例。Minecraft 里子目标是 $(\text{verb},\text{item},n)$，例如 $(\text{craft},\texttt{iron\_pickaxe},1)$，verifier 只看背包计数；actor 做导航、合成、放置。没有 Google Search。

Qwen3.5-9B 在 OSWorld-Verified 上从 27.0% 到 46.9%，27B 从 31.6% 到 62.2%，MiniMax-M3 到 78.9%。失败审计里 actor、planner、verifier 大约各占三成。它解决的是“进度必须有证据才能提交”，不是“行动前预判执行会怎样”。

因此 StructAgent 不能直接换成 Planner 世界模型：

- 它去掉了约 80 秒的搜索，换成本地、可检查、可重置的桌面或游戏；
- Planner 不输出对 actor 参数和工具返回的预判。Actor 先做，Verifier 事后盖章；
- 学习信号仍是任务是否完成，不是认知误差；
- 它适合当执行环境（快、轨迹结构化、失败可归因），不适合当认知算法。

### 5.4 更接近“先预判再执行”的近邻

v1.3 按 [文献调研](世界模型驱动MAS-文献调研.md) 扩为四组。每一组都是一次答辩必问，须逐项划开。

**组一：双输出 / 想象执行（形最近）**

- **DyMo**（arXiv:2506.02918）：同一 LLM 微调输出 tool call **和** 调用后工具引擎状态，SVS 在想象中给 N 个候选打分选优、不查真实环境。**与双输出形最近**。缺：单 agent 自我工具调用（无 Planner–Executor 分工）；预测对象是环境侧状态，不分层；残差只用于当轮打分、不写成持久信念；无信息量判据。
- **WebDreamer**（Gu et al., arXiv:2411.06559）：LLM 当互联网世界模型，行动前用自然语言想象页面会怎么变（change / a11y / html），再打分，只执行分数最高的动作。差别在目标：它最大化任务进度，不更新 Executor / Tool / Capability 信念，也不把预判误差当训练信号。想象的是网页状态，不是“Executor 会怎么填参数、工具会怎么搜集证据”。

**组二：预测-对比-重规划（机制最近）**

- **ReasonPlanner**（arXiv:2410.09252）：TKG 世界模型生成假设轨迹；actor 执行后 critic 对比预测与真实下一状态、反思差异、触发重规划。**预测-观测残差驱动重规划的字面先例，可引作动机**。缺：世界模型对象是环境状态图景，不预判 executor 参数与工具动力学；反思是 prompt 级，不训练、不沉淀为信念——同类工具错误下次照旧。
- **WorldEvolver**（arXiv:2606.30639）：部署期从预测-观测不匹配中修订情景/语义记忆，agent 参数冻结。它证明「残差 → 信念」路径本身有效（本框架可借证），但修订是被动检索记忆、不训练 Planner，也无主动实验设计。
- **RWML**（arXiv:2602.05842）：GRPO 最小化模拟与真实下一状态的 embedding 差。占「预测误差当 RL 信号」半边，但残差是环境状态标量距离、无三层结构与信念体系。

**组三：能力自评（认知对象最近，粒度最远）**

- **MUSE**（arXiv:2411.13537）：可学习内部模型预测所提计划的成功概率，用于策略选择。单 agent、标量自评，不预判执行内容。
- **MAGELLAN**（arXiv:2502.07709）预测自身胜任度与学习进度、指导课程；**CSA**（arXiv:2606.00251）用 RL 教 LLM 知道自己不会什么（其 RL>SFT 的结论可为本框架训练设计背书）。共同点：都是「行不行」的标量/路由判断，不是「怎么执行、哪层断」的机制预判。
- **BayesBeliefAgent / MetaMind / CoBel-World**：认知他者（队友技能、意图、信念），非本系统执行链。BayesBeliefAgent 的 belief–action gap 可借作指标。

**组四：想象门控执行**

- **SiRA**（arXiv:2507.23773）：planner 在内部世界模型上 simulative reasoning 选动作、actor 落地；world model 对象是环境信念态，actor 是自己的执行器。**FOREAGENT**（ACL 2026）：隐式世界模型执行前剪枝（9 小时物理执行 → 1 秒推理，动机叙事可引）。均非 MAS 分工、均无信念修订。
- **HiPER**（arXiv:2602.16165，ICML 2026）：同一模型先出子目标再出原子动作，ALFWorld 97.4%、WebShop 83.3%。环境是文本家居和模拟购物站，动作是 `go` / `take` / `open` / `heat` 和 `search` / `click` / `buy`，没有在线 Google。没有执行预判，奖励仍是任务成功。它说明 plan–execute 不必绑在搜索上。
- **RAP**（Hao et al.）：LLM 同时当策略和世界模型，在 blocksworld 一类小环境里做树搜索。动力学太窄，接不上多工具 MAS。

划界可以收成一句：这些工作想象的是环境下一步长什么样，用来选更可能成功的动作；本框架要想象的是 MAS 自己会怎么执行，用来发现信念哪里错了，并因此改下一份 plan。

逐项对照：

| 工作 | 做了什么 | 相对本框架缺什么 |
|---|---|---|
| AgentFlow | 优化工具选择，轨迹级成败广播到 Planner | 不监督 Executor 为何失败 |
| EPC-AW | 用跨 agent 一致性校准可行性判断 | 仍真调 Google；候选计划不在想象中滚执行 |
| WebDreamer / SiRA / FOREAGENT | 想象环境/网页/方案状态以选动作或剪枝 | 预判误差不是学习信号，也不更新能力/工具信念 |
| DyMo | 双输出 tool call + 引擎后状态，想象采样免调环境 | 单 agent、无分层残差、无持久信念、无 \(I\) 判据 |
| ReasonPlanner / WorldEvolver | 预测-观测差异触发反思/记忆修订 | 对象是环境状态；反思/修订不训练 Planner、不持久分层 |
| RWML | 预测误差当 RL 训练信号 | 环境状态 embedding 标量残差、无信念结构 |
| MUSE / MAGELLAN / CSA | 标量能力自评、知道何时委托 | 不预判执行内容、无三层残差；正确但太粗 |
| BayesBeliefAgent / MetaMind | 认知队友技能/意图 | 认知对象是他人智能体，不是本系统执行链 |
| StructAgent | 用证据提交进度；本地 GUI / shell / 应用工具 | Planner 不预判执行过程 |

都没有把“对 Executor–Tool 执行过程的预判误差”当成 Planner 的学习对象。

### 5.5 建议的系统形态

不要在 AgentFlow 的五工具上继续加 Diagnoser。保留三角色，改两件事：Planner 的输出契约，以及什么时候允许真调用。

```
M_t
  → Planner 世界模型
       plan_i                         子目标、工具、待检验假设
       imag_i = (executor_args,
                 tool_dynamics,
                 predicted_obs,
                 failure_locus)
  → 在想象里算  TaskUtility + λ·IG − β·Cost
  → 只把 argmax 的那一条交给 Executor 真执行
  → 记录真实 args、tool 返回、verifier 裁决
  → 三层残差更新 Capability / Tool / Environment / Recovery
  → M_{t+1} 条件下重新规划
```

训练环境二选一，不要用在线 Google 当默认动力学：

- **检索任务**：Hotpot / 2Wiki 的本地语料或 Wikipedia dump + BM25。工具返回是可复现文本，预判有明确对比目标，单次调用是毫秒到秒级。Google 只留作少量分布外探针，用来测 Tool Reliability，不进每步主循环。
- **结构化环境**：StructAgent 的 OSWorld，或 HiPER 的 ALFWorld。动作和观测本来就是类型化的（点击坐标、shell 参数、背包计数），三层残差可以直接对齐，不经过搜索摘要模型。

Python Coder 可以留下：它快、可重置、失败信息具体（报错栈），适合当能力边界实验。Base Generator 不是工具，不应和 Google 放在同一个动作空间里竞争。

---

## 6 路线图与定位

### 6.1 Agentic RL 的三阶段演化

| 阶段 | 范式 | 奖励/学习信号 | 核心优化对象 |
|---|---|---|---|
| **Stage 1** | Outcome-driven RL（当前） | Reward = Task Success | Task Policy |
| **Stage 2** | Epistemic RL | Reward = Success + Calibration + Capability Prediction | Cognitive Model |
| **Stage 3** | World Model RL | Reward = Success + Information Gain + World Model Improvement | Internal World Model |

三个阶段分别对应 §2 的校准与能力盲区（Stage 2）和 §3–§5 的完整闭环（Stage 3）。§5 把 Stage 3 写成可执行形态：双输出、三层残差、以及不用在线 Google 当默认动力学。这也与"认知校准（知道自己知道什么）→ 因果建模（知道为什么）→ 世界模型（形成可迁移认知）"的研究主线一一对应。

### 6.2 与现有 Agentic RL 的逐项对照

| 现有 Agent RL | 本框架 |
|---|---|
| 学 Task | 学 MAS |
| 学 Policy | 学 World Model |
| Success Reward | Prediction / Information Reward |
| 学 Trajectory | 学 Capability |
| Reactive Repair | Proactive Prediction |
| Outcome Supervision | Cognitive Supervision |

对现有工作四个核心信条的替换：

| 现有工作认为 | 本框架主张 |
|---|---|
| Reward Hacking 是奖励设计问题 | Reward Hacking 是认知学习目标错误导致的表象 |
| Planner 应学习获得高 Reward 的策略 | Planner 应学习建立对 MAS 的正确认知 |
| Success 是唯一学习信号 | Prediction Error、Capability Update、Information Gain 同为学习信号 |
| RL 最终得到 Task-specific Policy | RL 最终得到可迁移的 System Cognition / World Model |

### 6.3 研究问题的重新定义

不要把问题写成 *How to mitigate reward hacking in Agentic RL?*——这个问题太小。更有潜力的表述是：

> **How can an agent learn to plan experiments that turn failures into useful information for future success?**
> 或更学术化：**How can Agentic RL jointly optimize task progress and epistemic progress through failure-guided exploration?**

这里出现两个优化维度——$\text{Task Progress}$ 与 $\text{Epistemic Progress}$；传统 RL 只优化前者，本框架同时优化 $\text{Task Progress}+\lambda\,\text{Epistemic Progress}$。且需强调：Epistemic Progress **不是"预测得准"本身**，而是完整因果链：

$$\text{预测}\rightarrow\text{决定探索}\rightarrow\text{获得信息}\rightarrow\text{更新模型}\rightarrow\text{改变后续 Plan}\rightarrow\text{最终提高成功能力}.$$

若最终成文，核心机制建议命名为 **Epistemic Planning for Agentic RL**。Planner 每步不再选择 $a^*=\arg\max_a P(\text{Success}\mid a)$，而是

$$a^* = \arg\max_a\ \underbrace{P(\text{task progress}\mid a)}_{\text{Exploitation}} + \lambda\,\underbrace{IG(M;\,a)}_{\text{Exploration}},\qquad\text{更完整地：}\ a^*=\arg\max_a\big[\Delta\text{Task Success}+\lambda\,\Delta\text{Model Quality}-\beta\,\text{Cost}(a)\big].$$

公式本身不是重点；重点是它定义了一种新的 Agent 行为：**当不知道怎么成功时，不是盲目尝试，而是主动设计一个能最大化学习价值的尝试。**

### 6.4 一句话总结与待解决问题

整个思想的最终压缩：

> **Prediction-guided epistemic planning**：利用对执行结果的预测，主动决定下一步如何试错，使失败成为诊断系统能力、环境规律和因果关系的**实验**，从而持续更新 World Model，并反过来改变后续 Planning。
> 相应地，学习信号的原则是：**不是奖励失败，而是奖励"通过失败获得了多少能改变未来决策的信息"。**

这与 prediction accuracy、failure recovery、process reward、reward hacking mitigation 均有明确区别。将哲学直觉转化为可形式化 RL 问题后，剩余的核心技术问题为：

1. **如何定义"有价值的失败"**——信息增益 $I(\text{Failure})$ 的可计算形式（Type I/II/III 的判别准则）；
2. **如何让 Planner 选择诊断性试错**——$\arg\max_e IG(e;H)$ 在 LLM Planner 上的实现（假设表示、实验生成）；
3. **如何把试错信息沉淀为可迁移认知**——Capability Graph / World Model 的表示、更新与跨任务复用机制；
4. **如何证明认知探索带来长期收益**——epistemic exploration 最终确实提高长期任务成功率的实验验证方案。
5. **如何表示想象轨迹，以及哪一层残差算“改变了后续决策的信息”**——Plan 头与 Imagined Execution 头的结构化表示（子目标、工具、待检验假设；executor 参数、工具动力学、预期观测、失败位点）；参数层 / 工具动力学层 / 结局层的残差里，只有因此改写了下一步 plan 的部分计入 $I(\hat{e}_t, e_t)$（§5.1）。

---

## 7 实验底座：HIVE 做主环，StructAgent 做迁移

§5 规定了双输出与三层残差，§6.4 留下「如何证明认知探索带来长期收益」。本节把实验底座落到现有两个代码库上。论文要学的是：行动前预判 Executor–Tool 过程，再用三层残差改下一步 plan。HIVE 已经有这条因果链的日志和失败分类；StructAgent 已经有可重置、可核验证据的环境。两边都缺同一件事：Planner 的想象执行头，以及把残差当学习信号的训练环。§1–§4 的问题诊断不因本节改写。

### 7.1 两个代码库各自已经有什么

**HIVE**（`HIVE/MAS/hive/solver.py` 的 `solve`）是推理环：`analyze → propose → executor → diagnoser → commit / recovery`。仓库里没有 GRPO 或参数更新；`data/get_train_data.py` 只是把 NQ 和 DeepMath 合成 parquet。

- Planner 的结构化输出是 `formatters.py` 的 `NextStep`：`justification, context, sub_goal, tool_name`。没有执行参数、工具返回形态、失败位点。`generate_next_step` 里 `Epistemic_Constraint=""` 是写死的空字符串。
- Executor 另有 `ToolCommand`。真正的 query / code / URL 在执行时才生成。这正好是三层残差的参数层对照：预判的 args 对真实 command。
- Diagnoser 的 `REASON_INSTRUCTION` 已经把失败拆开：参数（`entity_missing`、`schema_mismatch`）、工具类（`capability_mismatch`）、环境（`paywalled`、`backend_timeout`、`import_whitelist`）、信念（`belief_conflict`）。`capability.py` 是一张写死的四能力表（retrieve / compute / perceive / synthesize），不是随经验更新的世界模型。
- Sibling pair 是真实执行之后的对照（换工具或换参数），写进 memory。它证明失败有结构，但发生在真调用之后，不是行动前的想象轨迹。
- 默认工具仍是 Base Generator、Python Coder、Wikipedia、Web Search、Google Search。评测集与 EPC-AW 相同：Bamboogle、2Wiki、HotpotQA、Musique、GAIA、MedQA。HIVE 自己的论文是 sibling probing（README 写明投 ICLR 2027）。本稿不把 sibling 对照再写成贡献。

**StructAgent**（`StructAgent/mm_agents/structagent/loop.py`）也是推理系统。Planner 的 XML 强制每个 `<step>` 带 `<text>` 和 `<expected_post_state>`（`core/planner/planner.py`）。这是「成功后屏幕应长什么样」的结局预判，不是「actor 会怎么填参数、工具会返回什么」。进度只由 verifier 探针提交。`_init_plan_teacher_demo_block` 写明离线训练不在发布代码里。环境是 OSWorld 虚拟机和 Mind2Web，一次任务是百步 GUI，不适合当第一篇的 RL 主循环。类型化动作（`cli_run`、`calc_*`、`navigate(url=...)`、文件探针）的参数是离散的，三层残差可以对齐；pyautogui 坐标不行。

### 7.2 主实验：在 HIVE 的 `solve` 上改三处

不重写 Diagnoser 的失败分类。

1. **工具动力学换成可复现的。** 主循环只留 `Python_Coder_Tool` 和本地检索（Hotpot / 2Wiki 语料或 Wikipedia dump + BM25）。Google / Web / Wikipedia 的在线调用退出每步循环，只在一小段分布外探针里测 Tool Reliability。Base Generator 移出工具竞争，继续只做最终作答。
2. **Planner 双输出。** 在 `NextStep` 旁增加想象执行：`executor_args`、`tool_dynamics`、`predicted_obs`、`failure_locus`、待检验假设。候选在想象里打分，只把 argmax 交给现有 `execute_tool_command`。
3. **三层残差当学习信号。** 参数层：想象 args 对 `ToolCommand`。工具层：想象的返回形态对工具原文（报错栈、检索片段），不经过第二遍 LLM 摘要。结局层：想象的子目标是否满足，对 Diagnoser 的 `SUBGOAL_COMPLETE` 和 `failure_type`。只有因此改写了下一步 plan 的残差计入 $I(\hat{e}_t, e_t)$。`capability.py` 的静态表当作先验，被残差更新，不再继续当唯一的能力判断。

训练分两截，都在本地工具上。先用 HIVE 现有 `inference_log` 轨迹做双头的监督：想象轨迹对齐真实 command 和 diagnoser 标签。再对 $U_{\text{task}}+\lambda I-\beta C$ 做 on-policy 更新，只更新 Planner。Executor 和 Diagnoser 保持冻结，否则残差所对照的真实量会跟着策略一起漂。

### 7.3 指标、五条对照、StructAgent 迁移

主表不只报 pass@1。同一预算下同时报：

- 任务成功率：相对冻结的 HIVE 不下降；
- 三层残差：参数 / 工具 / 结局分开；
- 无信息 Retry 比例：重复调用且下一步 plan 不变；
- **belief–action gap（v1.3，借 BayesBeliefAgent 指标）**：「信念更新了但下一步 plan 未变」的比例——按 §3.4 的信息判据，这类更新不应计入 \(I\)；该比例应随训练单调下降；
- 每成功一次的真实工具调用次数；
- 跨任务迁移：在任务 A 上学到的工具可靠性，是否改变任务 B 的首步工具选择；
- **hacking 行为计数（v1.3）**：纯 Retry 成功、换工具而不改 plan 的频率——直接检验 §4.8 的「釜底抽薪」是经验断言还是设计性质（该节 v1.3 已改为待验证假设）。

对照五条，都用同一套本地工具：

| 对照 | 设定 | 要分开的因素 |
|---|---|---|
| HIVE 原环 | sibling 在真执行之后，无想象头 | 现有推理系统 |
| 标量校准 | 只加成功率 $p$ | §2 盲区二 |
| **value-only 双头（v1.3）** | 第二头换成标量成功概率头（= ORM 式 outcome critic），其余训练流程完全一致 | **核心 ablation**：检验 §3.5 的论断——critic 形态下三层残差不可分、belief revision 退化。预期：选择不差，但残差定位、信念更新、plan 改写、跨工具迁移全面退化 |
| 双头、全执行 | 每个候选都真执行 | 有预判，无调用门控 |
| 完整环 | 想象打分、只执行一条、残差改 plan | 本稿的可执行形态 |

StructAgent 不进主表。迁移实验只取类型化子集：LibreOffice Calc 的单元格动作，加只读 `file_grep` / `shell_command`。把 `<expected_post_state>` 扩成同样的想象执行字段，verifier 探针当作结局层标签。要看的是 HIVE 上更新过的能力/工具信念能否改变 StructAgent planner 的下一步，而不是去追 OSWorld 上 78.9% 的成功率。GUI 点击坐标不纳入残差。

---

## 附：与源文档的映射（整理备忘）

| 本稿章节 | 源文档内容 |
|---|---|
| §1 | 开篇主张 + 一、Reward Hacking 是表象 + "被困在 reward 范式"段 |
| §2 盲区一~七 | 二~八（System Knowledge / Prediction / Capability / Failure / Recovery / Counterfactual / Internal Model） |
| §2 小结双目标 | "比 Reward Hacking 更深层的问题"段（$\max\mathbb{E}[R]-\lambda\mathcal{E}_{\text{epistemic}}$） |
| §3.1–3.3 | 5. 为什么失败更值钱 + 6. 重新定义 Reward + 7. 改变 RL 目标 + 8. 方向比 Reward Hacking 大 |
| §3.4 | "真正的 Novelty 在哪里" + Active Inference 提醒段（v1.3 重写为三项合取） |
| §3.5 | （v1.3 新增）世界模型 vs 价值/评论模型的理论边界；非源文档内容 |
| §4 | "并不是单纯的预测"修正后的 1–9 节（epistemic planning 完整展开）；§4.2 闭环图在 PREDICTION 处拆成 Plan 头与 Imagined Execution 头；§4.8 附加结论强度校准（v1.3） |
| §5 | 本稿新增：双输出世界模型、三层残差、AgentFlow / EPC-AW / StructAgent 结构选型；§5.4 v1.3 扩为四组近邻划界（双输出 / 预测-对比 / 能力自评 / 想象门控） |
| §6.1 | 三阶段表 |
| §6.2 | 两组对照表 |
| §6.3–6.4 | 10–11 节 + 最终循环图与压缩表述；§6.4 第 5 问为想象轨迹表示与残差计入条件 |
| §7 | 本稿新增：实验底座。HIVE 主环（本地工具、双输出、三层残差、五条对照含 value-only ablation），StructAgent 类型化子集做迁移（非源文档；§7.3 v1.3 增补 belief–action gap 与 hacking 计数指标） |
| （已去重） | 源文档中重复出现两次的 1–11 节与"结合过去几周讨论"段落仅保留一份 |
