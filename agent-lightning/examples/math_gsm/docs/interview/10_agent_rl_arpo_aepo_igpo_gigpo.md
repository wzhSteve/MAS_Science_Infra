# Agent RL 算法对照：ARPO / AEPO / IGPO / GiGPO

基线始终是 **vanilla GRPO**（本仓库 `algorithm.adv_estimator=grpo`，`rollout.n=4`，见 [train_math_agent.py](../../train_math_agent.py)）。四篇论文都是 GRPO 族在 **多轮 LLM Agent** 上的补丁，只是补的位置不同：

```text
Vanilla GRPO
  ├─ 改 rollout（树采样 / 熵分支）     → ARPO → AEPO
  ├─ 改 reward（稠密内在过程奖励）     → IGPO
  └─ 改 advantage（episode + step 两层）→ GiGPO
```

先会 GRPO 组内相对优势（[03_ppo_grpo_dpo.md](03_ppo_grpo_dpo.md)、[09_grpo_logit_backprop.md](09_grpo_logit_backprop.md)），再读本文。口述时先答「补哪个瓶颈」，再展开三层：rollout、reward、loss。

**结论先行**：没有 `pip install arpo` 这种独立算法包。全部是 **veRL fork / Hydra 补丁**。上游 verl 的 `AdvantageEstimator` 枚举里没有这四个名字。想跑论文设定，分别 clone 官方 repo。


| 算法 | 会议 | 论文 | 官方代码 |
| ---- | ---- | ---- | -------- |
| ARPO | ICLR 2026 | [arxiv:2507.19849](https://arxiv.org/abs/2507.19849) | [dongguanting/ARPO](https://github.com/dongguanting/ARPO) |
| AEPO | WWW 2026 Oral | [arxiv:2510.14545](https://arxiv.org/abs/2510.14545) | 同上仓库 `AEPO/` |
| IGPO | ICLR 2026 | [arxiv:2510.14967](https://arxiv.org/abs/2510.14967) | [GuoqingWang1/IGPO](https://github.com/GuoqingWang1/IGPO) |
| GiGPO | NeurIPS 2025 | [arxiv:2505.10978](https://arxiv.org/abs/2505.10978) | [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) |


---

## 1. 各算法一句话 + 要解决的坑

### 1.1 ARPO：Agentic Reinforced Policy Optimization

**坑**：工具返回后，模型接下来的前若干 token **熵飙升**——外部反馈把推理分布打散了。vanilla GRPO 只做整条轨迹独立采样，把高不确定的 tool-call 步和低不确定的纯推理步同等对待，探索预算浪费在已经确定的前缀上。

**补丁**：在高熵 tool-call 处 **自适应分支（partial / tree rollout）**。全局采 N 条完整轨迹，剩下 M−N 预算留给「工具刚返回、熵跳变大」的节点做 beam。Advantage 用 GRPO 的 **soft** 设定（共享前缀的 importance ratio 相同，梯度上等价于对共享段用平均优势）。

**场景**：多工具 TIR（search + python）、Deep Search（GAIA / HLE）。Qwen3-14B + ARPO 在 GAIA Pass@5 约 61.2%，训练 tool-call 次数约为 GRPO 的一半。

### 1.2 AEPO：Agentic Entropy-Balanced Policy Optimization

**坑**：ARPO 把熵当分支信号之后，出现两个新问题：

1. **High-Entropy Rollout Collapse**：高熵 tool 步经常连续出现，分支预算打在少数几条链上（论文统计 93.4% 的分支集中在 1–3 条轨迹）。
2. **High-Entropy Token Gradient Clipping**：树采样刻意保留的探索 token，一进 PPO/GRPO clip 就被裁掉梯度，探索刚开始就被掐死。

**补丁**：

- Rollout：先跑 1 条完整轨迹做 **熵预监控**，按 \(H_{\text{root}}\) vs \(H_{\text{tool}}^{\text{avg}}\) 动态分全局/分支预算；连续高熵步再乘分支惩罚。
- Loss：clip 里对高熵侧加 **stop-gradient**（来自 GPPO）；advantage 再乘 token 熵项。

**场景**：Web agent / Deep Search。同组后续工作，论文称在 14 个数据集上超过 GRPO、DAPO、ARPO、GiGPO 等 7 个算法。

### 1.3 IGPO：Information Gain-based Policy Optimization

**坑**：多轮 search agent 只有终局 outcome。组内全对或全错 → **advantage collapse**（标准化后 Â≈0，见 [04_algorithm_self_qa.md](04_algorithm_self_qa.md) Q5）；中间哪次 search 有用完全看不见。外部过程 RM 贵且有偏，Monte Carlo 逐步估值方差大。

**补丁**：不改采样拓扑。每轮用策略对 **ground-truth 答案的 logprob 增量** 当内在过程奖励：

\[
r_{i,t}^{\mathrm{IG}}=\log\pi_\theta(a\mid q,o_{i,\le t})-\log\pi_\theta(a\mid q,o_{i,\le t-1})
\]

再与 F1 outcome **分开组内 z-norm**，γ 折现成 turn 级回报，塞进 GRPO clip。reward 本身 stop-gradient，避免模型通过改 logπ 来刷奖励。

**场景**：多跳 QA / agentic search（NQ、HotpotQA、2Wiki、Musique）。需要 **可获取的 GT 答案** 才能算 IG。

### 1.4 GiGPO：Group-in-Group Policy Optimization

**坑**：长地平线（ALFWorld 可达 50 步、上万 token）稀疏奖励下，轨迹级 GRPO 把整条 episode 打同一个 Â，逐步决策没有 credit。对每个状态再 rollout 一组动作又贵得不可行。

**补丁**：同任务、**同一初始环境状态** 采一组轨迹（与 GRPO 相同的 LLM 前向次数）。两层相对优势：

- Episode 级 \(A^E\)：整条轨迹总回报的组内相对（就是 GRPO）。
- Step 级 \(A^S\)：事后把 **重复出现的环境状态**（同一房间、同一搜索结果页）聚成 anchor 组，比较从该状态出发的不同动作的折扣回报。

无 Critic、无额外 rollout。没有重复状态时 \(A^S=0\)，退化为 GRPO。

**场景**：ALFWorld、WebShop、有循环/回访的 embodied / web 环境。论文相对 GRPO：ALFWorld +12%、WebShop +9%。

---

## 2. 三层对照总表

读表顺序：先看「补哪一层」，再看「额外成本」。


| 维度 | GRPO（本仓库） | ARPO | AEPO | IGPO | GiGPO |
| ---- | -------------- | ---- | ---- | ---- | ----- |
| 主要补丁层 | — | **rollout** | **rollout + clip/Â** | **reward** | **advantage** |
| 采样拓扑 | 同 prompt 独立 G 条完整轨迹 | 全局 N + 高熵 tool 步 branch | 预监控后动态分 m vs k−m；连续高熵惩罚 | 同 GRPO | 同 GRPO，但必须共享初始 env 状态 |
| 分支判据 | 无 | \(\Delta H_t\) 超阈 → Branch(Z) | \(P_t=(\alpha+\gamma\Delta H_t)(1-\hat P(l))>\tau\) | 无 | 无；事后按状态哈希分组 |
| Reward | 轨迹级 outcome | 轨迹级：Acc + 格式 + 多工具 \(r_M\) | 同 ARPO 族 outcome | turn 级 IG + 终局 F1，分开 z-norm | 环境 r_t（可稀疏）；step 比较用折扣回报 |
| Advantage | 组内 \( (R-\mu)/\sigma \) | GRPO soft（共享前缀同 IS） | \(\hat A_{\mathrm{Acc}}(1+a\hat A_{\Delta H})\) | turn 级折现 \(\tilde R_{i,t}\) 替换 \(\hat A_i\) | \(A^E+\omega A^S\) |
| Clip | 标准 PPO clip | 标准 GRPO clip | 高熵侧 stop-grad rescale | 标准 GRPO clip | 标准 GRPO clip |
| 额外前向 | 无 | 每 tool 步再解 k token 算熵 | 预监控 1 条 + 同上 | 每轮（或向量化一次）算 GT logprob | 无（hashmap，论文称 <0.002% 时间） |
| 无 Critic | 是 | 是 | 是 | 是 | 是 |
| 工具 token | 本仓库按 triplet mask | tool response **不进 loss** | 同左 | 同左 | 逐步交互，每步独立前向 |


---

## 3. Rollout 采样

### 3.1 GRPO（对照点）

同一 prompt 采 \(G\) 条 **互不共享前缀** 的完整轨迹。本仓库 `rollout.n=4`。多轮 tool agent 时，每条轨迹自己走 think → tool → obs → … → answer，轨迹之间不分支。

### 3.2 ARPO：熵驱动的全局 + 部分采样

总预算 \(M\)。先做 \(N\) 条 **trajectory-level** 全局采样，剩下 \(M-N\) 留给 partial。

1. **初始化**：对每条全局轨迹，算开头 \(k\) 个 token 的熵矩阵 \(H_{\mathrm{initial}}\in\mathbb{R}^{1\times k}\)。token 熵：

\[
H_t=-\sum_{j=1}^{V}p_{t,j}\log p_{t,j},\quad p_t=\mathrm{softmax}(z_t/\tau)
\]

2. **熵变化监控**：每次 tool 返回后，再生成 \(k\) 个 token，得到步级熵 \(H_t\)，标准化相对变化 \(\Delta H_t=\mathrm{Normalize}(H_t-H_{\mathrm{initial}})\)。

3. **自适应分支**：步 \(t\) 的分支概率（示意）\(P_t=\alpha+\beta\cdot\Delta H_t\)。\(P_t>\tau\) 则从当前节点 `Branch(Z)` 条部分路径，否则沿当前轨迹继续。

4. **终止**：分支条数用尽 \(M-N\)，或所有路径先结束则补全局采样凑满预算。

直觉：工具刚返回时分布最不确定，把采样预算压在这里，比把 16 条完整轨迹从头再走一遍更值。共享前缀只算一次前向，后面分叉。

官方脚本里常见 Hydra 项（名称随 fork 略有出入）：

```text
actor_rollout_ref.rollout.n=16              # 总预算 M
actor_rollout_ref.rollout.initial_rollouts=8  # 全局 N
actor_rollout_ref.rollout.beam_size=2         # Branch(Z)
actor_rollout_ref.rollout.branch_probability=0.5
actor_rollout_ref.rollout.entropy_weight=0.2
```

### 3.3 AEPO：先分预算，再罚连续分支

ARPO 的 \(\Delta H\) 阈值在「连续高熵」时会把树抽成一条深链。AEPO 拆成两段。

**熵预监控。** 先生成 **1 条** 完整 tool 轨迹，比较问题根熵 \(H_{\mathrm{root}}\) 与工具平均熵 \(H_{\mathrm{tool}}^{\mathrm{avg}}\)。信息增益视角：

\[
I_{\mathrm{Gain}}\propto m\cdot H_{\mathrm{root}}+(k-m)\cdot H_{\mathrm{tool}}^{\mathrm{avg}}
\]

全局条数：

\[
m=k\cdot\sigma\bigl(\beta(H_{\mathrm{root}}-H_{\mathrm{tool}}^{\mathrm{avg}})\bigr)
\]

根更不确定 → 加大 \(m\)（多探索不同开局）；工具更不确定 → 减小 \(m\)（把预算留给 tool 后分支）。

**连续分支惩罚。** 记当前链在步 \(t\) 之前已连续高熵分支 \(l\) 次：

\[
P_t=(\alpha+\gamma\cdot\Delta H_t)\,(1-\hat P(l))
\]

\(\hat P(l)\) 随 \(l\) 增大。超过 \(\tau\) 才 `Branch(Z)`。这样不会在同一条链上把预算花光。

### 3.4 IGPO：采样与 GRPO 相同

标准组采样：每题 \(G\) 条独立多轮 search 轨迹（官方 `agent_grpo.n=16`，`max_turns=10`）。**不树采样**。额外计算发生在 reward：对 GT 答案做 teacher-forcing logprob（见第 4 节）。工程上可用自定义 attention mask 把 \(T\) 份 GT 拼在轨迹末尾，一次前向算完所有 turn，复杂度从 \(\sum_t L_t^2\) 降到约 \(L_{T-1}^2\)。

### 3.5 GiGPO：采样次数同 GRPO，约束更强

LLM 前向次数与 GRPO 相同，但有硬约束：

- 一组 \(N\) 条轨迹必须 **同一任务 \(x\)、同一初始环境状态** \(s_1^{(1)}=\cdots=s_1^{(N)}\)。这是事后能对齐「同一网页 / 同一房间」的前提。
- verl-agent 用 **step-independent 多轮**：每步单独前向，不把 50 步历史拼成一条超长序列（对比 RAGEN）。这对 ALFWorld 这种超长地平线是刚需。

分支不在采样时发生，而在 advantage 时用 hashmap 把相同 `anchor_obs` 聚在一起。

---

## 4. Reward 设计

### 4.1 ARPO：层次 outcome（Tool-Star）

轨迹级标量，赋给整条（含共享前缀）：

\[
R=
\begin{cases}
\max(\mathrm{Acc}+r_M,\ \mathrm{Acc}) & \text{格式正确且 Acc}>0 \\
0 & \text{格式正确且 Acc}=0 \\
-1 & \text{否则}
\end{cases}
\qquad
r_M=
\begin{cases}
0.1 & \text{同时用了 search 与 python} \\
0 & \text{否则}
\end{cases}
\]

- 格式坏直接 −1，压非法 tool 调用。
- 答对才给 Acc；多工具协作再加小额 \(r_M\)，鼓励 search+code 而不是单工具刷分。
- **没有逐步过程奖励**。共享前缀会同时被多条分支的 \(R_i\) 影响（soft advantage 下通过相同 IS 间接平均）。

### 4.2 AEPO：创新不在 reward

仍是正确性 / 格式 / 工具协作这类 outcome。熵用在 **何时分支** 和 **clip / Â 怎么改**，不另造过程 RM。

### 4.3 IGPO：内在信息增益 + 终局 F1

这是四者里唯一把「过程奖励」做成算法核心的。

**Turn 级 IG。** \(a=(a_1,\ldots,a_L)\) 为 GT token。在 rollout \(i\) 的第 \(t\) 轮结束后：

\[
\log\pi_\theta(a\mid q,o_{i,\le t})=\frac{1}{L}\sum_{j=1}^{L}\log\pi_\theta(a_j\mid q,o_{i,\le t},a_{<j})
\]

\[
r_{i,t}^{\mathrm{IG}}=\log\pi_\theta(a\mid q,o_{i,\le t})-\log\pi_\theta(a\mid q,o_{i,\le t-1}),\quad 1\le t<T
\]

实践上把 GT 包进与预测相同的 schema（例如 `Now there's enough information to answer …`），保证 teacher forcing 格式一致。**对该奖励 stop-gradient**：IG 只当标量监督，不让 \(\pi\) 通过改自己的 logπ 来刷分。

**终局 outcome。** 格式合法则词级 F1；否则常数罚 \(\lambda_{\mathrm{fmt}}<0\)：

\[
r^O=
\begin{cases}
\mathrm{F1}(\hat a,a)\in[0,1] & \text{格式合法} \\
\lambda_{\mathrm{fmt}} & \text{否则}
\end{cases}
\]

**分开标准化再折现。** 组内所有 \(r^{\mathrm{IG}}\) 一套 \(\mu_{\mathrm{IG}},\sigma_{\mathrm{IG}}\)，所有 \(r^O\) 另一套，避免量纲吞掉另一路：

\[
\tilde r_{i,t}=
\begin{cases}
(r_{i,t}^{\mathrm{IG}}-\mu_{\mathrm{IG}})/\sigma_{\mathrm{IG}} & t<T \\
(r_i^O-\mu_O)/\sigma_O & t=T
\end{cases}
\qquad
\tilde R_{i,t}=\sum_{k=t}^{T}\gamma^{k-t}\tilde r_{i,k}
\]

\(\tilde R_{i,t}\) 赋给该 turn 的所有决策 token。消融：只 IG 或只 F1 都明显弱于两者相加；只 IG 仍能学（有 GT 锚定，不像瞎标过程 RM 那样容易 hacking）。

官方开关：`+algorithm.info_gain_type=log_prob_diff|prob_diff`，`+algorithm.info_gain_norm_mode=separate|joint`，`algorithm.gamma`（论文主实验常用 1.0）。

### 4.4 GiGPO：沿用环境奖励，比较时用折扣回报

不发明新 reward。环境给 \(r_t\)（ALFWorld 常是终局 0/1；WebShop 可有中间分）。step 组内比较用：

\[
R_t^{(i)}=\sum_{k=t}^{T}\gamma^{k-t}r_k^{(i)}
\]

同一 anchor 状态下，「先点错商品再返回买对」的早步 \(R_t\) 低于「直接点对」；「点 Next Page 最终失败」更低。这样 \(A^S\) 能排出 `1st Item > 2nd Item > Next Page`，轨迹级 GRPO 做不到。

---

## 5. Loss / Advantage 设计

四者都是 **clip 策略梯度 + 可选 KL**，tool / env 返回 token 通常 mask。差别只在 \(\hat A\) 从哪来、clip 是否改。

GRPO 骨架（本仓库也是这个形状）：

\[
\mathcal{J}=\mathbb{E}\Big[\frac{1}{G}\sum_i\frac{1}{|o_i|}\sum_t\min\big(\rho_{i,t}\hat A,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A\big)\Big]-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{\mathrm{ref}})
\]

\[
\rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid o_{i,<t})}{\pi_{\theta_{\mathrm{old}}}(o_{i,t}\mid o_{i,<t})}
\]

### 5.1 ARPO：默认 soft GRPO

**Hard：** 分叉后的 token 用各自 \(R_i\) 的组内 \(\hat A_i\)；共享前缀 token 显式取这 \(d\) 条轨迹 \(\hat A\) 的平均。

**Soft（默认）：** 直接套 GRPO。树采样使 \(y_{i,<t}=y_{j,<t}\) 时 \(\rho_{i,t}=\rho_{j,t}\)，共享段的梯度贡献自然对齐到「组内平均优势」。论文对比 hard vs soft：soft 奖励更高更稳。

理论侧用 GPG（Generalized Policy Gradient）：把一次 tool 前后的 token 段看成 macro-action，策略梯度对 macro-action 仍然成立，所以 partial rollout 不是 heuristically 切序列，而是有 PG 合法性。

### 5.2 AEPO：保高熵梯度 + 熵感知优势

**Entropy Clipping-Balanced（GPPO 风格）。** 前向仍是 clip，但上界写成 \((1+\epsilon_h)/\mathrm{sg}(\delta)\cdot\delta\)。因为 \(\delta\cdot\mathrm{sg}(\delta)\) 在数值上为 1，前向不变；反向在 \(\delta>1+\epsilon_h\) 且 \(\hat A>0\) 时把因子钉在 \(1+\epsilon_h\)，高熵探索 token 的正优势不会被 clip 成 0。负优势且 \(\delta<1-\epsilon\) 仍按标准 clip 丢掉（过滤低置信负样本）。

**Entropy-aware advantage：**

\[
\tilde A_{\mathrm{Acc}}^{(t)}=\frac{r_t-\mathrm{mean}(\{R_i\})}{\mathrm{std}(\{R_i\})},\qquad
\tilde A_{\Delta H}^{(t)}=\frac{H_t-\mathrm{mean}(\{H\})}{\mathrm{std}(\{H\})}
\]

\[
\tilde A^{(t)}=\tilde A_{\mathrm{Acc}}^{(t)}\cdot\bigl(1+a\cdot\tilde A_{\Delta H}^{(t)}\bigr)
\]

答对且高熵的 token 放大更新；答错的高熵 token 惩罚也更大。\(a\) 是熵项权重。论文 Algorithm 2 里也写成 \(\hat A_{\mathrm{Acc}}(1+\hat A_{\Delta H})^\alpha\) 的等价变体，面试说「准确率优势乘上熵正则」即可。

### 5.3 IGPO：用 \(\tilde R_{i,t}\) 替换轨迹级 \(\hat A_i\)

loss 形状同 GRPO，token \(t\) 所属 turn 的 \(\tilde R_{i,t}\) 代替 \(\hat A_i\)。即使 16 条都答错（outcome 全 0），只要中间 search 让 \(\log\pi(\mathrm{GT})\) 有升有降，组内仍有非零梯度 → 直接打 advantage collapse。

### 5.4 GiGPO：两层相对优势相加

Episode 组：

\[
A^E(\tau_i)=\frac{R(\tau_i)-\mathrm{mean}(\{R(\tau_j)\})}{F_{\mathrm{norm}}(\{R(\tau_j)\})}
\]

\(F_{\mathrm{norm}}\) 可选 \(\mathrm{std}\)（标准 GRPO）或 \(1\)（减轻「极难/极简组 std 很小 → 梯度爆炸」，接近 RLOO / Dr.GRPO 讨论）。

Anchor 组：所有满足 \(s_t^{(i)}=\tilde s\) 的 \((a_t^{(i)}, R_t^{(i)})\) 放进 \(G^S(\tilde s)\)，再做同样的相对标准化得 \(A^S(a_t^{(i)})\)。

\[
A(a_t^{(i)})=A^E(\tau_i)+\omega\,A^S(a_t^{(i)})
\]

\(\omega\) 默认 1。消融：去掉 \(A^E\) 或 \(A^S\) 都明显掉点；\(F_{\mathrm{norm}}\) 的 std vs 1 是二阶因素。极端情况没有任何重复状态 → \(A^S=0\) → 退回 GRPO，下界有保证。

代码对应（verl-agent / 第三方拷贝 `gigpo/core_gigpo.py`）：

```text
episode_advantages = episode_norm_reward(...)          # Eq.3
step_group_uids    = build_step_group(anchor_obs, ...) # Eq.6
step_advantages    = step_norm_reward(...)             # Eq.7
scores = episode_advantages + step_advantage_w * step_advantages  # Eq.8
```

配置：`algorithm.adv_estimator=gigpo`，`algorithm.gigpo.step_advantage_w=1.0`，`algorithm.gigpo.mode=mean_norm`。

---

## 6. 代码如何实现、能否直接调用

### 6.1 没有现成 pip 算法包

| 你想要的 | 实际现状 |
| -------- | -------- |
| `pip install arpo` 然后 `ARPOTrainer(...)` | **不存在** |
| 官方 `verl` 里 `adv_estimator=arpo\|aepo\|igpo\|gigpo` | **不存在**。枚举是 `gae/grpo/rloo/reinforce_plus_plus/...` |
| HuggingFace 上的 Qwen3-14B-ARPO 等 | **已训权重**，不是训练算法 |
| 能跑通论文设定 | clone 对应 **veRL fork**，按 README 配环境后 `python -m verl.trainer.main_ppo` 或仓库脚本 |


三套官方入口都是「改过的 verl + Hydra」，彼此 **不能** 在同一份 verl 上用三个 flag 切换。

### 6.2 ARPO / AEPO

- 仓库：[https://github.com/dongguanting/ARPO](https://github.com/dongguanting/ARPO)（AEPO 在 `AEPO/` 子目录）
- 流程：可选 LLaMA-Factory SFT 冷启动（`ARPO-SFT-54K`）→ 自定义 verl 做树采样 RL → 评测。
- 启动形态：`python -m verl.trainer.main_ppo` + 上文 `initial_rollouts / beam_size / entropy_weight`。
- 数据：`ARPO-RL-Reasoning-10K`、`ARPO-RL-DeepSearch-1K`。
- 权重：HuggingFace collection `dongguanting/arpo`、`dongguanting/aepo`（3B–32B）。
- **不能**当 library import；要跟他们的 vLLM 多轮 / 工具缓存改动绑在一起。

### 6.3 IGPO

- 仓库：[https://github.com/GuoqingWang1/IGPO](https://github.com/GuoqingWang1/IGPO)
- 启动：`bash train.sh` → `python -m verl.trainer.main_ppo`，关键覆盖：

```text
algorithm.gamma=1.0
+algorithm.info_gain_type=log_prob_diff
+algorithm.info_gain_norm_mode=separate
+algorithm.use_vectorized_gt_logprob=false
agent_grpo.n=16
max_turns=10
```

- 第三方复现（同样不是 pip 包）：
  - [Fireworks cookbook / multihop_qa](https://github.com/fw-ai/cookbook)（`--ig-weight`，0 即退回纯 GRPO）
  - [inclusionAI/DR-Venus](https://github.com/inclusionAI/DR-Venus)（DeepResearcher 脚手架上的 IGPO）

### 6.4 GiGPO

- 仓库：[https://github.com/langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)（论文官方）
- 安装：conda 环境里 `pip install -e .`（装的是 **这份 fork 的 verl**，不是 PyPI `verl` 上的 GiGPO 插件）
- 训练：`bash examples/gigpo_trainer/run_alfworld.sh` 等；核心 flag `algorithm.adv_estimator=gigpo`
- 环境：ALFWorld、WebShop、Search、Sokoban、Gym Cards；另有 GiGPO+DAPO dynamic sampling 变体

### 6.5 本仓库 Agent-Lightning / math_gsm

math_gsm **只接了 GRPO**。contrib 里有 GiGPO 的「半截管道」，不能算可调用。


| 位置 | 现状 |
| ---- | ---- |
| [train_math_agent.py](../../train_math_agent.py) | `adv_estimator: grpo` |
| [contrib/.../train_env_agent.py](../../../../contrib/recipes/envs/train_env_agent.py) | algorithm 名含 `gigpo` 时把 `log_env_obs=True` |
| [contrib/.../daemon.py](../../../../contrib/agentlightning/contrib/algorithm/env_verl/daemon.py) `get_train_data_batch(..., is_gigpo=)` | `is_gigpo=True` 时把 `anchor_obs` 写入 `non_tensor_batch` |
| [contrib/.../env_verl/trainer.py](../../../../contrib/agentlightning/contrib/algorithm/env_verl/trainer.py) | advantage 仍走 `compute_advantage` / GRPO，**没有** `compute_gigpo_outcome_advantage` |
| `contrib/recipes/envs/config_verl/` | 只有 `grpo.yaml` / `empo2_*.yaml`，**没有** `gigpo.yaml` |


想在本仓库试四算法的远近：

1. **GiGPO 最近**：已有 `anchor_obs` 钩子，还缺 core 函数、yaml、以及「同初始状态并行 env」的 rollout 约定。适合 ALFWorld / ScienceWorld 这类会回访状态的 env agent，不适合 math_gsm。
2. **IGPO 次近**：GSM8K 有 GT，可在现有 triplet 上加「每 tool 步后算 \(\Delta\log\pi(\mathrm{answer})\)」。要改 reward 与 advantage 聚合，不必改树采样。代价是每步（或向量化）多一次 GT 前向，且数学题「search 检索」语义弱于 QA，IG 信号可能偏稀。
3. **ARPO / AEPO 最远**：必须改 vLLM / rollout 引擎做熵监控和 mid-trajectory branch。Agent-Lightning 当前是「完整轨迹 → triplet adapter → GRPO」，树结构共享前缀对不上。

### 6.6 和 math_gsm 选型

- **继续 GRPO**：任务短、答案可验证、组内相对足够。见 [09_grpo_logit_backprop.md](09_grpo_logit_backprop.md)。
- 面试被问「多轮 tool 下一步改哪」：
  - 有 GT、中间步要 credit → 讲 **IGPO**。
  - 环境会回到同一状态（网页、房间）→ 讲 **GiGPO**。
  - tool 返回后模型很不确定、想提高采样效率 → 讲 **ARPO**，再补 **AEPO** 如何防止熵把训练弄崩。
- 本文不实现算法，只给对照与代码地图。

---

## 7. 面试口述（每题 1–2 分钟）

### Q1. 这四个和 GRPO 是什么关系？

**结论**：都是 critic-free 的 group RL；GRPO 的 \(\hat A\) 来自同题一组轨迹的标量奖励。四者分别补多轮 Agent 的采样、奖励稀疏、逐步 credit。

**落到仓库**：我们现在就是最左边那一格——outcome + `adv_estimator=grpo`。

### Q2. ARPO 和 AEPO 差在哪？不要说「AEPO 更新」。

**结论**：ARPO 用熵 **加大** 高不确定 tool 步的采样；AEPO 发现这样会 **分支塌缩 + 高熵梯度被 clip**，所以改成「先按根/工具熵分预算、连续分支惩罚」以及「clip 里 stop-grad + Â 乘熵」。

**一句话**：ARPO 是熵驱动探索，AEPO 是熵 **平衡**。

### Q3. IGPO 的过程奖励会不会被 hacking？

**结论**：比外部过程 RM 难 hack。IG 是 \(\Delta\log\pi(\mathrm{GT})\)，有标准答案锚定；并对该标量 stop-gradient。没有 outcome 时只 IG 也不会立刻崩（消融里只 IG 仍强于瞎标过程）。但仍可能鼓励「提高 GT 概率的口头承诺」而不是真检索——所以论文把 F1 outcome 留着当锚。

### Q4. GiGPO 为什么几乎不增加时间？数学题能不能用？

**结论**：不额外 rollout，只对已有轨迹做状态哈希分组，论文测 ALFWorld 上分组+算术 <0.002% step 时间。数学 tool agent **很少出现可哈希的相同环境状态**（没有房间/搜索结果页），\(A^S\) 经常为 0，等于白加一套代码。适合 ALFWorld / WebShop，不适合 GSM8K。

### Q5. 能直接在 Agent-Lightning 里切 `adv_estimator=arpo` 吗？

**结论**：不能。官方 verl 没有这些 estimator。ARPO/AEPO/IGPO/GiGPO 各是一份 fork。本仓库 contrib 只给 GiGPO 留了 `anchor_obs` 字段，trainer 仍算 GRPO。要复现就 clone 官方仓库。

### Q6. 若只能加一个，math_gsm 加哪个？

**结论**：多数情况 **不加**，GRPO+规则 outcome 已经匹配任务。若一定要做研究点：IGPO 工程路径最短（有 GT）；ARPO 要动 rollout；GiGPO 收益预期低。

---

## 8. 一张「补丁位置」白板图

```text
采样      reward         advantage/clip
──────    ──────         ─────────────
GRPO      轨迹 outcome    组内 (R-μ)/σ
ARPO ★树  层次 outcome    GRPO soft
AEPO ★树+惩罚 同 ARPO     ★熵 Â + stop-grad clip
IGPO      ★IG+F1 折现     用 R̃ 替换 Â
GiGPO     环境 r_t        ★ A^E + ω A^S
```

星号 = 该算法真正改的层。面试画这三列，比背四个全称更有区分度。
