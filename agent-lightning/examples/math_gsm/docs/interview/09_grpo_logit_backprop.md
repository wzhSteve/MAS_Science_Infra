# GRPO 下 Rollout Logit 计算与长轨迹反传（对照 math_gsm）

与 [03_ppo_grpo_dpo.md](03_ppo_grpo_dpo.md) 的分工：`03` 讲算法直觉对比；本文讲 **logit → logπ → GRPO Â → clip loss → backward** 的实现级推导，全部对照 Agent-Lightning + VERL 源码。

---

## 0. 一句话结论

math_gsm 配置 `adv_estimator=grpo` 后：同题采样 `rollout.n` 条轨迹 → 组内 outcome advantage（**无 Critic**）→ **clip PPO policy loss**。  
vLLM rollout **只存 token_ids**（训练不用其 logprobs）；FSDP Actor 用 teacher-forcing **重算** $\log\pi_{\theta_{\mathrm{old}}}$；真正反传到 LLM 的路径只在 Actor `update_policy` 的**第二次前向**。

---

## 1. 端到端数据流

```mermaid
flowchart LR
  parquet[GSM8K parquet] --> daemon[Daemon x n]
  daemon --> agent[LitMathAgent vLLM]
  agent --> spans[token_ids + reward]
  spans --> adapter[Triplet adapter]
  adapter --> batch[DataProto]
  batch --> oldlp["compute_log_prob no_grad"]
  oldlp --> adv[GRPO advantage]
  adv --> upd["update_policy with grad"]
  upd --> sync[sync weights to vLLM]
```

| 步骤 | 代码入口 | 产物 |
|------|----------|------|
| 选 GRPO | [`train_math_agent.py`](../../train_math_agent.py) `algorithm.adv_estimator=grpo` | `use_critic=False` |
| 同题 ×n 采样 | [`daemon.py`](../../../../agentlightning/verl/daemon.py) `_async_set_up` | 同 `data_id` 的 n 条轨迹 |
| Agent + 规则奖励 | [`math_agent.py`](../../math_agent.py) `emit_reward` | 标量 $R\in\{0,1\}$ |
| 组 batch | `daemon.get_train_data_batch` | `input_ids`, `responses`, `token_level_scores` |
| uid | [`trainer.py`](../../../../agentlightning/verl/trainer.py) `_train_step` | `uid = data_id_list` |
| 重算 old logπ | `actor_rollout_wg.compute_log_prob` | `old_log_probs`（无梯度） |
| GRPO Â | `compute_advantage(..., adv_estimator=grpo)` | `advantages` |
| 更新 | `actor_rollout_wg.update_actor` → `update_policy` | `loss.backward()` → Adam |

VERL 在 `adv_estimator ∈ {GRPO, ...}` 时关闭 Critic；math_gsm 另设 `use_kl_in_reward=False`、`use_kl_loss=False`，因此 loss 里通常**没有** value / KL 项。

---

## 2. Logit / logπ 怎么算（核心）

### 2.1 符号

对一条训练样本，令完整序列

$$
x = \underbrace{(x_1,\ldots,x_{L_p})}_{\text{prompt}} \;\Vert\;
\underbrace{(y_1,\ldots,y_{L_r})}_{\text{response}},\quad
L = L_p + L_r.
$$

FSDP Actor 对整段 $x$ 做一次因果前向，位置 $t$ 得到词汇表 logits $z_t\in\mathbb{R}^{V}$。采样温度 $T$（与 rollout 一致，来自 `meta_info["temperature"]`）：

$$
\tilde{z}_t = \frac{z_t}{T}.
$$

下一 token 预测：位置 $t$ 的标签是 $x_{t+1}$（对 response 段即 $y$）。则

$$
\log\pi_\theta(a_t\mid s_t)
= \tilde{z}_{t,a_t} - \log\sum_{v=1}^{V} e^{\tilde{z}_{t,v}}
= -\mathrm{CE}(\tilde{z}_t,\, a_t).
$$

实现：`logits.div_(temperature)` 后调用 `logprobs_from_logits(logits, labels)`（Flash-Attn CE 或 log-softmax+gather），见 VERL `verl/workers/actor/dp_actor.py` 的 `_forward_micro_batch` 与 `verl/utils/torch_functional.py`。

### 2.2 只取 response 段

因果 LM 在位置 $t$ 预测 $x_{t+1}$。对长度为 $L_r$ 的 response，取切片

$$
\log\pi(y_{1:L_r}\mid\mathrm{prompt})
\;\leftarrow\;
\texttt{log\_probs[:, -L\_r-1 : -1]},
$$

即丢掉 prompt 内部的 logπ，只保留「生成 response 每个 token」的那一段。`responses` 张量形状为 `(bs, L_r)`，与上述 logπ 逐位相乘进 loss。

无 remove-padding 时的等价写法：

```text
logits = model(input_ids).logits
logits = logits / T
logits = logits[:, -response_length-1 : -1, :]   # (bs, L_r, V)
log_probs = logprobs_from_logits(logits, responses)
```

### 2.3 remove_padding（math_gsm 默认开）

`use_remove_padding=True` 时：

1. `unpad_input` 去掉 pad，得到 `total_nnz` 有效 token  
2. 对有效 token 做 varlen / flash-attn 前向  
3. `torch.roll(input_ids, -1)` 得到 next-token labels  
4. `logprobs_from_logits` → `pad_input` 填回 `(bs, seqlen)`  
5. 再切 response 段  

这不改变公式，只省掉 pad 上的无效算力；梯度仍只对有效 response token 有意义（再乘 `response_mask`）。

### 2.4 两阶段：old vs current（关键）

| 阶段 | 调用 | 梯度 | 符号 | 用途 |
|------|------|------|------|------|
| 重算旧策略 | `DataParallelPPOActor.compute_log_prob` | `torch.no_grad()` | $\log\pi_{\theta_{\mathrm{old}}}(y_t\mid s_t)$ | importance ratio 分母 |
| 策略更新 | `update_policy` → `_forward_micro_batch` | **有** | $\log\pi_\theta(y_t\mid s_t)$ | ratio 分子 + 反传 |

**Rollout 阶段 vLLM 返回的 logprobs 不进入 GRPO loss。** Agent-Lightning daemon 组 batch 只用 `prompt_ids` / `response_ids`；注释惯例是 *recompute old log prob with actor*（HybridEngine：推理引擎与训练数值不完全一致，且多轮拼接后必须在最终 token 序列上 teacher-forcing）。

序列对数似然（可选理解用）：

$$
\log\pi(y\mid x_{\mathrm{prompt}}) = \sum_{t=1}^{L_r} m_t\,\log\pi(y_t\mid s_t),
$$

其中 $m_t$ 为 `response_mask`（见第 5 节：trajectory 模式下观测 token 为 0）。

---

## 3. Reward → token_level → GRPO Advantage

### 3.1 Outcome 标量如何进张量

Daemon 把每条样本的标量 reward $R_i$ 写到**最后一个有效 token（EOS）** 上，其余位置为 0：

$$
r_{i,t} =
\begin{cases}
R_i & t = t_{\mathrm{EOS}} \\
0 & \text{otherwise}
\end{cases}
\quad\Rightarrow\quad
\texttt{token\_level\_scores}_i \in \mathbb{R}^{L_r}.
$$

`use_kl_in_reward=False` 时：

$$
\texttt{token\_level\_rewards} = \texttt{token\_level\_scores}.
$$

### 3.2 GRPO 组内标准化

`compute_grpo_outcome_advantage`（`verl/trainer/ppo/core_algos.py`）：

$$
R_i = \sum_{t=1}^{L_r} r_{i,t}
\quad\text{（outcome：求和后仍是原标量奖励）}.
$$

按 `uid`（math_gsm 里对齐为 `data_id`）分组 $g(i)$。组内：

$$
\mu_g = \mathrm{mean}_{j\in g}(R_j),\qquad
\sigma_g = \mathrm{std}_{j\in g}(R_j).
$$

默认 `norm_adv_by_std_in_grpo=True`（原版 GRPO）：

$$
\hat{A}_i = \frac{R_i - \mu_{g(i)}}{\sigma_{g(i)} + \varepsilon},\quad \varepsilon=10^{-6}.
$$

若为 `False`（Dr.GRPO）：$\hat{A}_i = R_i - \mu_{g(i)}$。

组内仅 1 条时实现置 $\mu=0,\sigma=1$（避免除零，但信号退化）。

再广播到每个 response token：

$$
\hat{A}_{i,t} = \hat{A}_i \cdot m_{i,t}.
$$

**Outcome credit assignment**：同一条轨迹上所有 $m_{i,t}=1$ 的 token **共享同一个** $\hat{A}_i$（粗信用分配；数学 RLVR 的标准做法）。

math_gsm：`rollout.n = G = 4`，同题 4 条完整 agent 轨迹构成一组。

---

## 4. Policy loss 与反传到 LLM

### 4.1 Importance ratio

$$
\Delta_t = \mathrm{clamp}\big(\log\pi_\theta(y_t\mid s_t) - \log\pi_{\theta_{\mathrm{old}}}(y_t\mid s_t),\, -20,\, 20\big),
$$

$$
\rho_t = \exp(\Delta_t)
= \frac{\pi_\theta(y_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(y_t\mid s_t)}.
$$

（clamp 仅为数值稳定，不改变算法含义。）

### 4.2 非对称 clip + dual-clip

math_gsm：`clip_ratio_low = ε_L = 0.2`，`clip_ratio_high = ε_H = 0.3`；dual-clip 下界系数默认 $c = 3.0$（`clip_ratio_c`）。

令

$$
\begin{aligned}
L_t^{(1)} &= -\hat{A}_t\,\rho_t,\\
L_t^{(2)} &= -\hat{A}_t\,\mathrm{clip}(\rho_t,\, 1-\varepsilon_L,\, 1+\varepsilon_H),\\
L_t^{\mathrm{clip}} &= \max\big(L_t^{(1)},\, L_t^{(2)}\big).
\end{aligned}
$$

当 $\hat{A}_t < 0$ 时再套 dual-clip：

$$
L_t =
\begin{cases}
L_t^{\mathrm{clip}} & \hat{A}_t \ge 0,\\
\min\big(L_t^{\mathrm{clip}},\, -\hat{A}_t\, c\big) & \hat{A}_t < 0.
\end{cases}
$$

直觉：正向 advantage 用 PPO clip 限制「推得太猛」；负向 advantage 时 dual-clip 防止 $\rho$ 过小导致过大的惩罚梯度。

### 4.3 聚合为标量

默认 `loss_agg_mode = "token-mean"`：

$$
L_{\mathrm{pg}} = \frac{\sum_{i,t} m_{i,t}\, L_{i,t}}{\sum_{i,t} m_{i,t}}.
$$

其它模式（实现里可选）：`seq-mean-token-sum`、`seq-mean-token-mean`、`seq-mean-token-sum-norm`（与 Dr.GRPO 论文归一化相关）。

math_gsm：`entropy_coeff=0`、`use_kl_loss=False`，因此

$$
L = L_{\mathrm{pg}}.
$$

### 4.4 反传与优化器

`update_policy` 内：

$$
L_{\mathrm{micro}} = \frac{L}{\texttt{gradient\_accumulation}},\quad
\texttt{gradient\_accumulation}
= \frac{\texttt{ppo\_mini\_batch\_size}}{\texttt{ppo\_micro\_batch\_size\_per\_gpu}}.
$$

然后 `L_micro.backward()` → FSDP `clip_grad_norm_` → `optimizer.step()`（lr $=10^{-6}$）。

**可微计算图（唯一）：**

```text
θ (Transformer) → logits z_t → log π_θ(y_t|s_t) → ρ_t → L_t → L → ∂L/∂θ
```

常数（`detach` / 无梯度）：$\log\pi_{\theta_{\mathrm{old}}}$、$\hat{A}_{i,t}$、$m_{i,t}$、reward。  
**没有** Critic；也没有从 vLLM 采样图反传（采样是离散的，训练靠 likelihood ratio）。

更新后权重同步回 vLLM，下一步继续 on-policy 采样。

---

## 5. 长 Agent Rollout：transition vs trajectory

配置：[`agentlightning/verl/config.yaml`](../../../../agentlightning/verl/config.yaml) 的 `agentlightning.trace_aggregator`。  
math_gsm **默认 `level: transition`**。

### 5.1 `transition`（默认）

多轮 tool 调用时，**每一次 LLM 调用**变成一条独立训练样本（一个 triplet）：

```text
Turn1: prompt₁ → response₁ (tool call)     → 样本 A，reward = R_final
环境执行工具 → ToolMessage
Turn2: prompt₂ → response₂ (最终答案)       → 样本 B，reward = R_final
```

要点：

1. **同一终局 outcome $R$** 复制到每一轮  
2. 每条样本各自做 teacher-forcing，序列长度 ≈ 该轮 `max_prompt + max_response`，**不会**把整条多轮轨迹拼成一条超长反传  
3. 同题多条轨迹仍共享 `data_id` → GRPO 分组  
4. 副作用：轮数多 → 训练样本数↑；组内 mean/std 会被「轮数多的题」加权；超长 prompt 先算 advantage，再用 `is_drop_mask` 丢掉（保证算 Â 时组仍完整）

对「特别长」的 agent：默认策略是**拆开算、拆开反传**，而不是一次吃完整轨迹。

### 5.2 `trajectory`（可选）

把同一次 rollout 的多轮拼成**一条**长序列。伪代码逻辑（`daemon.get_train_data_batch`）：

```text
prompt = 首轮 prompt
response_ids = 首轮 response
response_mask = [1] * len(首轮 response)

for 后续轮:
  new_obs = 本轮 prompt 相对已有上下文的新增部分   # 常含 tool 结果
  response_ids += new_obs + 本轮 response
  response_mask += [0]*len(new_obs) + [1]*len(本轮 response)
```

公式上：

$$
m_t =
\begin{cases}
1 & t\text{ 属于模型生成的 assistant token}\\
0 & t\text{ 属于拼接进来的环境/观测 token}
\end{cases}
$$

因此：

- $\log\pi$ 仍可对整段 causal 前向算出，但 **loss / Â 广播只乘 $m_t=1$**  
- 环境 token **不贡献策略梯度**（模型并未「选择」它们）  
- 长度上限：`trajectory_max_prompt_length`（默认 2048）、`trajectory_max_response_length`（默认 8192）

### 5.3 两轮 tool 小例子

假设一道题、一轮 tool、一轮作答（示意 token）：

```text
[系统+题面]  [assistant: call python]  [tool: 42]  [assistant: ### 42 ###]
     P              Y1                    O            Y2
```

| 模式 | 训练样本 | 进 loss 的 token | 共享的 $\hat{A}$ |
|------|----------|------------------|-------------------|
| transition | 样本1：`P→Y1`；样本2：`(P+Y1+O)→Y2` | 各样本的 $Y$ | 两样本都用同一终局 $R$ 算出的 $\hat{A}$（再进各自组内标准化） |
| trajectory | 一条：`P ‖ Y1 ‖ O ‖ Y2` | 仅 $Y1,Y2$（$O$ 的 $m=0$） | 整条轨迹一个 $\hat{A}$，广播到 $Y1\cup Y2$ |

无论哪种，**反传都只经过「模型生成 token」上的 $\log\pi_\theta$**；trajectory 只是把多轮拼进同一次前向，并用 mask 挡掉观测。

### 5.4 长序列时显存如何扛住

反传按 micro-batch 切，**不会**把整个 step 的所有轨迹一次 backward：

- `ppo_micro_batch_size_per_gpu`（math_gsm=4）：单次反传条数  
- `enable_gradient_checkpointing=True`：用算力换激活显存  
- `use_remove_padding=True`：变长去 pad  
- `max_prompt_length` / `max_response_length`：单样本上限  

细节与 OOM 决策树见 [08_gpu_memory_oom_hyperparams.md](08_gpu_memory_oom_hyperparams.md)。

---

## 6. 公式 ↔ 源码速查

| 公式符号 | 变量 / 函数 | 文件 |
|----------|-------------|------|
| $z_t / T$ | `logits.div_(temperature)` | `verl/.../dp_actor.py` `_forward_micro_batch` |
| $\log\pi(a_t\mid s_t)$ | `logprobs_from_logits` | `verl/utils/torch_functional.py` |
| $\log\pi_{\theta_{\mathrm{old}}}$ | `old_log_probs` | `compute_log_prob` → trainer `_train_step` |
| $\log\pi_\theta$ | `log_prob` | `update_policy` |
| $R_i$ | `token_level_rewards.sum(-1)` | `compute_grpo_outcome_advantage` |
| $\hat{A}_i$ | 组内 mean/std 标准化 | 同上 |
| $m_t$ | `response_mask` | daemon（trajectory）或 `compute_response_mask` |
| $\rho_t$ | `exp(clamp(log_prob - old_log_prob))` | `compute_policy_loss` |
| $L$ | `agg_loss(..., token-mean)` | 同上 → `loss.backward()` |
| 分组键 | `uid = data_id_list` | `agentlightning/verl/trainer.py` |
| 聚合模式 | `trace_aggregator.level` | `agentlightning/verl/config.yaml` |

---

## 7. 面试 60 秒口述

「我们用 VERL 的 GRPO：同题采 4 条轨迹，组内标准化 reward 当 advantage，没有 Critic。vLLM 只负责采样出 token；训练时 FSDP Actor 对 prompt+response 做 teacher-forcing，先无梯度重算 old logπ，再有梯度算当前 logπ，用 PPO clip 的 ratio 乘 advantage 做 loss，backward 更新 LLM。Agent 多轮默认按 transition 拆开，每轮共享终局 0/1 奖励；若开 trajectory，就把多轮拼长，并对 tool 观测 token 置 mask=0，梯度只走模型自己生成的 token。」

---

## 8. 自检清单

1. 能否写出 $\log\pi = z_a/T - \log\sum_v e^{z_v/T}$ 并指出只取 response 切片？  
2. 能否说清 **old_log_prob 无梯度、update 有梯度**，且 vLLM logprob 不用？  
3. 能否手推 $\hat{A}_i = (R_i-\mu_g)/(\sigma_g+\varepsilon)$ 并说明 outcome 广播到所有 response token？  
4. 能否对比 transition / trajectory 下哪些 token 的 $m_t=1$？  
5. 能否画出可微链：$\theta \to z\to\log\pi_\theta\to\rho\to L\to\partial\theta$？
