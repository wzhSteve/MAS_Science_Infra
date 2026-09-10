# PPO / GRPO / DPO：面试公式直觉（对照 math_gsm）

## 1. 共同目标

都想让策略 π_θ 生成「更好」的回复，但「更好」的定义与优化路径不同：


|                 | 信号从哪来               | 要不要 online 采样  | 要不要 Critic / RM    |
| --------------- | ------------------- | -------------- | ------------------ |
| PPO (RLHF/RLVR) | reward（RM 或规则）      | 要（on-policy）   | 通常要 value；RM 可选    |
| GRPO            | 同上                  | 要（同 prompt 一组） | **不要 value**；用组内相对 |
| DPO             | 人类/合成偏好对 (y_w, y_l) | 通常不要（离线）       | 不要显式 RM（隐式在偏好里）    |


你的 math_gsm：**规则 outcome reward + GRPO online** → 典型 **RLVR**。

---

## 2. PPO（LLM 版）核心

对一条回复 y，令比值


r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}


目标（示意）：


L^{\mathrm{CLIP}}=\mathbb{E}_t\Big[\min\big(r_t\hat{A}_t,\ \mathrm{clip}(r_t,1-\epsilon,1+\epsilon)\hat{A}_t\big)\Big]


- **Importance sampling**：用旧策略采的数据更新新策略
- **Clip**：防止单步更新过大（你配置里还有非对称 `clip_ratio_low/high`）
- **Advantage \hat{A}**：常用 GAE；需要 **Critic V_φ** 估状态价值
- 常加 **KL(ππ_ref)**：别离开 SFT/基座太远

内存画像：Actor + Critic + Ref（+ Reward Model）→ 贵。

---



## 3. GRPO：用「组内同学」代替 Critic

对同一 prompt x，采样一组 y_1,\ldots,y_G（你这里 `rollout.n = G = 4`），得分 R_i。

组内标准化 advantage（示意，与实现细节可能差常数项）：


\hat{A}_i=\frac{R_i-\mathrm{mean}(R)}{\mathrm{std}(R)+\delta}


再套与 PPO 类似的 clip 策略梯度（对 token logprob 加权）。

### 为什么 LLM 喜欢 GRPO

1. **去掉 value net**：大模型上 Critic 又贵又难训（token 级价值难估）
2. **同题对比**天然适合「这题答没答对」的相对信号
3. 与 **rule reward / RM 标量分** 都兼容



### 代价与坑


| 现象               | 直觉                                        |
| ---------------- | ----------------------------------------- |
| G 太小             | mean/std 极噪；全对或全错时 advantage≈0（**无学习信号**） |
| G 太大             | 算力近似 ×G                                   |
| Reward 全 0/全 1   | 组内无区分度 → 更新停滞                             |
| Entropy collapse | 多样性↓，组内更同质，相对信号更弱                         |
| 无 KL             | 易 reward hacking / 格式崩                    |


---



## 4. DPO 一眼看懂

偏好对：赢回复 y_w、输回复 y_l。目标把「隐式 reward」推到偏好一致：


L_{\mathrm{DPO}}=-\log\sigma\Big(\beta\big[\log\frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)}-\log\frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\big]\Big)


- **离线、稳定、实现简单**
- **不探索**新轨迹：分布外行为靠数据覆盖
- Agent 多轮 tool、环境会变时，**online GRPO/PPO 更合适**；DPO 更适合「已有成对偏好、单轮对话」

---



## 5. 一张对比表（建议背）


| 维度     | PPO     | GRPO            | DPO        |
| ------ | ------- | --------------- | ---------- |
| 采样     | online  | online（分组）      | 通常 offline |
| Critic | 要       | 不要              | 不要         |
| 偏好/奖励  | RM 或规则  | 同左              | 成对偏好       |
| 显存     | 高       | 中               | 低          |
| 探索     | 有       | 有               | 弱          |
| 典型场景   | 通用 RLHF | 数学/代码 RLVR、R1 系 | 对齐、聊天偏好    |


---



## 6. 映射回你的 config

```text
adv_estimator: grpo          → 用组内相对 advantage
rollout.n: 4                 → G=4
use_kl_loss: False           → 无 KL 锚定（更大探索/更大跑偏风险）
clip_ratio_low/high: 0.2/0.3 → 非对称 clip，限制策略比
entropy_coeff: 0             → 不靠熵奖励，靠 temperature 探索
```



### Outcome vs Process reward

- **Outcome（你现在）**：只看最终对错，实现简单、难 hacking 过程标注；credit 粗
- **Process**：逐步打分，信用分配更好，但标注贵、易被过程 RM hacking

数学/代码工业界主流仍是 outcome + 强验证器（执行/数值匹配）。

---



## 7. R1-Zero vs R1（口述 30 秒）

- **R1-Zero**：基座上直接大规模 RL（GRPO + 规则奖励），可涌现长思考；副作用是可读性/语言混杂
- **R1**：冷启动 SFT → 推理 RL → 拒绝采样再 SFT → 全场景对齐 RL；更稳、更好用

「能否跳过 SFT」取决于基座是否已有足够能力与格式先验；小模型上通常 **SFT warm-start 更稳**。