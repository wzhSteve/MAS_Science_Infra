# 高频面试题：2 分钟标准答（对照 math_gsm）

答题结构：**结论 → 机制 → 落到我的项目 → （可选）边界/监控**。

---

## 1. GRPO 和 PPO 的本质区别？为什么 LLM 喜欢 GRPO？

**结论**：都是 clip 的 on-policy 更新；GRPO 用 **同题一组样本的相对奖励** 当 advantage，**去掉 Critic**。

**机制**：PPO 要训 value net；LLM 上 value 又贵又难。数学/代码题天然适合组内对比。

**项目**：`adv_estimator=grpo`，`rollout.n=4`。

---

## 2. 为什么数学/代码常用 rule reward，而不是学一个 RM？

**结论**：对错可自动验证，**噪声低、可扩展、难被「讨好人」绕过**。

**机制**：RM 适合主观偏好；可验证任务用执行器/数值匹配更直接（RLVR）。

**项目**：GSM8K 答案匹配 → 0/1；SQL 示例则是执行匹配。

---

## 3. Outcome reward 下，中间错误 tool-call 会被惩罚吗？如何缓解？

**结论**：若最终答对，中间错步仍可能拿 **正 advantage**（整段共享终局分）。

**缓解**：更多组内采样、过程奖励（贵）、关键步骤约束、错 tool 直接终止并给 0、或 SFT 先教好工具使用。

**项目**：当前是标准 outcome；消融里用格式/长度 shaping 只调「表达」，不解决细粒度 credit。

---

## 4. KL 在 post-training 里干什么？你为什么实验里关了它？

**结论**：KL(π\|\|π_ref) **锚定参考策略**，防跑偏与遗忘。

**为何可关**：纯 RLVR 想让模型为准确率大胆改行为；R1 系也常强调探索。

**风险**：格式崩、hacking。应用熵、长度、val、tool 率监控；不稳再开 `use_kl_loss`。

---

## 5. SFT → RL 顺序能否颠倒？R1-Zero 说明了什么？

**结论**：强基座上 **可以** 直接 RL 涌现推理（R1-Zero）；小模型/弱基座通常需要 **SFT 冷启动** 稳格式。

**R1**：冷启动 SFT → 推理 RL → 再 SFT/对齐，可读性与通用性更好。

**项目**：Qwen3-4B instruct 已有格式先验，相当于轻量冷启动。

---

## 6. Agent RL 和单轮 LLM RL 的工程难点差在哪？

**结论**：难点从「算 logprob」变成 **变长多轮、工具、异步调度、token mask**。

**项目拆分**：vLLM 采样 ↔ FSDP 训练；Runner 跑 LangGraph；Adapter 转 triplet；`n_runners` 提吞吐。

---

## 7. 如何发现 reward hacking？举本项目的例子

**结论**：优化指标涨、真实能力不涨，或策略钻奖励漏洞。

**例子**：

- `format_only`：输出 `### 错误数字 ###` 仍得 1 分（dry-run 已展示）
- 解析过宽：从无关文本抓到数字假匹配
- 刷很长 CoT 若奖励含长度相关项

**检测**：固定 binary 准确率做验收；看格式率、解析失败率、人工抽检。

---

## 8. `rollout.n`、batch、micro-batch、TP 分别卡什么？

| 旋钮 | 卡什么 |
|------|--------|
| `rollout.n` | 每题采样倍数 → **算力/时间**，advantage 方差 |
| `train_batch_size` | 每步题量 → **梯度方差与步成本** |
| `ppo_micro_batch_size_per_gpu` | **激活显存** vs 速度 |
| `tensor_model_parallel_size` | 单卡装不下的大模型切分；4B 通常 1 |

另：`gpu_memory_utilization` 是 **vLLM 与 FSDP 抢显存** 的阀门。

---

## 9. Val reward 不涨但 train reward 涨，怎么查？

按序查：

1. **泄漏/分布**：train 过拟合、val 更难  
2. **评测温度**：train 探索、val 应低温度（本项目 val 默认 0）  
3. **Reward hacking**：train 钻解析/格式，val 用更严口径  
4. **Checkpoint/同步**：rollout 是否用到最新权重  
5. **方差**：步数少、group 全对/全错无信号  

---

## 10. DPO 能否替代 online GRPO？适用边界？

**结论**：有成对偏好、单轮对齐 → DPO 香；**要与工具/环境交互并探索** → online GRPO/PPO。

**边界**：DPO 不产生新环境轨迹；tool agent 状态空间靠离线对盖不住。

**实践**：DPO/SFT 稳格式 + GRPO 冲可验证指标，常优于单阶段。
