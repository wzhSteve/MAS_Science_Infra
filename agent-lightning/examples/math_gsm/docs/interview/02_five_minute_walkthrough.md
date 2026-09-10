# 5 分钟白板讲解稿：math_gsm 从数据到一次 GRPO step

目标时长：4–6 分钟。面试官打断时，优先保住「Reward → Advantage → Update」三段。

---

## 开场（20 秒）

「我做了一个 **GSM8K 数学 Agent 的 online RL**：模型可以调用受限 Python 工具，用 **规则 0/1 奖励**，算法是 **VERL 上的 GRPO**，框架是 Agent-Lightning 把 LangGraph agent 接到训练环路上。」

---

## 1. 任务与 Agent（60 秒）

画简图：

```text
Human 问题 → LLM(可 bind tools) ⇄ execute_python → 最终 ### 数字 ###
```

要点口述：

1. **状态**：LangGraph 的 `messages` + 轮数 + 是否已 finalize
2. **动作**：生成文本 / tool-call；工具结果以 `ToolMessage` 写回
3. **终止**：解析到答案，或达到 `max_turns` 后强制 finalize
4. **Reward**：`compute_reward` 把预测与金标做数值/字符串匹配 → `1.0/0.0`，`agl.emit_reward` 上报

一句差异化：「这是 **RLVR**，不是学偏好 RM；对错可自动验证。」

---



## 2. 训练环路（90 秒）

画：

```text
parquet 任务
    → n_runners × LitMathAgent.rollout
         → 打到 vLLM 暴露的 OpenAI endpoint（当前策略）
         → 多轮 tool 轨迹 + reward
    → Adapter：trace → training triplets
    → GRPO：同题 n 条响应，组内相对优势
    → FSDP Actor 更新 → 权重回灌 vLLM
```

必须讲清的三个角色：


| 角色                   | 做什么                                   |
| -------------------- | ------------------------------------- |
| **Algorithm（VERL）**  | 管 vLLM rollout、算 logprob、GRPO 更新、FSDP |
| **Runner（LitAgent）** | 只关心「拿题 → 跑图 → 给 reward」               |
| **Adapter**          | 把 tracing 的 span 变成可训练 token 序列       |


解耦价值：「Agent 图逻辑不绑死训练后端，换 spider SQL 只需换图和 reward。」

---



## 3. 为什么是 GRPO（60 秒）

「每个 prompt 采样 `n=4` 条完整轨迹，用组内 reward 均值（和标准差）当 baseline，得到 advantage，**不需要 Critic**。对 LLM 来说省掉一套 value 网络，显存和工程都更简单。」

追问预防：

- Outcome reward：整段轨迹共享同一个终局 advantage（credit assignment 粗，但是数学任务标准做法）
- 本配置 `use_kl_loss=False`：更靠 reward 驱动；风险是偏离基座/格式崩，需要靠监控熵和 val

---



## 4. 工程与结果口径（40 秒）

- 推荐 **2×A800**：完整 batch/序列；单卡是回退不是同等实验
- 评测：`val_before_train` 先打基线；看 **准确率**，并盯 **解析失败率、工具调用率、平均轮数**（防 hacking）
- 冒烟：`fast` 只证明「能跑通 1 个 GRPO step」，不证明效果

---



## 收尾（20 秒）

「项目价值不只是涨点，而是我能把 **Agent 多轮 tool + 可验证奖励 + GRPO online 训练系统** 串起来，并且知道每个超参动的是方差、算力还是稳定性。」

---



## 若被打断：30 秒极简版

「GSM8K 题 → 多轮 Python tool agent → 对错 0/1 → 同题采 4 条做 GRPO → vLLM 采样、FSDP 更新。SFT 教格式，RL 用可验证奖励挖搜索策略。」