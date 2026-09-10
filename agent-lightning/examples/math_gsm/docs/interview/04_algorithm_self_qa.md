# 算法自问自答（Week 2 模拟追问）

每题控制在 1–2 分钟。先答结论，再补机制，最后落到 math_gsm。

---

## Q1. GRPO 和 PPO 的本质区别？为什么 LLM 爱 GRPO？

**结论**：都是带 clip 的 on-policy 策略梯度；差别在 **advantage 怎么来**——PPO 靠 Critic，GRPO 靠 **同 prompt 一组样本的相对奖励**。

**机制**：LLM 的 value 网络贵且难准；数学题天然是「同题多答案对比」。

**落地**：`adv_estimator=grpo`，`rollout.n=4` 就是 group size。

---

## Q2. `rollout.n` 从 2 调到 8，期望发生什么？

**结论**：advantage 估计更稳，但 **每步采样成本近似线性增加**。

**边界**：

- n=2：两条一好一坏还行；两条都对/都错 → 信号≈0
- n 过大：算力墙，且收益递减

**实验口径**：应同时看 val 准确率、每 step 时间、组内 reward 方差。

---

## Q3. 为什么数学任务常用 outcome reward 而不是逐步 process reward？

**结论**：最终答案 **可自动验证**，成本低、标注一致；process 要逐步标或训过程 RM，贵且易被「看起来很对」欺骗。

**代价**：中间错误 tool-call 仍可能拿到正 advantage（只要最后蒙对）——credit assignment 粗糙，靠多采样与相对比较缓解。

---



## Q4. 本配置关掉 KL，风险与收益？

**收益**：少被 ref 拉回，更允许策略为了涨分改变行为（工具多用、更长推理）。

**风险**：格式崩、胡言、reward hacking、偏离指令跟随。

**监控**：熵、平均回复长度、tool 调用率、解析失败率、val 与 train 差距。必要时开 `use_kl_loss` 或减小 lr。

---



## Q5. 全组 reward 都是 1 或都是 0 时 GRPO 在学什么？

**结论**：**几乎不学**——标准化后 advantage≈0。

**工程含义**：任务太难（全 0）或太简单/泄漏（全 1）都会让 RL「空转」。要用 curriculum、更好基座/SFT、或调温度增加组内多样性。

---



## Q6. DPO 能不能替代现在的 online GRPO？

**结论**：对齐聊天偏好可以；**带工具、环境反馈、要探索新解题路径**时，online RL 更合适。

**原因**：DPO 不滚动与环境交互的新轨迹；tool agent 的状态空间靠离线对覆盖不住。

**折中**：先 SFT/DPO 稳格式，再 GRPO 冲准确率（类似 R1 多阶段）。