# math_gsm 超参词典

对照 `[train_math_agent.py](../../train_math_agent.py)` 的 `RL_TRAINING_CONFIG`（推荐配置 `a800_2gpu`）。

**显存怎么算、OOM 先调谁、各参数对实验的影响**：见 [08_gpu_memory_oom_hyperparams.md](08_gpu_memory_oom_hyperparams.md)。

## algorithm


| Key                | 默认      | 一句话                            | 改大/打开                       | 改小/关掉             |
| ------------------ | ------- | ------------------------------ | --------------------------- | ----------------- |
| `adv_estimator`    | `grpo`  | 用组内相对奖励估 advantage，无 value net | 换成 `gae`/`ppo` 需 Critic，显存↑ | —                 |
| `use_kl_in_reward` | `False` | 是否把 KL 惩罚加进 reward             | 更保守、少跑偏，可能抑制探索              | 更敢探索，易格式崩/hacking |




## data


| Key                   | 默认      | 一句话                         | 改大                   | 改小                          |
| --------------------- | ------- | --------------------------- | -------------------- | --------------------------- |
| `train_batch_size`    | `32`    | 每步多少 prompt（再 × group size） | 梯度更稳、吞吐需求↑、显存/时间↑    | 噪声大、更新抖                     |
| `max_prompt_length`   | `4096`  | prompt 截断上限                 | 多轮/长题不易截断，显存↑        | 长对话 `truncation=error` 直接失败 |
| `max_response_length` | `2048`  | 单段生成上限                      | 长 CoT/多 tool 更完整，算力↑ | 易截断答案→reward 变 0            |
| `truncation`          | `error` | 超长直接报错                      | —                    | 若改 `left`/`right` 会静默丢上下文   |




## actor_rollout_ref.rollout


| Key                                          | 默认                                  | 一句话                                                  | 改大                        | 改小                    |
| -------------------------------------------- | ----------------------------------- | ---------------------------------------------------- | ------------------------- | --------------------- |
| `n`                                          | `4`                                 | **GRPO group size**：同题采样几条                           | advantage 更稳，算力近似线性↑      | `n=2` 时组内方差估计很噪       |
| `name`                                       | `vllm`                              | 用 vLLM 做 online rollout                              | —                         | —                     |
| `gpu_memory_utilization`                     | `0.35`（`a800_2gpu`；`a800` 为 `0.45`） | vLLM 相对**整卡**容量的 KV 预留（非剩余显存比例）；FSDP colocated 后须留余量 | 吞吐↑，易与 FSDP 抢显存 OOM       | 更安全，KV cache 小→慢      |
| `tensor_model_parallel_size`                 | `1`                                 | 推理 TP                                                | 大模型切卡；小模型 TP>1 反而慢        | 保持 1（4B 足够）           |
| `log_prob_micro_batch_size_per_gpu`          | `4`                                 | 重算 logprob 的 micro-batch                             | 更快但显存↑                    | 更慢更省显存                |
| `multi_turn.format`                          | `hermes`                            | Qwen tool-call chat 格式                               | 必须与 `tool_call_parser` 一致 | 不一致→工具解析失败            |
| `engine_kwargs.vllm.enable_auto_tool_choice` | `True`                              | 允许自动 tool 选择                                         | Agent 能调工具                | 关掉则退化为纯文本             |
| `engine_kwargs.vllm.tool_call_parser`        | `hermes`                            | 解析 tool XML/JSON                                     | 换模型要换 parser              | 错 parser→轨迹坏、reward 噪 |




## actor_rollout_ref.actor


| Key                             | 默认      | 一句话                          | 改大               | 改小            |
| ------------------------------- | ------- | ---------------------------- | ---------------- | ------------- |
| `ppo_mini_batch_size`           | `32`    | 每次策略更新的 mini-batch           | 更稳，步内更新次数可能↓     | 更新更频繁、方差↑     |
| `ppo_micro_batch_size_per_gpu`  | `4`     | 反传 micro-batch               | 快但 OOM           | 慢但能训长序列       |
| `optim.lr`                      | `1e-6`  | 策略学习率                        | 学得快，易崩/遗忘        | 稳但慢，可能“看起来不涨” |
| `use_kl_loss`                   | `False` | 是否在 loss 里加 KL(π             |                  | π_ref)        |
| `kl_loss_coef`                  | `0.0`   | KL 系数                        | 越大越像“别离开 SFT/基座” | 0 = 无 KL loss |
| `entropy_coeff`                 | `0`     | 熵奖励系数                        | >0 鼓励探索，防塌缩      | 0：靠采样温度探索     |
| `clip_ratio_low`                | `0.2`   | PPO/GRPO clip 下界（ratio 下限相关） | 更紧→更新更保守         | 更松→单步步子大      |
| `clip_ratio_high`               | `0.3`   | clip 上界（非对称 clip）            | 允许好样本推得更猛一点      | 两侧对称更保守       |
| `fsdp_config.param_offload`     | `True`  | 参数卸到 CPU                     | 省 GPU 显存，训练变慢    | 关则更快但易 OOM    |
| `fsdp_config.optimizer_offload` | `True`  | 优化器状态卸 CPU                   | 同上               | 同上            |




## actor_rollout_ref.ref


| Key                                 | 默认     | 一句话                      | 作用                                            |
| ----------------------------------- | ------ | ------------------------ | --------------------------------------------- |
| `log_prob_micro_batch_size_per_gpu` | `8`    | ref 模型算 logπ_ref 的 batch | 即使 `use_kl_loss=False`，框架仍常保留 ref 路径；开 KL 时必需 |
| `fsdp_config.param_offload`         | `True` | ref 参数 offload           | 省显存                                           |




## actor_rollout_ref.model


| Key                             | 默认            | 一句话               |
| ------------------------------- | ------------- | ----------------- |
| `path`                          | Qwen3-4B 本地路径 | 策略初始化权重           |
| `use_remove_padding`            | `True`        | 变长序列去 padding，省算力 |
| `enable_gradient_checkpointing` | `True`        | 用算力换显存            |




## trainer / Trainer 进程


| Key                     | 默认        | 一句话                          | 改大                    | 改小                  |
| ----------------------- | --------- | ---------------------------- | --------------------- | ------------------- |
| `n_gpus_per_node`       | `2`（推荐）   | FSDP+vLLM 可用卡数               | 完整超参；1 卡必须砍 batch/mem | 1 卡回退见 `a800`       |
| `total_epochs`          | `2`       | 数据扫几遍                        | 过拟合风险↑                | 可能欠训                |
| `test_freq`             | `16`      | 每隔多少 step 验证                 | 监控密、打断训练              | 省时间但盲飞              |
| `val_before_train`      | `True`    | 训前先测基线                       | 必须有，才能说“RL 涨了多少”      | 冒烟可关                |
| `save_freq`             | `32`      | checkpoint 频率                | 更安全                   | 盘占用↑                |
| `n_runners`（Trainer 参数） | `8`（2gpu） | 并行跑 agent rollout 的 worker 数 | 吞吐↑，调度/显存/API 压力↑     | 瓶颈在 agent 侧时会闲置 GPU |




## 配置变体速查


| 模式          | 用途        | 相对 `a800_2gpu` 砍了什么        |
| ----------- | --------- | -------------------------- |
| `a800_2gpu` | 正式训练      | 无                          |
| `a800`      | 1 卡回退     | batch/micro-batch/vLLM mem |
| `fast`      | 1-step 冒烟 | GPU=1、n=2、短序列、1 step       |




## 面试一句话

> `train_batch_size` **决定“多少道题”，**`rollout.n` **决定“每道题几条轨迹”，真实采样量约等于二者乘积；显存同时被 vLLM KV 与 FSDP 训练瓜分，所以** `gpu_memory_utilization` **和 micro-batch 是一对耦合旋钮。**

