# Agent RL 掌握清单 × APPO/ARPO 源码对照

日期：2026-10-16

本文按 [Agent_RL_博士掌握清单.md](Agent_RL_博士掌握清单.md) 的 **Must / Bonus 概念卡**，把每一项落到 APPO 与 ARPO 的**真实源码位置**。读完应能指着文件讲清机制，而不是只背概念。

路径约定（相对仓库根）：

| 简称 | 根目录 |
| --- | --- |
| **ARPO** | `ref_Rep/APPO-main/ARPO/verl_arpo_entropy/` |
| **APPO** | `ref_Rep/APPO-main/APPO/verl_arpo_entropy/` |
| 脚本 ARPO | `ref_Rep/APPO-main/ARPO/scripts/` |
| 脚本 APPO | `ref_Rep/APPO-main/APPO/scripts/` |

两边 `main_ppo.py` 内容相同；算法差在启动配置 + `ray_trainer.py` / `vllm_rollout_with_tools.py` / advantage / loss。

配套文档：[源码学习指南](APPO_ARPO_源码学习指南.md)（按调用链读）、[技术报告](APPO_ARPO_技术报告.md)（公式与差异表）。

---

## 总览：清单条目 → 源码锚点

| 清单 Must / 概念卡 | ARPO 主锚点 | APPO 主锚点 |
| --- | --- | --- |
| M1 On-policy step | `ray_trainer.py` `fit` L948–1166 | 同文件更长版 `fit` L1803–2110 |
| M2 Agent 轨迹 / loss_mask | `vllm_rollout_with_tools.py` L340–450, L598 | 同逻辑 L783–899, L1032；分支行 L401–457 |
| M3 Outcome reward 落点 | `naive.py` L89 + `deep_research.py` L296 | 同构 |
| M4 多轨迹比较 (uid / GRPO) | `fit` L1039–1041；`core_algos.compute_grpo_outcome_advantage` | APPO 另用 `compute_appo_advantage` L286 |
| M5 分叉 | 同一次 generate 内 L458–511 | `fit` 第二次生成 L1971–1984 + `build_appo_branch_request` L379 |
| 3.3 rollout vs old logprob | `fit` L1068–1104；rollout 填 `rollout_log_probs` | APPO 分支选择强依赖二者差 L408–418 |
| 3.6 Reward 雷区 | `deep_research.compute_score` L331–366 | 同构 |
| 3.7 KL 挂着不用 | `main_ppo.py` L127–129；脚本 `kl_loss_coef=0` | APPO `kl_loss_coef=0.001` + `future_kl` |

---

## M1. On-policy 一步在干什么

### 口述版

Driver（`RayPPOTrainer.fit`）编排：采样 → 打分 → 算 advantage → 更新 actor。生成与反传在 Ray GPU worker 上。

### ARPO：只有一次 `generate_sequences`

入口脚本打开 GRPO + 工具同步 rollout：

```113:142:ref_Rep/APPO-main/ARPO/scripts/ARPO_7B_Reasoning_1node.sh
    algorithm.adv_estimator=grpo \
    ...
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
```

`fit` 主序（ARPO `ray_trainer.py`）：

```1010:1166:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/trainer/ppo/ray_trainer.py
                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    # ...
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # reward → old_log_prob → (ref) → advantage(GRPO) → update_actor
```

逐步对应：

| 阶段 | 行号 (ARPO) | 调用 |
| --- | --- | --- |
| 生成（含工具+分叉） | L1014 | `actor_rollout_wg.generate_sequences` |
| 赋 uid + repeat | L1039–1042 | 与 `rollout.n` 对齐 |
| reward | L1065 | `compute_reward` → `NaiveRewardManager` |
| 重算 logprob | L1069 | `compute_log_prob` → `old_log_probs` |
| ref（若挂了） | L1106–1113 | `compute_ref_log_prob` |
| advantage | L1143–1154 | `compute_advantage(..., grpo)` |
| 更新 | L1164 起 | `update_actor` |

Worker 转发（两边同构）：

```652:667:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/workers/fsdp_workers.py
    def generate_sequences(self, prompts: DataProto):
        ...
        with self.rollout_sharding_manager:
            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)
            output = self.rollout_sharding_manager.postprocess_data(output)
```

`rollout.mode=sync_with_tool` 时类为 `vLLMRolloutWithTools`（`fsdp_workers.py` L432–434）。

### APPO：两次 `generate_sequences`

开关（`_appo_common.sh`）：`adv_estimator=appo` + `appo_dynamic_branching=True`。

```1895:2005:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/ppo/ray_trainer.py
                    appo_dynamic_branching = (
                        self.config.algorithm.adv_estimator == AdvantageEstimator.APPO
                        and self.config.actor_rollout_ref.rollout.get("appo_dynamic_branching", True)
                    )
                    rollout_repeat = self.config.actor_rollout_ref.rollout.n
                    if appo_dynamic_branching:
                        rollout_repeat = len(gen_batch_output.batch) // len(batch.batch)
                    ...
                    if appo_dynamic_branching:
                        ...
                            branch_request = build_appo_branch_request(...)
                            if branch_request is not None:
                                branch_output = self.actor_rollout_wg.generate_sequences(branch_request)
                                batch = align_appo_branch_batch(batch, branch_output)
                        with _timer("reward", timing_raw):
                            ...
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
```

要点：

1. 第一次生成常只返回 `initial_rollouts` 条，故 `rollout_repeat` 按**实际返回行数**算（L1900–1901）。  
2. 先 `compute_log_prob`，再 `build_appo_branch_request`（需要 `old_log_probs`）。  
3. **reward 推迟到第二次生成之后**（L1996），保证分支轨迹也被打分。  
4. advantage 前先 `apply_appo_reward_scale_M`（L2068–2072）。

### 白板对照图

```text
ARPO:  gen(工具+熵分叉→凑满 n) → reward → old_lp → GRPO → update
APPO:  gen(init) → old_lp → gen(branch) → reward → M → APPO-adv → update(future_kl)
```

---

## M2. Agent 轨迹 ≠ 纯文本；`loss_mask` 决定谁吃梯度

### 工具循环：模型 token=1，工具回填=0

ARPO（APPO 初始轨迹同构）：

```340:450:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
                    curr_inputs[out_idx].extend(generated_tokens)
                    result_masks[out_idx].extend([1] * len(generated_tokens))
                    ...
                    is_tool_call = finish_reason == 'stop' and stop_reason in self.stop_sequences
                    ...
                        formatted_result = f" <result>\n{result_text}\n</result>"
                        result_tokens = self.tokenizer.encode(formatted_result)
                        curr_inputs[idx].extend(result_tokens)
                        result_masks[idx].extend([0] * len(result_tokens))
```

停词（ARPO 只有工具闭合标签）：

```113:114:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
        self.stop_sequences = [f"</{tag}>" for tag in self.tools.keys()]
        self.logprobs = 10 # entropy
```

组 batch 时 `result_mask` → `loss_mask`：

```598:653:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
            loss_mask = torch.stack(padded_result_mask_list, dim=0).to(input_ids.device)
            ...
            loss_mask = loss_mask * response_attention_mask
            ...
                "loss_mask": loss_mask,
```

### `response_mask` vs `loss_mask`

`response_mask` 来自 attention 的 response 段（**含**工具文本）：

```201:216:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/ppo/ray_trainer.py
def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]
```

Actor 更新时：**有 `loss_mask` 就用它**，不用 `response_mask`：

```370:373:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/workers/actor/dp_actor.py
                    if multi_turn or "loss_mask" in data.keys():
                        response_mask = data["loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
```

| 掩码 | 含义 | 工具 `<result>` |
| --- | --- | --- |
| `attention_mask` / `response_mask` | 位置有效、可被模型看见 | 通常为 1 |
| `loss_mask` | 允许策略梯度 | **0** |

面试一句：attention 看得见 ≠ 要为它负责。

### APPO 分支行：整段 mask 置 0

第二次生成入口：`meta_info["appo_branch_mode"]=True`（`build_appo_branch_request` L506）→ `_generate_appo_selected_branches`。

```401:457:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
        result_masks = [
            [0] * min(max_len, max(0, len(prefix_ids) - len(prompt_ids)))
            for prefix_ids, prompt_ids in zip(prefix_ids_list, prompt_ids_list)
        ]
        ...
                curr_inputs[out_idx].extend(generated_tokens)
                result_masks[out_idx].extend([0] * len(generated_tokens))
```

前缀已有 response token 先写 0；续写也写 0。分支轨迹仍跑工具、仍被打分，但 **actor 不直接更新分支行 token**；信用在 advantage 阶段映回初始前缀（见 M5）。

---

## M3. Outcome reward 写在哪个 token

### 装载：`main_ppo` → `load_reward_manager` → `deep_research.compute_score`

```131:132:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/trainer/main_ppo.py
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, ...)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=1, ...)
```

```81:99:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/trainer/ppo/reward.py
    compute_score = get_custom_reward_fn(config)
    ...
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=final_compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs,
    )
```

脚本：`reward_manager=naive`，`custom_reward_function` → `deep_research.compute_score`。

### 标量落点

```65:89:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/workers/reward_manager/naive.py
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            ...
            score = self.compute_score(...)
            ...
            reward_tensor[i, valid_response_length - 1] = reward
```

形状与 `responses` 相同，**非零只在最后一个有效 response 下标**。后续 GRPO/APPO 都先 `sum(dim=-1)` 得到轨迹标量。

### `deep_research.compute_score` 规则（Reward 雷区源码）

```331:366:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/utils/reward_score/deep_research.py
    if mode == "strict":
        if not strict_ok:
            result["score"] = -1.0
            ...
        if answer is None:
            result["score"] = -1.0
            ...
    else:
        if answer is None:
            result["score"] = SOFT_UNPARSEABLE_SCORE  # -0.3
            ...
    f1_score = get_f1_score(answer, ground_truth)
    ...
    if f1_score > 0 and "</search>" in response and "</python>" in response:
        score = f1_score + 0.1 - penalty
    elif f1_score > 0:
        score = f1_score - penalty
    else:
        score = 0.0
```

| 现象 | 源码含义 | 研究风险 |
| --- | --- | --- |
| soft 抽不出答案 | `-0.3`，不是 -1 | 与 strict 不可比 |
| 格式不严但能抽答案 | 仅扣小 penalty | 易刷「能 parse」 |
| `+0.1` 多工具 | 要 F1>0 **且** 文本里同时出现两个闭合标签 | 诱导无意义双工具 |
| 分数在末 token | outcome-only | 过程归功难 → 分叉/M 的动机 |

---

## M4. 多轨迹怎么比：`uid` + GRPO

### 分组键怎么来

ARPO（APPO 首次对齐同理）：

```1039:1042:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/trainer/ppo/ray_trainer.py
                    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
```

先给每条 **prompt** 一个 `uid`，再 `repeat(n)`，故同一 prompt 的 n 条轨迹共享 `uid`。

配置语义：

| 配置 | 含义 |
| --- | --- |
| `data.train_batch_size` | 每 step 多少条 **prompt** |
| `rollout.n` | 每条 prompt 多少条 **轨迹** |
| 有效优化样本量 | 约 `train_batch_size × n`（APPO 动态分支时含 init+branch） |

### GRPO 实现

调用处（ARPO，`multi_turn=True` 时用 `loss_mask`）：

```255:270:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/trainer/ppo/ray_trainer.py
    elif adv_estimator == AdvantageEstimator.GRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = grpo_calculation_mask.size(1)
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
```

核心算法（两边同构）：

```140:166:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/ppo/core_algos.py
    scores = token_level_rewards.sum(dim=-1)
    ...
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        ...
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask
    return scores, scores
```

自检答案（对着源码）：

- `n=1`：组内只有一条，`id2mean=0, id2std=1`（L151–153），advantage 退化。  
- 组内全同分：减均值后全 0，无学习信号。  
- 工具 token：`multi_turn` 下乘的是 `loss_mask`，advantage 在工具位为 0。

### APPO：双组再回映

```317:366:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/ppo/ray_trainer.py
    for idx in range(bsz):
        group_key = (str(tree_ids[idx]), str(rollout_kind[idx]))  # (uid, init|branch)
        group_to_indices[group_key].append(idx)
    ...
    # init 行：标量 adv × response_mask
    advantages[init_mask] = base_adv_values[init_mask].unsqueeze(-1) * response_mask[init_mask]
    ...
    # 分支标量加到对应 init 行、分裂点及之前
        aux_sum[orig_row, :end] += base_adv_values[branch_row] * valid_prefix
    ...
    advantages = advantages * response_mask
```

分组键比 GRPO **多一项** `appo_rollout_kind`。分支行最终不贡献自身 token 的非零 advantage 用于更新（再叠加 `loss_mask` 全 0）。

---

## M5. 分叉：ARPO vs APPO（核心对照）

### ARPO：同一次 `generate_sequences` 内、工具轮后复制前缀

超参读入：

```83:86:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
        self.initial_rollouts = self.config.get("initial_rollouts", self.config['n'])
        self.beam_size = self.config.get("beam_size", 1)
        self.branch_probability = self.config.get("branch_probability", 0.5)
        self.entropy_weight = self.config.get("entropy_weight", 0.5)
```

每轮用 top-k logprob 估熵（L305–327），工具回填后：

```492:509:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
                            entropy_now = current_entropy_dict.get(source_idx, 0.0)
                            entropy_init = self.initial_entropy_dict.get(source_idx, 0.0)
                            entropy_delta = entropy_now - entropy_init
                            prob = random.random() - self.entropy_weight * entropy_delta
                            prob = max(0.0, min(1.0, prob))
                            if prob > self.branch_probability: 
                                continue
                            new_inputs.append(curr_inputs[source_idx].copy())
                            ...
                            new_result_masks.append(result_masks[source_idx].copy())
```

| 问题 | ARPO 答案 |
| --- | --- |
| 分叉点 | 某一工具轮结束后的**整段前缀** |
| 复制什么 | `curr_inputs`（含已回填的 `<result>`） |
| 梯度 | 新轨迹有自己的 `loss_mask`，模型 token 仍为 1，进 GRPO 组内比较并更新 |
| 回到 trainer | 已凑满 `n`，**无**第二次 generate |

### APPO：token 级选点 → 第二次生成 → 信用回映

**选分裂点**（依赖 `old_log_probs - rollout_log_probs` 的 future value × 熵 z-score）：

```408:426:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/ppo/ray_trainer.py
        log_ratio = current_log_probs[row] - behavior_log_probs[row]
        ...
        future_log_ratio = torch.matmul((log_ratio * valid_positions.to(log_ratio.dtype)).unsqueeze(0), decay.t()).squeeze(0)
        future_values = torch.exp(torch.clamp(future_log_ratio, min=-20.0, max=20.0))
        ...
        entropy_z[valid_positions] = _tensor_zscore(rollout_entropys[row][valid_positions])
        future_z[valid_positions] = _tensor_zscore(future_values[valid_positions])
        branch_scores = entropy_z * future_z
```

前缀 = prompt + `responses[:token_idx+1]`（L466–474）；`meta_info={"appo_branch_mode": True}`（L506）。

**Reward 缩放 M**（只改初始行 reward 向量）：

```193:219:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/appo_branching.py
def compute_reward_scale_M(...):
    """
    M_base = branch_score / max_branch_score
    t <= 分裂点：M[t] = 1 + M_base * discount^(分裂点 - t)
    t >  分裂点：M[t] = 1
    """
```

在 `fit` 中、`compute_advantage` 之前调用（L2068–2072）。因 outcome 分只在末 token，若分裂点在答案前，末 token 的 M 常为 1——M 对标量分的影响取决于分裂点位置；读代码时不要默认「每个位置原本都有非零 reward」。

**信用回传**：见上节 `compute_appo_advantage`——分支组标量优势加到初始轨迹前缀；分支行 `loss_mask=0`。

### 对照表（贴清单 3.5，补源码锚点）

| | ARPO | APPO 官方动态分支 |
| --- | --- | --- |
| 代码入口 | `vllm_rollout_with_tools.py` L458–511 | `ray_trainer.py` L1971–1984 |
| 分叉粒度 | 轮级（工具后前缀） | token 级 |
| 触发信号 | `entropy_delta` + 随机门 | `Z(H)×Z(FV)` |
| FV / log ratio | 不需要 `old_log_probs` | **需要**先 `compute_log_prob` |
| 分支是否进 loss | 是（正常 mask） | 否（mask 全 0） |
| advantage | 单组 GRPO | `(uid, kind)` 双组 + 回映 |
| loss | 默认 PPO clip | `future_kl`（FIPO） |

---

## 概念卡 3.3：`rollout_log_probs` vs `old_log_probs`

| 字段 | 谁写 | 何时 | 用途 |
| --- | --- | --- | --- |
| `rollout_log_probs` / `rollout_entropys` | vLLM 采样时（`logprobs=10`） | 第一次生成 | 行为策略；APPO 选分裂点；监控 |
| `old_log_probs` | actor `compute_log_prob` | 采样后、更新前 | PPO/GRPO 的「旧策略」；APPO 的 `log_ratio` |

ARPO `fit` 里对比二者做监控（L1082–1104）。APPO 选点公式用差值（上文 L408）。

分支续写 `logprobs=0`（APPO L441），分支行的 `rollout_log_probs` 在 `align_appo_branch_batch` 里补零（L518 一带）。

**为什么不只用 vLLM 的 logprob？** 采样引擎与训练图可能数值不一致；PPO 需要当前可微策略上的 logprob；APPO 还要「当前相对行为」的 future value。

---

## 概念卡 3.7：KL「挂着不用」

`main_ppo` 注册 RefPolicy 的条件：

```127:129:ref_Rep/APPO-main/ARPO/verl_arpo_entropy/verl/trainer/main_ppo.py
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
```

ARPO 脚本：`use_kl_loss=True` 且 `kl_loss_coef=0.0` → **仍会建 ref worker、仍会算 ref logprob**，但乘系数为 0（`dp_actor.py` L435–441）。

两条 KL 通道：

| 通道 | 配置 | 代码 |
| --- | --- | --- |
| reward 内扣 KL | `algorithm.use_kl_in_reward` | `fit` 里 `apply_kl_penalty` |
| actor loss 加 KL | `actor.use_kl_loss` + `kl_loss_coef` | `dp_actor.update_policy` |

两边官方 reward 侧一般关闭；APPO actor 侧常用 `kl_loss_coef=0.001`。

---

## Bonus：策略 loss（APPO future KL vs ARPO PPO）

```391:425:ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/workers/actor/dp_actor.py
                    fipo_cfg = self.config.get("policy_loss", {})
                    use_future_kl = fipo_cfg.get("loss_mode", "vanilla") == "future_kl"
                    if use_future_kl:
                        pg_loss, ... = compute_policy_loss_future_kl(...)
                    else:
                        pg_loss, ... = compute_policy_loss(...)
```

- ARPO：不传 `loss_mode=future_kl` → `compute_policy_loss`（标准 clip）。  
- APPO：`policy_loss.loss_mode=future_kl` → `core_algos.compute_policy_loss_future_kl`（L688 起）。  

注意：选分裂点用的 future value（`build_appo_branch_request`）与 loss 里的 future-KL **是两次独立计算**，共用「未来 log-ratio 折扣」思想，不要当成同一个张量。

---

## 配置开关速查（改算法先改这里）

| 想改什么 | ARPO | APPO |
| --- | --- | --- |
| advantage 算法 | `algorithm.adv_estimator=grpo` | `=appo` |
| 是否工具 rollout | `rollout.mode=sync_with_tool` | 同 |
| 轨迹数 / 初始条数 | `n`, `initial_rollouts` | 同 |
| 分叉强度 | `beam_size`, `branch_probability`, `entropy_weight` | `appo_dynamic_branching`, `n - initial` 作 budget |
| 打分 | `custom_reward_function.path/name` | 同 |
| loss | 默认 PPO | `policy_loss.loss_mode=future_kl` |
| KL | `kl_loss_coef`（可为 0） | 常非 0 |

---

## 面试口述稿（对着源码）

**90 秒版（ARPO）**

> 我们用 veRL：driver 的 `fit` 调一次 `generate_sequences`。`vLLMRolloutWithTools` 里模型生成停在 `</search>`/`</python>`，工具结果塞进 `<result>`，`result_mask` 对工具写 0，最后变成 `loss_mask`。工具轮后按熵增做自适应复制，凑满 `n` 条。`NaiveRewardManager` 把 `deep_research` 的标量写在最后一个有效 token。按 `uid` 做 GRPO，`update_policy` 用 `loss_mask`，所以不为搜索引擎的字更新策略。

**再加 30 秒（APPO 差异）**

> APPO 第一次只采初始轨迹，用 `old_log_probs - rollout_log_probs` 的 future value 和熵选 token，再 `generate_sequences` 一次续写；分支行 `loss_mask` 全 0。advantage 按 `(uid, init/branch)` 两组算，把分支优势加回初始前缀；loss 走 future-KL。

---

## 建议阅读顺序（按清单，不按文件名）

1. 本页 M1 → 打开对应 `fit` 走一遍。  
2. M2 → ARPO `vllm_rollout_with_tools.py` 工具循环 + mask。  
3. M3 → `naive.py` L89 + `deep_research.compute_score`。  
4. M4 → `compute_grpo_outcome_advantage`。  
5. M5 → ARPO L492 与 APPO `build_appo_branch_request` + `compute_appo_advantage`。  
6. Bonus → `dp_actor.update_policy` 的 `loss_mode` 分支。

需要逐行跟 `main_ppo` 装配时，回 [源码学习指南 §1](APPO_ARPO_源码学习指南.md)。需要公式与本地实测时，回 [技术报告](APPO_ARPO_技术报告.md)。

---

## 清单自测题 → 源码答案索引

| 题（掌握清单 §8） | 到哪里核对 |
| --- | --- |
| 画 training step | 本文 M1；ARPO `fit` L1010+ / APPO L1856+ |
| 工具回填进不进 loss | M2；ARPO L450 vs L341 |
| GRPO 分组键 / n | M4；`uid` L1039；`index=uid` in GRPO |
| reward 非零下标 | M3；`naive.py` L89 |
| ARPO vs APPO 分叉 | M5 对照表 |
| 提高格式分风险 | `deep_research` soft/strict 与 penalty |
| 分叉多但 adv 方差小 | 组内分数趋同 → GRPO L160–163 近 0 |
| `use_kl_loss` 且 coef=0 | `main_ppo` L127；仍挂 RefPolicy |

---

## 文档关系

| 文档 | 角色 |
| --- | --- |
| [Agent_RL_博士掌握清单.md](Agent_RL_博士掌握清单.md) | 掌握边界与面试策略 |
| **本文** | 清单每一项的源码定位与对照 |
| [APPO_ARPO_源码学习指南.md](APPO_ARPO_源码学习指南.md) | 调用链精读 |
| [APPO_ARPO_技术报告.md](APPO_ARPO_技术报告.md) | 公式、差异总表、实测 |
