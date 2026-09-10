# ARPO Rollout 采样与熵分支详解

本文只讲一件事：ARPO 在 rollout 里 **如何采样、如何算熵、如何在“熵高的地方” fork**。所有行号对应

`[ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py](ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py)`

类：`vLLMRolloutWithTools`。官方训练 `rollout.mode=sync_with_tool` 走的就是这份文件。

先给结论，避免和论文表述对不上：

- 代码 **不会** 扫描整条 response、找到某个高熵 token 下标、再从那个 token 切开。
- 真正做的是：把一条多轮轨迹按 **生成轮次**（每次生成到 `</search>` / `</python>` / EOS）切开；每一轮算出 **一个标量熵**；若这一轮还要继续（通常刚吃完工具结果），就用「当前熵 − 该轨迹第一次观测到的熵」去调 fork 概率。
- 论文说的「tool 反馈后的前若干 token 熵高」在工程上被近似成：**每一轮只看本轮新生成的前 20 个 token 的 top-10 logprob**。Fork 发生在 **轮次边界**（工具结果已经写回 context 之后），复制的是到目前为止的整条前缀。

---



## 1. 和标准 GRPO rollout 差在哪

标准 GRPO：每个 prompt 独立采样 `n` 条完整轨迹，互不共享前缀。

ARPO：每个 prompt 先并行开 `initial_rollouts` 条；之后只在 **仍 active 的轮次边界** 上，按熵决定要不要从当前前缀再复制出新轨迹，直到该 prompt 凑满 `n` 条。


| 超参         | 配置字段                         | 官方脚本典型值 | 含义                                 |
| ---------- | ---------------------------- | ------- | ---------------------------------- |
| 最终组大小      | `rollout.n`                  | 16      | 每个 prompt 最终必须交出的轨迹数               |
| 初始并行       | `rollout.initial_rollouts`   | 8       | 一开始从同一 prompt 复制几份                 |
| 单源 fork 上限 | `rollout.beam_size`          | 2       | 每个 active 源一次最多再复制 `beam_size-1` 份 |
| 分支阈值       | `rollout.branch_probability` | 0.5     | 见 §5                               |
| 熵权重        | `rollout.entropy_weight`     | 0.2     | 见 §5                               |


读取位置：`__init__` L82–86，以及 `generate_sequences` 里 `num_samples = self.sampling_params.n`（L233）。

验证或 greedy 时 `beam_size` 被强制设为 1（L213–227），**整套熵分支关闭**。

---



## 2. 轨迹状态机：fork 发生在哪一类“位置”

主循环是 `generate_sequences` 里的 `while active_indices:`（L264）。一轮循环对应一次 vLLM 生成。

每条轨迹用这些 list 对齐存放（下标是全局 rollout 下标 `out_idx`，不是原始 prompt 下标）：


| 变量                          | 行号               | 含义                                 |
| --------------------------- | ---------------- | ---------------------------------- |
| `curr_inputs[i]`            | L242, L318, L425 | 当前完整 token 序列（prompt + 已生成 + 工具结果） |
| `init_inputs[i]`            | L243, L252       | 该轨迹的原始 prompt，用来算 response 长度      |
| `result_masks[i]`           | L244, L319, L426 | 1=模型生成 token，0=工具回填 token          |
| `call_counters[i]`          | L245             | 已调用工具次数                            |
| `rollouts_per_sample[orig]` | L258             | 这个 prompt 已经有几条轨迹                  |
| `sample_to_indices[orig]`   | L260             | prompt → 其所有轨迹下标                   |




### 2.1 初始化：共享 prompt，不共享后续

L237–260：每个原始 prompt 复制 `initial_rollouts` 份。此时它们 token 完全一样，之后各自独立采样，自然分叉。

```text
prompt_0 ──► traj 0,1,...,7     （假设 initial_rollouts=8）
prompt_1 ──► traj 8,9,...,15
```



### 2.2 每一轮生成停在工具 tag 或 EOS

L271–282：对所有 active 轨迹调用 vLLM，关键采样参数：

- `n=1`：每条 active 轨迹只续写 1 段
- `stop=self.stop_sequences`：`["</search>", "</python>"]`（L113，由工具 `trigger_tag` 拼出）
- `logprobs=self.logprobs`：`10`（L114），为后面算熵准备 top-10
- `max_tokens`：剩余 response 预算

因此 **“定位”的粒度是一轮生成，不是一个 token**。一轮通常是：

```text
... <think> 推理 </think> <search> query </search>     ← 停在这里
```

或吃完工具后再开一轮：

```text
... </search> <result> ... </result> <think> 基于结果继续 ...
```



### 2.3 谁还能 fork：只有 `final_active_indices`

生成结束后（L311–361）：


| `finish_reason` / `stop_reason`                      | 行为         | 还会进下一轮吗     |
| ---------------------------------------------------- | ---------- | ----------- |
| `stop` 且命中 `</search>` / `</python>`，未超 `call_limit` | 抽内容、排队执行工具 | 是（L346）     |
| 工具次数打满                                               | 补 EOS，结束   | 否           |
| `length` 且未到 `max_len`                               | 继续生成       | 是（L357–358） |
| `stop`（普通 EOS）                                       | 轨迹结束       | 否（L360–361） |


工具跑完后把  `<result>\n...\n</result>` 追加进 `curr_inputs`，mask 为 0（L422–426）。然后用长度再滤一遍，得到 `final_active_indices`（L428–432）。

**Fork 只对** `final_active_indices` **做**（L444 起）。也就是：

- 刚做完一次 tool-call、结果已经写回的轨迹 —— 这是论文关心的「tool 反馈后」位置
- 或被 length 截断、还要继续写的轨迹

已经 EOS 的轨迹 **不会** 在这一点上被熵 fork。它们若所在 prompt 还不满 `n` 条，走的是另一条路：从 **原始 prompt 重开**（L490–508），与熵无关。

```text
时间轴（一条轨迹）

prompt
  │  第 0 轮生成（baseline 熵写入 initial_entropy_dict）
  ▼
</search>  ── 执行 search ── 写入 <result>
  │  ★ 这里是 fork 点 1（若仍 active）
  ▼
第 1 轮生成（tool 后的前 20 token 用来算 entropy_now）
  │
</python> ── 执行 python ── 写入 <result>
  │  ★ 这里是 fork 点 2
  ▼
第 2 轮生成 → EOS
```

Fork 复制的是 **当前整条** `curr_inputs`（含已有 tool 交互），不是从某个高熵 token 切开。两份拷贝下一轮各自独立采样，这才叫前缀共享。

---



## 3. 熵到底怎么算

对应代码：L114、L172–177、L283–306。

### 3.1 先要 vLLM 返回每步 top-10 logprob

```114:115:ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
        self.logprobs = 10 # entropy
        self.initial_entropy_dict = {}  # record initial entropy of active indice
```

L276：`logprobs = self.logprobs`。vLLM 会给每个生成位置一个 dict：token_id → 对象（含 `.logprob`），一般包括采样到的 token 和若干备选。

### 3.2 每一轮、每条 active 轨迹 → 一个标量

```283:306:ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
                # ========== Entropy Variation Monitoring ==========
                vocab_size = len(self.tokenizer.get_vocab())
                entropy_norm_factor = math.log(vocab_size)
                current_entropy_dict = {}
                for i, out_idx in enumerate(active_indices):
                    output = outputs[i]
                    logprobs = []
                    tokens = output.outputs[0].token_ids
                    for j in range(min(20, len(tokens))):
                        try:
                            logprob_info = output.outputs[0].logprobs[j]
                        except Exception:
                            logprob_info = output.outputs[0].logprobs[-1]
                        token_list = list(logprob_info.values())
                        token_logprobs = [token.logprob for token in token_list]
                        logprobs.extend(token_logprobs)
                    if logprobs:
                        entropy = self._calc_entropy(logprobs) / entropy_norm_factor
                    else:
                        entropy = 0.0
                    current_entropy_dict[out_idx] = entropy
                    if out_idx not in self.initial_entropy_dict:
                        self.initial_entropy_dict[out_idx] = entropy
```

逐步拆开：

**（1）只看本轮新生成的前 20 个 token**

`tokens = output.outputs[0].token_ids` 是 **这一轮** 新生成的，不含 prompt、也不含上一轮已经写进去的 tool result。这就是论文「tool 返回后最初几个 token」的工程近似：工具结果在上一轮末尾注入，本轮开头的 20 个 token 正好是模型看到 `<result>` 之后的第一段续写。

第 0 轮还没有 tool，这 20 个 token 就是对 prompt 的第一段回复，用来当 baseline。

**（2）每个位置取出 top-10 的 logprob，全部拼成一张长列表**

位置 `j=0..19`，每个位置最多 10 个数，列表最长约 200 项。注意：这 **不是**「对每个位置算熵再平均」。所有位置的候选 logprob 被 `extend` 进同一个 list。

**（3）**`_calc_entropy` **是未归一化的 -\sum p\log p**

```172:177:ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
    def _calc_entropy(self, logprobs):
            if not logprobs:
                return 0.0
            p_list = [math.exp(l) for l in logprobs]
            entropy = -sum(p * l for p, l in zip(p_list, logprobs))
            return entropy
```

若 `logprobs = [ℓ_1, …, ℓ_M]`，则 p_k=e^{ℓ_k}，返回 -\sum_k p_kℓ_k。

标准 next-token 熵应是：对 **单个位置** 的完整词表分布 H=-\sum_v \pi(v)\log\pi(v)，再对 20 个位置取平均。这里有两处启发式：

1. 每个位置只用 top-10，不是满词表。
2. 20 个位置的候选被当成 **一个袋子**，并且 **没有把 p_k 重新归一化到和为 1**。若每位置 top-10 质量接近 1，则 \sum p_k \approx 20，原始值大约是「20 个位置熵的和」，量级比单步熵大。

**（4）再除以 \log|V|**

`entropy_norm_factor = math.log(vocab_size)`。单步满词表均匀分布的熵是 \log|V|，除完以后意图是把分数压到可比较的范围。因为第（3）步不是标准单步熵，这个值 **不必落在 [0, 1]**；它只是一个相对分数，后面只用 **差值** `entropy_now - entropy_init`。

### 3.3 数值直觉（不必精确）

设某位置 top-2 为 \log 0.8, \log 0.2，很自信：局部 -\sum p\log p 小。  
设 top-10 几乎均匀：局部值接近 \log 10。  
tool 反馈后模型经常在「该信结果、该再搜、还是该改计划」之间犹豫，前 20 token 的 top-k 更平，拼起来的分数就会升高。这就是后面 `entropy_delta > 0` 的来源。

它 **定位的是“这一轮开头是否变不确定”，不是“序列里哪一个 token 熵最大”**。

---



## 4. baseline：`initial_entropy_dict` 记的是什么

L304–305：某个 `out_idx` **第一次** 出现在熵统计里时，把当时的标量熵存成 baseline。

因此：


| 轨迹                        | baseline 是哪一轮                         |
| ------------------------- | ------------------------------------- |
| 初始 `initial_rollouts` 那几条 | **第 0 轮**（对 prompt 的第一段生成，尚未 tool）    |
| fork 出来的新下标               | 该新轨迹 **自己的第一轮生成**（通常已经带着父轨迹的 tool 历史） |


比较的是 **同一条轨迹自己的“现在 vs 出生时”**，不是和 batch 里别人比，也不是和词表均匀分布比。

论文图里「tool 之后熵升高」在这里变成：


\Delta H = H_{\text{now}} - H_{\text{init}}


- 第 0 轮：H_{\text{now}}=H_{\text{init}}，\Delta H=0，熵项不提供额外 fork 倾向（还纯随机）。
- tool 之后第 1 轮：若前 20 token 更平，\Delta H>0，fork 变容易。

实现细节：`self.initial_entropy_dict` 建在 `__init__`（L115），`generate_sequences` **开头没有清空**。下标从 0 重新数起时，上一 batch 留下的 baseline 可能被复用。改代码若要严格按「本 batch 第一条轨迹」算 \Delta H，应在 `generate_sequences` 入口加 `self.initial_entropy_dict = {}`。

---



## 5. 如何用熵决定 fork（核心判定）

对应 L434–487。先按原始 prompt 把仍 active 的轨迹归组，再在名额内尝试复制。

### 5.1 名额与 beam

对原始样本 `orig_sample`：

```text
remaining_slots = n - rollouts_per_sample[orig]     # 这个 prompt 还能再要几条
branches_per_idx = min(beam_size - 1, remaining_slots - 已建数)
```

`beam_size=2` ⇒ 每个 active 源一次最多再复制 **1** 份。`n=16`、`initial_rollouts=8` 时，整个 prompt 一共还能再长出 8 条，分散在后续若干轮 tool 边界上。

名额用尽后，即使熵再高也不再 fork（L460–461、L465–466）。这就是「把有限采样预算压到高不确定步骤」的硬约束。

### 5.2 判定公式

```468:478:ARPO/verl_arpo_entropy/verl/workers/rollout/vllm_rollout/vllm_rollout_with_tools.py
                            # ==== Entropy-based Adaptive Beaming ====
                            
                            entropy_now = current_entropy_dict.get(source_idx, 0.0)
                            entropy_init = self.initial_entropy_dict.get(source_idx, 0.0)
                            entropy_delta = entropy_now - entropy_init
                            prob = random.random() - self.entropy_weight * entropy_delta
                    
                            prob = max(0.0, min(1.0, prob))
                            if prob > self.branch_probability: 
                                continue
```

写成式子（官方脚本 `entropy_weight=w=0.2`，`branch_probability=τ=0.5`）：


u \sim \mathrm{Unif}(0,1),\qquad
\tilde{p} = \mathrm{clip}\bigl(u - w\cdot\Delta H, 0, 1\bigr)



\text{fork if } \tilde{p} \le \tau


`continue` 表示 **跳过这次复制**。所以：

- \tilde{p} 越小越容易 fork
- \Delta H>0（比出生时更不确定）→ 从均匀随机数里减掉 w\Delta H → \tilde{p} 变小 → 更容易 \le 0.5
- \Delta H=0 → \tilde{p}=u → P(\text{fork})=\tau=0.5
- \Delta H<0（比出生时更确定）→ \tilde{p} 变大 → 更常被 `continue` 掉

`w=0` 时整项消失，退化为每轮以概率 `τ` 随机 fork。

**没有**「熵超过某绝对阈值就 fork」的分支。高熵是通过 **相对升高 \Delta H** 提高成功概率，再叠一层均匀随机，避免每条高熵轨迹都无脑翻倍。

### 5.3 复制什么

判定通过之后（L479–485）：

```python
new_inputs.append(curr_inputs[source_idx].copy())      # 整条当前序列
new_init_inputs.append(init_inputs[source_idx].copy())  # 仍指向同一 prompt
new_result_masks.append(result_masks[source_idx].copy())
new_call_counters.append(call_counters[source_idx])     # 工具次数一并继承
```

这是 token 级 copy-on-write，**不是** KV-cache 共享。父轨迹和子轨迹从下一轮 `while` 开始各自 `n=1` 续写。子轨迹拿到新的全局下标，下一轮会给它写一条新的 `initial_entropy_dict[new_idx]`。

新下标随后被 `extend` 进 `final_active_indices`（L518–524），同一轮循环末尾 `active_indices = final_active_indices`（L526），下一轮就会一起生成。

---



## 6. 用一条轨迹把「定位 + fork」走通

假设 `n=4`，`initial_rollouts=2`，`beam_size=2`，`w=0.2`，`τ=0.5`。只看 prompt A。

```text
t=0  初始化
     A0, A1  都等于 prompt

t=1  第 0 轮生成（尚无 tool）
     算 H(A0)、H(A1)，写入 initial_entropy_dict
     ΔH = 0
     两条都停在 </search> → 执行搜索 → 写入 <result>
     仍 active，尝试 fork：
       每条最多复制 1 份，剩余名额 = 4-2 = 2
       ΔH=0 ⇒ 各以约 50% 随机决定
       假设 A0 复制出 A2（前缀 = prompt + 第0轮 + search结果）
       A1 没中
     现在 3 条：A0, A1, A2

t=2  第 1 轮生成（tool 之后，论文说的高熵段）
     每条用「本轮前 20 token」算 H_now
     对 A0：ΔH = H_now(A0) - H_init(A0)    ← 通常 > 0
     对 A2：这是 A2 的第一轮，H_init(A2)=H_now(A2)，ΔH=0
     假设 A0、A1、A2 又各自停在 </python>，执行后仍 active
     剩余名额 = 4-3 = 1
     先遍历到 A0：ΔH 较大，u - 0.2ΔH 很容易 ≤ 0.5 → 复制出 A3
     名额用尽，A1/A2 不再 fork

t=3  四条都 EOS
     输出 A0,A1,A2,A3 共 4 条给 GRPO
```

要点：

1. **定位**发生在 t=2：比较的是 tool 后前 20 token 的熵相对第 0 轮是否升高。
2. **切开位置**是 t=1 结束时的前缀（已经含 search 结果），不是某个内部 token。
3. 新轨迹 A2 在 t=2 的 \Delta H 为 0，不会因为「人在高熵段」就自动再叉；它要等 **再下一轮** 相对自己的出生熵再升高。

若某条在 t=1 就 EOS，它不进 `active_by_sample`。若此时该 prompt 还不满 `n`，L490–508 会从 **裸 prompt** 再开一条（`init_inputs` 拷贝、空 mask、call_counter=0）。这不是熵 fork，是补采样。

---



## 7. 循环结束后如何凑满 `n` 条

L536–555：按 `sample_to_indices` 取前 `n` 个下标。不够就 **重复最后一条**。分支太少、过早 EOS、或熵一直偏低导致很少 fork 时，GRPO 组里会出现重复轨迹，组内标准差被压低。这是改分支策略时要盯的副作用。

随后 pad 到 `response_length`，`result_masks` 变成 batch 里的 `loss_mask`（L573–629）：工具 token 为 0，不进 actor 梯度。

---



## 8. 整张控制流（只保留和熵/fork 有关的边）

```text
generate_sequences
  ├─ 每个 prompt 复制 initial_rollouts 份          L249-255
  └─ while active:
        vLLM 生成到 </search></python>/EOS        L271-282
        用本轮前 20 token × top-10 算标量熵       L283-306
        首次见到的 out_idx 写入 baseline           L304-305
        追加生成 token；若 tool-call 则执行并注入   L311-426
        仍未结束的下标 → final_active_indices      L428-432
        按 prompt 分组，在 n 的名额内：             L458-487
           ΔH = H_now - H_init
           p̃ = clip(U - w·ΔH, 0, 1)
           若 p̃ ≤ τ：copy 整条 curr_inputs 作为新轨迹
        已结束但不满 n：从裸 prompt 再开 1 条       L490-508  （与熵无关）
        新轨迹加入 active，进入下一轮
  每个 prompt 截取/补齐到恰好 n 条                 L536-555
```

---



## 9. 改这段逻辑时动哪里


| 目的                                  | 改哪里                                                                                        |
| ----------------------------------- | ------------------------------------------------------------------------------------------ |
| 换熵定义（按位置平均、用满词表、只用采样 token）         | `_calc_entropy` L172；收集 logprob 的循环 L289–298                                               |
| 前 20 token 改成前 k 个，或改成「tool 后整段」    | L291 `min(20, ...)`                                                                        |
| 绝对阈值（熵高于 θ 才允许 fork）                | L468–477，在 `entropy_delta` 之外加判断                                                           |
| 关掉随机、高熵必 fork                       | 去掉 `random.random()`，直接用 \Delta H 或 H_{\text{now}}                                         |
| 从某个高熵 **token** 切开而不是整轮前缀           | 当前没有 token 级熵序列；需要按位置存 `H_j`，fork 时 `curr_inputs[source][:cut]`                            |
| 只在 tool-call 后 fork、length 截断不 fork | 在 L444 前把非 tool-call 的 idx 从 `final_active_indices` 拿掉（需自己打标）                              |
| 调整预算                                | 脚本里的 `n` / `initial_rollouts` / `beam_size` / `entropy_weight` / `branch_probability`      |
| 验证集也要分支                             | L220–227 不要把 `beam_size` 设成 1                                                              |
| baseline 跨 batch 污染                 | `generate_sequences` 开头清空 `initial_entropy_dict`                                           |
| `mode=agent` 那套                     | `verl/workers/agent/tool_agent.py` 只有 `random() > branch_probability`，**没有熵**；改 ARPO 不要走那边 |


超参入口：`ARPO/scripts/ARPO_*.sh` 中 `INITIAL_ROLLOUTS`、`BEAM_SIZE`、`BRANCH_PROBABILITY`、`Entropy_weight`。yaml 默认值会被脚本覆盖。

---



## 10. 和论文表述的对齐与差距

论文：tool-call 反馈会把不确定性注入后续推理，**最初若干 token 熵高**，应在这些轮次加大采样。

代码对齐的部分：

- 停在 `</search>` / `</python>`，注入 `<result>` 后再生成，自然对准「反馈后的下一轮」
- 只用本轮前 20 token 估计该轮不确定度
- 用 \Delta H 相对第 0 轮升高来加大 fork 概率
- 用 `n` 做全局预算，避免每步都指数爆炸

代码没有做、或只是近似的部分：

- **没有 token 级定位**，不能回答「是第 3 个 token 还是第 17 个该切开」
- 熵是 top-10 logprob 拼袋后的启发式分数，不是标准 H(\pi(\cdot\mid x_{<t}))
- 第 0 轮（无 tool）也会参与同一套公式，只是 \Delta H=0
- length 截断的续写同样可以 fork，不限于 tool 边界
- fork 出的子轨迹以 **自己第一轮** 为新 baseline，不会把父轨迹的高熵一直遗传进 \Delta H

若要更贴近「找到高熵 token 再 fork」，最小改动是：在 L289–298 按位置存 20 个单步熵，取 \arg\max_j H_j 或第一个超过阈值的 j，复制 `generated_tokens[:j]` 而不是整轮 `curr_inputs`。那已经是算法变体，不是当前 ARPO 实现。