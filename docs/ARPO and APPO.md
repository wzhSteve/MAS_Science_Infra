# APRO 和 APPO的本质

ARPO 和 APPO 都属于“前缀分支采样”算法，它们都会选择合适的分叉点保留中间状态，从这个状态重新采样多个后续

本质都是采样Rollout 方法中的一类
```
Init-A   = P + U
Branch-A = P + V
```

## ARPO  

> 分叉点是工具级别（熵）, 分支进入直接进入 Actor loss

```
原始轨迹 ──→ Reward ──→ Actor loss
分支轨迹 ──→ Reward ──→ Actor loss
分支轨迹 ──→ Reward ──→ Actor loss
```
ARPO 把分支当作新的完整训练轨迹

### 训练过程

（1）初始化
```
问题 Q1
问题 Q2
问题 Q3

rollout.n = 16
initial_rollouts = 8
beam_size = 2
```
表示每道题最终希望获得 16 条轨迹，先独立生成 8 条，再通过分支补足

（2）在工具边界计算熵（这里有一个计算熵的公式）

ARPO 不扫描任意 token，而是在每轮工具执行后计算当前生成的不确定性。找到这些需要进行分支的节点位置，因此
真实的情况是要记录轨迹的需要分支的位置（前缀其实不用再重新生成了，因此前缀哪些状态都要保存）

（3）分支生成新的轨迹

从需要分支的位置开始生成，也就是之前的轨迹前缀需要保留, 形成类似下面的轨迹

```
共享前缀
├─ Branch-1 → 最终答对 → reward=1
├─ Branch-2 → 最终答错 → reward=0
└─ Branch-3 → 格式错误 → reward=-1
```
（4）GRPO Advantage

初始轨迹和分支轨迹计算 Advantage


## APPO 



> 分叉点是Token级别，分支不进入 Actor loss

```
原始轨迹 ──→ Reward ──→ 经分支对比缩放 ──→ Actor loss

分支轨迹 ──→ Reward ──┐
分支轨迹 ──→ Reward ──┼─→ 对比/归因信号
分支轨迹 ──→ Reward ──┘

分支 token ──X──→ 不直接进入 Actor loss
```

- 分支仍然有最终 reward；
- 分支仍参与 advantage 或对比计算；
- 但分支生成的 token 不直接计算 policy gradient；
- 分支结果主要用于调整原始 init 轨迹关键位置的 credit。
  
### 训练过程（常规情况）

| 对象 | 是什么 | 是否被梯度更新 |
| --- | --- | --- |
| Actor | 具有可训练权重的 Qwen | 是 |
| Rollout 引擎 | 用当前权重生成回答，例如 vLLM | 负责采样，并接收更新后的权重 |
| 工具和评分函数 | 搜索、Python、答案判定 | 通常不参与梯度更新 |

（1）第一步：生成初始轨迹

```
rollout.n = 16
initial_rollouts = 8

Q1
├─ Init-1
├─ Init-2
├─ ...
└─ Init-8

Rollout 过程中还保留了：

rollout_log_probs
rollout_entropys
response_mask
```
（2）第二步：分析初始轨迹，找分叉点

APPO 查看初始轨迹中的模型输出 token，给候选位置打分

一般是下面的公式理解：
```
Branching Score
= 标准化的 token entropy （该位置的预测分布有多不确定）
× 标准化的 Future Value（当前位置之后的累计策略 log-probability ratio 构造的代理量）
```
选择高分位置
```
Init-A：
token 0 …… token 20 …… token 60 …… 最终答案
               ↑             ↑
            候选分数高     候选分数低
```

> Actor 重新计算的 log probability，需要发送的后续情况

（3）第三步：复制前缀，重新采样后续


```
                    ┌─ 原始后续 U → 原始答案
共享前缀 P ─────────┤
                    └─ 新后续 V   → 分支答案
```

（4）给 Init 和 Branch 都计算 Reward和Advantage

```
Init-A   → 答错 → reward=0
Init-B   → 答对 → reward=1

Branch-A → 答对 → reward=1
Branch-B → 答错 → reward=0
```
各自在组内比较 reward，得到相对 advantage
```
Init 组：所有初始轨迹
Branch 组：所有辅助分支轨迹
```

（5）第六步：把分支的 Advantage 写回父轨迹前缀

```
Branch-A 的组内 advantage
       ↓
找到它来自 Init-A
       ↓
找到分叉位置 token 20
       ↓
加到 Init-A 的 token 0…20 的 advantage 上
```

| 数据 | 基础 Advantage |
| --- | --- |
| Init-A | -1 |
| Init-B | +1 |
| Branch-A，来自 Init-A | +1 |
| Branch-B，来自 Init-B | -1 |

以例子就可以得到：
```
Init-A：
  分叉前缀 P：-1 + 1 = 0
  原始后续 U：-1

Init-B：
  分叉前缀 S：+1 - 1 = 0
  原始后续 W：+1
```
> Init-A 虽然最终失败，但沿相同前缀重新续写可以成功，因此减轻对原始前缀的惩罚，而保留对原始失败后续的惩罚。

可以理解为：但如果从同一个 P 重新续写，分支得到较好的结果，APPO 就获得了额外证据：
不应仅仅因为原始后续 v 失败，就同等程度地惩罚前缀 P。


（6）计算 Actor loss，更新模型

> 改变对原始前缀 P 的训练信号，

### 本质

```
分支 V
  ↓
得到终局 Reward
  ↓
产生对比信号
  ↓
修改父轨迹 P 的 Advantage
  ↓
父轨迹 P 参与 Actor loss
```


# 一些疑问

## Advantage 到底是整个任务的，还是每个动作的？

**Reward 可以是任务级标量，但最终用于 Actor loss 的 advantage 通常是逐 token 的数组。**

可以分三层：

| 层次 | 例子 |
| --- | --- |
| 任务终局 reward | 这次完整执行得分 1 |
| 轨迹基础 advantage | 相比同题其他轨迹，这次是 +0.6 |
| token 最终 advantage （信用分配） | 广播基础值，再按分支覆盖、mask、额外权重修正 |

token 最终 advantage 因不同的算法会有不同的：

- RRPO: 广播基础值
- RPPO: 再按分支覆盖、mask

## 整个过程的更新本质是什么
轨迹是交互产生的**训练数据**；训练时根据它计算 loss，更新的是 LLM 的参数，而不是对LLM 调用 → 工具执行 → 下一次调用整个流程做端到端反向传播

即：分为**采样阶段和训练阶段**

（1）采样阶段：记录数据Rollout等形成loss需要的数据

比如如下的：一个Rollout涉及3个输入输出

| 样本 | 固定输入 | 固定输出 |
| --- | --- | --- |
| 调用 1 | 问题及当时上下文 | 搜索请求 |
| 调用 2 | 问题、历史、搜索结果 | 计算代码 |
| 调用 3 | 问题、历史、工具结果 | 最终答案 |

（2）训练阶段：汇总loss，对模型进行反向传播

```
固定输入 1 + 固定输出 1 → Qwen(θ) → log probability → loss₁
固定输入 2 + 固定输出 2 → Qwen(θ) → log probability → loss₂
固定输入 3 + 固定输出 3 → Qwen(θ) → log probability → loss₃
```

```
                  同一组参数 θ
                 /     |     \
              前向 1  前向 2  前向 3
                ↓       ↓       ↓
              loss₁   loss₂   loss₃
                 \      |      /
                    总 loss
                       ↓
                 对参数 θ 求梯度
```

## “汇总 loss、梯度累积”究竟是什么？

为了突出原理，先不用完整 PPO 公式。第 `i` 次调用的简化策略梯度 loss 可以写成以下纯文本形式：

```text
logp[i,t] = 模型对已记录 token y[i,t] 计算的条件 log probability
           条件：本次输入 x[i] + 本次输出中第 t 个 token 之前的内容

token_loss[i,t] = -mask[i,t] × A[i,t] × logp[i,t]

loss_i = 对本次调用所有输出 token 的 token_loss[i,t] 求和
```

这里：

- `i`：调用编号；`t`：这次调用中的输出 token 位置；
- `x[i]`：这次调用实际收到的上下文；
- `y[i,t]`：这次调用记录的第 `t` 个输出 token；
- `A[i,t]`：这个 token 分配到的 advantage；
- `mask[i,t]`：是否作为训练目标，参与为 1，不参与为 0；
- `θ`：这次调用所属的可训练模型参数，决定 `logp[i,t]` 的值。

输入、输出 token 和 advantage 都作为固定训练数据；可求导的是模型给这些输出计算的 log probability。实际 PPO 使用新旧概率比及裁剪等机制，以上公式只用于解释梯度的来源。

如果三次调用使用同一组参数，将它们的 loss 相加：

```text
total_loss = loss_1 + loss_2 + loss_3
```

其梯度就是各自梯度之和：

```text
梯度 g1 = loss_1 对参数 θ 的梯度
梯度 g2 = loss_2 对参数 θ 的梯度
梯度 g3 = loss_3 对参数 θ 的梯度

总梯度 g = g1 + g2 + g3
```

三个调用不需要通过工具或离散文本建立连续的可微计算图，只需要各自的 loss 都能对同一组参数求导。可以将样本组成 batch 后反向传播，也可以按 micro-batch 分别反向传播，将梯度累积到同一组参数上，再执行一次 `optimizer.step()`。实际训练需按配置进行样本或 token 归一化，梯度累积时也要使用对应的缩放。

**`backward()` 计算并累积梯度，`optimizer.step()` 才修改参数。采样时的调用次数不等于参数更新次数。**
