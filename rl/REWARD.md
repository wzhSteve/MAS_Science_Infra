# 训练 Reward

四算法对比的默认训练标量是终局答案对错，由 `EVALUATION_MODEL`（当前为 gpt-4o）打 0/1。主曲线用这一列，不要把格式加减分混进同一张图。

## 默认：答案对错

| 算法 | `algorithm.reward.outcome` | 训练标量 |
|------|----------------------------|----------|
| GRPO | `llm_binary` | gpt-4o 0/1 |
| Flow-GRPO | `llm_binary` | gpt-4o 0/1 |
| ARPO | `llm_binary` | gpt-4o 0/1 |
| APPO | `llm_binary` | gpt-4o 0/1 |

Judge 读 `.env` 的 `EVALUATION_MODEL_API_KEY` / `EVALUATION_MODEL_API_BASE` / `EVALUATION_MODEL`，与训练用的 `OPENAI_MODEL` 分开。实现在 `rl/rewards/llm_judge.py`。抽取出的终局答案送进 Flow-GRPO 同款 prompt；调用失败记 0，训练不中断。

`rl/hooks/overlay.py` 的 `_apply_reward_outcome` 写入上述 outcome。`mas/train_tir_agent.py` 的 `reward_outcome_for` 把同一结果放到环境变量 `TIR_REWARD_OUTCOME`，采集端 `mas/workflow/collector.py` 据此打分。

手写的 `algorithm.reward.outcome`（且 `meta.outcome_source` 不是 `preset`）优先于上表。

## ARPO / APPO 可选：格式 reward

只对 ARPO、APPO 生效。配置键：

```yaml
algorithm:
  reward:
    include_format: false   # 默认。true 才叠加格式分
```

界面：采样设置或训练方案里，算法为 ARPO / APPO 时出现 Reward 下拉。

- `仅答案对错（gpt-4o）` → `include_format: false`，outcome 仍是 `llm_binary`。
- `答案对错 + 格式` → `include_format: true`，outcome 改为 `deep_research_llm`。

格式项来自 ARPO/APPO 的 `deep_research.py`，准确度已从 token F1 换成 gpt-4o 0/1。实现在 `rl/rewards/deep_research.py`。答错保持 0，格式项只改「抽不出答案」和「已经判对」的样本：

| 项 | 何时 | 幅度 |
|----|------|------|
| 不可解析 | 抽不出答案 | -0.3 |
| strict 不符 | 已判对，工具标签或 `\boxed{}` 不完整 | -0.05 |
| 缺 boxed | 已判对，且解析路径不是 `<answer>` 内的 `\boxed{}` | -0.05 |
| 双工具 | 已判对，且 search 与 python 都用过 | +0.1 |

GRPO / Flow-GRPO 忽略 `include_format`，始终是 `llm_binary`。

## Loss（与 reward 开关无关）

`rl/hooks/overlay.py` 的 `_apply_algo_actor_loss` 按源码写入，不随格式开关变化。

| 算法 | KL | clip low / high | 其它 |
|------|----|-----------------|------|
| GRPO、Flow-GRPO | `use_kl_loss=true`，`kl_loss_coef=0.001` | 0.2 / 0.3 | `entropy_coeff=0` |
| ARPO | `use_kl_loss=true`，`kl_loss_coef=0.0` | 0.2 / 0.3 | `loss_mode=vanilla` |
| APPO | `use_kl_loss=true`，`kl_loss_coef=0.001` | 0.2 / 0.28 | `loss_mode=future_kl` |
