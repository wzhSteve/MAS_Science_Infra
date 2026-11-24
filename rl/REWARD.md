# 训练 Reward

四算法对比的默认训练标量是终局答案对错，由 `EVALUATION_MODEL`（当前为 gpt-4o）打 0/1。主曲线用这一列，不要把格式加减分混进同一张图。

## FCR-GRPO 记账恒等式

每轨迹总分与 token 注入严格同式：

```
R_i = w_a·Ra_i + w_c·Rc_i + w_f·rate_i
    默认 w_a=0.5, w_c=0.5, w_f=0.1

Ra_i ∈ [0, 1.1]   soft F1（accuracy / tir.reward；+ 可选 multi_tool；format 不打 -1）
Rc_i ∈ [0, 1]     correct pattern：N_i/N_max（弱 require_prev_fail + 硬分母）
rate_i ∈ [0, 1]   planner scoreable_rate → Rf = w_f·rate
EM_i ∈ {0,1}      binary_correct → fcr/em_mean（仅统计，不进 R）
```

注入点（`rl/hooks/advantage.py`）：**先清零**该 rollout 所有 response token，再在
**最后一行末 token** 写入一次 `R_i`（避免多轮放大）。

指标口径（同名必须同义）：

| 键 | 含义 | 分母 |
|----|------|------|
| `val/reward` / `fcr/R_answer_mean` | mean(Ra soft F1) | per-rollout |
| `fcr/em_mean` | mean(binary_correct) | per-rollout |
| `fcr/outcome_mean` | 同 Ra（诊断别名） | per-rollout |
| `fcr/R_correct_mean` | mean(Rc 原始 pattern，注入前) | per-rollout |
| `fcr/Rf_mean` | mean(w_f·rate) | per-rollout |
| `fcr/R_total_mean` | `w_a·Ra + w_c·Rc + Rf` | per-rollout |
| `fcr/row_*` / `fcr/reward_identity_gap` | 注入后 row 口径；gap 对 `Ra_part+Rc_part+Rf` | watch >1e-6 STOP |

`training/reward`（per-rollout final_reward 均值）与 `fcr/R_total_mean` 分母不同，
对账只看 `fcr/reward_identity_gap`。

## 默认：答案对错

| 算法 | `algorithm.reward.outcome` | 训练标量 |
|------|----------------------------|----------|
| GRPO | `llm_binary` | gpt-4o 0/1 |
| Flow-GRPO | `llm_binary` | gpt-4o 0/1 |
| ARPO | `llm_binary` | gpt-4o 0/1 |
| APPO | `llm_binary` | gpt-4o 0/1 |
| FCR-GRPO | `hierarchical` | Ra soft F1（见上恒等式） |

Judge 读 `.env` 的 `EVALUATION_MODEL_*`。实现在 `rl/rewards/llm_judge.py`。

## ARPO / APPO 可选：格式 reward

ARPO/APPO 默认无格式；`include_format: true` 时 soft 格式揉进 outcome 进 GRPO。
FCR 用独立 Rf（`w_f·scoreable_rate`）末 token 注入，默认 `w_f=0.1`。
