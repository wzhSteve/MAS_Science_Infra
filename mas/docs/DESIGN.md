# TIR Agent 设计说明

对照 [math_gsm interview/10](../../math_gsm/docs/interview/10_agent_rl_arpo_aepo_igpo_gigpo.md)。本目录把五套 GRPO 族算法接到 **Agent-lightning 的两处钩子**，不 fork 官方 veRL 仓库，也不改 `examples/math_gsm` 的训练默认。

## 1. 三层怎么落到文件

```text
rollout  →  tir_agent.py (LangGraph) + algos/daemon.py (enqueue / resume)
reward   →  algos/rewards.py + LitTirAgent.emit_reward / emit_annotation
loss / Â →  algos/advantage.py + algos/gigpo_core.py（trainer 在 compute_advantage 处替换）
```

入口 [`train_tir_agent.py`](../train_tir_agent.py) 与 math_gsm 同形：

```python
algorithm = agl.VERL(
    config,
    trainer_cls=TirAgentLightningTrainer,
    daemon_cls=bound_daemon_cls(tir_algo, tir_cfg),
)
trainer = agl.Trainer(n_runners=..., algorithm=algorithm)
trainer.fit(agent, train_dataset, val_dataset)
```

`algorithm.adv_estimator` **始终** `"grpo"`，避免 VERL 拉起 Critic。CLI `--algo` 写入 `algorithm.tir_algo` 和 `algorithm.tir.*`（见 `algos/overlay.py`）。

## 2. 工具为什么要 copy 而不是 import epc_aw

`examples/CD_TAA/MAS/epc_aw/tools` 内部会 `create_llm_engine` 做页面摘要，那是 **不可训的第二套 LLM**，会污染 credit 并拖慢 rollout。本目录只保留：

- DuckDuckGo HTML 检索（`tools/search.py`），默认 `LOCAL_SEARCH` 风格，无 Chrome / Gemini / Playwright
- Wikipedia API + REST summary（`tools/wikipedia.py`）
- math_gsm 同款 Python 沙箱（`python_tool.py`：禁 import、timeout、`result=`）

无网时 `TIR_OFFLINE_SEARCH=1` 或 `prepare_data.py --offline`。

## 3. 数据

统一 schema：`id, question, answer, answers, source, split`。`source` 为 `gsm8k|hotpot|nq`。Reward 按 source 分支：数字 exact/rel_tol vs 词级 F1。

## 4. 各算法实现与论文差距

### GRPO

Daemon 对同一 parquet 行 enqueue `rollout.n` 次；adapter 转 triplet；VERL `compute_advantage(grpo)`。层次 outcome 见 README。

### IGPO

论文：每个 assistant turn 后算 \(\log\pi(\text{GT}\mid\text{prefix}_t)-\log\pi(\text{GT}\mid\text{prefix}_{t-1})\)，stop-gradient，与 outcome 分开 z-norm 再 \(\gamma\) 折现。

本目录 MVP：

- 采样同 GRPO（不改拓扑）
- Agent 侧用「工具观测文本是否新包含 GT」作稠密代理，写入 `tir.ig_deltas`
- Trainer `_apply_igpo`：组内对 IG / outcome 分别 z-norm，非最后 turn 用 IG，最后 turn 用 outcome，折现后广播到 response token

**未做**：actor 对 GT 的 teacher-forcing 一次（或按 turn）forward。这需要额外序列构造和自定义 mask，可在不改采样的前提下后续补。

### GiGPO

`gigpo_core.compute_gigpo_advantages`：episode \(A^E\)（同题组内相对总回报）+ step \(A^S\)（同 `(data_id, anchor_obs)` 上折扣回报相对），\(A=A^E+\omega A^S\)，默认 \(\omega=1\)。

`anchor_obs = sha1(tool_name + "\\n" + obs[:512])[:16]`。同一题、同一检索结果可对齐。数学题很少撞 anchor，\(A^S=0\)，自动退化 GRPO。VERL 已对同一行 `rollout.n` 次独立跑，满足「同初始状态」。

实现自包含，不依赖 contrib 未完成 yaml。contrib 的 `anchor_obs` 钩子只作参考。

### ARPO

论文：工具返回后前 k 个 token 熵高跳 \(\Delta H\) 则树分支；vLLM 内核 partial rollout；soft GRPO（共享前缀 IS 相同）。

本目录近似：

1. 第一波只 enqueue `tir.initial_rollouts`（fast 默认 2，且不超过 `n-1`）条完整轨迹
2. Agent 在第一次 tool 后把 messages dump 到 `.resume_cache/{rollout_id}.json`，并估计 turn 级熵（有 logprobs 则用 \(-\mathbb{E}\log p\)，否则 tool-call≈1.0 / 纯文本≈0.3）
3. Daemon 若 \(\Delta H\) 超阈且预算未满：再 enqueue 带 `resume_messages` 的任务（`beam_size`）；不足则补全局轨迹
4. Agent 见 `resume_messages` 则从该前缀继续。**不**把已生成 token 的梯度合并——M 条完整轨迹进 GRPO，共享前缀自然相同 IS → soft 设定

**未做**：逐 token 在 decode 中途 beam；不改 vLLM。熵不是论文里的 decode 中途 token 熵。

与官方 ARPO（`examples/ARPO`，`vLLMRolloutWithTools` 内真前缀 fork）的 Agent / Tool / 算法对照见 [ARPO_VS_TIR.md](ARPO_VS_TIR.md)。

### AEPO-lite

论文额外两点：预监控分预算 \(m=M\cdot\sigma(\beta(H_{\text{root}}-H_{\text{tool}}))\)；连续高熵分支惩罚 \(P_t=(\alpha+\gamma\Delta H_t)(1-\hat P(l))\)；clip 里对高熵侧 stop-grad；Â 乘熵项。

本目录：

- 第一波每题 1 条探针轨迹，再按上式分全局 vs 分支预算，并乘连续分支惩罚
- `compute_advantage` 之后 \(\tilde A=\tilde A_{\text{Acc}}(1+a\tilde A_{\Delta H})\)，熵来自 `entropys` 或 \(|\text{old_log_probs}|\) 代理

**未做（二期）**：actor loss 里的 stop-grad clip。不要声称完整 AEPO。

## 5. Annotation → batch

`LitTirAgent` `agl.emit_annotation` 键：`tir.source / n_search / n_python / format_ok / obs_hashes / ig_deltas / h_root / h_tool / consecutive_high / resume_parent_id`。

Daemon 从 span attributes 抽出，写入 `DataProto.non_tensor_batch` 的 `anchor_obs`、`ig_logprob_delta`、`h_root`。

## 6. 验收

- `python prepare_data.py --train-limit 32 --val-limit 8`
- `python train_tir_agent.py fast --algo grpo`
- `python tir_agent.py`（需 `OPENAI_API_BASE`）

## 7. MAS_structagent RL（独立入口）

与 LangGraph `LitTirAgent` **并存**：可训对象是 `MAS_structagent`（Planner→Actor→Verifier）。

```text
parquet → agl.VERL → main_llm (vLLM Qwen3-4B)
       → LitMASAgent.rollout
       → llm_runtime_override(base_url, model) + OPENAI_* 双保险
       → construct_solver(...).solve(question)
       → compute_mas_outcome_reward → emit_reward
```

| 文件 | 角色 |
|------|------|
| `mas_agent.py` | `bootstrap_mas_package` + `LitMASAgent` + debug |
| `train_mas_agent.py` | GRPO 训练入口（关 Hermes tool parser） |
| `MAS_structagent/epc_aw/engine/runtime_override.py` | 强制全部 `ChatOpenAI` 走 VERL endpoint |
| `algos/rewards.py` → `compute_mas_outcome_reward` | 无 `<answer>` format 惩罚，仅 Acc/F1 |

要点：

- **不读** `.env` 的 dmxapi；训练 LLM 一律来自 `resources["main_llm"].get_base_url(...)`
- 工具内二次 LLM（摘要 / 代码生成）也走同一策略模型（`tool_engine=[policy]*N` + override）
- 不启用破损的 `Google_Search_Tool`；默认 `Base_Generator` / `Python_Coder` / `Wikipedia` / `Web_Search`
- 首版只接 GRPO；Tir 的 ARPO/GiGPO annotation 未接

验收：

```bash
OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_MODEL=/root/autodl-tmp/LLM/Qwen3-4B \
  python mas_agent.py
python train_mas_agent.py fast
# 或 bash scripts/train_mas_2gpu.sh
```


## 8. Workflow 分层（本目录落地，不改 agentlightning 核心）

在 `workflow/` 增加 **无 RL 可独立运行** 的数据层；训练仍走 `LitTirAgent` + `algos/` + `train_tir_agent.py`。

```text
workflow/contracts.py   — Trajectory / BranchPoint / Snapshot
workflow/archive.py     — Archive + dump_resume_with_archive（兼容 resume_messages）
workflow/collector.py   — Collector：TirAgent 或 --mock → Trajectory + reward
workflow/train_signal.py — AdvantageSpec / LossSpec → RL
scripts/collect_rollouts.py
scripts/check_workflow_deps.py   — workflow 禁止 import agentlightning/verl
```

### 无 RL 采集

```bash
cd mas
PYTHONPATH=. python scripts/collect_rollouts.py --mock --out /tmp/traj.json
PYTHONPATH=. python scripts/check_workflow_deps.py
```

`Collector(mock=True)` **不**启动 VERL；`--mock` 路径不加载 `agentlightning`。

### Archive ↔ ARPO

- `lit_tir_agent` 在 dump 时调用 `dump_resume_with_archive`（只写 Archive；`.resume_cache` 仅作读取回退）
- `algos/daemon.py` 第二波用 `load_resume_messages`；优先 task.`resume_from`（BranchPoint）
