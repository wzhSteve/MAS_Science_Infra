# 层布局：AGL 黑盒 vs 可改钩子

日期：2026-09-15

## 原则

- **不**拆开重建 `agent-lightning`（Trainer / VERL / Store / LitAgent 基类仍是依赖）。
- 只把**常改、需灵活**的部分放到顶层 `rl/`。
- MAS 构图与 Runtime 在 `mas/`，`workflow/` **禁止** import `agentlightning` / `verl` / `ray`。

## 三栏


| 类型  | 路径                                              | 职责                                                                                                    |
| --- | ----------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 黑盒  | `agent-lightning/`                              | `agl.Trainer`、`agl.VERL`、Store、Dashboard、`emit_reward`、父类 `AgentLightningTrainer` / `AgentModeDaemon` |
| 钩子  | `rl/`                                           | outcome reward、`LossSpec`、algo overlay、advantage、Daemon/Trainer **子类**                                |
| 胶水  | `mas/lit_tir_agent.py`、`mas/train_tir_agent.py` | 跑图 + 调 `rl.*` + `agl.Trainer.fit`                                                                     |




## `rl/` 包

```text
rl/
  rewards/outcome.py   # 权威 reward 公式（零 AGL）
  loss.py              # LossSpec / AdvantageSpec
  train_signal.py      # TrainSignal + batch_to_train_signal
  hooks/               # overlay, advantage, daemon, trainer, arpo_rollout, …
```



## 数据流

```text
MASSpec.sampling → Collector / LitTirAgent → TrajectoryBatch
                 → rl.rewards.compute_outcome_reward → agl.emit_reward
TrainSignal (rl) → rl.hooks.overlay → Hydra → agl.VERL → agl.Trainer
```

兼容：`mas/algos`、`mas/workflow/rewards.py`、`mas/workflow/train_signal.py` 为薄 re-export。