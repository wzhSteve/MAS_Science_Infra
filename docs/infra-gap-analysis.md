# MAS Research Infra：design.md 与 agent-lightning 联合分析

**日期**：2026-09-06  
**范围**：以 [design.md](../design.md) 四层为产品真源，以 [agent-lightning](../agent-lightning)（含 `examples/tir_agent`）为落地真源。  
**非范围**：不吸收 [MAS_INF/ENGINEERING_SPEC.md](/root/autodl-tmp/MAS_INF/ENGINEERING_SPEC.md) 为合同；仅借用其「AGL 是 backend、不要自研 PPO、Event 与 Trajectory 分层」三条边界。  
**本轮不改训练代码**；下文「如何改」是改造规格，供后续工作包使用。

> **2026-09 初版本已落地**（四层部分功能）：`science-infra collect/diagnose/status/doctor`、`workflow/` 单路径 episode、TrainSignal overlay、Harness `log_error`/`loss_volatility`。可复制命令见 [design.md Phase 0](../design.md) 与仓库 [README.md](../README.md)。下文保留当时的差距分析，个别「不存在」陈述以 Phase 0 为准。

---

## 0. 一句话结论

**现状不是四层 research infra，而是「能训的 TIR 示例 + 半截 workflow 原型」。**

- RL 主路径已经能跑：LangGraph 单 hub ReAct -> `LitTirAgent` -> Agent-Lightning `Trainer`/`VERL` -> 五套 GRPO 族算法挂钩。
- design.md 要的 **Framework / Runtime 分层、格式化跨层合同、层独立、Harness 插件、UI** 大部分未落地。
- [pyproject.toml](../pyproject.toml) 声明的 `science_infra` 包和 `science-infra` CLI **不存在**；Phase 0 表格里的「已实现」是过时陈述。
- 正确策略：**不 fork `agentlightning/` 核心**；把 science 契约先在 `examples/tir_agent/workflow/` 长成，稳定后再抽包。TIR 图作为默认 Runtime adapter，而不是产品 Runtime。

---

## 1. 范围与原则

### 1.1 产品要回答的问题

design.md 的目的：研究人员配置 MAS workflow，然后直接做 rollout / RL / evaluation。层与层由**单一 Boundary API** 连接，每层可单独工作（例如 MAS 完全不需要 RL 和 Harness）。

研究优先级（design.md）：

| 优先级 | 层 | 本轮态度 |
|--------|----|----------|
| 1 最高 | RL | 不自研训练循环；挂 AGL/VERL adapter |
| 2 | MAS workflow（构建 + 数据） | 先单 Agent，接口为 MAS 预留 |
| 3 | Harness | 插件注册表，先留口后实现 |
| 4 | UI | 只定传参接口，不设计前端 |

### 1.2 必须守住的原则

1. **层独立**：MAS 构建 / 数据层不得 import `agentlightning` / `verl` / `ray`。RL 不得持有 LangGraph `StateGraph` 或 `TirAgent` 类。Harness 不得 `from tir_agent import TirAgent`。
2. **接口格式化**：「单一接口」= 每层一个 Boundary API，内部可有多个 typed 合同（Pydantic，可 JSON），不是整个系统只有一个对象。
3. **层内插件**：AgentRole / Skill / Tool / LLMProvider / MemoryStore / RewardFn / Diagnoser 均可替换。
4. **不改 AGL 核心**：`agentlightning/` 的 `LitAgent`、`Trainer`、`Runner`、`LightningStore`、`VERL(trainer_cls, daemon_cls)`、`emit_reward` / `emit_annotation`、`LLM.get_base_url` 已经是稳定基板。tir 算法已通过 `trainer_cls`/`daemon_cls` 挂钩，符合「精简用户改动面」。
5. **Phase 0 单 Agent**：默认拓扑是单个 hub ReAct（`web_search` + `wikipedia_search` + `execute_python`）。多智能体只留插件槽，不把 `MAS_structagent` 当成第二套 infra。

### 1.3 从 MAS_INF 只借三条边界（非合同）

- Agent-Lightning 是 **Optimizer / 执行基板**，禁止再做一套 Store / Trainer / PPO。
- 运行时真相应是事件流；`Trajectory` 是投影，不要让 Trajectory 承担「什么都有」。
- Reward 的 **canonical 计算**在数据层；Agent 侧 `emit_reward` 只转发同一数字（AGL 训练需要 span）。

### 1.4 命名冲突（必须先记住）

| 名字 | 位置 | 实际含义 |
|------|------|----------|
| `ExecutionEvent` | `agentlightning.execution.events` | 进程取消信号（`threading.Event` 同类） |
| `ExecutionEvent` | `tir_agent/workflow/contracts.py` | 研究用执行事件（task_start / tool_call / ...） |

后续契约建议 science 侧改名为 `MasEvent` 或 `TraceEvent`，避免与 AGL 撞名。下文仍用 workflow 现名，并在首次出现处标注。

---

## 2. AGL 基板地图（不要重造的部分）

Agent-Lightning 0.3.x 已经把「怎么训一个 agent」拆成稳定角色。tir_agent 只应 **挂上去**，不应复制。

```text
agl.Trainer.fit(agent, train_dataset, val_dataset)
        |
        +- ExecutionStrategy（进程编排）
        +- Algorithm = agl.VERL(config, trainer_cls=?, daemon_cls=?)
        |       +- verl PPO loop + AgentModeDaemon
        |              enqueue Rollout -> 等 span -> TracerTraceToTriplet -> DataProto
        +- Runner x n  （LitAgentRunner）
        |       +- LitAgent.rollout(task, resources, rollout)
        |              resources["main_llm"] -> OpenAI-compat endpoint（RL 第三路）
        |              agl.emit_reward / emit_annotation -> OTel span -> Store
        +- LightningStore + Tracer + TraceAdapter
```

### 2.1 与 design.md 三钩子的真实对应

design.md 要求 RL 层从三个结构抽象，精简用户修改面：rollout / advantage / loss。

| design.md 钩子 | AGL / tir 落点 | 用户今天实际改哪里 |
|----------------|----------------|-------------------|
| rollout 采样 | `AgentModeDaemon` enqueue；ARPO/AEPO 在 `TirAgentModeDaemon._enqueue_tree_branches` 二次 enqueue | [`algos/daemon.py`](../agent-lightning/examples/tir_agent/algos/daemon.py) |
| reward | `LitTirAgent` 内 `compute_outcome_reward` + `agl.emit_reward`；adapter 把 span 收成 `Triplet.reward` | [`algos/rewards.py`](../agent-lightning/examples/tir_agent/algos/rewards.py) + [`lit_tir_agent.py`](../agent-lightning/examples/tir_agent/lit_tir_agent.py) |
| advantage | Hydra `algorithm.adv_estimator` **永远是 `grpo`**（关 Critic）；真实算法名在 `algorithm.tir_algo`；`TirAgentLightningTrainer` 在 `compute_advantage` 处替换 | [`algos/trainer.py`](../agent-lightning/examples/tir_agent/algos/trainer.py) + [`algos/advantage.py`](../agent-lightning/examples/tir_agent/algos/advantage.py) |
| loss | 未单独挂钩；走 VERL PPO clip（`clip_ratio_low/high` 等在 Hydra config） | [`train_tir_agent.py`](../agent-lightning/examples/tir_agent/train_tir_agent.py) 的 `RL_TRAINING_CONFIG` |

这已经满足「基于 AGL/verl 修改、不 fork 官方 veRL 仓库」。**缺口不在 AGL，而在 MAS 契约没有接到这三钩子上。**

### 2.2 AGL 已覆盖的 LLM 三路

[`agentlightning/types/resources.py`](../agent-lightning/agentlightning/types/resources.py) 的 `LLM` / `ProxyLLM`：

- 第三方 API：把 `endpoint` 指到 OpenAI-compat 网关。
- 本地 LLM：指到 vLLM / 任意 `OPENAI_API_BASE`。
- RL 返回的 endpoint：训练时 `resources["main_llm"].get_base_url(rollout_id, attempt_id)`（`ProxyLLM` 会改写 URL 以便归因）。

Phase 0 不需要再发明 `LLMProvider` 协议才能训起来；需要的是 **Framework 里用同一绑定描述这三路**，Runtime 只消费绑定，不直读 `.env`。

### 2.3 禁止对 AGL 做的事

- 改 `agentlightning/verl/trainer.py` 主循环来「支持 TrainSignal」。
- 把 `verl.DataProto` 写进 science 契约。
- 自研第二套 LightningStore / Trace UI / PPOTrainer。
- 让 `class Trainer: def __init__(self, runtime: TirAgent)` 这种持有 Runtime 的写法进入 RL 层。

合法扩展点（已经在用）：

```python
algorithm = agl.VERL(config, trainer_cls=TirAgentLightningTrainer, daemon_cls=bound_daemon_cls(...))
trainer = agl.Trainer(n_runners=..., algorithm=algorithm)
trainer.fit(LitTirAgent(), train_data, val_data)
```

GRPO 甚至不用自定义 class：`agl.VERL(config)` 即可（[`train_tir_agent.py`](../agent-lightning/examples/tir_agent/train_tir_agent.py) `tir_algo == "grpo"` 分支）。

---

## 3. tir_agent 落地图

代码主线在 `agent-lightning/examples/tir_agent/`。这是 **示例目录承担了 infra 职责**，所以层边界必然发糊。

### 3.1 文件职责

```text
examples/tir_agent/
  tir_agent.py          LangGraph ReAct 图（MAS 构建的「Runtime 实现」，无 AGL import）
  lit_tir_agent.py      LitAgent 包装：跑图 + emit_reward/annotation（RL 入口）
  python_tool.py        数学沙箱（禁 import、timeout、result=）
  tools/                web_search / wikipedia / langchain TOOL_MAP
  algos/                RL 三钩子实现（依赖 AGL + verl）
    rewards.py          outcome / format / multi-tool；MAS 另有 compute_mas_outcome_reward
    daemon.py           span->tir_meta；ARPO/AEPO 二次 enqueue
    trainer.py          拦截 compute_advantage
    advantage.py        IGPO / GiGPO / AEPO-lite
    overlay.py          Hydra 上写 tir_algo，adv_estimator 锁 grpo
    arpo_rollout.py     resume_cache 序列化、熵估计
    gigpo_core.py       A^E + omega A^S
  workflow/             无 AGL 的数据层原型（仍 import tir_agent / algos）
    contracts.py        Trajectory / ExecutionEvent / Snapshot / BranchPoint
    archive.py          磁盘档案 + 兼容 .resume_cache
    collector.py        mock 或 TirAgent.invoke -> Trajectory + reward
    train_signal.py     AdvantageSpec / LossSpec 外壳（无消费端）
  scripts/collect_rollouts.py     MAS-only CLI
  scripts/check_workflow_deps.py  禁止 workflow import agentlightning/verl/ray
  train_tir_agent.py    训练入口
  mas_agent.py / train_mas_agent.py / MAS_structagent/   平行多智能体栈
```

算法对照与论文差距见 [examples/tir_agent/docs/DESIGN.md](../agent-lightning/examples/tir_agent/docs/DESIGN.md)，本文不重复论文细节。

### 3.2 两条执行路径（核心结构问题）

今天「跑一次 TIR」有两套几乎重复的代码：

```text
路径 A — 训练（占 GPU / 要 AGL）
  parquet 行
    -> LitTirAgent.rollout
    -> TirAgent.graph().invoke
    -> compute_outcome_reward
    -> agl.emit_reward + emit_annotation
    -> （可选）dump_resume_with_archive
    -> Daemon 从 span 抽 tir.* -> DataProto
    -> VERL compute_advantage / PPO

路径 B — 离线采集（可 --mock，不启 VERL）
  task dict
    -> Collector.collect_one
    -> mock 或同一 TirAgent.graph().invoke
    -> default_reward_fn -> Trajectory.final_reward
    -> batch_to_train_signal  （只写 JSON，无人读取）
```

两条路径都调用 `TirAgent`，但：

- **事件**：路径 B 写 `ExecutionEvent` jsonl；路径 A 几乎不写（只在 ARPO dump 时 snapshot）。
- **reward**：两边都调 `compute_outcome_reward`，但路径 A 在 Agent 内算完就 emit；路径 B 在 Collector 再算一遍。公式今天碰巧相同，没有单点强制。
- **TrainSignal**：只在路径 B 的 CLI 出现；路径 A 的 trainer **零引用** `AdvantageSpec`/`LossSpec`。

这直接违反 design.md「数据层提供接口传递给 RL」——接口造了，RL 没接。

### 3.3 两条 MAS（第二套未契约化）

| | TIR hub ReAct | MAS_structagent |
|--|---------------|-----------------|
| 入口 | `train_tir_agent.py` / `lit_tir_agent.py` | `train_mas_agent.py` / `mas_agent.py` |
| 编排 | LangGraph `StateGraph` | `Solver`：Planner -> Executor -> Diagnoser/Verifier |
| 训练包装 | `LitTirAgent` | `LitMASAgent` + `llm_runtime_override` + 环境变量双保险 |
| workflow 契约 | 部分（Collector/Archive） | **无** |
| Archive / BranchPoint | ARPO 路径有 | 无 |
| 算法 | GRPO/ARPO/AEPO/IGPO/GiGPO | 仅股票 GRPO |
| LLM | `resources["main_llm"]` | 同样来自 VERL，但靠改 `OPENAI_*` 防止漏到 `.env` dmxapi |
| Memory | 只有 LangGraph `messages` | `SystemMemory` + 各角色 Memory，未进跨层 schema |

`MAS_structagent` 是可运行的 PEV 研究代码，**不是** design.md 的「可扩充 subagent 插件」。后续若要多智能体，应把角色收成 Framework 插件，Runtime 仍输出同一事件合同；不要让 `workflow/` 和 `MAS_structagent/` 各长一套。

### 3.4 五算法挂钩点（已实现，且应保持在 algos/）

| `tir_algo` | rollout | advantage | 诚实边界（已有文档） |
|------------|---------|---|----------------------|
| grpo | `rollout.n` 条独立轨迹 | VERL 组内相对 outcome | 与 math_gsm 同形 |
| arpo | 先 `initial_rollouts`，高熵 tool 后 `resume_messages` 再采 | 同 GRPO（soft：完整轨迹进组） | 不改 vLLM 内核 |
| aepo | 1 条探针再分预算 + 连续高熵惩罚 | GRPO 后再乘熵项 | 无论文 clip stop-grad |
| igpo | 同 GRPO | 工具观测是否新含 GT 作 IG 代理 | 非 actor teacher-forcing |
| gigpo | 同 GRPO | A^E + omega A^S，anchor=`hash(tool+obs)` | 无重复状态时退化为 GRPO |

这些实现位置正确：**留在 `algos/`，通过 AGL 公开钩子工作**。infra 要做的是让 `AdvantageSpec.name` / daemon 采样策略读同一份 spec，而不是把 GiGPO 公式搬进 `workflow/contracts.py`。

---

## 4. 最优四层设计

保持 design.md 的四层与优先级。层与层只走下表合同；层内插件可换。

```text
L4 UI  --------------------------------------------- 最后做
  |  init params / algo name / 外链 dashboard
  v
L0 MAS 构建  Framework（静态 spec） + Runtime（动态执行） + Archive
  |  MasEvent[]  +  Snapshot / BranchPoint
  v
L0 MAS 数据  Collector + RewardFn
  |  TrajectoryBatch + TrainSignal      --> L2 RL adapter（AGL/VERL）
  |  Trajectory + ArchiveRef            --> L3 Harness plugins
  |
  <-- L2 写回 resources["main_llm"] endpoint
  <-- L3 写回 BranchPoint + Intervention（fork，不改父事件）
```

### 4.1 Boundary API（每层一个门面）

#### MAS 构建：`ExecutionService`

```text
run(spec_ref, task, config) -> ExecutionResult   # events + 可选 messages 投影
fork(branch_point, intervention) -> ExecutionRef
collect(ref) -> ExecutionResult
```

- Phase 0 默认实现：**现有 LangGraph `TirAgent`** + Event Normalizer。
- 不把 `AgentState` / `StateGraph` 放进契约。
- `spec_ref` 可以是内联 YAML（`hub_react.yaml`）或以后的 artifact id。

#### MAS 数据：`Collector` + `RewardFn`

```text
Collector.collect(tasks, n) -> TrajectoryBatch
RewardFn(traj, task) -> float          # canonical
batch_to_train_signal(batch, algo) -> TrainSignal
```

Runtime **可以没有 reward 就结束**。训练时 LitAgent 再调用 **同一** `RewardFn`，然后 `emit_reward`。禁止 Agent 内手写第二套公式。

#### RL：`OptimizerBackend`（adapter，不是自研循环）

```text
optimize(train_signal | live_rollouts, algo_config) -> PolicyEndpoint
```

实现 = 今天的 `agl.VERL` + 可选 `trainer_cls`/`daemon_cls`。  
`TrainSignal.advantage` / `.loss` 映射到 Hydra `algorithm.tir` 与 actor clip 超参；**真正的 advantage 仍在 verl `compute_advantage` 里算**。数据层只传 spec，不算 token-level advantage。这与 design.md「advantage 设计：reward − critic」的心智一致：GRPO 的 critic 就是组内均值，算在 backend。

#### Harness：`Diagnoser` 注册表

```text
diagnose(ctx: Trajectory | RLTrainLog) -> Hypothesis[]
```

未实现的插件返回 `[]`。反馈只经 `ExecutionService.fork`。禁止 import Runtime 类。

#### UI：`StudioParams`

只读写上述合同的字段（spec 路径、endpoint、`--algo`、n、dashboard URL）。不持有 graph 实例。

### 4.2 跨层 typed 合同

| 合同 | 生产者 | 消费者 | 今日代码 | 目标 |
|------|--------|--------|----------|------|
| `MASSpec` / Framework YAML | 用户 / UI | Runtime | **无** | 新增；Phase 0 一份 `hub_react.yaml` |
| `MasEvent`（现 `workflow.ExecutionEvent`） | Runtime Normalizer | 数据层、Harness | 有形状，训练路径几乎不写 | 训练与 collect 都写；收紧 kind，少用 `payload` dict |
| `Snapshot` / `BranchPoint` / `ArchiveRef` | Archive | ARPO daemon、Harness fork | 有，且与 `.resume_cache` 双写 | Archive 为唯一 resume 源 |
| `Trajectory` / `TrajectoryBatch` | Collector 或从 events 投影 | RL adapter、Harness | 有；TIR 专用字段（`n_search`/`n_python`）焊死 | 稳定面 + `extensions.tir` |
| `TrainSignal` | 数据层 | RL overlay | 仅 CLI dump | `apply_algo_overlay` 读取 AdvantageSpec/LossSpec |
| `RewardSignal` | RewardFn | emit_reward、Trajectory.final_reward | 散落 float | 同一函数 |
| `RLTrainLog` | AGL metrics jsonl / TB | Harness、UI 外链 | `ensure_metrics_env` 已有路径 | 不自研 SPA，只存 URI |
| `Intervention` | Harness | `fork` | **无** | P4 再做；P0 只需 BranchPoint 续跑 |

`Trajectory` 不要继续堆训练私货。`n_search` / `n_python` / `format_ok` 应迁到 `meta` 或 `extensions.tir`，否则换拓扑就要改 schema。

### 4.3 MAS 构建：Framework vs Runtime

design.md 已点名这两个概念，代码未落地。

**Framework（静态、可序列化）**

Phase 0 最小字段：

```yaml
# 建议路径：examples/tir_agent/specs/hub_react.yaml
schema_version: "0.1.0"
topology: hub_react          # preset；不是公理
hub:
  role: orchestrator
  skills: [react_loop]
tools: [web_search, wikipedia_search, execute_python]
llm:
  kind: api | local | rl_endpoint
  model: ...
  base_url: ...              # rl_endpoint 时由 AGL 覆盖
memory:
  agent: messages            # Phase 0 = 对话缓冲
  system: none               # 接口留空
archive:
  window: post_first_tool    # ARPO 用
```

以后扩 PEV：增加 `agents[]`、`edges[]`、`skills[]`（例如 verifier 上挂 `causal_analysis`，`SkillResult.route` 指向 executor 或 planner）。**中枢是 preset，不是 Runtime 死循环。**

**Runtime（动态）**

- 默认 adapter：`TirAgent` LangGraph 图。
- 职责：按 spec 装配 LLM/Tool、执行、在屏障处 `Archive.snapshot`、把 LangGraph 状态 **翻译** 成 `MasEvent`。
- 明确不负责：算 reward、算 advantage、启 VERL、画曲线。

**LangGraph 结论（回应 design.md「是否建议 LangGraph」）**

- **建议作为 Phase 0 默认 Runtime adapter。** 理由：TirAgent 已闭环；AGL 的 LangChain callback handler 已接 tracer；不必再写一套调度器才能训。
- **不建议作为跨层 schema。** `AgentState` TypedDict 留在 `tir_agent.py`。
- 不必上 AutoGen/AgentScope 来「更像 MAS」——那是换 adapter，不是换合同。
- 不必做 MAS_INF 式 NativeRuntime 产品。reference 调度器的 ROI 低于把现有图的事件写全。

**五插件槽**

| 槽 | Phase 0 | 后续 |
|----|---------|------|
| AgentRole | 单一 `hub` | planner / executor / verifier，带 level |
| Skill | ReAct 循环写死在图里 | 可注册；verifier 因果分析经 `route` |
| Tool | `tools/` 三件套 | Provider 化；禁止工具内第二套不可训 LLM（tir DESIGN.md 已踩过坑） |
| LLMProvider | `init_chat_model` + AGL `LLM` | Framework 绑定三路 |
| MemoryStore | messages | agent-local + system；读写都要落 event |
| Archive | `workflow/archive.py` | 窗口策略可配；fork 复制 `seq<=barrier` 的事件 |

Archive 语义：**可分支续跑**（ARPO resume、Harness 回退），不是 bit-identical replay。LLM/检索再次执行可以不同。

### 4.4 MAS 数据层

- `Collector.collect(..., n=)` = GRPO 组采样的 MAS 侧接口；daemon 的 `rollout.n` 是同一概念在训练时的实现。
- Reward 层次（已实现，应成为唯一实现）：格式坏 -> -1；格式好 Acc=0 -> 0；Acc>0 -> Acc（GSM8K 0/1，QA 词级 F1）；search 与 python 都用过 -> +0.1。
- `TrainSignal` 只携带 **算法名与超参**，不携带 token tensor。
- 交给 Harness 的就是 `Trajectory` + `ArchiveRef`，不要第二套 event 模型。

### 4.5 RL 层（优先级最高，但是 adapter）

用户可改的三处继续是：

1. `RolloutHook`：对应 daemon 如何 enqueue / 是否从 `BranchPoint` fork。
2. `AdvantageFn`：对应 `algos/advantage.py`。
3. `LossFn`：Phase 0 继续委托 VERL clip；`LossSpec` 只覆盖 `clip_ratio_*` / `entropy_coeff` / `kl_loss_coef`。

训练数据流目标态：

```text
Runtime -> MasEvent[] -> project Trajectory
       ->  live: LitAgent 调同一 RewardFn -> emit_reward/annotation
Collector（离线）-> TrajectoryBatch -> TrainSignal  （评测 / 无 GPU 实验）
live spans -> Triplet / DataProto                 （AGL 内部，不进 science schema）
```

**两条路径必须共用 Trajectory/annotation 字段。** 禁止 Collector 一套、LitTirAgent 一套。

### 4.6 Harness（插件，即插即用）

注册表名字先占位，未实现返回空：

| 插件 id | 输入 | 输出 | 实现优先级 |
|---------|------|------|------------|
| `log_error` | events 中 `ERROR` | Hypothesis + event_id | P4 骨架 |
| `epc_aw_consensus` | planner 后候选 | 多数决/加权；k 与阈值进配置 | 后做 |
| `cognitive_convergence` | 同类错误轨迹 | 对齐 vs 反复失败 | 后做 |
| `reward_hacking` | Trajectory + 日志 + **外部** Judge LLM | 是否虚高 reward | 后做；Judge 不得是当前 policy |
| `loss_volatility` | RLTrainLog | 波动/不下降归因 | 后做；曲线仍走 TB/W&B |

可视化：跳转 TensorBoard / W&B / SwanLab。AGL 已有 `ensure_metrics_env` 与 `TENSORBOARD_DIR`。不自研 loss SPA。

### 4.7 UI（第四优先）

本轮只定接口：

- 把 `MASSpec` 路径、LLM 三路、`--algo`、`n`、温度传给 MAS/RL。
- `uv` + 本地 GPU 探测是工程脚本，不是 Runtime。
- 「拖拽搭 MAS」依赖 Framework YAML 先存在。
- Dashboard = 打开已有 TB/W&B URL。

[`pyproject.toml`](../pyproject.toml) 里的 `science-infra = science_infra.ui.cli:main` 在包复活前应视为无效，不要再文档里当可运行命令。

### 4.8 包树建议

**不要立刻复活空的 `science_infra/`。** 根目录 pyproject 已经声明该包，但磁盘上没有模块，会再次撒谎。

推荐分两步：

1. **现在（P0–P2）**：契约与 Collector/Archive/Reward **继续长在** `examples/tir_agent/workflow/`。训练代码继续在 `algos/`、`lit_tir_agent.py`。用 `check_workflow_deps.py` 守住「workflow 不 import AGL」。
2. **稳定后（P3+）**：把 `workflow/` + `algos/rewards.py` 的纯函数抽到 `Agent_Science_Infra/science_infra/{schema,runtime,data,rl,harness,cli}`，tir_agent 变成「LangGraph adapter + 训练入口」示例。抽包标准：合同 JSON round-trip 稳定、collect 与 train 共用 RewardFn、TrainSignal 已被 overlay 消费。

```text
# 抽包后的目标（不是现在就建空目录）
science_infra/
  schema/          MasEvent, Trajectory, TrainSignal, MASSpec, BranchPoint
  runtime/         ExecutionService Protocol + langgraph adapter
  data/            Collector, RewardFn, project_trajectory
  rl/              overlay: TrainSignal -> Hydra；不 import DataProto
  harness/         Diagnoser registry
  ui/              CLI 传参
examples/tir_agent/   TirAgent 图、algos 挂钩、LitTirAgent、MAS_structagent 过渡
agentlightning/       只读依赖，不改核心
```

依赖方向：`schema !-> runtime !-> rl`；`harness` 只依赖 `schema` + `ExecutionService` Protocol。

---

## 5. 实现对照矩阵

图例：[done] 已实现且可用 | [partial] 部分 / 错位 | [missing] 缺失

### 5.1 总表

| design.md 能力 | 状态 | 代码位置 | 无 RL 可独立？ |
|----------------|------|----------|----------------|
| 单 hub ReAct + search/python | [done] | `tir_agent.py`, `tools/` | 是（需 API 或 `--mock`） |
| Framework 静态图 | [missing] | 无 `MASSpec` / YAML | — |
| Runtime 抽象 / ExecutionService | [missing] | 图逻辑焊在 `TirAgent` 类上 | — |
| Event Normalizer | [partial] | Collector 只写 `TASK_START`/`SNAPSHOT`/`ERROR`；训练路径基本不写 event | collect 是 / train 否 |
| Archive snapshot + 回退 | [partial] | `workflow/archive.py`；与 `.resume_cache` 双写 | 是 |
| 双层 Memory | [missing] | TIR 仅 messages；MAS_structagent 有内存对象但无契约 | — |
| 角色 / 技能插件（含 verifier->planner/executor 路由） | [missing] | PEV 在平行栈里写死 | — |
| Collector n 采样 | [done] | `Collector.collect` | 是（`--mock`） |
| Reward 数据层单点 | [partial] | 函数在 `algos/rewards.py`；Agent 与 Collector 各调一次 | collect 是 |
| TrainSignal -> RL | [partial] | 有模型；trainer 不消费 | CLI 是，训练否 |
| LLM 三路 | [partial] | 训练第三路已通；Framework 未描述 | 冒烟是 |
| GRPO 训练 | [done] | `train_tir_agent.py --algo grpo` | 否 |
| ARPO/AEPO/IGPO/GiGPO | [done]（有文档化差距） | `algos/` | 否 |
| LitAgent 与 Collector 单路径 | [missing] | 两套 invoke | — |
| MAS_structagent 接同一契约 | [missing] | 独立入口 | 冒烟是 |
| Harness 插件 | [missing] | 无 | — |
| UI / `science-infra` CLI | [missing] | pyproject 虚声明 | — |
| `science_infra` 包 | [missing] | 不存在 | — |

### 5.2 已基本具备（保持，少动）

**单 hub ReAct + 三工具**  
[`tir_agent.py`](../agent-lightning/examples/tir_agent/tir_agent.py) 图：`agent <-> tools`，无答案则 `finalize`，再失败则 `react`。工具：DuckDuckGo HTML、Wikipedia REST、math_gsm 风格 Python 沙箱。无网可用 `TIR_OFFLINE_SEARCH=1`。  
**不要改**：把工具内摘要 LLM 加回来（会污染 credit）。

**无 VERL mock collect**  
[`scripts/collect_rollouts.py`](../agent-lightning/examples/tir_agent/scripts/collect_rollouts.py) + [`collector.py`](../agent-lightning/examples/tir_agent/workflow/collector.py)。`--mock` 不加载 agentlightning（脚本会警告若被导入）。  
改完后这条必须仍为绿：

```bash
cd agent-lightning/examples/tir_agent
PYTHONPATH=. python scripts/check_workflow_deps.py
PYTHONPATH=. python scripts/collect_rollouts.py --mock --n 2 --out /tmp/traj.json
```

**Archive ↔ ARPO**  
[`archive.py`](../agent-lightning/examples/tir_agent/workflow/archive.py) 的 `dump_resume_with_archive` / `load_resume_messages`；daemon 二次 enqueue 写入 `resume_messages` 与可选 `resume_from`。语义已覆盖 design.md「RL 从历史状态再采样」。

**Outcome reward 与五算法**  
奖励与 advantage 的论文差距见 tir `DESIGN.md`。infra 不要把这些公式搬进 schema。

**AGL endpoint 注入**  
`LitTirAgent` 用 `llm.get_base_url(...)`；`LitMASAgent` 用 override + `OPENAI_*`。第三路 LLM 在训练中已成立。

### 5.3 部分具备 -> 必须改

每条格式：违反哪条原则 / 改哪里 / 改完验收。

#### G1. `design.md` Phase 0「已实现」表与真实代码不符

- **违反**：文档把目标写成现状（`science_infra.mas`、`science-infra collect`）。
- **改**：本文 + 修正 Phase 0 表格（见同目录提交）。不要在 design.md 里继续写不存在的包。
- **验收**：新同学按 design.md 找不到 `science-infra` 命令——修正后应指向 `scripts/collect_rollouts.py` 与本文。

#### G2. Collector 与 LitTirAgent 双路径重复

- **违反**：层内不模块化；同一 Runtime 逻辑复制，事件/reward 易分叉。
- **改**：
  1. 抽 `runtime/tir.py`（或 `TirAgent.run_episode(task, llm_cfg) -> EpisodeRaw`），只返回 messages / 计数 / 熵 / 可序列化 messages。
  2. `Collector` 与 `LitTirAgent.rollout` **只调这一函数**。
  3. Collector 负责：写 events、snapshot、RewardFn、组装 Trajectory。
  4. LitTirAgent 负责：从 AGL 填 llm_cfg、调同一 RewardFn、`emit_*`、按需 archive dump。
- **文件**：新建薄封装；删掉 `collector._tir_agent_rollout` 与 `LitTirAgent.rollout` 中重复的 `initial` state 构造。
- **验收**：`--mock` 仍绿；`python lit_tir_agent.py` 冒烟仍只需 `OPENAI_API_BASE`。

#### G3. `TrainSignal` 断头

- **违反**：数据层「提供接口传递给 RL」未闭合。
- **改**：
  1. `apply_algo_overlay(config, algo)` 改为也可接受 `TrainSignal`（或 `AdvantageSpec`+`LossSpec`）：`tir_algo`、`clip_ratio_*`、`entropy_coeff` 从 spec 写入 Hydra。
  2. 训练入口默认仍用 CLI `--algo` 构造 spec（不必先离线 collect）。
  3. 离线 JSON 的 `train_signal` 可用于复现实验超参，不用于替代 DataProto。
- **不要改**：让 VERL 直接吃 `Trajectory.messages` 当 batch（那是重造 adapter）。
- **验收**：`collect_rollouts.py --mock` 写出的 `advantage.name` 与 `train_tir_agent.py --algo` 使用同一字符串集合 `grpo|arpo|aepo|igpo|gigpo`。

#### G4. Reward 双计，canonical 不清晰

- **违反**：MAS_INF 对照原则「不要在 Agent.execute 里发明另一套 reward」；两处调用没有单测锁死一致性。
- **改**：`RewardFn` 只定义在数据层可 import 的模块。建议把 `compute_outcome_reward` 从 `algos/rewards.py` 挪到 `workflow/rewards.py`（或抽包后 `science_infra.data.rewards`），`algos/` 再 re-export 以免打断训练。`LitTirAgent` 只调用它然后 `emit_reward`。
- **验收**：改 format 惩罚时只改一处；Collector 与训练日志中同一 mock 轨迹 reward 相同。`--mock` 不 import `agentlightning`。

#### G5. 事件几乎只在 collect 时出现

- **违反**：档案库要支撑 Harness 失败归因；训练路径没有事件 SSOT。
- **改**：TirAgent 或 Normalizer 在 tool_call / tool_result / final_answer 各 append 一条 `MasEvent`。训练 dump archive 时写入同一 jsonl。不必把 OTel span 再译一遍当 SSOT——span 归 AGL；science 事件归 Archive。
- **今日**：`_mock_rollout` 写 `TASK_START`+`SNAPSHOT`；真实 `_tir_agent_rollout` 同样偏少，中间 tool 没有独立 event。
- **验收**：一次非 mock collect 的 `events.jsonl` 能看出「搜了什么、算了什么、最终答案」；Harness `log_error` 将来能用 `event_id` 定位。

#### G6. Archive 与 `.resume_cache` 双写

- **违反**：单一数据源。daemon 已能 `load_resume_messages` 走 Archive，但仍 fallback 到 `algos.arpo_rollout.load_resume`。
- **改**：P2 以 Archive 为唯一源；`.resume_cache` 仅兼容读一层后删除写路径。`BranchPoint` 成为 daemon enqueue 的正式字段（已部分写入 `resume_from`）。
- **验收**：清空 `.resume_cache` 后 ARPO 仍能从 `.tir_archives/**/rollout_index.json` 续跑。

#### G7. Trajectory 过窄且 TIR 字段焊死

- **违反**：可扩展接口；换 PEV 就要改 Pydantic。
- **改**：稳定面保留 `trajectory_id, task, events, messages, final_answer, final_reward, archive, branch_points, resume_from, meta`。`n_search`/`n_python`/`format_ok` 迁 `meta` 或 `extensions`。`ExecutionEvent.payload` 对已知 kind 改为可选 typed 子模型，未知进 `extensions`。
- **验收**：现有 JSON 能读；`extra=forbid` 仍在；MAS_structagent 将来不必伪造 `n_python`。

#### G8. `workflow/` 对 `tir_agent`/`algos` 的硬依赖

- **违反**：数据层应可独立；`check_workflow_deps.py` 只禁了 AGL，没禁反方向耦合。
- **事实**：`collector.py` import `algos.rewards`、`tir_agent`；`archive.py` import `algos.arpo_rollout.dump_resume`。
- **改**：mock 路径零 import 图；真实路径通过 `Runtime` Protocol 注入。`check_workflow_deps.py` 增加：`workflow/` 不得 import `lit_tir_agent`。Tir 图可作为 *optional* runtime 插件放在 `tir_agent.py`，由 Collector 在非 mock 时延迟 import（今日已有延迟 import，但 Reward 仍顶层绑 `algos`）。
- **验收**：把 `tir_agent.py` 改名仍能跑 `--mock`（mock 不该需要它）。

#### G9. MAS_structagent 平行栈

- **违反**：可扩充 subagent 应走同一插件槽，而不是第二套训练入口。
- **改（P3，不要现在搬）**：保留 `train_mas_agent.py` 能训；新增 `AgentRole` 接口文档，列出要把 `Planner`/`Executor`/`Diagnoser` 收成 skill 时必须实现的方法。禁止在 P0 把 Solver 塞进 Collector。
- **验收**：P0 文档明确「PEV 是后续 preset，不是当前默认拓扑」。

### 5.4 缺失 -> 如何加（先接口后实现）

#### M1. Framework / `MASSpec`

- **为什么要**：UI 拖拽、Harness 改路由、RL 写回 endpoint，都需要一份可序列化的静态图。
- **改**：`workflow/spec.py` + `specs/hub_react.yaml`。Runtime `TirAgent.from_spec(spec)`。
- **验收**：改 YAML 的 `tools` 列表即可关掉 wikipedia，无需改 Python 图拓扑代码（图仍可硬编码 ReAct，但工具集来自 spec）。

#### M2. 角色 / 技能插件与 verifier 路由

- **为什么要**：design.md 的「重点抽象 point」。
- **改**：`Skill.run(ctx) -> SkillResult`（含可选 `route: agent_id`）。Phase 0 的 ReAct 不必走 Skill 接口；只把 Protocol 和空 registry 放上。PEV 实现推迟。
- **验收**：`registry.get("causal_analysis")` 未注册时返回明确错误，而不是 import Solver。

#### M3. 双层 Memory

- **为什么要**：design.md 明确 agent memory + MAS memory。
- **改**：契约 `MemoryItem { scope: agent|system, owner }`；Phase 0 Runtime 把 LangGraph messages 标成 `scope=agent, owner=hub`。system memory 可空实现。读写若发生，落 `memory_read`/`memory_write` 事件（可后加 kind）。
- **验收**：Trajectory 能区分「对话缓冲」与「系统档案」；空实现不阻断 ReAct。

#### M4. Harness 注册表

- **为什么要**：层独立前提下的即插即用。
- **改**：`workflow/harness.py` 或日后 `science_infra/harness/`：`Diagnoser` Protocol + `registry`。P4 只实现 `log_error`（扫 `EventKind.ERROR`）。其余名字 `register` 后 `diagnose` 返回 `[]`。
- **验收**：Collector 输出可被 `log_error` 消费；模块不 import `TirAgent`。`--mock` 仍绿。

#### M5. UI / CLI

- **为什么要**：第四优先；现在 pyproject 入口是假的。
- **改**：P5 再做。在此之前 README 只宣传 `scripts/collect_rollouts.py`。若保留 `science-infra` 脚本名，必须先有模块。
- **验收**：文档中的命令都能复制执行。

#### M6. 数据层「计算 advantage」

- **不要做。** design.md 写了 advantage 设计，但 token-level advantage 属于 VERL。数据层只给 `AdvantageSpec`。把 advantage 算进 Trajectory 会迫使 schema 依赖 torch。

---

## 6. 必须改的耦合与错误（清单）

按修复成本从低到高：

1. **文档撒谎**：`science_infra.mas` 已实现、`science-infra collect --mock`、`examples/smoke_mas_only.py` —— 均不存在。
2. **命名冲突**：两套 `ExecutionEvent`。
3. **TrainSignal 无人消费**。
4. **Reward 两处调用无单点模块**（函数已共享文件，但层边界不清：数据层顶层 import `algos`）。
5. **Collector vs LitTirAgent 复制 graph 启动逻辑**。
6. **训练不写 MasEvent jsonl**。
7. **Archive + resume_cache 双写**。
8. **MAS_structagent 完全在契约外**。
9. **Harness/UI 为空，却在 Phase 0 表里占了一行「已实现」**。

---

## 7. 改造路线（只规划）

原则：每步结束后 `--mock` collect 与 `check_workflow_deps` 必须为绿；不改 `agentlightning/` 核心。

### P0 — 契约诚实 + 单路径 Runtime（infra 最小闭环）

- 修正 design.md / README（本文配套提交）。
- 抽 `run_episode`，Collector 与 LitTirAgent 共用。
- `RewardFn` 模块归属数据层；algos re-export。
- Event：至少 tool_call / tool_result / final_answer。
- 可选：`MasEvent` 改名计划写进 contracts 注释，实际改名可随 P0 或 P1。

验收：`collect_rollouts.py --mock`；`lit_tir_agent.py` 冒烟；训练路径行为不变。

### P1 — TrainSignal ↔ AGL overlay

- `apply_algo_overlay` 读 `AdvantageSpec`/`LossSpec`。
- CLI `--algo` 生成 spec，与 collect JSON 字段对齐。
- 仍然 **live span -> Triplet**，不把 Trajectory 塞进 PPO。

验收：`--algo igpo` 与一份手写 `TrainSignal(advantage.name=igpo)` 产生相同 Hydra `tir_algo`。

### P2 — Archive 成为唯一 resume

- daemon 只认 `BranchPoint` / Archive。
- 停止写 `.resume_cache`（读兼容可留一个版本）。
- LitTirAgent annotation 填真实 `archive_id`/`snapshot_id`（今日先 emit 空字符串再补写，易丢）。

验收：删 `.resume_cache` 后 `tir_algo=arpo` 仍能二次 enqueue。

### P3 — Framework YAML + 角色插件口（不搬完整 PEV）

- `hub_react.yaml` + `TirAgent.from_spec`。
- `Skill`/`AgentRole` Protocol + registry；默认只有 hub。
- MAS_structagent 保持独立入口，文档标明「过渡」。

验收：改 YAML 工具列表生效；registry 对未知 skill 失败清晰。

### P4 — Harness 骨架

- `log_error` 真实现；其余 stub。
- `diagnose` 输入为 Trajectory JSON，输出 Hypothesis 列表（含 `event_id`）。
- 不 import Runtime。

验收：对带 `ERROR` 的 mock 轨迹返回非空 hypothesis；无 ERROR 返回 `[]`。

### P5 — UI / 真 CLI

- 薄 CLI：collect / 打印 TrainSignal / 打印 dashboard 路径。
- 拖拽与 GPU uv 探测更后。

验收：文档命令与 pyproject script 一致。

### 明确不做

- 不改 `agentlightning/` 核心源码。
- 不自研 PPO / LightningStore / Trace SPA。
- 不把 MAS_INF NativeRuntime 当产品。
- 不把 `DataProto` 或 LangGraph State 放进 science schema。
- 不在 P0 合并 MAS_structagent 与 TIR 图。
- 不声称 Archive replay bit-identical。

---

## 8. 建议的短期文件改动图（供后续 PR，非本轮）

```text
workflow/
  spec.py            新增 MASSpec
  rewards.py         从 algos/rewards.py 迁纯函数
  runtime.py         ExecutionService Protocol + TirAdapter.run_episode
  harness.py         Diagnoser registry（P4）
  contracts.py       Trajectory 瘦身；事件 kind 补全
  collector.py       只依赖 runtime + rewards
  archive.py         去掉对 dump_resume 的硬依赖（反向：arpo 调 archive）
algos/
  rewards.py         from workflow.rewards import *
  overlay.py         接受 TrainSignal
  daemon.py          resume 只走 Archive
lit_tir_agent.py     调 runtime.run_episode + 同一 RewardFn
tir_agent.py         图实现；from_spec
specs/hub_react.yaml 新增
```

依赖方向修正：

```text
今日（错误）：
  workflow.collector -> tir_agent -> algos.rewards
  workflow.archive   -> algos.arpo_rollout

目标：
  tir_agent 图  -> 不依赖 workflow 契约（或只依赖 schema）
  workflow.runtime adapter -> tir_agent 图（延迟 import）
  workflow.rewards  ← algos.rewards 再导出
  algos.arpo_rollout / daemon -> workflow.archive
  lit_tir_agent -> workflow.runtime + workflow.rewards + agl
```

---

## 9. 附录：现有命令对照（以代码为准）

在 `agent-lightning/examples/tir_agent`：

```bash
# MAS 数据层，不占训练 GPU（仓库根目录）
pip install -e .
science-infra collect --mock --n 2 --out /tmp/traj.json
science-infra diagnose /tmp/traj.json
science-infra status /tmp/traj.json --html /tmp/status.html
science-infra dashboard /tmp/traj.json --html /tmp/science-dashboard.html
science-infra doctor

# 或在 tir_agent 目录
PYTHONPATH=. python scripts/check_workflow_deps.py
PYTHONPATH=. python scripts/collect_rollouts.py --mock --out /tmp/traj.json

# 单 Agent 冒烟（要 OpenAI-compat endpoint，不启 VERL）
OPENAI_API_BASE=http://127.0.0.1:8000/v1 OPENAI_API_KEY=dummy python tir_agent.py

# 训练（要 GPU + agentlightning）
python train_tir_agent.py fast --algo grpo
python train_tir_agent.py a800_2gpu --algo arpo

# 平行 MAS 栈（契约外）
python train_mas_agent.py fast
```

`python examples/smoke_mas_only.py` 仍不存在。

---

## 10. 附录：与 design.md 原文条目的逐条闭合

| 原文要点 | 最优设计结论 | 代码 |
|----------|--------------|------|
| 层可单独工作 | Collector `--mock` 独立；训练可选 | [partial] mock 行；训练未消费 TrainSignal |
| 接口严格格式化 | 每层 Boundary + 上表合同 | [partial] 有 Pydantic，未贯通 |
| 层内插件 | 六槽 + Diagnoser 注册表 | [missing] 焊死在类里 |
| UI 传参与可视化 | StudioParams + 外链 TB/W&B | [missing] |
| Framework + Runtime | YAML spec + LangGraph adapter | [missing] / [partial] 只有 Runtime 实现 |
| 中枢 agent | topology preset `hub_react` | [partial] 硬编码单图 |
| 可扩 subagent/skill/路由 | SkillResult.route | [missing]（PEV 在平行栈） |
| LLM 三路 | AGL `LLM` + spec.kind | [partial] 训练通，spec 无 |
| Tool：search + python | 已有 + wikipedia | [done] |
| 双层 memory | MemoryItem scope | [missing] |
| 档案库回退 | Archive + BranchPoint | [partial] 双写 cache |
| rollout 采样 | Collector.n 与 daemon n | [partial] 两套 |
| advantage = reward − critic | AdvantageSpec；算在 VERL | [done] 心智正确，spec 未接 |
| loss 接口 | LossSpec -> Hydra clip | [partial] 有模型无消费 |
| 基于 AGL/verl 改三钩子 | trainer_cls / daemon_cls | [done] |
| Harness 插件 | Diagnoser registry | [missing] |
| MAS harness：日志错误 / EPC-AW / 认知收敛 | 先留名 | [missing] |
| RL harness：hacking / 不上升 / loss 波动 | 先留名；Judge 非 policy | [missing] |

---

**文档状态**：分析与改造规格，2026-09-06。实现以后续工作包为准；以本文对照矩阵覆盖 [design.md](../design.md) 中已过时的「已实现」表。
