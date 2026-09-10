# MAS Research Infrastructure 技术设计报告

面向 Agent / MAS Research、Rollout、RL、Evaluation 与自动化诊断的一体化研究基础设施。

| 项 | 内容 |
|---|---|
| 版本 | v1.0 Architecture Proposal |
| 定位 | Research Infrastructure，而非通用 Agent Framework |
| 文档性质 | 架构提案与优先级判断，不是实现规格 |

研究人员真正关心的是：发生了什么、为什么失败、该改什么、有没有变好、能不能复现。他们不该先理解底层 Runtime、分布式训练框架、Trajectory 存储或监控系统。

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心设计原则](#2-核心设计原则)
3. [总体系统架构](#3-总体系统架构)
4. [Universal Data Plane](#4-universal-data-plane)
5. [MAS Layer](#5-mas-layer)
6. [RL Layer](#6-rl-layer)
7. [Harness Layer](#7-harness-layer)
8. [Experiment Layer](#8-experiment-layer)
9. [UI Layer](#9-ui-layer)
10. [工程落地](#10-工程落地)
11. [分阶段路线图](#11-分阶段路线图)
12. [优势、风险与战略判断](#12-优势风险与战略判断)

---

## 1. 项目概述

### 1.1 目标

构建面向 LLM-based Agent / Multi-Agent System（MAS）研究的统一基础设施，使研究人员通过统一接口完成完整研究闭环：

```text
MAS Definition
      ↓
Execution / Rollout
      ↓
Trajectory Collection
      ↓
Evaluation
      ↓
Failure Diagnosis
      ↓
RL / Optimization
      ↓
Checkpoint / Replay / Intervention
      ↓
再次 Rollout
```

最终形态：定义一次 Agent / MAS，即可无缝进行 Rollout、Evaluation、Harness Diagnosis、RL Training、Replay 和 Experiment Comparison。

### 1.2 成功标准

系统成功与否，不取决于「能不能启动一个 LangGraph」，而取决于基础设施能否稳定回答：

- What happened?
- Why failed?
- What should change?
- Did it improve?
- Can I reproduce it?

---

## 2. 核心设计原则

六条原则不是并列口号。P1–P2 决定层边界，P3–P4 决定生态位置，P5 决定数据原语，P6 把产品从运行时抬成研究平台。

### P1. Layer Independence

每一层必须可以独立运行。

- `MAS → Trajectory` 不依赖 RL。
- `Trajectory → Harness` 不依赖 MAS Runtime。

有 Trajectory，就可以单独做评价、诊断或训练。

### P2. Contract-First

层与层之间不能直接依赖具体框架的数据结构。

禁止：

```text
LangGraphState → VERL
```

应采用：

```text
LangGraphState
      ↓
Adapter
      ↓
Universal Trajectory
      ↓
RL Adapter
      ↓
VERL
```

### P3. Plugin-Oriented

每一层内部采用插件架构：

```text
Interface
   │
   ├── Default Implementation
   ├── Implementation A
   ├── Implementation B
   └── User Custom Plugin
```

研究人员可以替换 Runtime、LLM、Tool、Memory、Skill、Reward、Evaluation、Harness、RL Algorithm、Trainer，而不修改核心系统。

### P4. Framework Agnostic

系统不应该成为「另一个 LangGraph」。

- LangGraph、AgentScope、AutoGen 等是 **Runtime Backend**。
- VERL、Agent-Lightning、TRL 是 **Training Backend**。

核心系统不绑定具体第三方框架。

### P5. Event-Centric

Agent execution 的最小数据原语是 `Event`，而不是某个具体框架的 State。典型事件包括：

`AgentAction`、`ToolCall`、`ToolObservation`、`Message`、`LLMRequest`、`LLMResponse`、`Reward`、`Error`、`Checkpoint`、`Feedback`、`Termination`

所有 Runtime 最终都映射到统一 Event。

### P6. Reproducibility

所有实验必须支持 Experiment、Run、Episode、Checkpoint、Seed、Configuration、Artifact，并保证：同一个 Experiment Configuration + Seed 可以尽可能重建对应实验。

---

## 3. 总体系统架构

### 3.1 六层 + 贯穿式 Data Plane

```text
┌───────────────────────────────────────────────────────────┐
│                    UI / CLI Layer                         │
│       Configuration / Visualization / Control             │
└─────────────────────────┬─────────────────────────────────┘
                          │
                          ▼
┌───────────────────────────────────────────────────────────┐
│                 Experiment / Control Layer                │
│ Experiment / Run / Trial / Seed / Artifact / Lifecycle    │
└─────────────────────────┬─────────────────────────────────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
┌────────────────┐ ┌──────────────┐ ┌────────────────┐
│   MAS Layer    │ │   RL Layer   │ │ Harness Layer  │
│ Definition     │ │ Optimization │ │ Evaluation     │
│ Runtime        │ │ Training     │ │ Diagnosis      │
│ Agent / Tool   │ │              │ │ Intervention   │
│ Memory / Skill │ │              │ │ Visualization  │
└───────┬────────┘ └──────┬───────┘ └───────┬────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────┐
│                 Universal Data Plane                      │
│ Event → Episode → Trajectory → Reward → TrainingSignal    │
│ Checkpoint / Snapshot / Fork / Replay / Artifact          │
└───────────────────────────────────────────────────────────┘
```

| 层 | 职责 |
|---|---|
| UI / CLI | 人机交互：配置、可视化、控制 |
| Experiment | 管理实验生命周期 |
| MAS | 定义并执行 Agent System |
| RL | 训练和优化 Agent |
| Harness | 自动评价、诊断和干预 |
| Data Plane | 连接所有模块的数据基础设施 |

UI 不参与核心计算。MAS、RL、Harness 是三条平行能力，互不直接调用，全部通过 Data Plane 交换对象。

### 3.2 为什么需要 Data Plane

整个项目最核心的设计不是 UI，也不是 RL，而是 **Universal Agent Data Model**。

LangGraph、AutoGen、AgentScope、Agent-Lightning、VERL、OpenAI Agents SDK 的数据结构完全不同。若直接互相通信，最终会产生 N × M 个 Adapter。

```text
Framework A ─┐
Framework B ─┤
Framework C ─┤
Framework D ─┤
              ▼
       Universal Data Model
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
      RL   Harness  Eval
```

复杂度从 O(NM) 降为 O(N+M)。这是后面所有解耦能成立的前提。

### 3.3 Control Plane 与 Data Plane

推荐采用 **控制面 + 数据面分离**。

```text
             Control Plane
                  │
      Config / Command / Lifecycle
                  │
                  ▼
             Components
             Data Plane
                  │
                  ▼
       Event / Trajectory / Signal
```

**Control Plane** 负责 Create / Start / Stop / Pause / Resume / Fork / Configure：

```text
UI
 ↓
Experiment Controller
 ↓
Runtime
```

**Data Plane** 负责 Event、Trajectory、Reward、TrainingSignal、Diagnostic、Artifact：

```text
MAS
 ↓
Trajectory Store
 ├── RL
 ├── Harness
 └── Evaluation
```

若不分平面，很容易出现 Harness 直接调用 Runtime、修改 Agent，最终变成 `A → B → C → A → D → B` 的循环依赖。

正确结构：

```text
                 Control Plane
                      ↓
             Experiment Controller
                      ↓
       ┌──────────────┼──────────────┐
       ▼              ▼              ▼
      MAS             RL          Harness
       │              │              │
       └──────────────┼──────────────┘
                      ▼
                 Data Plane
```

Harness 只输出 Diagnostic / Intervention；Experiment Controller 决定是否 pause / fork / replay。

### 3.4 最终推荐形态

```text
                 ┌─────────────────────┐
                 │      UI / CLI       │
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  Experiment Plane   │
                 │ Experiment / Run    │
                 │ Trial / Seed        │
                 │ Artifact / Result   │
                 └──────────┬──────────┘
                            │
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐
       │   MAS    │   │    RL    │   │ Harness  │
       │Definition│   │Advantage │   │Evaluate  │
       │Runtime   │   │Objective │   │Diagnose  │
       │Agent     │   │Trainer   │   │Intervene │
       │Skill     │   │Endpoint  │   │          │
       │Tool      │   │          │   │          │
       │Memory    │   │          │   │          │
       └────┬─────┘   └────┬─────┘   └────┬─────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                ┌───────────────────────┐
                │  Universal Data Plane │
                │ Event / Episode       │
                │ Trajectory / Reward   │
                │ TrainingSignal        │
                │ Diagnostic            │
                │ Intervention          │
                │ Checkpoint            │
                └───────────┬───────────┘
                            │
                  ┌─────────▼─────────┐
                  │ Replay / Fork     │
                  │ Experiment Loop   │
                  └───────────────────┘
```

---

## 4. Universal Data Plane

数据对象层层推导：越往上越语义化，越往下越可训练、可诊断。MAS 不直接产 PPO Loss，Harness 不直接改 Agent。

### 4.1 Event

Event 是最底层原语。

```text
Event {
    event_id
    episode_id
    timestamp
    event_type
    agent_id
    input
    output
    parent_event_id
    metadata
}
```

事件类型：

| 类别 | 类型 |
|---|---|
| 模型 | `LLMRequest`、`LLMResponse` |
| 决策 | `AgentAction`、`AgentMessage` |
| 工具 | `ToolCall`、`ToolObservation` |
| 结果 | `Reward`、`Error`、`Termination` |
| 控制 | `Checkpoint`、`Feedback` |

### 4.2 Episode

Episode 表示一次 Agent 任务执行。

```text
Episode {
    experiment_id
    run_id
    episode_id
    task
    events[]
    outcome
    reward
    checkpoints[]
    metadata
}
```

```text
Episode
 │
 ├── Event 1: AgentAction
 ├── Event 2: LLMResponse
 ├── Event 3: ToolCall
 ├── Event 4: ToolObservation
 ├── Event 5: AgentAction
 ├── Event 6: Reward
 └── Event 7: Termination
```

### 4.3 Trajectory

Trajectory 是提供给 Evaluation / RL / Harness 的标准数据对象。

```text
Trajectory {
    trajectory_id
    task
    states[]
    events[]
    actions[]
    observations[]
    reward
    outcome
    checkpoints[]
    metadata
}
```

**核心原则：Trajectory 不包含任何 LangGraph / VERL 特有对象。**

Schema 必须遵守：**Core Schema 最小化，差异全部进入 `metadata`**。若 Trajectory 膨胀成几十个 Optional 字段，所有框架用起来都会不舒服。

### 4.4 TrainingSignal

MAS 不直接生成 GRPO / PPO Loss。MAS 只负责 `Trajectory → Reward`；RL 再计算 `Reward → Advantage → Objective → Loss`。

```text
TrainingSignal {
    trajectory_id
    reward
    advantage: Optional
    return_: Optional
    log_prob: Optional
    reference_log_prob: Optional
    token_mask: Optional
    metadata
}
```

由此可以支持 REINFORCE、PPO、GRPO、RLOO、DAPO 与自定义 RL。

### 4.5 Diagnostic

Harness 输出诊断合同，而不是直接改系统：

```text
Diagnostic {
    diagnostic_id
    target_episode
    target_event
    failure_type
    severity
    evidence
    confidence
    recommendation
}
```

示例：

```json
{
  "failure_type": "tool_selection_error",
  "target_event": "event_27",
  "severity": "high",
  "confidence": 0.91,
  "recommendation": {
    "action": "fork",
    "checkpoint": "checkpoint_25"
  }
}
```

### 4.6 Intervention

Harness 不应该直接修改 MAS，而应输出：

```text
Intervention {
    type
    target
    checkpoint
    modification
    rationale
}
```

执行路径：

```text
Harness
 ↓
Diagnosis
 ↓
Intervention
 ↓
Experiment Controller
 ↓
MAS Runtime
```

这样可以避免 Harness 与 MAS 强耦合。这是层独立性能否保住的关键闸门。

### 4.7 必须冻结的六个 Contract

| Contract | 粒度 | 职责 | 明确不包含 |
|---|---|---|---|
| `WorkflowSpec` | 静态定义 | 描述 Agent / Tool / Memory / Topology / Runtime | 具体框架的 Graph 对象 |
| `ExecutionEvent` | 一次原子发生 | 统一 Runtime 输出 | LangGraph State、图节点对象 |
| `Trajectory` | 标准研究数据 | 给 Eval / RL / Harness 的框架无关对象 | 任何 LangGraph / VERL 特有类型 |
| `TrainingSignal` | 训练输入 | reward / advantage / log_prob / mask | MAS 内部的 Agent / Tool / Memory |
| `Diagnostic` | 失败解释 | failure_type、evidence、confidence、recommendation | 对 Runtime 的直接副作用 |
| `Intervention` | 建议动作 | type / target / checkpoint / modification | Harness 自己去改 MAS |

这六个 Contract 一旦设计正确，后面的 LangGraph、AgentScope、Agent-Lightning、VERL、TRL、各种 Agent、Skill、Tool、Harness 都只是插件。反过来，哪怕 UI、MAS、RL、Harness 全部先做出来，这六个合同不稳定，后期依然很可能重构。

---

## 5. MAS Layer

MAS Layer 分成 Definition Plane、Runtime Plane 和 Data Collection。

### 5.1 Definition Plane

MAS Definition 描述 Agent、Model、Tool、Memory、Skill、Topology、Runtime、Checkpoint。建议 `WorkflowSpec` 使用 YAML / JSON。

```yaml
workflow:
  controller:
    type: planner
    model:
      type: local
      endpoint: ...
    skills:
      - decomposition
      - planning
  agents:
    executor:
      role: executor
      skills:
        - tool_selection
      memory:
        scope: private
    verifier:
      role: verifier
      skills:
        - factual_verification
        - causal_analysis
  tools:
    - web_search
    - python
  memory:
    system: shared
  runtime:
    backend: langgraph
```

### 5.2 Agent

不要把 Agent 类型写死。

```text
Agent {
    id
    role
    model
    skills
    memory
    policy
    metadata
}
```

`role` 只是语义信息，例如 planner、executor、verifier、researcher、critic、router，都可以存在。

### 5.3 Controller

「核心 Agent 中枢」应实现成 **Controller**，而不是直接叫 Planner。这样才能在未来替换为 Planner、Supervisor、Router、Manager、Policy、World Model。

```text
Controller.decide(context) -> Decision
```

### 5.4 Skill

Skill 是 MAS 最重要的可扩展点之一。

```text
Skill {
    name
    execute(context) -> SkillResult
}
```

例如：`PlanningSkill`、`ToolSelectionSkill`、`CausalAnalysisSkill`、`VerificationSkill`、`ReasoningSkill`。

一个 Agent 可以拥有多个 Skill，从而避免把 Agent 类型和能力绑定：

```text
Verifier
 ├── factual_verification
 ├── consistency_check
 └── causal_analysis
```

### 5.5 Model

LLM 统一抽象：

```text
Model.generate(messages, parameters) -> ModelResponse
```

实现包括 `ThirdPartyModel`、`LocalModel`、`RLModelEndpoint`、`MockModel`。其中 RL Model Endpoint 使训练中的模型可以重新进入 MAS：

```text
RL Trainer
     ↓
Inference Server
     ↓
MAS Runtime
```

### 5.6 Tool

```text
Tool.execute(arguments) -> ToolResult
```

- 第一版：WebSearch、Wikipedia、Python
- 未来：Database、Browser、CodeExecution、FileSystem、API、MCP

### 5.7 Memory

Memory 不与 Archive 混合。Memory 存知识，Archive 存执行状态。

```text
Memory
│
├── Private Memory
│
└── Shared MAS Memory
```

更长期可扩展为 Working Memory、Episodic Memory、Semantic Memory、Shared Memory。统一接口：

```text
MemoryStore.read()
MemoryStore.write()
MemoryStore.update()
MemoryStore.snapshot()
MemoryStore.restore()
```

### 5.8 Archive / Checkpoint

Archive 专门负责 Execution State，而不是 Knowledge。

```text
Checkpoint 0
    ↓
Checkpoint 1
    ↓
Checkpoint 2
    ↓
Checkpoint 3
```

提供 `checkpoint()`、`restore()`、`fork()`。其中 `fork()` 非常重要。

### 5.9 Forkable Execution

系统允许从同一 Checkpoint 分出原路径和干预路径：

```text
Episode A
      │
      └── Checkpoint 10
              │
       ┌──────┴──────┐
       ▼             ▼
Episode B        Episode C
Original         Intervention
```

这可以直接支持：

- Counterfactual analysis
- Failure attribution
- Harness intervention
- RL replay
- Planning search
- Agent debugging

因此应将其作为 **核心 Runtime Primitive**，而不仅仅是辅助功能。

### 5.10 Runtime

Runtime 负责把静态 Workflow 转化成动态执行。

```text
Runtime.run(workflow, task) -> Episode
Runtime.pause()
Runtime.resume()
Runtime.checkpoint()
Runtime.restore()
Runtime.fork()
```

### 5.11 Runtime Adapter

```text
ExecutionService
        │
        ├── LangGraphAdapter
        ├── AgentScopeAdapter
        ├── AutoGenAdapter
        └── NativeRuntime
```

Phase 0 使用 LangGraph 是合理的，但必须保持：

```text
WorkflowSpec
      ↓
ExecutionService
      ↓
LangGraph
```

而不是：

```text
WorkflowSpec = LangGraph Graph
```

不要承诺所有 Runtime 100% feature parity。应区分 Core Capability 与 Optional Capability：

```text
RuntimeCapabilities {
    checkpoint=True
    fork=True
    streaming=True
    human_interrupt=False
}
```

LangGraph 的 Graph State、Checkpoint、Interrupt 与其他 Runtime 的执行语义可能不同。

### 5.12 Data Collection

Runtime 产生 Event；Collector 负责聚合成研究数据：

```text
Event
 ↓
Episode
 ↓
Trajectory
 ├── Evaluation
 ├── Harness
 └── RL
```

---

## 6. RL Layer

### 6.1 边界

RL Layer 不应该感知 LangGraph、Agent、Tool、Memory。它只应该看到 Trajectory、TrainingSignal、ModelEndpoint。

```text
Trajectory
    ↓
Reward
    ↓
Advantage
    ↓
Objective
    ↓
Loss
    ↓
Optimizer
    ↓
Model Update
```

### 6.2 内部抽象

```text
RL Engine
│
├── Reward Adapter
├── Advantage Estimator
├── Objective
├── Optimizer
├── Trainer
└── Inference Endpoint
```

| 模块 | 可替换实现 |
|---|---|
| Advantage | GAE、GRPO、RLOO、Custom |
| Objective | PPO、GRPO、DAPO、REINFORCE、Custom |
| Trainer | VERL、Agent-Lightning、TRL、Native |

### 6.3 Agent-Lightning / VERL 的位置

不建议把它们放到核心 API。它们是 Training Backend：

```text
                 RL Interface
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
          VERL       AGL         TRL
```

| 系统负责 | 底层训练框架负责 |
|---|---|
| Trajectory | GPU |
| TrainingSignal | Distributed Training |
| Experiment | FSDP |
| Model Endpoint | Optimizer / Checkpoint |

这样可以最大限度复用成熟基础设施。

RL Adapter 必须独立于 MAS Data Contract。Agent RL 本身就存在 Rollout、Reward、Advantage、Token Alignment、Log Probability、Policy、Reference Model；不同 Trainer 对数据要求不同。

---

## 7. Harness Layer

Harness 是 Agent Research 的自动化分析与干预系统，不是日志查看器。

### 7.1 结构

```text
Harness
│
├── Observation
├── Evaluation
├── Diagnosis
├── Intervention
└── Visualization
```

插件消费路径：

```text
Harness Plugin
       ↓
Trajectory
       ↓
Diagnostic
       ↓
Intervention
```

### 7.2 演进

| 阶段 | 插件 |
|---|---|
| 第一阶段 | `LogErrorDetector`、`RewardAnalyzer`、`TrajectoryAnalyzer` |
| 后续 | `EPC-AW`、`CognitiveConvergence`、`FailureAttribution`、`LLMJudge`、`CausalAnalysis` |

### 7.3 Cognitive Convergence

可抽象为：

```text
Trajectory Cluster
       ↓
Error Signature
       ↓
Repeated Failure
       ↓
Cognitive Deviation
```

- 同类错误连续出现（Episode 1–4 均失败）→ High Deviation
- 失败后收敛到成功（Episode 1–2 失败，3–4 成功）→ Convergence

以后可以逐渐从 heuristic 升级为真正的 semantic / epistemic analysis。

### 7.4 Reward Hacking

第一版判定：

```text
reward ≥ threshold
AND
task output invalid
→ RewardHackingDiagnostic
```

后续可以升级为：

```text
Trajectory
   ↓
External Judge
   ↓
Ground Truth / Log
   ↓
Reward Validity
```

这会成为很有价值的 RL Research Harness。

---

## 8. Experiment Layer

这是建议新增的核心层。它使系统从 Agent Framework 升级成 Agent Research Platform。

### 8.1 对象模型

```text
Experiment
│
├── Run
├── Trial
├── Episode
├── Seed
├── Checkpoint
├── Artifact
└── Result
```

示例配置：

```yaml
experiment:
  name: mas_grpo
  workflow: planner_executor.yaml
  model: qwen
  rl:
    algorithm: grpo
  harness:
    - reward_hacking
    - cognitive_convergence
  seeds:
    - 1
    - 2
    - 3
```

### 8.2 生命周期

```text
CREATE
  ↓
CONFIGURE
  ↓
BUILD
  ↓
ROLLOUT
  ↓
EVALUATE
  ↓
DIAGNOSE
  ↓
TRAIN
  ↓
CHECKPOINT
  ↓
REPLAY / FORK
  ↓
COMPARE
```

---

## 9. UI Layer

UI 不参与核心计算，主要承担 Configuration、Visualization、Experiment Control。

### 9.1 第一阶段

不要马上做拖拽。先实现配置、控制和可视化。

| 类别 | 内容 |
|---|---|
| Configuration | Model、Workflow、Tools、Memory、RL、Harness |
| Control | Run、Stop、Pause、Resume、Fork |
| Visualization | Reward、Loss、Trajectory、Token、Latency、Failure、GPU |

### 9.2 第二阶段：Graph Editor

前端 Graph 由 Node、Edge、Property 组成，最终生成 **WorkflowSpec**，而不是直接生成 LangGraph 代码。

```text
              ┌──────────┐
              │ Planner  │
              └────┬─────┘
                   │
            ┌──────┴──────┐
            ▼             ▼
       ┌────────┐    ┌─────────┐
       │Executor│    │Verifier │
       └────────┘    └────┬────┘
                           │
                        feedback
                           │
                           ▼
                        Planner
```

Drag & Drop 看起来简单，实际涉及 Graph Editor、Schema Validation、Node Registry、Edge Semantics、Versioning、Serialization、Runtime Compilation、Error Handling。因此应该最后做。

### 9.3 与后端通信

```text
UI
 │
 ├── REST API
 ├── WebSocket / SSE
 └── Config API
       │
       ▼
Experiment Controller
```

REST 管实验生命周期：

```text
POST /experiments
POST /runs
POST /runs/{id}/pause
POST /runs/{id}/fork
GET  /runs/{id}
GET  /trajectories/{id}
```

实时数据走 WebSocket / SSE：

```json
{
  "type": "reward_update",
  "run_id": "run_01",
  "step": 100,
  "reward": 0.82
}
```

---

## 10. 工程落地

### 10.1 通信技术

第一阶段不需要微服务化。这是典型的过度工程。

| 规模 | 建议 |
|---|---|
| Local / Single Machine | Python API |
| Multi-process | gRPC 或 Message Queue |
| 大规模分布式（未来） | Kafka / NATS、Object Storage、Redis |

第一版不要直接上 Kafka + Kubernetes + 微服务。

### 10.2 数据存储

分三类存储，不要混用。

| 类别 | 介质 | 内容 |
|---|---|---|
| Metadata | SQLite / PostgreSQL | Experiment、Run、Trial、Configuration、Artifact metadata |
| Trajectory | Parquet / JSONL / Object Storage | 轨迹与事件 |
| Checkpoint | Object Storage（S3 / MinIO） | 执行快照 |

### 10.3 推荐目录结构

```text
science-infra/
├── core/
│   ├── contracts/
│   │   ├── workflow.py
│   │   ├── event.py
│   │   ├── trajectory.py
│   │   ├── training_signal.py
│   │   ├── diagnostic.py
│   │   └── intervention.py
│   ├── experiment/
│   └── registry/
├── mas/
│   ├── definition/
│   ├── runtime/
│   │   ├── langgraph/
│   │   ├── agentscope/
│   │   └── native/
│   ├── agent/
│   ├── skill/
│   ├── tool/
│   ├── memory/
│   └── checkpoint/
├── data/
│   ├── collector/
│   ├── trajectory/
│   ├── storage/
│   └── replay/
├── rl/
│   ├── advantage/
│   ├── objective/
│   ├── trainer/
│   │   ├── verl/
│   │   ├── agent_lightning/
│   │   └── trl/
│   └── endpoint/
├── harness/
│   ├── evaluator/
│   ├── diagnostics/
│   ├── interventions/
│   └── plugins/
├── experiment/
│   ├── scheduler/
│   ├── runner/
│   └── registry/
├── ui/
│   ├── api/
│   ├── dashboard/
│   └── graph_editor/
└── plugins/
```

### 10.4 一次完整实验

研究人员配置：

```yaml
workflow:
  controller: planner
  agents:
    - executor
    - verifier
rl:
  algorithm: grpo
harness:
  - reward_hacking
  - cognitive_convergence
```

系统执行：

```text
① Experiment Controller
          ↓
② WorkflowSpec
          ↓
③ Runtime
          ↓
④ Agent Execution
          ↓
⑤ Events
          ↓
⑥ Trajectory
       ↙     ↘
Harness       RL
   ↓           ↓
Diagnostic   Training
   ↓           ↓
Intervention Model Update
       ↘     ↙
         MAS
```

关键边界：

1. Experiment Controller 只发命令，不持有框架对象。
2. `WorkflowSpec ≠ LangGraph Graph`。
3. Runtime 产出只能是 Event。
4. Collector 去掉框架特有对象后再生成 Trajectory。
5. Harness 异步消费 Trajectory，输出 Diagnostic，不改 MAS。
6. RL 异步消费 Trajectory，输出 TrainingSignal / Model Update。
7. Controller 接受 Intervention，从 Checkpoint fork 再跑。

### 10.5 异步解耦

不要要求：

```text
Rollout → 等待 Harness → 等待 RL → 下一次 Rollout
```

默认应支持三者异步消费 Trajectory：

```text
Rollout
  │
  ├────→ Harness
  │
  ├────→ Evaluation
  │
  └────→ RL
```

RL training 不得阻塞 Harness analysis。

### 10.6 插件系统

所有模块统一注册：

```python
Registry.register("grpo", GRPOAlgorithm)
```

注册表包括：`ModelRegistry`、`ToolRegistry`、`SkillRegistry`、`MemoryRegistry`、`RuntimeRegistry`、`RewardRegistry`、`HarnessRegistry`、`RLRegistry`。

用户配置后由系统动态加载：

```yaml
runtime:
  backend: langgraph
rl:
  algorithm: grpo
harness:
  plugins:
    - reward_hacking
    - cognitive_convergence
```

---

## 11. 分阶段路线图

阶段顺序是：**先契约，后拓扑，再算法，最后 UI**。不要先做多 Agent，也不要先做拖拽。

### 11.1 Phase 0：当前 MVP

当前 Phase 0 选择是正确的：

- single Agent
- hub ReAct
- LangGraph
- Agent-Lightning
- VERL
- Harness
- CLI
- Dashboard

尤其值得保留：`workflow/` 不 import `agentlightning`。这是正确的架构边界。

### 11.2 Phase 1：冻结研究原语

- Universal Event
- Universal Trajectory
- Experiment
- Checkpoint
- Fork

不要急着多 Agent。

### 11.3 Phase 2：加入 MAS

加入 Planner、Executor、Verifier、Router，同时实现 Skill、Role、Topology、Private Memory、Shared Memory。

### 11.4 Phase 3：完善 RL

完善 GRPO、PPO、RLOO、DAPO，并接入 VERL、Agent-Lightning、TRL。

### 11.5 Phase 4：研究型 Harness

Failure Attribution、Cognitive Convergence、Reward Hacking、Epistemic Evaluation、LLM Judge。

### 11.6 Phase 5：UI

Workflow Builder、Drag & Drop、Experiment Manager、Trajectory Viewer、Harness Visualization、RL Dashboard。

---

## 12. 优势、风险与战略判断

### 12.1 优势

**高度解耦。** MAS 可以不需要 RL；RL 可以脱离具体 Runtime；Harness 可以直接分析历史 Trajectory。非常适合 Research。

**生态兼容。** 不会被 LangGraph、Agent-Lightning、VERL 锁死。

**研究自由度。** 研究人员可以只使用 MAS，也可以 MAS + Harness、MAS + RL，甚至只拿 Trajectory 接自定义 RL。

**面向研究问题。** 基础设施直接回答 What happened / Why failed / What should change / Did it improve / Can I reproduce it。

**天然支持未来 Marketplace。** Agent、Skill、Tool、Memory、Harness、RL Algorithm 都已 Plugin 化，远期可形成 Agent / Skill / Harness Marketplace。

### 12.2 劣势与风险

**抽象成本很高。** 必须先定义 Agent、Event、Trajectory、TrainingSignal、Diagnostic、Intervention。这些 API 一旦设计错误，后期修改成本极高。这是整个项目最大的技术风险。

**Universal Schema 可能过度抽象。** Trajectory 若变成几十个 Optional Field，所有框架都会不舒服。必须遵守 Core Schema 最小化、Metadata 扩展。

**Runtime 能力很难完全统一。** 不要承诺所有 Runtime 100% feature parity，应声明 Core / Optional Capability。

**RL 集成复杂。** Token Alignment、Log Probability、Reference Model 等要求因 Trainer 而异，所以 RL Adapter 必须独立于 MAS Data Contract。

**系统容易过度工程化。** 最大的工程诱惑是 Microservices、Kafka、Kubernetes、Ray、PostgreSQL、Redis、Object Storage、WebSocket 全部一起上。第一版绝对不要这样做。

推荐第一版技术栈：

```text
Python
+ Pydantic
+ Local Storage
+ LangGraph
+ Agent-Lightning
+ VERL
```

先把 Contract 跑通。

**UI 会严重拖慢项目。** Graph Editor 涉及的校验、版本、序列化和编译成本很高，应最后做。

### 12.3 最大战略风险

如果最终项目变成 LangGraph + Agent-Lightning + VERL + Dashboard + 一些 Harness，定位会退化成「一个不错的 Agent Research Wrapper」，技术壁垒有限。

真正应该形成的是：

- Universal Agent Data Plane
- Forkable Execution
- Experiment Plane
- Research Harness
- Optimization Interface

### 12.4 优先级

从项目管理角度，能力优先级应理解为：

| 优先级 | 核心能力 | 重要程度 | 原因 |
|---|---|---|---|
| P0 | Universal Data Contract | 最高 | 全系统枢纽，错了必重构 |
| P0 | Runtime Adapter | 最高 | 框架无关的第一落点 |
| P0 | Experiment / Run | 最高 | 从 Framework 升级成 Research Platform |
| P0 | Checkpoint / Fork | 最高 | 诊断、反事实、replay 的共同原语 |
| P1 | RL Adapter | 高 | 优化闭环，但可后于契约 |
| P1 | Harness | 高 | 回答 why failed，形成研究差异化 |
| P2 | MAS Multi-Agent | 高但可后置 | 拓扑是增量，不是地基 |
| P3 | UI / Drag & Drop | 中 | 会严重拖慢项目 |
| P4 | Marketplace | 低 | Plugin 化之后的远期形态 |

### 12.5 下一阶段

下一阶段第一优先级应从「继续开发功能」调整为 **冻结 Core Contract + 做端到端 Vertical Slice**：

```text
WorkflowSpec
      ↓
LangGraph Runtime
      ↓
Universal Event
      ↓
Trajectory
      ↓
Reward
      ↓
GRPO / VERL Adapter
      ↓
Checkpoint
      ↓
Harness Diagnosis
      ↓
Fork
      ↓
重新 Rollout
```

只要这一条链路真正跑通，就已经拥有一个区别于普通 Agent Framework 的 Research Infrastructure 骨架。

在 Slice 稳定前：不扩多 Agent，不上拖拽编辑器，不微服务化。
