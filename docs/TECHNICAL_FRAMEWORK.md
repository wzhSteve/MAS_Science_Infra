# Agent_Science_Infra 技术框架说明与实现差距分析


| 项       | 内容                                              |
| ------- | ----------------------------------------------- |
| 版本      | 2026-09-08                                      |
| 对照文档    | [GPT_analysis.md](./GPT_analysis.md)（架构提案 v1.0） |
| 产品原则    | [design.md](../design.md)                       |
| UI 落地   | [CONTROL_UI.md](./CONTROL_UI.md)                |
| 实现主仓    | `agent-lightning/examples/tir_agent/workflow/`  |
| CLI 入口  | `science_infra.ui.cli`（`science-infra`）         |
| Control | `science_infra.control`（`science-infra serve`）  |


本文描述**当前代码真实结构**，并对照 GPT 架构提案逐层标注：**已实现 / 部分实现 / 未实现**。

---



## 0. 一句话结论

当前仓库是 **Phase 0 MVP 已达成，并多走了一段 Control Plane / WebUI 切片（v1.5）**，且已补上 **GPU 勾选、Agent/Tool 画布、Topology Compiler 最小执行**。

执行后端仍是：**LangGraph** `TirAgent`（hub ReAct 为默认；Compiler 把合法 graph 编成同步交接 walk）。研究原语尚未冻结：六合同、独立 `Episode`、`Diagnostic` / `Intervention`、Experiment 生命周期仍缺。

- **已跑通**：`WorkflowSpec(YAML) → LangGraph TirAgent → Event/Trajectory → Reward/TrainSignal → Harness 诊断 → CLI HTML`；训练侧 `LitTirAgent → AGL Trainer/VERL → GRPO 族五算法`。
- **v1.5 新增**：实验 YAML bundle（`experiments/<id>/`）+ FastAPI REST/SSE + React 分层面板 + React Flow 图画成 **MASSpec YAML**（不生成 LangGraph 代码）；Collect / Diagnose / Train 经 Control API 调用下层。
- **未成型**：完整 Universal Data Plane、Experiment Controller 生命周期、多 Runtime Adapter、研究型 Harness 闭环、Intervention 自动执行。
- 与提案最大的偏差不是「UI 没做」，而是 **UI / 图编辑被提前做了，合同冻结与 Experiment 生命周期仍落后于提案 §11–§12 的优先级**。
- 与提案一致的关键边界已守住：`workflow/` **不** import `agentlightning`。

---



## 1. 仓库布局与职责

```text
Agent_Science_Infra/
├── design.md                          # 产品原则 / Phase 0 说明
├── pyproject.toml                     # science-infra 包 + CLI
├── run.sh                             # smoke / live-api / live-vllm / tests / train
├── scripts/setup_uv_env.sh            # 本地 uv 环境（非 UI 自动配）
├── science_infra/                     # 薄 UI / Control 层（不持有图 / 不训模型）
│   ├── env.py                         # 仓库根 .env 加载
│   ├── ui/
│   │   ├── cli.py                     # collect / diagnose / status / dashboard / doctor / serve
│   │   ├── status.py                  # 单文件 HTML 状态页
│   │   └── dashboard.py               # dashboard 视图模型（CLI 与 Monitor 共用）
│   └── control/
│       ├── app.py                     # FastAPI：实验 YAML、REST、SSE
│       ├── experiments.py             # ExperimentMeta + 四段 YAML bundle
│       ├── services.py                # LLM / collect / diagnose / train
│       ├── process_manager.py         # llm | collect | train 子进程
│       ├── events.py                  # 进程内 EventBus → SSE
│       └── paths.py
├── webui/                             # React + React Flow 分层面板
│   └── src/pages/                     # Experiment / LLM / MAS / RL / Harness / Monitor
├── experiments/<id>/                  # 实验配置真源（默认 demo）
│   ├── experiment.yaml
│   ├── llm.yaml / workflow.yaml / rl.yaml / harness.yaml
│   ├── .secrets.env                   # API key，不回显
│   └── artifacts/                     # collect.json / diagnose.json / archives / runs
├── docs/
│   ├── GPT_analysis.md                # 目标架构提案（对照真源）
│   ├── TECHNICAL_FRAMEWORK.md         # 本文
│   ├── CONTROL_UI.md                  # Control UI 操作说明
│   └── infra-gap-analysis.md          # 早期相对 design.md 的差距分析
└── agent-lightning/                   # AGL 基板（勿 fork 核心）
    └── examples/tir_agent/            # ★ 实现主仓
        ├── workflow/                  # MAS 构建 + 数据面（无 AGL）
        ├── specs/hub_react.yaml       # 默认 WorkflowSpec
        ├── tir_agent.py               # LangGraph ReAct Runtime
        ├── lit_tir_agent.py           # AGL LitAgent 适配
        ├── train_tir_agent.py         # 训练入口（支持 --rl-yaml）
        ├── algos/                     # VERL trainer/daemon/advantage 挂钩
        ├── tools/                     # web_search / wikipedia / python
        ├── MAS_structagent/           # 平行 RL 路径，非产品 Runtime
        ├── scripts/                   # collect_rollouts / check_workflow_deps
        └── tests/                     # stage1–10 回归
```


| 区域                                           | 角色                                                              | 可否独立于 RL                   |
| -------------------------------------------- | --------------------------------------------------------------- | -------------------------- |
| `science_infra/ui`                           | 传参、采集、诊断、静态可视化、启动 Control                                       | 是                          |
| `science_infra/control` + `webui/`           | 实验 YAML、REST/SSE、分层面板、图画 YAML                                   | collect/diagnose 是；train 否 |
| `experiments/<id>/`                          | 分层配置 + artifacts                                                | 是（配置本身）                    |
| `workflow/`                                  | Spec、Runtime facade、Event、Trajectory、Reward、Harness、TrainSignal | 是                          |
| `tir_agent.py` + `tools/`                    | LangGraph 执行后端                                                  | 是（需 LLM endpoint 或 mock）   |
| `lit_tir_agent.py` + `algos/` + `train_*.py` | RL 训练挂接 AGL/VERL                                                | 否（需 GPU + AGL）             |
| `MAS_structagent/`                           | EPC-AW 等多 Agent 参考 / 独立 GRPO 训练                                 | 未接入 `ExecutionService`     |


---



## 2. 当前运行时数据流

系统有两条面：**研究数据面**（MAS → Trajectory → Harness / RL）与 **控制面**（UI → YAML → 进程）。控制面不持有 `StateGraph`；训练 live 曲线仍在 AGL `/metrics`。

### 2.1 研究数据面

```text
YAML (specs/hub_react.yaml 或 experiments/<id>/workflow.yaml)
        │
        ▼
   MASSpec (workflow/spec.py)          ← Definition Plane
        │
        ▼
 ExecutionService.run / fork           ← Runtime Plane facade
        │
        ├─ mock: MockRunner → run_mock_episode
        └─ live: TirRunner  → TirAgent (LangGraph) via importlib
        │
        ▼
 Archive: ExecutionEvent[] + Snapshot  ← 执行态存档
        │
        ▼
 Trajectory (+ final_reward)           ← Collector / rewards
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
   TrainSignal        HARNESS.diagnose    science-infra status
   (Advantage/Loss)   → Hypothesis[]      → 静态 HTML / Monitor
        │
        ▼  (仅训练路径)
 apply_train_signal → Hydra/VERL config
        │
        ▼
 LitTirAgent + AGL Trainer + VERL
 (GRPO / ARPO / AEPO / IGPO / GIGPO)
```



### 2.2 控制面

```text
WebUI 六页  /  science-infra serve
        │
        ▼
 FastAPI (science_infra.control.app)
        │
        ├─ PUT  /api/experiments/{id}/{section}   → 写 llm/workflow/rl/harness.yaml
        ├─ POST /api/mas/collect                  → Collector → artifacts/collect.json
        ├─ POST /api/harness/diagnose             → HARNESS    → artifacts/diagnose.json
        ├─ POST /api/rl/train                     → train_tir_agent.py --rl-yaml
        ├─ POST /api/llm/start|stop               → 本地 vLLM 子进程
        └─ GET  /api/events                       → SSE（进程状态，非 token 曲线）
        │
        ▼
 ProcessManager  kind ∈ {llm, collect, train}
        │
        ▼
 experiments/<id>/artifacts/{collect,diagnose}.json
                 /artifacts/archives/
                 /artifacts/runs/{run_id}/
```

`ExperimentMeta.pipeline` 默认为 `["collect","diagnose","train"]`，**不会自动串行执行**；要在 UI 上分别点 Collect / Diagnose / Train。

层边界（与提案 Control/Data 分离一致的部分）：

1. **MAS 构建**产出 Event / Trajectory，不算 advantage / loss。
2. **Harness**只读 Trajectory，输出 `Hypothesis`，不改图；回退经 `ExecutionService.fork`（Control/UI **尚未接线**）。
3. **RL**只通过 `TrainSignal` + AGL span 消费，不持有 `StateGraph`。
4. **Control**只发命令、写 YAML、启停进程，不持有 LangGraph 对象。

---



## 3. 代码框架详解（按模块）



### 3.1 Definition：`workflow/spec.py`

静态 Framework，不是 Runtime。代码类名是 `MASSpec`（提案称 `WorkflowSpec`）。默认文件 `specs/hub_react.yaml`；Control 路径使用 `experiments/<id>/workflow.yaml`。

```yaml
# specs/hub_react.yaml（默认）/ experiments/demo/workflow.yaml
schema_version: 0.1.0
topology: hub_react          # 或 graph（见 is_executable）
hub:
  role: orchestrator
  skills: [react_loop]
  # verify: verifier          # v1.3 可选事后回边
  # max_feedback_hops: 1
tools: [web_search, wikipedia_search, execute_python]
llm: { kind: api, model: "", base_url: "" }
memory: { agent: messages, system: none }
archive: { window: post_first_tool }
agents: []                   # schema 0.2：图节点（Runtime 大多忽略）
edges: []                    # schema 0.2：from/to + kind
```


| 字段                 | 现状                                                                          |
| ------------------ | --------------------------------------------------------------------------- |
| `topology`         | `hub_react` / `single` 可执行；`graph` 经 `compile_spec` 校验边后可执行               |
| `hub.skills`       | 每次 `run` 真正 `Skill.run`                                                     |
| `hub.verify`       | 可选事后 verifier；失败 `route=hub` 再跑                                             |
| `hub.system_prompt` | 写入 TirAgent SystemMessage（hub / from_spec）                              |
| `tools`            | 反映到 mock 事件与 `TirAgent.from_spec`                                           |
| `llm.kind`         | Schema 有 `api/local/rl_endpoint`；collect 主要靠 env / CLI / Control `llm.yaml` |
| `memory.system`    | `none` 时不写系统层                                                               |
| `entry_agent`      | Rollout 起点；Compiler walk 从此 id 开始                                        |
| `agents` / `edges` | **可保存且可执行**（`tool_call`/`route`/`message`/`feedback`）；非法边 Collect 400     |
| Topology Compiler  | **有**：`workflow/compiler.py` 编成同步交接；每 hop 仍跑 TirAgent/mock               |


对应提案 `WorkflowSpec`：**部分实现**（单拓扑可执行子集 + 图 schema 槽位）。

### 3.2 Contracts：`workflow/contracts.py`


| 类型                                        | 职责                   | 对应提案             |
| ----------------------------------------- | -------------------- | ---------------- |
| `ExecutionEvent` + `EventKind`            | 原子执行事件               | `ExecutionEvent` |
| `Trajectory` / `TrajectoryBatch`          | 研究数据对象               | `Trajectory`     |
| `Snapshot` / `ArchiveRef` / `BranchPoint` | Checkpoint / Fork 输入 | Checkpoint 子集    |
| `MemoryItem`                              | 双层 Memory 条目         | Memory 子集        |
| （内部）`EpisodeRaw` dataclass                | 单次执行原始结果             | 非正式 `Episode`    |


**事件种类（已有）**：`task_start`、`agent_message`、`tool_call`、`tool_result`、`final_answer`、`snapshot`、`error`、`memory_read`、`memory_write`、`feedback`。

**提案有、代码无的事件语义**：独立 `LLMRequest`/`LLMResponse`、`Reward` 事件、`Checkpoint`/`Feedback` 作为一等控制事件合同（现有 `FEEDBACK` 仅 verifier 回边）、`Termination`。

`Trajectory` 仍带 TIR 顶层字段（`n_search` / `n_python` / `format_ok`），同时 `sync_tir_meta()` 镜像进 `meta`，尚未收敛为「Core 最小 + metadata」。

**注意命名冲突**：AGL 的 `agentlightning.execution.events.ExecutionEvent` 是进程取消信号；science 侧是研究事件。提案建议改名 `MasEvent`/`TraceEvent`，**尚未改**（记为债务，不作为当前阻断）。

### 3.3 Runtime：`workflow/runtime.py`

`ExecutionService` 是 MAS 构建门面：


| API                             | 行为                              | 提案对照             |
| ------------------------------- | ------------------------------- | ---------------- |
| `run(task) → Trajectory`        | mock 或 TirAgent；可选 verifier 回边  | `Runtime.run` 部分 |
| `fork(branch_point, task?)`     | 从 Snapshot/`resume_messages` 续跑 | `fork` 已有最小实现    |
| `collect(tasks)`                | 多次 `run`                        | 薄封装              |
| `pause` / `resume`              | **无**                           | 未实现              |
| `checkpoint` / `restore` 独立 API | 经 Archive，非 Runtime 一等方法        | 部分（Archive 有）    |


Adapter 形态：

```text
ExecutionService
    ├── EpisodeRunner (Protocol)          # 可注入；无 RuntimeCapabilities
    ├── MockRunner  →  run_mock_episode
    └── TirRunner   →  importlib("tir_agent").TirAgent   # 唯一产品 Runtime
```

无类名 `RuntimeAdapter` / `LangGraphAdapter`，无 AgentScope/AutoGen/Native 第二后端。TirAgent 通过 `importlib` 加载，保证 `workflow/` AST 上不依赖 AGL。

`fork` 语义是 **从 Archive Snapshot 续跑**（`resume_messages`），不是提案里的运行中 pause/resume。LangGraph 内部另有 `TirAgent.resume_react`（finalize 后无 `<answer>` 时继续 ReAct），与 Archive fork 不是同一原语。

### 3.4 Archive / Memory / Plugins


| 模块           | 能力                                                            | 缺口                                         |
| ------------ | ------------------------------------------------------------- | ------------------------------------------ |
| `archive.py` | events.jsonl、snapshot JSON、`restore`、`BranchPoint`、rollout 索引 | 无窗口淘汰策略产品化；无跨进程统一 Object Store             |
| `memory.py`  | `read/write/latest`；agent + system 两层                         | 无 `update/snapshot/restore`；非向量/Episodic   |
| `plugins.py` | `Skill`/`AgentRole` 注册表；`react_loop`、`verifier`；planner/executor role 已注册 | 无 Planning/Causal 等 Skill；无独立 Controller.decide |


Control 采集时 `archive_root` 落到 `experiments/<id>/artifacts/archives/`。

### 3.5 Data Plane：Collector / Rewards / TrainSignal

```text
Collector.collect(tasks, n=)
    → ExecutionService.run × n
    → reward_fn(traj, task)   # canonical：workflow.rewards
    → TrajectoryBatch
    → batch_to_train_signal → TrainSignal{advantage, loss, batch}
```

任务来源：CLI `--tasks` JSON/JSONL/parquet；Control `POST /api/mas/collect` 可传 `tasks` 或 `parquet` + `data_n` + `source`（对齐 `./run.sh live-api-data`）。


| 合同                   | 现状                                                                       | 提案                           |
| -------------------- | ------------------------------------------------------------------------ | ---------------------------- |
| Reward               | outcome reward（答案匹配 + format + 工具计数等）                                    | `Trajectory → Reward` 有      |
| `TrainSignal`        | `AdvantageSpec` + `LossSpec` 超参规格，**不含** token log_prob / advantage 数值张量 | 提案 `TrainingSignal` 更偏训练张量合同 |
| `apply_train_signal` | 写入 Hydra clip/entropy/kl/gamma + `tir_algo`                              | RL Adapter 雏形                |
| Episode 对象           | 隐式含在 Trajectory + Archive；内部 `EpisodeRaw`                                | 无独立 `Episode` 模型             |




### 3.6 Harness：`workflow/harness.py`

插件注册表 `HARNESS`：


| 插件                      | 状态   | 行为摘要                                              |
| ----------------------- | ---- | ------------------------------------------------- |
| `log_error`             | 已实现  | 扫描 `EventKind.ERROR`                              |
| `loss_volatility`       | 已实现  | reward 不升 / 波动过大（可用 metrics JSONL）                |
| `cognitive_convergence` | 薄实现  | ERROR 签名归一；≥2 条轨迹且后半段仍重复 → high deviation         |
| `reward_hacking`        | 薄实现  | reward≥0.9 且缺答案/format 或带 ERROR；**无外部 Judge LLM** |
| `epc_aw_consensus`      | stub | 恒返回 `[]`；Control 将其列入 `STUB_HARNESS`，UI 勾选禁用      |


输出类型为 `Hypothesis{plugin, event_id, message, meta}`，**不是**提案的 `Diagnostic` + `Intervention` 双合同。干预路径仅人工/脚本调用 `fork`，无 Experiment Controller 自动执行 Intervention，Control 也无 fork API。

### 3.7 RL 挂接（`algos/` + `lit_tir_agent.py` + `train_tir_agent.py`）

不在 `workflow/` 内，依赖 AGL：


| 能力              | 落点                                                                               |
| --------------- | -------------------------------------------------------------------------------- |
| rollout enqueue | `algos/daemon.py`（ARPO/AEPO 树分支）                                                 |
| reward span     | `LitTirAgent` + `agl.emit_reward`（与数据层同一 reward 函数）                              |
| advantage       | `algos/trainer.py` 在 VERL `compute_advantage` 替换；Hydra `adv_estimator` 固定 `grpo` |
| 算法              | `grpo` / `arpo` / `aepo` / `igpo` / `gigpo`                                      |
| loss            | VERL PPO clip；无独立 Loss 插件接口                                                      |
| Control 配置      | `train_tir_agent.py --rl-yaml experiments/<id>/rl.yaml` 深合并到 profile             |


提案中的 `Reward Adapter / Advantage Estimator / Objective / Trainer / Inference Endpoint` 五段抽象：**未抽成 science 包内接口**，直接挂 AGL/VERL。

`llm.kind=rl_endpoint` 可写入 YAML；Collect 阶段无专用运行时，训练中的 endpoint 仍由 AGL `ProxyLLM` 提供。

### 3.8 UI：CLI + Control Plane + WebUI



#### CLI（`science_infra.ui.cli`）


| 命令                                   | 作用                                                 |
| ------------------------------------ | -------------------------------------------------- |
| `science-infra collect`              | 无 VERL 采集 + 写出 batch/TrainSignal JSON              |
| `science-infra diagnose`             | 跑 Harness                                          |
| `science-infra status` / `dashboard` | 单文件 HTML：mean_reward、SVG 曲线、事件、Harness、TrainSignal |
| `science-infra doctor`               | Python / GPU / AGL / MASSpec 探测                    |
| `science-infra serve`                | 启动 FastAPI Control + 可选 `webui/dist`               |


AGL Dashboard 侧栏 **Science MAS**（`/science`）可加载同一 JSON。训练 live 曲线仍在 `/metrics`。

#### Control API（`science_infra.control.app`）


| Method   | Path                              | 行为                                               |
| -------- | --------------------------------- | ------------------------------------------------ |
| GET      | `/api/health`                     | 健康检查                                             |
| GET      | `/api/meta`                       | algos / harness 插件 / stub / profiles / llm_kinds |
| GET      | `/api/gpus`                      | nvidia-smi 列表 + 推荐档位                             |
| GET/POST | `/api/experiments`                | 列出 / 创建实验目录                                      |
| GET/PUT  | `/api/experiments/{id}`           | 读 bundle / 写 meta                                |
| GET/PUT  | `/api/experiments/{id}/workflow`  | 读/写 workflow.yaml + `executable`                 |
| PUT      | `/api/experiments/{id}/{section}` | `llm|workflow|rl|harness|meta`                   |
| GET      | `/api/mas/palette`                | skills/roles/tools/edge_kinds/templates          |
| POST     | `/api/llm/health|start|stop`      | 探测 OpenAI 兼容 `/models`；本地 vLLM 启停                |
| POST     | `/api/mas/sample-data`            | 从 parquet 抽样任务                                   |
| POST     | `/api/mas/collect`                | mock / live / parquet；不可执行图返回 400                |
| POST     | `/api/harness/diagnose`           | 对最近 collect 跑插件                                  |
| POST     | `/api/rl/train|stop`              | 子进程训练；可先停本地 LLM                                  |
| GET      | `/api/runs` `/api/runs/{id}`      | 进程状态 + log tail                                  |
| GET      | `/api/monitor/{id}`               | dashboard 模型（读 artifacts）                        |
| GET      | `/api/events`                     | **SSE** 进程事件（`event: message`）                   |


**无** WebSocket。**无** pause / resume / fork 端点。Collect 为同步 HTTP，长任务会阻塞该请求。

`ProcessManager`：kind ∈ `{llm, collect, train}`，每种同时只保留一个 active；生命周期仅 start / SIGINT→SIGKILL，不是暂停语义。

#### WebUI（`webui/src`）

无 React Router；`App.tsx` 侧栏 tab：`experiment | llm | mas | rl | harness | monitor`。SSE 更新顶栏状态 chip；Monitor 图表需手动刷新。


| 面板         | 已接线                                                             | 边界                                                                         |
| ---------- | --------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Experiment | 列表 / 新建 / name+seed                                             | `pipeline` / `agl_metrics_url` 未在表单编辑                                      |
| LLM        | kind `api|local|rl_endpoint`；health；local 启停；密钥写 `.secrets.env` | `rl_endpoint` 仅提示文案                                                        |
| MAS        | React Flow（Agent/Tool/Rollout 起点）→ YAML；模板 hub_react/hub_verify/pev_draft；Collect mock/live/parquet | 图版本化未做；Fork 徽章未做 |
| RL         | GPU 勾选 / 每题采样 / n_runners / TrainSignal 预填 / 启停训练 / 链到 AGL metrics                     | `GET /api/runs` 列表未在 UI 调用                                                 |
| Harness    | 勾选插件（stub 禁用）+ diagnose                                         | 无 Intervention 按钮                                                          |
| Monitor    | Recharts reward、事件、hypotheses                                   | 不订阅 SSE 刷新图表                                                               |


图编辑（`workflowGraph.ts`）序列化为 MASSpec YAML，**不生成 LangGraph 代码**——这一点与提案 §9.2 一致。Topology Compiler 把合法 `route/message/feedback/tool_call` 编成 ExecutionService walk。

### 3.9 Experiment YAML bundle

对象：`ExperimentMeta{id, seed, refs, pipeline, name, agl_metrics_url}`。

磁盘：

```text
experiments/{id}/
  experiment.yaml
  llm.yaml
  workflow.yaml
  rl.yaml
  harness.yaml
  .secrets.env              # chmod 600，API 只回 api_key_set
  artifacts/
    collect.json
    diagnose.json
    tasks_sampled.json      # parquet 抽样时
    archives/
    runs/{run_id}/stdout.log, status.json
```

这是提案 Experiment Layer 的 **配置+产物目录雏形**，不是 CREATE→COMPARE 状态机，没有 Trial / Result / 配置指纹复现 API，没有 SQLite/PG。

`MAS_structagent/`：`train_mas_agent.py` 等是**平行 GRPO 路径**，强制训练 LLM 走 VERL endpoint；**未**接入 `ExecutionService` / `MASSpec` 拓扑编译 / Harness `epc_aw_consensus`。

---



## 4. 与 GPT_analysis.md 的对照总表

图例：✅ 已实现 · 🟡 部分/雏形 · ❌ 未实现

### 4.1 设计原则（P1–P6）


| 原则                    | 判定  | 说明                                                                                      |
| --------------------- | --- | --------------------------------------------------------------------------------------- |
| P1 Layer Independence | 🟡  | `workflow/` 可无 RL；Harness 可只读 JSON。Control 可只跑 collect/diagnose。尚无独立 Experiment 层**调度** |
| P2 Contract-First     | 🟡  | 有 Pydantic 合同；未冻结提案六合同全集；TrainSignal ≠ 完整 TrainingSignal                                |
| P3 Plugin-Oriented    | 🟡  | Skill / Diagnoser / Role 有注册表；Model/Tool/Runtime/RL Algorithm 未统一 Registry              |
| P4 Framework Agnostic | 🟡  | Spec ≠ Graph；图编辑产出 YAML。但仅 TirAgent 一条可执行后端                                             |
| P5 Event-Centric      | 🟡  | Event 流已有；缺 LLMRequest/Response 等细粒度模型事件                                                |
| P6 Reproducibility    | 🟡  | 有 `ExperimentMeta.seed` 与 YAML bundle；无 Trial/Compare/按 seed 重建 API                     |




### 4.2 Universal Data Plane


| 提案对象           | 判定  | 代码落点 / 缺口                                               |
| -------------- | --- | ------------------------------------------------------- |
| Event          | 🟡  | `ExecutionEvent`；事件类型子集                                 |
| Episode        | ❌   | 无独立模型；逻辑散落在 Trajectory + 内部 `EpisodeRaw`                |
| Trajectory     | 🟡  | 有；偏 TIR 字段（`n_search`/`format_ok`），非极简 Core+metadata 终态 |
| TrainingSignal | 🟡  | `TrainSignal` 为算法超参规格 + batch，非 token 级信号               |
| Diagnostic     | 🟡  | `Hypothesis` 近似，缺 severity/confidence/recommendation 结构 |
| Intervention   | ❌   | 无正式合同；仅 `fork(BranchPoint)`                             |
| 六合同冻结          | ❌   | WorkflowSpec/Event/Trajectory 雏形；其余未冻结                  |




### 4.3 MAS Layer


| 提案能力                               | 判定  | 现状                                                     |
| ---------------------------------- | --- | ------------------------------------------------------ |
| WorkflowSpec YAML                  | 🟡  | `MASSpec` / `hub_react.yaml` + Control `workflow.yaml` |
| Agent / Role 可扩展                   | 🟡  | hub / verifier / planner / executor role；图上可训节点走 Compiler walk |
| Controller.decide                  | 🟡  | 无独立 Controller；边 `route` 由 Compiler 解释为交接                      |
| Skill 插件                           | 🟡  | `react_loop` + `verifier`；无规划/因果等                      |
| Model 三路抽象                         | 🟡  | Spec 有 kind；训练走 AGL `ProxyLLM`；无统一 `Model.generate` 接口 |
| Tool                               | 🟡  | web_search / wikipedia / python；无 MCP/Browser/DB       |
| Memory 双层                          | 🟡  | 进程内 `MemoryStore`；system 默认关闭                          |
| Archive / Checkpoint               | 🟡  | 文件归档 + restore                                         |
| Forkable Execution                 | 🟡  | `ExecutionService.fork` + resume_messages；UI 未接线       |
| Runtime pause/resume               | ❌   | 无                                                      |
| Runtime Adapter 多后端                | ❌   | 仅 LangGraph TirAgent；有 `EpisodeRunner` Protocol        |
| RuntimeCapabilities 声明             | ❌   | 无；仅有 `is_executable()`                                 |
| 多 Agent（Planner/Executor/Verifier） | 🟡  | Compiler 最小执行已有；每 hop 仍是 TirAgent ReAct，非独立 LangGraph 子图 |




### 4.4 RL Layer


| 提案能力                          | 判定  | 现状                                          |
| ----------------------------- | --- | ------------------------------------------- |
| 只看 Trajectory/Signal/Endpoint | 🟡  | 训练仍大量依赖 AGL span/DataProto                  |
| Advantage 可替换                 | 🟡  | 五算法在 VERL hook；非 science Advantage 插件层      |
| Objective PPO/GRPO/…          | 🟡  | GRPO 族；无独立 DAPO/RLOO/REINFORCE 产品接口         |
| Trainer 多后端 VERL/AGL/TRL      | 🟡  | AGL+VERL；无 TRL                              |
| Inference Endpoint 回灌 MAS     | 🟡  | 训练时 AGL 提供 endpoint；collect CLI/Control 需自配 |
| 独立 RL Engine 包                | ❌   | 逻辑在 `examples/tir_agent/algos`              |
| `--rl-yaml`                   | ✅   | Control `rl.yaml` 深合并进 train 入口             |




### 4.5 Harness Layer


| 提案能力                                                        | 判定  | 现状                                         |
| ----------------------------------------------------------- | --- | ------------------------------------------ |
| Observation/Evaluation/Diagnosis/Intervention/Visualization | 🟡  | Diagnosis + 静态/React 可视化；无 Intervention 合同 |
| LogError / Reward / Trajectory 分析                           | 🟡  | log_error、loss_volatility、薄 reward_hacking |
| Cognitive Convergence                                       | 🟡  | 启发式签名；非语义/认知对齐                             |
| Reward Hacking + 外部 Judge                                   | 🟡  | 规则版；无强 LLM Judge                           |
| EPC-AW / FailureAttribution / LLMJudge / Causal             | ❌   | stub 或未做                                   |
| 异步消费 Trajectory                                             | ❌   | 同步 CLI / 同步 Control HTTP；无消息队列             |




### 4.6 Experiment Layer


| 提案能力                                                | 判定  | 现状                                                 |
| --------------------------------------------------- | --- | -------------------------------------------------- |
| Experiment / Run / Trial / Seed / Artifact / Result | 🟡  | `ExperimentMeta` + YAML + artifacts；无 Trial/Result |
| 生命周期 CREATE…COMPARE                                 | ❌   | 无 Controller 状态机；pipeline 字段不自动跑                   |
| 配置驱动 MAS+RL+Harness 一体实验                            | 🟡  | YAML 已分层；执行仍靠 UI/CLI 分步按钮                          |
| REST 实验 API                                         | 🟡  | CRUD + collect/diagnose/train；无 fork/compare       |




### 4.7 UI Layer


| 提案能力                                          | 判定  | 现状                                             |
| --------------------------------------------- | --- | ---------------------------------------------- |
| Configuration / Control / Visualization（第一阶段） | 🟡  | 分层面板配置 + 启停进程 + Monitor；无 pause/resume/fork UI |
| REST + WebSocket/SSE                          | 🟡  | REST + SSE 已有；无 WebSocket；SSE 不推训练曲线           |
| Graph Editor / 拖拽                             | 🟡  | Agent+Tool 画布 + Rollout 起点；无图版本化          |
| GPU 检测 + uv 环境                                | 🟡  | `GET /api/gpus` + 顶栏勾选；`doctor` + `scripts/setup_uv_env.sh` |




### 4.8 工程落地（提案 §10）


| 项                                 | 判定  | 现状                                                            |
| --------------------------------- | --- | ------------------------------------------------------------- |
| 第一版单体 Python API                  | ✅   | 符合「不要微服务」                                                     |
| Metadata DB（SQLite/PG）            | ❌   | 实验元数据即 YAML 文件                                                |
| Trajectory Parquet/Object Storage | 🟡  | JSON / JSONL 本地文件；任务可从 parquet 抽样                             |
| Checkpoint Object Storage         | 🟡  | 本地 `.tir_archives` 或 experiment `artifacts/archives`          |
| 统一 Registry（Model/Tool/Skill/…）   | 🟡  | Skill + Diagnoser + Role；不全                                   |
| 插件动态 YAML 加载 Harness/RL           | 🟡  | `harness.yaml.plugins` 与 `rl.yaml.algo` 可被 Control 读取；非完整动态装配 |




### 4.9 路线图阶段对照（提案 §11）


| 阶段          | 提案目标                                                       | 当前进度                                                                 |
| ----------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| Phase 0 MVP | 单 Agent hub ReAct、LangGraph、AGL、VERL、Harness、CLI、Dashboard | **已达成**（design.md v1.0–v1.4 + Control v1.5）                          |
| Phase 1     | 冻结 Event/Trajectory/Experiment/Checkpoint/Fork             | **进行中**：Event/Trajectory/Fork 雏形；Experiment **YAML+API 已起步**，生命周期未开始 |
| Phase 2     | 多 Agent + Skill/Memory 拓扑                                  | **最小 Compiler 已起步**（PEV 可 Collect；无独立 Controller/Memory 产品化） |
| Phase 3     | 完善 RL + 多 Trainer                                          | **部分**：五算法挂 VERL + `--rl-yaml` + GPU 勾选；无 TRL/DAPO 产品层                        |
| Phase 4     | 研究型 Harness                                                | **薄实现 + stub**                                                       |
| Phase 5     | UI / Graph Editor                                          | **配置/控制/可视化 + Agent/Tool 画布已落地**；无 Graph 版本仓库                               |


相对提案顺序的战略偏差：Phase 5 的配置面板与 Graph Editor **提前于** Phase 1 合同冻结。编辑器方向正确（生成 Spec 而非 Graph）；Compiler 已能跑合法 graph，缺口转到合同/Experiment 生命周期。

---



## 5. 端到端能力矩阵（研究五问）

提案成功标准：What happened / Why failed / What should change / Did it improve / Can I reproduce。


| 问题                  | 当前能否回答 | 手段                                                       |
| ------------------- | ------ | -------------------------------------------------------- |
| What happened?      | 🟡     | Trajectory.events + Archive + status HTML / Monitor      |
| Why failed?         | 🟡     | log_error / cognitive_convergence / reward_hacking（浅）    |
| What should change? | ❌      | 无 Diagnostic.recommendation / Intervention 自动闭环          |
| Did it improve?     | 🟡     | mean_reward 曲线、训练 Metrics；无 Experiment Compare / fork 对比 |
| Can I reproduce?    | 🟡     | 有 seed 字段与 YAML；无按配置指纹重建的 API                            |


---



## 6. 已实现功能清单（可操作）

无训练 GPU：

```bash
pip install -e .
science-infra doctor
science-infra collect --mock --n 2 --out /tmp/traj.json
science-infra diagnose /tmp/traj.json
science-infra status /tmp/traj.json --html /tmp/status.html
science-infra dashboard /tmp/traj.json --html /tmp/dash.html
```

Control UI（仓库 uv `.venv`）：

```bash
./run.sh ui
# 浏览器 http://127.0.0.1:8787/   API 文档 /docs
./run.sh ui-test              # GPU/RL 控制面冒烟（start 后立刻 stop 训练子进程）
```

面板可：保存分层 YAML、mock Collect、parquet live Collect、Diagnose、启停本地 LLM、`--rl-yaml` 训练（需 GPU）。操作细节见 [CONTROL_UI.md](./CONTROL_UI.md) 与 [README.md](../README.md)。

等价直调：

```bash
cd agent-lightning/examples/tir_agent
PYTHONPATH=. python scripts/collect_rollouts.py --mock --n 2 --out /tmp/traj.json
PYTHONPATH=. python tests/test_infra_v1.py          # 兼容聚合
# stage1–11：含 test_stage10_control_ui.py / test_stage11_gpu_compiler.py
```

有 GPU + AGL：

```bash
cd agent-lightning/examples/tir_agent
PYTHONPATH=. python train_tir_agent.py fast --algo grpo --n-runners 1
PYTHONPATH=. python train_tir_agent.py fast --algo grpo --rl-yaml ../../../experiments/demo/rl.yaml
```

仓库脚本：`./run.sh smoke` / `live-api` / `live-api-data` / `live-vllm` / `./run.sh train fast --gpu 0 --n-runners 1`；环境 `bash scripts/setup_uv_env.sh`。

代码级已具备的原语：

- YAML Spec → ExecutionService（hub_react；graph 经 Compiler walk）
- Mock / Live episode → Event → Trajectory → Reward
- Archive snapshot + `fork`（脚本/测试级，非 Control）
- Skill 热路径 + verifier 回边（可选）
- TrainSignal → Hydra overlay；`--rl-yaml` 深合并；`run.sh train` 可选 GPU/`n_runners`
- Harness 四插件 + 一 stub
- CLI HTML Dashboard / AGL `/science` 页
- Experiment YAML bundle + REST/SSE + WebUI + React Flow→YAML + GPU 勾选
- stage1–11 回归（stage10：`is_executable`、Control collect；stage11：Compiler / `/api/gpus`）

---



## 7. 未实现 / 明确延后功能清单

按提案优先级（P0→P4）整理：

**P0 缺口（骨架尚未齐，下一阶段主攻）**

1. 冻结并统一命名六合同（含独立 `Episode`、`Diagnostic`、`Intervention`；`Hypothesis` 缺 severity/confidence/recommendation；`TrainSignal` ≠ token 级 `TrainingSignal`；`ExecutionEvent` 改名仍为债务）
2. Experiment / Run / Trial / Seed / Artifact / Result 与生命周期 Controller（CREATE…COMPARE）；现仅有 YAML 目录 + 分步按钮
3. Checkpoint/Fork 作为跨 Harness/RL/UI 的一等原语（现仅 TIR resume 友好；Control 无 fork API）
4. Runtime Adapter 接口 + `RuntimeCapabilities`（即使仍只有 LangGraph；现仅 `EpisodeRunner` + `is_executable`）

**P1 缺口**

1. RL Adapter 与 MAS 合同彻底分离（token alignment / logprob 合同）
2. `Diagnostic → Intervention → Controller → fork` 自动闭环
3. Harness 异步消费与更强诊断（Judge LLM、Failure Attribution）；`epc_aw_consensus` 仍 stub

**P2 及以后（刻意后置）**

1. Compiler 加深：独立 `Controller.decide`、并发 fork、非 Tir 子图
2. Planning/Causal Skill；Shared/Private Memory 产品化与可恢复
3. TRL / DAPO / RLOO 产品接口；Marketplace
4. WebSocket、运行中动态改参、Graph 版本化

**刻意不做的（Phase 0 / 当前）**

- 把 `MAS_structagent` 升为第二套产品 Runtime
- 微服务 / Kafka / K8s
- 拖拽生成 Python / LangGraph 代码

---



## 8. 与提案推荐目录的映射

提案 `science-infra/core|mas|data|rl|harness|experiment|ui` 扁平包结构；**当前尚未抽包**，映射如下：


| 提案路径                            | 当前路径                                                                         |
| ------------------------------- | ---------------------------------------------------------------------------- |
| `core/contracts/*`              | `workflow/contracts.py` + `train_signal.py`                                  |
| `mas/definition`                | `workflow/spec.py` + `specs/` + `experiments/*/workflow.yaml`                |
| `mas/runtime/langgraph`         | `tir_agent.py` + `workflow/runtime.py`（`EpisodeRunner`）                      |
| `mas/skill` / `memory`          | `workflow/plugins.py` / `memory.py`                                          |
| `data/collector` / `trajectory` | `workflow/collector.py` + `contracts.Trajectory`                             |
| `rl/*`                          | `algos/*` + `lit_tir_agent.py` + `train_tir_agent.py`                        |
| `harness/*`                     | `workflow/harness.py`                                                        |
| `experiment/*`                  | `science_infra/control/` + `experiments/`（雏形，非完整 Controller）                 |
| `ui/*`                          | `science_infra/ui/*` + `science_infra/control/*` + `webui/` + AGL `/science` |


稳定后再从 `examples/tir_agent/workflow/` 上抽到顶层 `science_infra`（或提案式 `core/`），是既定策略。Control 已先落在顶层 `science_infra`，workflow 仍留在 tir_agent 示例内。

---



## 9. 建议的下一刀（对齐提案 §12.5 Vertical Slice）

原则：**Vertical Slice 仍优先（合同 / Experiment / Fork）**。Compiler 最小 walk 已落地，下一步不是再堆画布，而是把 Checkpoint/Fork 接到 Control。

```text
WorkflowSpec
  → LangGraph Runtime（现有）
  → Universal Event（补齐类型 / 改名债务后置）
  → Trajectory（Core 最小化）
  → Reward
  → TrainSignal / VERL Adapter（现有挂钩加固）
  → Checkpoint + Fork（现有加固为合同，并接到 Control）
  → Diagnostic + Intervention
  → Experiment Controller 触发再 Rollout
```

建议工作包顺序：

1. **合同冻结（最小破坏）**
  在 `workflow/contracts.py` 增加 `Diagnostic` / `Intervention` / `Episode`（可与现有 `Hypothesis` / `Trajectory` 并存并映射）。Core Trajectory 字段收敛，TIR 计数只进 `meta`。事件补齐类型枚举或明确 metadata 扩展规则。暂不强制改名 `MasEvent`。
2. **Experiment 垂直切片**
  用现有 YAML bundle：同一 `experiment.yaml` + `seed` 能复现 mock/live collect。Harness 输出带 `recommendation`。Control 增加 `POST /api/runs/{id}/fork`（调用已有 `ExecutionService.fork`）。Monitor 能对比 fork 前后 mean_reward。这直接补提案五问中的 *What should change* / *Did it improve* / *Can I reproduce*。
3. **RuntimeCapabilities 声明**
  即使仍只有 TirAgent：`checkpoint=true, fork=true, pause=false, multi_agent=true`（graph 已 compile）。为非法边保留拒绝原因（`is_executable`）。
4. **Compiler 加深（非阻塞 Slice）**
  hub+verifier 与 PEV walk 已可跑。下一步才是独立 Controller、非 Tir 子图。UI 继续只写 Spec。
5. **Harness / RL 加深（并行、不阻塞 1–3）**
  Judge LLM 版 `reward_hacking`；EPC-AW 从 stub 变成可读 MAS 路径的诊断（不必先接入 `MAS_structagent` Runtime）；TrainSignal 与 VERL 的边界写进合同，不先抽 TRL。

验收标准：同一 Experiment 配置 + Seed，能复现采集；Harness 建议 fork 后能量化「是否变好」；图编辑仍只生成 Spec。

在 Slice 稳定前：不上拖拽生成代码、不微服务化。

---



## 10. 文档关系


| 文档                                                 | 用途                                            |
| -------------------------------------------------- | --------------------------------------------- |
| [GPT_analysis.md](./GPT_analysis.md)               | 目标架构与优先级（提案）                                  |
| [design.md](../design.md)                          | 产品原则与 Phase 0 已落地说明                           |
| [CONTROL_UI.md](./CONTROL_UI.md)                   | Control / WebUI 启动与面板操作                       |
| [TECHNICAL_FRAMEWORK.md](./TECHNICAL_FRAMEWORK.md) | **当前代码框架 + 与提案差距（本文）**                        |
| [infra-gap-analysis.md](./infra-gap-analysis.md)   | 早期相对 design.md / AGL 的改造规格（部分陈述已被 Phase 0 超越） |
| `tir_agent/docs/DESIGN.md`                         | TIR 算法细节                                      |


---

*生成说明：本文基于 2026-09-08 仓库实读（*`workflow/`*、*`science_infra/control/`*、*`science_infra/ui/`*、*`webui/src/`*、*`experiments/demo/`*、*`algos/`*、stage1–10 测试与 CLI），不以提案愿景替代实现状态。*