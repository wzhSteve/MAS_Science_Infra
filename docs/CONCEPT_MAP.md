# MAS 概念→文件地图（CONCEPT_MAP）

> 重构后的权威索引：每个可视化概念对应唯一文件+函数。设计细则见
> [MAS_design.md](MAS_design.md)；执行记录见 [refactor/phase0_*.md](refactor/)。

## 怎么读这张图

三层资产：**产品主栈**（`mas/workflow/` + `mas/tools/`，UI 唯一路径）、
**训练热路径**（`mas/tir_agent.py` 三件套，零移动红线）、**legacy 研究栈**
（`legacy/structagent/`，仅命令行）。

| 概念 | 唯一文件 | 关键符号 | 环 |
|---|---|---|---|
| **Agent 节点** | `mas/workflow/spec.py` | `AgentNodeSpec`（id/kind/system_prompt/profile） | spec |
| **编译图** | `mas/workflow/compiler.py` | `compile_spec` → `compiled.agents / routers / fan_out` | spec→runtime |
| **Router** | `mas/workflow/router.py` + `spec.py:RouterSpec` | `select()`（四契约：llm_choice=from_plan、score 回退候选自报分、round_robin 模块级计数器、fan-out 部分失败不丢成功） | runtime |
| **AgentMessage 信封** | `mas/workflow/protocol.py` | `AgentMessage`（8 kind），`make_message`/`validate_payload`（校验即抛错），`commit_to_fact`（bool 桥） | 协议 |
| **Memory（进程内）** | `mas/workflow/memory/store.py` | `MemoryStore`（三作用域 agent/shared hub/system；episode 生命周期，不跨 rollout） | runtime |
| **FactLedger（事实门控）** | `mas/workflow/memory/facts.py` | `FactLedger.classify` → COMMIT_FACT / KEEP_WINDOW_ONLY / REJECT（判定唯一，无存储） | 协议 |
| **Archive（快照/续采）** | `mas/workflow/memory/archive.py` | `Archive`（RLock 串行 append + tmp→rename 原子写），`archive_root()` 单点路径，`load_resume_messages` 三级回退 | 持久层 |
| **BranchSite（分支站点）** | `mas/workflow/contracts.py` | `BranchSite` / `BranchPoint` / `Snapshot`（schema_version=1） | 持久层 |
| **ToolAgent（工具 agent）** | `mas/tools/tool_agents.py` | `ToolAgent`（palette id→kernel+LLM 封装），`effective_tier`/`log_effective_tier`（lite/pro 解析 + mismatch 结构化日志） | runtime |
| **tier（pro/lite）** | `mas/tools/tool_agents.py` + `pro_tools.py` | `SCIENCE_INFRA_TOOL_KERNEL=1` 强制 lite（override_reason 注入） | runtime |
| **prompt 解析** | `mas/workflow/prompts.py` | `planner_prompt`（node>hub>default 空串规则）、`verifier_prompt`、`candidate_menu`、`planner_user_text`（三段式）、`prompt_provenance`（hash+source→Trajectory.meta） | runtime |
| **中央化循环** | `mas/workflow/centralized_runtime.py` | `run_centralized_episode`（planner→router fan-out→tool 窗口→verifier 环） | runtime |
| **单 hub ReAct** | `mas/workflow/runtime.py` | `run_episode`（TirAgent 训练路径，plain chat 无信封） | 训练热路径 |
| **Trajectory** | `mas/workflow/contracts.py` + `runtime.py:episode_to_trajectory` | `Trajectory`（meta 含 prompt_hash/prompt_source） | 数据层 |
| **MemorySpec 声明** | `mas/workflow/spec.py` | `MemorySpec`（scope/policy；Phase 6 增 `planner_ctx`） | spec |
| **采样策略** | `mas/workflow/active_set.py` + `sampling/` | `ActiveSetSession.plan_forks_from_raw`（第二波 enqueue） | 训练 |
| **用户区网关** | `mas/workflow/user_gateway/` | `loader.maybe_invoke_user_node`（HIVE/EPC-AW 隔离加载） | runtime |
| **Reward/TrainSignal** | `rl/rewards/outcome.py` + `rl/loss.py` + `rl/train_signal.py` | 唯一入口（Phase 3 起 shim 全删） | 数据层→训练 |

## 三条 prompt 链（只列读取者）

1. **tir_agent.py** 常量（SYSTEM_PROMPT 等）→ 单 hub 训练路径，原地不动。
2. **experiments/<id>/workflow.yaml**（`hub.system_prompt` + `agents[].system_prompt`）
   → `workflow/prompts.py` 解析（Phase 4 收口）→ centralized 循环窗口。
3. **legacy/structagent/epc_aw/prompts/*.txt** → StructAgent 栈内部 `_load_prompt`，
   不进产品主栈。

## 环别一覧（何时加载什么）

- **UI 采集/eval（无训练）**：`workflow` + `tools` 全走，`rl.*` 仅 reward 计算。
- **训练（VERL）**：`lit_tir_agent` → `run_episode`（+ ActiveSetScheduler 线程池，
  共享 Archive 由 RLock 保护）。
- **命令行研究（可选）**：`legacy/structagent/train_mas_agent.py`，
  `sys.modules["MAS"]` 注册名经 bootstrap 指向 `legacy/structagent/epc_aw`。
