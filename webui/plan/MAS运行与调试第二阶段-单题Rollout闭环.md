# Plan

本阶段定位为“统一 MAS 执行与单次 Rollout 采集”，依赖 [第一阶段：模型接入与最小执行契约](MAS运行与调试第一阶段-模型就绪与配置.md)。先使用远程 API 模型执行一个任务，在同一份 Workflow 上完成 Agent 协作、模型推理和工具交互，产生完整、结构清楚的 rollout。

UI 中的“运行 / 调试”只是该基础能力的入口。核心产物是轨迹，答案是轨迹的一个字段；以后增加自部署模型、训练策略、采样次数和局部分支，复用这套执行核心与记录结构，而不是另写执行器。

一个任务从入口执行一次、不分支，是本阶段的最小采样策略。Rollout 可以包含多次模型请求、工具调用和 Agent 交接，不等于一条回答，也不等于一次模型 HTTP 请求。

本阶段实施进度见 Action Items，具体接口与存储约定见 Implementation Notes。文件名保留以兼容原有链接；编译通过不代表已经执行真实模型采集或训练。

## Requirements

- 首期能通过远程 API 采集一次完整 Workflow 的 rollout，不要求本机训练 GPU、AGL 或参考答案。
- Control 单次运行、现有 Collector 与训练适配器复用 MAS 执行核心，HTTP 不是核心执行的必要依赖。
- 从首次运行就记录 Agent 执行片段、模型请求、工具调用、交接和错误，不能推迟到轨迹查看阶段再补采集。
- 复用现有 `Trajectory`、Archive 和 MemoryStore，增量补充字段，不建设另一套“调试专用轨迹”。
- 执行核心不强制计算奖励；Collector / 训练适配器仍可按各自职责附加奖励、优势配置或学习信号。
- 运行产物绑定本次任务、Workflow 快照、有效模型配置和可获得的策略信息。
- 成功、失败和中断都尽量保留已记录的轨迹，不把失败运行只处理成一个字符串弹窗。
- 保留现有示例 / parquet 采集接口及 `collect.json` 消费方式，不将“现有采集”当作废弃路径。
- 本阶段不启动训练，不宣称任意 API 轨迹都能直接用于任意 RL 算法。

## Scope

- In：单任务运行入口、统一 MAS 执行分派、最小 Agent 接线、调用级记录、运行存储与结果摘要。
- In：将现有训练适配器的 MAS 执行部分接到共同核心，保留模型资源、trace、奖励和训练输出契约。
- In：运行内上下文 / 记忆隔离、实际 Prompt / Skill / Tool 行为、错误与终止原因。
- Out：完整 Agent 平台、语言迁移、长期记忆、Agent 级多模型配置 UI。
- Out：多次采样调度、ARPO 自动分支、全系统快照恢复、奖励算法改版、训练参数更新。
- Out：后台任务队列、token 流式输出、中止功能和复杂历史比较。
- Out：轨迹的详细可视化，留给第三阶段；详细数据记录和最小读取能力不后置。

## Current State

以下为本阶段开始前的实现基线。

- `RunConfig` 只有示例任务与 parquet，尚无直接输入问题的单次运行入口。
- `/api/mas/collect` 调用 Collector，Collector 使用 ExecutionService 后计算奖励；Control 再包装训练信号。
- `ExecutionService.run()` 返回 `Trajectory`，自身不计算奖励，适合作为共同执行入口。
- `LitTirAgent.rollout()` 当前直接调用 `run_episode()` 和反馈处理，与 ExecutionService 的多 Agent 分派路径不完全一致。
- 部分 Skill、内存与事件默认 owner / agent_id 为 hub，不能据此认定所有 Agent 的配置都已真实执行。
- 部分事件在模型循环完成后根据消息归档，工具返回缺少完整关联信息，不能只靠整理已有日志获得调用级精度。
- 某些异常位于 `traj.meta.error`，HTTP 正常返回并不意味着执行成功。
- 现有编译失败分支、hub 兼容路径和空工具回退需要明确处理，不能在新入口中静默跑成另一种系统。

## Proposed Architecture

```text
UI / Control：一次任务       Collector：若干任务       训练 worker
          │                       │                 │
          └───────────────┬───────┴─────────────────┘
                          ↓
          统一 ExecutionService / EpisodeRunner
          输入：任务、Workflow、模型绑定、运行上下文
                          ↓
        编译 → Agent 循环 → 模型/工具 → 交接 → 终止
                          ↓
             统一 Trajectory + Archive 引用
                          ↓
      UI 查看 / 可选奖励评估 / 采样器 / 训练适配
```

- 业务核心放在 `workflow` 层，不能反向依赖 FastAPI、React、UI 状态或强制 import 训练框架。
- Control 负责请求校验、配置解析与运行结果访问，不在 API handler 内重写 Agent 循环。
- Collector 继续负责收集多条轨迹、应用 reward_fn；新单次入口使用同一执行器但不默认评分。
- 训练适配器把当前策略端点、采样参数、trace callback、rollout / attempt 身份传入核心，消费返回轨迹后继续现有奖励与训练桥接。
- 不改变训练优化器、分支调度或 loss，只消除训练与 API 入口间的 MAS 分派分叉。
- 已支持的 hub 兼容语义保留并记录实际执行模式；非法或未支持的拓扑明确拒绝，不降级成演示运行。
- 首期复用 Python 执行核心。接口按结构化数据设计，未来可对接服务端 TypeScript，但当前不增加第二套实现。

## Minimal Agent Wiring

第一阶段定义的接口在本阶段真实接通：

| 能力 | 本阶段执行要求 |
| --- | --- |
| Prompt | 应用当前 Agent 的实际 Prompt，记录最终生效值，不能全部回退使用 hub 提示词 |
| Context | 输入任务、上游输出、历史消息有明确来源；记录实际发给模型的上下文 |
| Memory | 运行内按 Agent owner 隔离，共享读写明确标注；不同 rollout 默认不共享可变状态 |
| Skill | 明确现有 Skill 的执行时机并记录调用与结果；无法支持的节点配置明确报错，不静默跳过 |
| Tool | 按明确规则形成有效工具集合，记录实际定义、调用参数、返回和失败 |
| Loop | 记录 Agent 进入 / 退出、模型调用、工具循环、交接及终止原因 |
| Recorder | 调用边界负责采集；不能从最后答案推断遗漏的执行过程 |

- 复用现有 Skill 注册表，不把技能名称自动解释成另一种 Prompt 文件或插件协议。
- 记录实际上下文裁剪与摘要结果。原始任务上下文不等于模型实际看到的上下文。
- 参考答案作为评估数据隔离，不自动进入模型消息。
- 自部署服务也可能走 API；“live”表示真实推理，不是供应商专属模式。
- 当前空工具回退必须区分未指定与显式空集合。沿用兼容入口时记录实际工具集合，不声称工具已禁用；必要的显式行为变化需清楚限定作用范围。
- 不新增跨任务长期记忆或通用状态管理框架，先做可解释的运行内状态。

## Rollout Recording Contract

首期采集粒度为“任务轨迹 → Agent 执行片段 → 模型调用 / 工具调用”，而不是只有整条消息列表。

```text
Trajectory
  ├─ AgentExecution：Planner 第一次执行
  │    └─ ModelCall
  ├─ Handoff：Planner → Executor
  ├─ AgentExecution：Executor 第一次执行
  │    ├─ ModelCall
  │    ├─ ToolCall → ToolResult
  │    └─ ModelCall
  └─ AgentExecution：Verifier 第一次执行
       └─ Skill / Feedback / Finish
```

| 层级 | 必需记录 | 用途 |
| --- | --- | --- |
| 运行与轨迹 | run_id、trajectory_id、任务身份、Workflow 快照、实际执行模式、起止与终止原因 | 可追溯的一次执行 |
| Agent 执行片段 | agent_id、agent_execution_id、顺序、来源交接、实际 Prompt / Skill / 工具 / 模型绑定 | 区分同一 Agent 多次进入及多个 Agent 共享模型 |
| 模型调用 | call_id、所属执行片段、实际请求上下文与工具定义、采样参数、可见输出、finish reason、错误 | 将动作与真实观测对应 |
| 工具调用 | 调用与返回关联 ID、所属模型调用 / Agent、参数、输出、状态 | 观察环境交互，不把工具结果当成模型动作 |
| 状态与快照 | snapshot_id、来源事件 / 执行片段、上下文 / 内存引用、覆盖范围 | 为未来续跑保留明确边界 |
| 可选模型信息 | 供应商实际返回的 token IDs、logprobs、usage、模型 / 策略 / tokenizer 标识 | 有则保留，缺失则标注能力限制 |
| 可选学习信息 | 已有 reward、parent trajectory、分支来源及后续适配字段 | 允许组合，执行核心不负责训练 |

- 使用有版本的结构和稳定 ID；`run_id` 与 `trajectory_id` 不混用，虽然首期一次运行只产生一条轨迹。
- 记录内容与关联可以内嵌或引用 Archive，但所有新采样必须能重建上述关联。
- 模型调用来源不能只依赖聚合消息里的 role；真实 Agent 和执行片段必须在调用时传递。
- 同一模型连续执行不同 Agent，仍分别记录 Agent 身份和模型身份。
- 接口凭据、Authorization 等不写入轨迹；完整上下文指模型可见业务输入，不包含传输密钥。
- 大型输出单独保存并引用，UI 可显示摘要；采集阶段不能仅截断为几百字符后当作完整训练记录。
- 若内容被明确裁剪、脱敏或缺失，记录其范围和原因，不假称严格可复现。
- 记录工具开始 / 结束和模型调用边界时间，区分实际发生时间与事后归档时间。
- 不请求隐藏思维内容，不合成 token 概率、模型版本或熵。不存在的信息明确缺失。
- 模型返回 logprobs 时原样保留和标注含义，不把选中 token 的负 logprob 自动称作完整分布熵。

## Snapshot and Sampling Boundary

- 首期从 Workflow 入口执行一次；“单独运行 Agent”“从某节点开始”“从状态分支续跑”是后续独立能力，不用一个入口参数含糊替代。
- 保留现有 Archive / BranchPoint，不要求本阶段实现任意位置可恢复的全系统快照。
- 消息快照不等于完整 MAS 状态。需标明是否包含执行游标、Agent 上下文、MemoryStore、工具外部状态和策略信息。
- 没有完整恢复证据时只提供快照记录，不开放“从这里分支”按钮。
- 采样策略以后在执行器之外安排次数、预算和分支；本次核心不硬编码 ARPO。
- 执行与奖励评估分开，后续采样策略可按算法需要调用评估器；不把四层写死为只能顺序执行。

## API Design

HTTP 只是单次执行的适配入口。以下路径以 Rollout 命名替代早期文档未实现的 `debug-runs`，不是迁移一个已存在的线上接口：

| 接口 | 作用 |
| --- | --- |
| `POST /api/mas/rollout-runs?experiment_id=...` | 同步执行一个任务的一次 Workflow rollout |
| `GET /api/mas/rollout-runs/{run_id}?experiment_id=...` | 读取运行摘要 |
| `GET /api/mas/rollout-runs/{run_id}/trajectory?experiment_id=...` | 读取同一轨迹契约，首期即提供基本数据访问 |

请求示意：

```ts
interface RolloutRunRequest {
  workflow: WorkflowSpec;
  task: { id?: string; question: string };
  execution: 'mock' | 'live';
}
```

- UI 捕获提交对象，保存成功后提交同一份 Workflow 快照；不在保存后读取更新中的最新草稿。
- 服务端执行该快照并再次校验，单次运行中不重读可能被另一页面覆盖的 YAML。
- 核心任务结构仍复用现有工作流契约；HTTP 首期限制为单题文本，不把问题字符串写死到未来所有 worker 接口。
- 服务端模型配置解析一次并记录实际摘要；训练适配器可以显式传入模型绑定，不经此 HTTP 请求。
- 不要求 RL 算法、GPU、奖励或训练组大小；不偷偷把单题参考答案填成空字符串用于评分。

响应为运行摘要，不替代完整轨迹：

```ts
interface RolloutRunSummary {
  run_id: string;
  trajectory_id?: string;
  status: 'running' | 'succeeded' | 'failed' | 'interrupted';
  execution: 'mock' | 'live';
  started_at: string;
  finished_at?: string;
  model: {
    source: 'api' | 'local' | 'rl_endpoint';
    name: string;
    policy_version: string | null;
  } | null; // mock 可以没有真实模型
  final_answer: string | null;
  termination_reason: string | null;
  format_ok: boolean | null;
  model_call_count: number;
  tool_call_count: number;
  trace_status: 'complete' | 'partial' | 'unavailable';
  error?: { stage: string; code: string; message: string };
}
```

- 完整轨迹继续以扩展后的 `Trajectory` 和 Archive 为准，并声明 schema 版本。
- `trace_status=complete` 仅指首期记录契约完整，不表示具备任何 RL 算法要求的全部数据。
- 执行前失败可以没有 trajectory_id；执行开始后尽量保留部分轨迹。
- POST 首期是同步请求，不承诺队列、取消、断线续传或服务重启后继续执行。
- `GET` 的运行与实验归属由服务端限定，不允许传任意本地文件路径。

## State and Persistence

- 持久化到 `experiments\<id>\artifacts\rollout-runs\<run_id>`，引用 / 保存统一轨迹和 Archive，不覆盖现有 `collect.json`。
- 保存本次任务、Workflow、实际模型配置摘要与有效 Agent 上下文；不包含 API Key。
- 状态文件完整写入后替换；读取方不能把半写入内容当作合法运行。
- 独立运行记录和现有批量产物可以有不同外层封装，内层轨迹不可分叉成两套协议。
- UI 使用 `useMasRolloutRun` 管理新运行状态，复用现有草稿保存；不在 hook 中实现业务执行。
- 保存与运行共享本页重复提交限制，输入、运行身份和后续草稿独立维护。
- 切换实验时隔离结果，旧请求返回不能污染新实验。
- 请求断开且无可读取 run ID 时显示“结果未知”，不自动补发。
- 后端重启后依据运行记录的服务实例归属明确标记遗留任务中断，不把未知状态标为成功。
- 底层轨迹写入失败必须显式报告；不能仅因答案已经返回就声称采集完成。

## Result and Error Semantics

| 情况 | 呈现 |
| --- | --- |
| 正常完成任务并得到有效终态 | 运行成功，不代表答案正确或训练数据充分 |
| 模型、工具或编排致命异常 | 运行失败，保留已有调用记录与错误阶段 |
| HTTP 正常返回但轨迹记录致命错误 | 按业务结果显示失败 |
| 工具局部错误后 Agent 恢复 | 保留错误事件，结合最终状态判断整体结果 |
| 达到轮次 / token / 编排步数上限 | 明确终止原因，不伪装成自然完成 |
| 无法提取最终答案 | 输出问题或失败，保留原始可见输出和部分轨迹 |
| 缺少 token/logprob/策略版本 | 可以形成基础 rollout，能力标记为缺失，不伪造字段 |
| 未进行奖励评估 | `final_reward` 未设置，不显示 0 分或通过率 |
| 模拟执行 | 明确 mock 轨迹，不混入真实推理记录 |
| 存储失败或断线 | 明确采集失败 / 结果未知，不自动重试外部调用 |

新逻辑必须由实际错误来源分类，不用 broad catch 返回空轨迹加成功状态。

## UI Layout

```text
[ 单次 Rollout ] [ 数据集采集 ]

模型：当前模型 · 远程 API                    配置模型
问题：[希望这个 Workflow 完成的任务……]
执行方式：模拟 / 真实推理
将从 Workflow 入口运行一次并记录完整过程。

运行前保存 Workflow                       采集一次 Rollout
```

- 默认单任务单次运行，不开放完整采样策略表单。
- 真实推理显示费用提示，不自动执行；模型摘要说明当前来源，而不是固定写成远程 API。
- 保留现有示例与 parquet 采集，在该用途区域保留奖励和训练信号相关配置。
- 结果首先显示 Rollout 身份、采集状态、最终答案摘要和记录完整性，提供“查看轨迹”入口。
- 答案卡片服务阅读体验，持久化的完整轨迹才是交付产物。
- 第二阶段轨迹读取可先用简洁结构化详情；第三阶段再做调用树和工具详情，不删减采集数据。
- 常驻工具箱、画布位置、问题输入在运行期间保留；新编辑明确不属于本次快照。
- 不显示假的实时节点高亮，不把工具绑定数量当调用次数。

## Files and Entry Points

| 文件或区域 | 调整职责 |
| --- | --- |
| `science_infra\control\app.py` | 单次 Rollout 请求与基本读取路由 |
| `science_infra\control\rollout_runs.py`（建议新增） | HTTP 执行适配、运行身份和存储 |
| `science_infra\control\llm_config.py` | 第一阶段模型配置复用 |
| `science_infra\control\services.py` | 现有采集与共同执行契约兼容 |
| `agent-lightning\examples\tir_agent\workflow\runtime.py` | 共同 MAS 执行入口、Agent 上下文、调用级记录 |
| `agent-lightning\examples\tir_agent\workflow\contracts.py` | 轨迹、片段、调用身份及完整性元数据 |
| `agent-lightning\examples\tir_agent\workflow\archive.py` | 原始记录、引用与快照范围 |
| `agent-lightning\examples\tir_agent\workflow\memory.py`、`plugins.py` | 运行内隔离与真实 Skill 接线 |
| `agent-lightning\examples\tir_agent\workflow\collector.py` | 复用执行核心并保留奖励职责 |
| `agent-lightning\examples\tir_agent\lit_tir_agent.py` | 训练资源 / trace / reward 适配，移除平行 MAS 分派 |
| `agent-lightning\examples\tir_agent\tir_agent.py` | 模型和工具边界、实际上下文、完整响应记录 |
| `webui\src\features\mas`、`webui\src\pages\MAS.tsx` | 单次运行入口、摘要、基本轨迹访问与草稿隔离 |
| `webui\src\shared\api\types.ts` | 与后端一致的运行及轨迹类型 |

## Action Items

- [x] 以 ExecutionService 收口 MAS 分派，保持执行核心不依赖 UI / HTTP / 训练框架。
- [x] 将现有 Collector 和训练适配器接到共同执行入口，保留各自 reward、trace 与资源契约。
- [x] 固定本次任务、Workflow 和模型绑定，阻止非法结构降级执行。
- [x] 接通实际 Agent 的 Prompt、上游上下文、Skill、工具与运行内记忆 owner。
- [x] 从首次采集记录 Agent 执行片段、模型输入输出、工具参数返回、交接和错误。
- [x] 保留可获得的 token/logprob/策略信息，缺失明确标注。
- [x] 区分日志摘要与完整记录，大输出保存引用而非只截断。
- [x] 标明快照范围，不把消息快照承诺为全系统可恢复状态。
- [x] 新增单次 Rollout HTTP 入口和摘要 / 基本轨迹读取，复用统一数据结构。
- [x] 持久化运行身份、终止原因、完整性与部分失败轨迹，不覆盖 collect.json。
- [x] 提供单题输入、模型摘要与显式执行按钮，保留现有数据集采集。
- [x] 显示 Rollout 摘要与轨迹入口，奖励不作为运行必需条件。
- [x] 保持重复提交限制、实验隔离和断线未知状态，不自动付费重试。
- [x] 更新直接相关的执行契约与接口说明。

## Implementation Notes

- Control 单次执行入口位于 `science_infra\control\rollout_runs.py`，不把奖励或训练信号包装进本次请求。
- 每次使用独立的 32 位 UUID 运行目录，`run.json` 保存摘要、任务、Workflow 快照与实际模型摘要，`trajectory.json` 保存统一轨迹；Archive 位于同一运行目录的 `archives` 下。
- 状态文件通过临时文件完整写入后原子替换。执行期间持有操作系统文件锁，多个读取请求不会因服务实例不同而误判仍在运行的任务。
- 执行进程退出后锁由操作系统释放；读取遗留 running 记录时再次确认状态，再标记 interrupted，不自动重跑。
- 实际模型绑定与提交的 Workflow 快照分别记录，避免把运行期间或提交后变化的模型配置误认为原始画布快照。
- 轨迹读取返回运行摘要、Workflow、任务、实际模型摘要和完整轨迹。因凭据脱敏改变记录时同时标记 `redaction_applied`，不将其宣称为可严格复现的完整输入。
- 结构校验失败不会执行模型；依赖、配置或执行错误明确返回失败信息，存储错误返回 HTTP 500，不以已有答案伪装采集成功。
- `useMasRolloutRun` 管理单次输入、摘要、日志与按需轨迹读取；通过 `executeWithSavedWorkflow` 与保存、Collect、预览共用操作锁，提交本次保存捕获的同一份 Workflow。
- Workflow 中的 LLM 绑定只包含协议允许的 kind、model、base_url，不把本地部署参数或就绪状态写入执行协议。
- Control、Collector 和训练适配使用 `ExecutionService`；Control 与执行器复用 `validate_execution_spec()` 的前置规则，不在 HTTP 层维护另一套编排判断。
- 轨迹 schema_version 为 2；事件关联 run_id、trajectory_id、agent_id、agent_execution_id、model_call_id 和 tool_call_id。模型次数统计 model_call 起始事件，不重复统计 model_result。
- 模型边界记录实际消息、工具定义、采样参数、重试来源与上下文处理；返回记录保留实际提供的模型名、usage 和 logprobs，不补造未提供的策略信息。
- mock、消息续跑、凭据脱敏或缺失模型边界的轨迹标为 partial，并通过 trace_coverage 说明限制。非模型验证 Skill 不伪造模型调用事件。
- 基础轨迹查看以按需 JSON 详情为主，完整时间轴与画布定位继续留到第三阶段；本阶段没有发起真实模型或训练验证。

## Delivery Boundary

本阶段交付“通过远程 API 执行当前 MAS，并采集一条可信、可追溯的 rollout”。UI 可以用于调试，但不是只交付聊天框和答案。

前端 TypeScript / Vite 与后端模块、接口契约需能正常编译 / 导入；本阶段不引入测试框架，不要求真实 GPU 或训练运行。训练适配统一接线不等于已验证训练收敛；外部模型请求由用户明确触发。

## Risks and Edge Cases

- 训练入口若仍绕开统一多 Agent 分派，就不满足模型来源可替换的目标。
- 修改旧 hub 或 Skill 行为必须明确兼容范围，不能借统一入口擅自改变算法。
- UI 上的 Agent 节点不是执行时刻，首期不能提供缺少状态依据的任意节点续跑。
- 文本轨迹不天然具备策略梯度训练所需的 token 对齐、mask 和行为策略概率。
- 同一个 API 模型名称可能对应供应商内部不同版本，未知版本必须诚实记录。
- 工具外部副作用不一定可恢复，消息重放不等于环境恢复。
- 协议复用不代表训练 worker 必须向 Control HTTP 服务提交请求。

## Next Phase

完成后实施 [第三阶段：Rollout 查看、定位与可用性说明](MAS运行与调试第三阶段-轨迹查看与问题定位.md)。第三阶段展示本阶段已经采到的数据，不承担事后恢复缺失身份或调用过程的职责。
