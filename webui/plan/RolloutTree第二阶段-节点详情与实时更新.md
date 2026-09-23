# Plan

本阶段为新 WebUI 建设按训练 run 查看 Rollout Tree 的页面：先稳定呈现树与节点详情，再用可靠的同 run 增量事件更新；刷新、断线和 Control 重启后仍以持久树快照为准。

依赖[第三阶段：按 run 存储与结果回填](RolloutTree第一阶段-按run存储与结果回填.md)。这不是照搬 `main` 的旧 RolloutTree 标签页：旧版虽有 React Flow 图和 SSE 提示，但没有 run 级隔离或完整节点结果合同。

## Position in the Project

这里是训练结果视图，不是采样配置编辑器。Sampling 画布展示“可能在哪儿分支”；Rollout Tree 只展示“本次 run 实际发生了什么”。单题调试的 `RolloutRun`/`TrajectoryDetails` 属于另一种记录，不与训练 Tree 混称或复用无关的 run ID。

```text
实验 → 训练记录 → 指定 run → 树列表 → 单题树 → 节点详情
                           └──────────同 run 事件流──→ 局部刷新/对账
```

## Requirements

- 仅通过已校验的 `experiment_id + run_id` 访问本次运行的树；旧运行没有新树产物时明确提示，不回退显示共享 `.local_expansion`。
- 树列表分页、可按任务与状态筛选，默认只加载选中的一棵；大量节点时不一次性渲染所有 run 的全部树。
- query root、独立初始 rollout、branch child、planned/运行/失败/完成状态视觉区分；没有分支的树明确标为“独立采样，不代表 branch”。
- 节点详情展示 parent、Site、真实事件、前缀引用、Gate 指标与决策、状态、最终 outcome、rollout reward 和节点 credit/verdict；缺失字段标为“未记录/尚未计算”，不填默认值。
- 不通过前端展示原始 Archive 消息或敏感配置；从树定位 Workflow Site 时依据本次运行快照，不能误指向后来修改过的草稿。
- 实时事件必须带 run/tree/node 身份，支持断线重连、去重和终态对账；状态最终以持久树为准，不能只依靠易丢的 SSE。

## Scope

- In：运行记录入口、树列表、图布局、节点详情、按 run 事件流与重连对账。
- In：树读失败、旧格式、不完整运行和空结果的明确 UI 状态。
- Out：在浏览器里生成或编辑树、前端计算 reward/credit、通用 Harness 流平台或六算法训练验收。

## Current State

- `main` 的 `webui/src/pages/RolloutTree.tsx` 曾按实验展示旧全局 API 的图，并定时刷新；集成分支已删除，当前 `app/navigation.ts` 无树入口。
- 新 UI 有按 run 的训练记录、控制台、日志和单题轨迹；`features/mas/components/RolloutRun.tsx` 不是训练树。
- Daemon 将标记的树事件写 stdout。`/api/events` 中 `_drain_tree_frames()` 只在 EventBus 有消息、循环继续时读取 stdout，不保证每个节点发生后立刻推送。
- `/api/rl/runs/<run_id>/events` 已提供绑定 run 的日志流、续读 offset 和心跳；可复用其安全读取逻辑，不需要新建全局消息基础设施。

## Proposed Architecture

### 页面与节点交互

从训练记录和控制台的固定 run ID 打开树视图，页头显示算法/运行状态、树数目和最后更新时间。左侧分页任务列表；中间仅绘制所选树，布局按 query、独立 rollout、分支深度展开，边标 Site；右侧选中节点的详情区显示因果链“在哪个窗口 → Gate 为什么通过/未通过 → 从哪里续跑 → 得到什么结果”。选择节点不改变草稿或训练。老树、空树、只有计划的树各有清晰状态，不以彩色节点暗示已经训练成功。

可复用现有 `@xyflow/react`，但节点 key 使用持久身份，位置/展开状态仅为前端视图状态；大树分页或按子树展开，避免一次性创建所有 React Flow 节点。

### 同 run 的实时事件

基于第三阶段持久树快照提供 `node_added / status_changed / outcome / reward` 等结构化变更，事件含 run、tree、node 及单调序号或稳定日志 offset。Control 可复用 `training_logs.py` 的按 run 字节续读实现，解析已注册 run 的标记帧，再转为有界结构化事件；不要让页面解析任意 stdout 或依赖全局 `/api/events` 的其他事件唤醒。事件只提示哪棵树变化，页面增量读取或重取该树；重复事件幂等。SSE 中断时标明“实时连接中断”，恢复后重拉快照；训练终态也做一次快照对账。

## Files and Entry Points

| 区域 | 调整职责 |
| --- | --- |
| `science_infra/control/training_logs.py`、树路由 | 按 run 恢复事件、身份校验和持久树对账 |
| `webui/src/features/training/components/TrainingHistory.tsx`、`TrainingConsole.tsx` | 按当前真实 run ID 进入树视图 |
| `webui/src/app/navigation.ts`、`WorkspacePanels.tsx` | run 级入口和刷新后可恢复的路由 |
| `webui/src/features/rollout-tree/`（拟新增） | 树 API、列表、图、节点详情和实时订阅 |
| `webui/src/shared/api/types.ts` | 带版本和缺失态的树/事件类型 |
| `webui/src/features/mas/` | 仅复用必要的定位模式，不把单题调试轨迹当训练树 |

## Action Items

- [ ] 在训练运行中增加固定 run ID 的树入口及可刷新路由。
- [ ] 建树列表、单树按需加载、节点/边状态和详情视图。
- [ ] 显示 planned、失败、无分支、旧运行无树等诚实状态。
- [ ] 接入按 run 的可靠事件续读、去重、断线重连和终态快照对账。
- [ ] 从本次 Workflow 快照而非当前草稿定位可确认的 Site。

## Lightweight Validation

前端 fixture 覆盖 root-only、两条独立 rollout、多层 branch、失败/未完成、缺失指标、旧 run 无树和跨 run 切换；Control 的无 GPU 检查覆盖事件重放、重复、断线及归属校验。真实 GPU 树更新只能由用户在服务器显式启动训练后验收。

## Acceptance and Delivery Boundary

- 从某 run 进入时只显示其树；直接刷新 URL 仍打开同一 run 和所选树，不显示其他实验结果。
- 能逐节点回答“在哪分支、从何续跑、是否真的完成、实际结果是什么”；未知值明确为未知。
- 训练进行时新节点或 outcome 到来后更新；断线重连、进程结束和 Control 重启后与持久树一致。
- 完成本阶段只代表树的记录和 UI 闭环，不能据此将六算法、节点级训练 credit 或实时 Harness 全部标记完成。

## Risks and Edge Cases

- stdout 可能被分块截断且夹杂普通日志；解析要处理半行、重放和 generation 变化，不依赖一次 `read()` 恰好得到整帧。
- 查询多、树大时应分页和按需加载；SSE 通知不是全量树传输。
- 树的运行身份必须服务端校验，不能由浏览器提供路径或自行拼装文件名。

## Next Phase

回到[训练运行第四阶段：RL 算法收口与真实训练验收](训练运行第四阶段-RL算法收口与真实训练验收.md)，逐算法确认真实行为；Monitor/Harness 的实时事件产品入口另行收口。
