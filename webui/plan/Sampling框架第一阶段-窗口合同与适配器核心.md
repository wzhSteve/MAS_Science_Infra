# Plan

本阶段冻结算法无关的 Sampling Core：统一 Window、Opportunity、Decision 和 ExpansionPlan 合同；建立 Strategy Adapter 接口；将旧 `BranchSite.anchor` 只作为输入边界的兼容声明。阶段结束时核心不再根据 ARPO/AEPO/RAE 名称决定窗口或 Gate。

依赖现有 `SamplePolicy`、Archive、`ActiveSetSession`、ForkPlan 和 Workflow compiler。下一阶段见：[ARPO Tool Result Window 适配](Sampling框架第二阶段-ARPO工具窗口适配.md)。

## Position in the Project

Sampling Core 位于 MAS Runtime 与训练适配器之间：

```text
Runtime 产生真实 Window + Snapshot
                ↓
Sampling Core 匹配机会、调用 Adapter、生成 ExpansionPlan
                ↓
Daemon / Store 执行计划
```

`mas/workflow/` 保持不依赖 AGL、VERL 和 Ray；Adapter 可以提供训练 overlay，但核心合同不得包含 Hydra 或 VERL 类型。

## Requirements

- 使用统一 Window 合同表示 `tool_result`、未来的 `agent_complete`、`router_decision`、`verification_complete` 等窗口。
- Window 至少包含稳定 ID、owner Agent、kind、sequence、Snapshot 引用、metrics 和 interaction metadata。
- 一个 Window 必须对应自己的可恢复 Snapshot；没有 Snapshot 的 Window 不能生成 ExpansionPlan。
- Opportunity 只描述 Adapter 支持的窗口 selector、允许 Gate 和可恢复能力。
- Decision 记录 Adapter 的判定事实；ExpansionPlan 只消费通过的 Decision 和预算。
- Core 不读取 `tir_algo`，不根据 mode 名称选择窗口、Gate 或预算算法。
- Adapter 注册表是能力真源；Preflight 和 UI projection 均从注册表读取。
- 旧 anchor 在解析边界归一化，核心内部不继续传播 `after_tool`/`after_agent_turn` 的模糊别名。
- 旧 YAML、旧 ForkPlan 和 `.local_expansion` 在迁移期保持可读。

## Scope

- In：Pydantic/dataclass 合同、Adapter Protocol、注册表、anchor 归一化、Window/Snapshot 一一对应、核心调度职责重构。
- In：将现有 `ActiveSetSession` 拆分为算法无关编排和 Adapter 调用，不改变 Daemon enqueue 形态。
- Out：实现普通 Agent/Router/Verifier Window 的运行时生产者。
- Out：定义 ARPO entropy 公式、AEPO budget 或 RAE verdict；属于具体 Adapter。
- Out：修改 UI 和 Rollout Tree。

## Current State

- `WindowEndEvent` 使用 `kind=after_tool/after_agent_turn/...`，同时承担真实事件与旧 anchor 语义。
- `ActiveSetSession` 直接继承全局 ARPO Gate 参数，并在 `_sites()` 中补 Gate 参数。
- `apply_sample_policy()` 同时完成 policy 映射、算法选择和算法 overlay。
- `site_policy.py` 手写当前能力矩阵，尚不是 Adapter projection。
- `ForkPlan` 已携带 event/snapshot/site，但没有独立 Decision 对象，判定和计划生成耦合。

## Proposed Architecture

### Canonical Window Contract

新增或升级合同：

```python
class SamplingWindow(BaseModel):
    window_id: str
    owner_agent_id: str
    kind: Literal[
        "tool_result",
        "agent_complete",
        "router_decision",
        "verification_complete",
    ]
    sequence: int
    snapshot_ref: str
    metrics: Dict[str, Any] = {}
    interaction: Dict[str, Any] = {}
```

旧事件映射：

```text
after_tool(hub, execute_python)
→ kind=tool_result
→ owner_agent_id=hub
→ interaction.tool_id=execute_python

after_agent_turn(planner)
→ kind=agent_complete
→ owner_agent_id=planner
```

没有真实生产者的 Window 只能存在于目标枚举或 Adapter 声明中，不得由解析器伪造为已发生事件。

### Adapter Protocol and Registry

建立 `workflow/sampling/` 或等价独立模块：

```text
contracts.py
core.py
adapters/base.py
adapters/registry.py
compat.py
```

Adapter 返回：

- 支持的 Opportunity。
- 允许的 Gate。
- Window 判定结果。
- 预算分配。
- 可选训练 overlay。

Core 负责：

- 按 selector 匹配 Window。
- 执行 `first/every/nth`。
- 维护全局剩余预算。
- 调用 Adapter。
- 将 Decision 转为 ExpansionPlan。

### Compatibility Boundary

`BranchSite` 继续作为 YAML 合同；加载后归一化为内部 `WindowSelector`。旧 `after_agent_turn(hub)` 若只能映射 Tool Window，必须保留 `compatibility` 标记，不能在核心中伪装为 `agent_complete`。

## Files and Entry Points

| 区域 | 调整职责 |
| --- | --- |
| `mas/workflow/contracts.py` | SamplingWindow、Opportunity、Decision、ExpansionPlan 合同 |
| `mas/workflow/sampling/`（拟新增） | Core、Adapter Protocol、注册表和兼容映射 |
| `mas/workflow/archive.py` | Window 与 Snapshot 一一绑定 |
| `mas/workflow/active_set.py` | 收口为 Core 编排或薄兼容入口 |
| `mas/workflow/gates.py` | 纯 Gate primitive，不决定 Adapter 支持范围 |
| `rl/hooks/overlay.py` | 分离通用 policy 合并与 Adapter overlay |
| `science_infra/control/training.py` | 从 Adapter 注册表执行能力检查 |

## Action Items

- [x] 冻结 SamplingWindow、Opportunity、Decision 和 ExpansionPlan 合同。
- [x] 建立 Adapter Protocol 与注册表。
- [x] 将旧 anchor 映射集中到兼容层。
- [x] 将命中次数、预算和计划生成移入算法无关 Core。
- [x] 让 Window 必须携带可恢复 Snapshot 才能参与扩展。
- [x] 让 Preflight 与 capability projection 使用 Adapter 注册表。
- [x] 保留旧 API/ForkPlan 的迁移期序列化兼容。

## Implementation Notes

- 新增 `workflow/sampling/` 包，将合同、compat、Core、Adapter Protocol、注册表和 adapter 实现拆分；Core 只认识 Window/Selector/Decision/Expansion，不读取算法名。
- `SamplingWindow.window_id` 与 `snapshot_ref` 在合同层要求非空；`SamplingCore` 还要求实际 `resume_messages`，缺任一条件都不会产生计划。
- `first/every/nth`、剩余预算、max depth 和 ExpansionPlan 来源统一由 Core 处理；Adapter 只判断 Window 并分配通过后的数量。
- `ActiveSetSession` 现在是 EpisodeRaw/旧 BranchSite/ForkPlan 的兼容入口：将真实事件与 Snapshot 转成 `ResumableWindow`，调用 Core，再回写 Daemon 当前消费的 ForkPlan。
- 注册表当前提供 independent adapter 和 configured-gate compatibility adapter。`grpo/grpo_n/igpo/gigpo` 映射 independent；`arpo/aepo/appo/rae` 暂映射 compatibility adapter，下一阶段由正式 ARPO Adapter 取代 ARPO 映射，其他算法不在本阶段宣称完成。
- `site_policy.py` 与 `/api/mas/sampling/preview` 已改用 Adapter registry；ARPO 当前只投影 Tool Result Opportunity，旧 `after_agent_turn(hub)` 进入 legacy 列表，GRPO 不返回 Branch Opportunity。
- Adapter 在 Gate 前校验必要指标：entropy Gate 缺少 `h_root/h_tool`、dual entropy 缺少 `h_branch` 时返回不通过 Decision，不再以零值或其他 Gate 语义补齐。
- capability projection 同时返回原生 opportunities 与独立 `legacy_sites`；WebUI 提供兼容配置的禁用/删除入口。Router blank-agent 的 runtime tool ID 与 canvas node ID 分离，避免 `blank:` 前缀破坏画布定位。
- 保留旧 ForkPlan JSON、Daemon metadata 和无显式 sites 时的首次 Tool legacy 入口；显式 Site 不再在缺少真实 Window/Snapshot 时自匹配。

## Lightweight Validation

只使用纯 Python fixture 检查 Window 匹配、Snapshot 缺失、次数规则、预算耗尽、Adapter 注册和 legacy anchor 映射；不导入 AGL/VERL，不启动 GPU。

## Acceptance and Delivery Boundary

- Core 文件中不出现具体算法名分支。
- 旧 `after_tool` 可映射到 `tool_result` selector，但普通 `after_agent_turn` 不再兼容匹配 Tool Event。
- 同一 Window ID 只引用一个 Snapshot；ExpansionPlan 保留 Window/Decision 来源。
- Adapter 注册表可独立回答某策略支持哪些 Window/Gate。
- 本阶段完成不代表 ARPO 已恢复；只有下一阶段完成 ARPO Adapter 后才能训练。

## Risks and Edge Cases

- 合同重构不能破坏现有 Daemon 所需的 `resume_messages` 和 Store metadata。
- Window sequence 必须按父 rollout 独立计数，不能跨 rollout 共享。
- 兼容层不能用缺失身份字段的事件匹配具名 selector。
- 不能把训练 overlay 重新塞回 Core，避免下一算法继续增加条件分支。

## Next Phase

[第二阶段：ARPO Tool Result Window 适配](Sampling框架第二阶段-ARPO工具窗口适配.md)。
