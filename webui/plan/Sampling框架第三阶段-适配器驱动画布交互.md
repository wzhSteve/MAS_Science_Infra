# Plan

本阶段重做 Sampling 设计态 UI，使画布完全由当前 Strategy Adapter 的 capability 驱动。ARPO 下只展示 Tool Result Window；普通 Agent、Router、Verifier 和 token Window 不作为原生可配置点。旧兼容配置单独呈现，不与原生机会竞争画布视觉权重。

依赖[ARPO Tool Result Window 适配](Sampling框架第二阶段-ARPO工具窗口适配.md)。本阶段完成后，其他算法和 Rollout Tree 才进入后续工作。

## Position in the Project

同一画布有两个模式：

```text
Workflow
  = 静态能力和可能路由

Sampling
  = 当前 Adapter 在静态关系上提供的设计态 Window

Rollout Tree（后续）
  = 本次 run 实际命中的 Window、Decision、child 和 outcome
```

Sampling Canvas 不预测真实 rollout 顺序，不展示未实现的目标窗口，也不使用运行态文案。

## Requirements

- Sampling 模式只消费 Adapter projection，不从节点类型自行推导可用窗口。
- ARPO 只显示 Tool Result Window。
- Tool Result Marker 挂在 Agent—Tool `tool_call` 交互边界，不挂 Tool 卡片，也不挂普通 Agent 卡片。
- Marker 文案明确“Tool 结果返回 Agent 后，重新采样 Agent 后续决策”。
- 普通 Agent、Router、Verifier 和 `on_edge` 关系只保留静态拓扑上下文；当前不显示可开启 Marker。
- `after_agent_turn(hub)` 等旧配置进入兼容区，提供禁用、删除和后续迁移，不作为画布原生机会。
- 所有 Workflow 边在 Sampling 模式保留为低对比度静态关系。
- 选中一个 Window 时，高亮其 Agent—Tool 关系；允许轻量流动效果表达“设计态重新采样范围”，但不得冒充实时执行。
- 配置态、可执行态和运行态文案严格分离。
- 全局策略以行为语言呈现，Adapter 和训练 objective 放入高级信息。
- 使用统一 CanvasToolbar、Tooltip、pressed 状态和响应式尺寸；不引入第二张线性轨迹图。

## Scope

- In：Adapter projection API、边界 Marker、关系高亮、Inspector、legacy 区、预算摘要和无损保存。
- In：ARPO/GRPO 设计态；GRPO 显示独立采样，不显示 Branch Window。
- Out：真实运行动画、Tree 节点、reward 和 Gate 命中结果。
- Out：未实现算法 Adapter 的虚拟机会。
- Out：普通 Agent/Router/Verifier Window 执行。

## Current State

- 当前 Sampling 模式已经有统一工具栏、能力 projection 和位置—条件—动作 Inspector。
- capability projection 同时返回 native、compatibility 和 unavailable 目标窗口，使 UI 混入当前 Adapter 不支持的目标能力。
- Tool Result Site 当前主要高亮 Tool 卡片，容易误解为重新执行 Tool。
- `after_agent_turn(hub)` 兼容项显示在 Hub 卡片上，容易误解为 ARPO 普通 Agent Window。
- Workflow 边统一降透明度，尚未将选中 Window 映射到具体 Agent—Tool 关系。

## Proposed Architecture

### Adapter Projection Contract

`POST /api/mas/sampling/preview` 返回：

```json
{
  "strategy": {
    "id": "arpo",
    "label": "自适应分支采样",
    "training_objective": "arpo"
  },
  "policy": {
    "group_n": 4,
    "initial_rollouts": 2,
    "remaining_budget": 2
  },
  "opportunities": [
    {
      "window_kind": "tool_result",
      "owner_agent_id": "hub",
      "interaction": {"tool_id": "execute_python"},
      "edge_id": "hub→execute_python/tool_call",
      "configured": true,
      "site_id": "site_after_tool_hub_execute_python"
    }
  ],
  "legacy_sites": [],
  "diagnostics": []
}
```

只返回当前 Adapter 原生机会；目标模型中未实现的 Window 不进入 `opportunities`。

### Canvas Interaction

ARPO 示例：

```text
Hub ───────────── execute_python
          ● Tool Result Window
```

- Marker 位于 `tool_call` 边靠近 Tool 端的语义边界。
- 选择 Marker 后，边和上游 Hub 高亮，Tool 卡片保持实体样式。
- Inspector 显示共享前缀包含 Tool Call/Result，重新采样 Hub 后续决策。
- 其他关系保留静态低对比度。
- 可选缓慢虚线动画只作用于当前选中 Window 的“可能重新采样”提示，并带设计态说明；默认及未选中边不动画。

### Legacy Configuration

单独区域：

```text
兼容配置
after_agent_turn(hub)
当前映射：Hub 内 Tool Result Window
状态：不属于 ARPO 原生机会
操作：禁用 / 删除 / 转换
```

legacy Site 不在画布节点上显示 Marker。

### Inspector

按人类决策顺序：

```text
位置
execute_python 结果返回 Hub 后

条件
第一次到达；后续响应不确定性提高

动作
共享 Tool Result 前缀；重新采样 Hub 后续；最多新增 N 条
```

底层 selector、Window kind、Snapshot 格式和 Adapter ID 置于高级合同信息。

## Files and Entry Points

| 区域 | 调整职责 |
| --- | --- |
| Adapter projection API | 仅返回当前 Adapter 原生机会、legacy 和诊断 |
| `features/mas/components/WorkflowEdge.tsx` | Tool Result Marker、选中关系与设计态动画 |
| `features/mas/components/MasGraphEditor.tsx` | opportunity → edge 映射和上下文高亮 |
| `features/sampling/components/SamplingSettings.tsx` | 行为策略、Inspector 和 legacy 区 |
| `features/mas/components/CanvasToolbar.tsx` | Workflow/Sampling 模式和画布动作 |
| `styles/integrations/react-flow.css` | 静态拓扑、Marker、选中范围和响应式 |

## Action Items

- [ ] 让 projection 由 Adapter 返回，移除 UI 中 unavailable 目标窗口。
- [ ] 将 ARPO Site 从 Tool/Agent 卡片迁移到 Agent—Tool 关系 Marker。
- [ ] 让选中 Marker 与 Inspector 双向同步，并高亮所属关系。
- [ ] 将 legacy Site 移出主画布，建立独立兼容区。
- [ ] GRPO 模式不显示 Branch Marker。
- [ ] 收口设计态线条、轻量流动提示和运行态文案边界。
- [ ] 完成宽屏、窄画布和 2K 下的视觉验收。

## Lightweight Validation

默认运行 WebUI production build 和本地页面交互检查。检查 ARPO、GRPO、legacy Site、删除 Tool 后失效 Site、多 Agent 静态关系和窄画布工具栏；不启动真实训练。

## Acceptance and Delivery Boundary

- ARPO 主画布只显示 Tool Result Window Marker。
- Marker 位于 Agent—Tool 交互边界，用户不会理解成 Tool 被重新执行。
- Hub/普通 Agent 不显示 ARPO 原生 Sampling Marker。
- legacy 配置可见但不占用主画布。
- 所有静态关系仍可理解；只有当前选中机会被强调。
- UI 不展示当前 Adapter 未实现的目标 Window。
- 本阶段完成不代表其他算法已适配，也不代表 Rollout Tree 已实现。

## Risks and Edge Cases

- 一条 Tool relation 可能同时对应多个 Site 次数规则；Marker 必须能打开列表而不是覆盖。
- 动态 Router 候选不能预画成固定真实路径。
- 动画必须遵守 `prefers-reduced-motion`，并明确是设计态提示。
- 删除 Tool 后的 legacy/失效 Site 必须在兼容区保留，不能静默丢失。

## Next Phase

完成 ARPO 服务器闭环后，再决定先实现下一个算法 Adapter，或进入[按 run 存储 Rollout Tree](RolloutTree第一阶段-按run存储与结果回填.md)。两者都不属于本轮 Sampling 三阶段。
