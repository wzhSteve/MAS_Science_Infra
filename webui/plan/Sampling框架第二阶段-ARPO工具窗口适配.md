# Plan

本阶段将 ARPO 实现为 Sampling Core 的第一个正式 Strategy Adapter。ARPO 只支持 Tool Result Window：Tool Observation 已进入共享前缀后，使用 Agent 下一次响应的 entropy 指标判断是否扩展，并通过现有 Daemon/Store 二波 enqueue 创建 child rollout。

依赖[第一阶段：窗口合同与适配器核心](Sampling框架第一阶段-窗口合同与适配器核心.md)。下一阶段见：[适配器驱动画布交互](Sampling框架第三阶段-适配器驱动画布交互.md)。

## Position in the Project

ARPO Adapter 是通用核心的第一个纵向切片：

```text
Tool Call
→ Tool Result
→ Tool Result Window + Snapshot
→ Agent 下一次响应 entropy
→ ARPO Adapter Decision
→ ExpansionPlan
→ Daemon enqueue child
```

采样位置属于 Tool Result Window；重新采样对象是上游 Agent 的后续生成。Tool 本身不重新执行，普通 Agent 完成也不是本阶段的 ARPO 原生机会。

## Requirements

- Adapter 只声明 `tool_result` Window。
- Window selector 包含 owner Agent 和可选 Tool ID。
- Snapshot 包含 Tool Call 与 Tool Result，但不包含待重新采样的后续 Agent 回复。
- entropy metrics 绑定同一 Window；不能使用整条父轨迹的最终滑动平均替代具体窗口指标。
- 支持 Official ARPO entropy Gate 和明确的 `always` 验收 Gate；其他 Gate 必须由 Adapter 显式声明。
- `initial_rollouts`、最终 group budget、Site beam 和 max depth 由 Adapter/Core 合作解析，不能分别由 UI、overlay、Runner 和 Daemon决定。
- child rollout 保留 parent/window/site/snapshot/decision 身份，并继续参与现有 actor training。
- `after_agent_turn(hub)` 仅作为 legacy 兼容输入，迁移到一个或多个明确 Tool selector；不得作为 ARPO 原生机会。
- 服务器验收必须证明有效模型响应、窗口命中、child enqueue、参数更新和终态。

## Scope

- In：ARPO capability、Tool Window 生产、entropy Decision、预算 overlay、二波 enqueue metadata、Preflight 和服务器验收。
- In：保留 messages Snapshot 与 AGL ToolMessage 兼容协议。
- Out：普通 Agent、Router、Verifier 和 token Window。
- Out：AEPO/RAE/IGPO/GIGPO。
- Out：Rollout Tree 页面和节点 credit。

## Current State

- `tir_agent.call_tools()` 已在 Tool Result 后记录 messages Window。
- 下一次 `call_model()` 会为前一窗口补 entropy 指标。
- 第一轮实施已使不同 Tool Window 使用不同消息前缀。
- `ActiveSetSession` 仍直接持有 ARPO 参数并决定 Gate/预算，尚未成为正式 Adapter。
- `arpo_e2e` 仍同时保存明确 Tool Site 和 `after_agent_turn(hub)` 兼容 Site，UI 容易误解为两个原生位置。

## Proposed Architecture

### ARPO Adapter Capability

```python
class ArpoSamplingAdapter(SamplingStrategyAdapter):
    id = "arpo"

    def opportunities(self, workflow):
        return every_agent_tool_result_relation(workflow)

    def allowed_gates(self):
        return {"entropy_delta", "always"}

    def decide(self, window, site, policy):
        return official_arpo_decision(window.metrics, site.gate)

    def allocate(self, decision, budget, site):
        return min(site.beam_or(policy.beam_size), budget)
```

AEPO/RAE 特有 probe、verdict 和 credit 不进入该 Adapter。

### Tool Result Window

对于：

```text
Hub → execute_python
```

Window 表示：

```text
owner_agent_id = hub
kind = tool_result
interaction.tool_id = execute_python
snapshot = [问题, Hub Tool Call, Tool Result]
```

Decision 使用 Hub 收到结果后生成下一回复时得到的 entropy；child 从 Snapshot 继续生成 Hub 后续回复。

### Legacy Migration

Preflight 与 UI 将 `after_agent_turn(hub)` 标为 legacy compatibility。提供明确转换：

```text
任意 Hub Tool 返回
→ 为 Hub 当前可调用的每个 Tool 建立 after_tool selector
```

转换必须由用户确认；不能无损证明的旧语义继续保留并阻断或警告，不静默展开。

## Files and Entry Points

| 区域 | 调整职责 |
| --- | --- |
| `mas/workflow/sampling/adapters/arpo.py`（拟新增） | ARPO capability、Decision 和预算 |
| `mas/tir_agent.py` | 产生 Tool Result Window 与窗口 entropy |
| `mas/workflow/runtime.py`、`archive.py` | 保存对应 messages Snapshot |
| `mas/workflow/active_set.py` | 调用 Adapter/Core，不内嵌 ARPO |
| `rl/hooks/overlay.py`、`daemon.py` | ARPO overlay 与 ExpansionPlan enqueue |
| `science_infra/control/training.py` | ARPO capability Preflight |
| `experiments/arpo_e2e` | 服务器验收 fixture；不在验收后遗留临时配置 |

## Action Items

- [x] 实现 ArpoSamplingAdapter 并注册。
- [x] 将 Tool Result Window/entropy 指标切换到统一核心合同。
- [x] 从 ActiveSetSession 移出 ARPO Gate 和预算分支。
- [x] 统一 `initial_rollouts`、group budget、beam 和 max depth 的有效解析。
- [x] 保留 parent/window/site/snapshot/decision metadata 到 child rollout。
- [x] 将 `after_agent_turn(hub)` 从原生能力降为 legacy 迁移项。
- [ ] 完成服务器 baseline 与双 Tool Window 验收。

## Implementation Notes

- 新增正式 `ArpoSamplingAdapter`，注册名为 `arpo`；它只支持 `tool_result` Window，只允许 `entropy_delta` / `arpo` / `always` Gate。AEPO/APPO/RAE 继续留在 compatibility adapter，未借用 ARPO 的完成状态。
- Tool Result opportunity 生成提取为共享模块；ARPO 与 compatibility adapter 可复用 Workflow compiler 的 Agent→Tool 关系，但能力和 Gate 集合分别声明。
- ARPO Decision 复用 Official ARPO gate primitive；缺少 `h_root/h_tool` 或使用 `dual_entropy` 等不支持 Gate 时明确返回不通过 Decision，不创建 child。
- `apply_sample_policy()` 通过 Adapter registry 应用 training overlay，将 `algorithm.tir.sampling_strategy=arpo` 固定进 effective RL；Runner 不再从深层 `tir_algo` 猜策略。
- Daemon 在第一波样本与预算重分配中注入 `sampling_strategy`；ExpansionPlan/ForkPlan 的 `window_id`、`snapshot_ref` 和完整 Decision 继续传入 child sample。
- LitTirAgent 使用任务中的 strategy 创建 ActiveSet Core，并在 annotation 中记录 strategy、Window、Snapshot 和 Decision reason，不记录原始前缀。
- `arpo_e2e` 本地配置解析已确认：`tir_algo=arpo`、`sampling_strategy=arpo`、`rollout.n=4`；原生 capability 只有明确的 `execute_python` Tool Site，旧 Hub Site 进入 legacy 列表。
- 本阶段新增快速纯测试，覆盖 ARPO Tool Result capability、Official entropy Decision、beam 分配、不支持 Gate、overlay 和 child metadata；未自动启动 GPU 训练。

## Manual GPU Acceptance

1. 基线 `arpo_e2e`：模型返回有效 token，至少一个 Tool Result Window 可判定并创建 child。
2. 双 Tool fixture：两个不同 Tool Result Window 具有不同 Window ID、Snapshot 和前缀长度。
3. child 真实入队并完成，日志越过 `compute_log_prob`，产生 `actor/pg_loss` 或 `global_step`。
4. 停止时目标进程和已识别子进程退出，可以再次启动。
5. 记录 run ID、Git commit、有效配置和结果；不把一次随机未通过 Gate 当作执行失败。

## Lightweight Validation

默认只编译受影响 Python，并用纯 fixture 验证 capability、Decision 和预算；不自动启动 A800、Ray、VERL 或 vLLM。真实验收由用户从 WebUI 明确触发。

## Acceptance and Delivery Boundary

- ARPO Adapter 只返回 Tool Result Window。
- Tool 节点不被描述为“被分支”；文档和 metadata 表达的是上游 Agent continuation。
- 明确 Tool Site 与 Window/Snapshot/Decision 一一对应。
- legacy Agent Site 不再与同一 Tool Event 形成重复原生 Decision。
- 完成正常训练 step 和停止场景后，ARPO Adapter 才能标为服务器已验证。

## Risks and Edge Cases

- 一次 Agent 响应并行调用多个 Tool 时，需要明确一个组合 Window 或拒绝具名 Tool selector，不能猜测某一个 Tool。
- 模型未调用目标 Tool 时没有 Window，这不是错误，也不能补全独立 rollout 为假 branch。
- entropy 属于 Tool Result 后的 Agent continuation，必须绑定正确 sequence。
- child 的 Tool Result 前缀可能包含敏感输出，公开日志只记录引用与摘要。

## Next Phase

[第三阶段：适配器驱动画布交互](Sampling框架第三阶段-适配器驱动画布交互.md)。
