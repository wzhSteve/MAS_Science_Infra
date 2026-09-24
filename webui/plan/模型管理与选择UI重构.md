# 模型管理与选择 UI 重构

本方案只解决实验画布中的模型选择混乱，不重做画布布局，不拆分“运行模型”和“训练模型”，不改变训练数据、Sampling、RL、GPU、Monitor、Harness 或 Rollout Tree。

## Core Decision

最终模型：

```text
每个 Agent 始终有一个模型
        +
trainable 只决定该 Agent 的模型调用是否进入优化
```

数据合同保持与 `docs/New_framework_design.md` 一致：

```yaml
agents:
  - id: hub
    model: inherit
    trainable: true

  - id: planner
    model: model_external_reasoner
    trainable: false
```

不新增：

- `training_model_id`
- 永久 `active_agent` UI
- “统一训练模型 / 按 Agent 指定”
- “执行模型 / 训练模型”双字段

## Concepts

### Agent Model

每个 Agent 都需要模型。即使 `trainable=false`，它仍然参与 Workflow 执行。

```text
hub       Qwen3-4B
planner   External Reasoner
verifier  Qwen3-4B
```

### Inherit

`model=inherit` 表示使用实验默认模型。

```text
实验默认模型 = Qwen3-4B

hub.model = inherit
  → Qwen3-4B

verifier.model = inherit
  → Qwen3-4B
```

继承本身就是批量配置的快捷方式，因此不需要额外的“统一模型模式”。

### Trainable

`trainable=true`：

```text
Agent 正常执行
Agent 的模型调用进入训练优化范围
有效模型必须具备本地可训练权重
```

`trainable=false`：

```text
Agent 仍正常执行
模型参数冻结
可以使用本地模型或 API 模型
```

`trainable` 不决定 Agent 是否参与 Workflow，也不决定 Agent 是否需要模型。

## Target UI

保持当前：

```text
左侧训练配置 + 中间 MAS 画布 + 右侧节点 Inspector
```

不增加新的顶部模型工作区，不改变训练数据、训练策略和执行资源布局。

### Left Training Panel

```text
01 模型与训练范围

实验默认模型
[Qwen3-4B ▼]

Agent 模型与训练状态

hub
Qwen3-4B · 继承                  参与训练

planner
External Reasoner · API          已冻结

verifier
Qwen3-4B · 继承                  已冻结
```

左侧只做总览：

- 选择实验默认模型。
- 查看所有 Agent 的有效模型。
- 查看每个 Agent 是参与训练还是冻结。
- 点击 Agent 行定位到画布节点并打开 Inspector。

删除：

- 本次训练 Agent。
- “一次训练只优化一个 Agent”的辅助说明。
- 使用统一训练模型。
- 按 Agent 指定模型。
- 初始权重。
- 现有权重路径。
- 独立执行模型。

### Agent Inspector

所有 Agent 始终显示：

```text
模型
[继承实验默认模型 · Qwen3-4B ▼]

参与训练 [✓]
```

非训练 Agent：

```text
模型
[External Reasoner · API ▼]

参与训练 [ ]

该 Agent 仍参与 Workflow 执行，但模型参数保持冻结。
```

训练 Agent：

```text
模型
[Qwen3-4B · 本地 · 可训练 ▼]

参与训练 [✓]

该 Agent 的模型调用进入训练优化。
```

### Model Filtering

`trainable=false`：

- 显示所有可推理模型。
- 包括 API 和本地模型。

`trainable=true`：

- 只显示具备可训练权重的本地模型。
- 当前模型是 API-only 时保留原值并显示错误。
- 不自动替换模型。
- 不静默关闭或开启 Trainable。

### Agent Card

卡片显示模型和优化状态：

```text
hub
Qwen3-4B · 参与训练
```

```text
planner
External Reasoner · 已冻结
```

```text
verifier
继承默认 · 已冻结
```

工具数量、入口状态和模型状态不能互相覆盖。

## Model Options

模型选项由现有资源和本地扫描合并：

```text
inference resources
  → API 或本地推理模型

training resources
  → 本地可训练模型

SCIENCE_MODEL_ROOTS / <repo>/LLM
  → 自动发现本地模型
```

扫描模型被选择时：

```text
校验 allowed Root
→ 复用已有 training resource
→ 不存在则自动创建
→ 返回稳定 resource ID
```

UI 不显示“登记并绑定”。

## Runtime Resolution

### Ordinary Execution

```text
Agent.model = inherit
  → 使用实验默认模型

Agent.model = resource_id
  → 使用对应资源模型
```

Graph Runtime 在进入每个 Agent 工作窗口时解析对应 LLM：

```text
run_compiled_episode
  → current Agent
  → agent_llms[current]
  → fallback experiment default LLM
```

因此冻结 Agent 的 API 模型仍会真实参与 Workflow，不只是 UI 标签。

### Training

训练范围直接来自：

```text
agents[].trainable == true
```

没有第二套 active Agent 配置。

训练 Resolver 对每个 trainable Agent 解析其有效模型：

```text
显式 resource_id
  → 对应模型

inherit
  → 实验默认模型 / 兼容 training binding
```

当前单个训练进程只有一套 Policy/Optimizer：

- 多个 trainable Agent 使用同一模型时，可以在同一 run 中按 Agent span 过滤并共同优化该模型。
- 多个 trainable Agent 使用不同权重时，Preflight 必须阻断，不能静默只训练第一个。
- 后续顺序训练调度器应把不同模型拆成独立阶段和独立 checkpoint。

当前交付不再使用 `trainable_agents[0]` 作为隐式选择。

### Agent Attribution

每个 Agent 的 LangGraph LLM 节点使用：

```text
agent:<agent_id>
```

Adapter 过滤表达式：

```text
^agent:(hub|planner)$
```

保证只消费 `trainable=true` Agent 的模型 span，冻结 Agent 的调用不进入梯度优化。

## Backend Contract

### Workflow

继续使用：

```python
class AgentNodeSpec(BaseModel):
    model: str = "inherit"
    trainable: bool = True
```

不增加训练专用模型字段。

### Preflight

返回：

```json
{
  "effective": {
    "trainable_agents": ["hub", "planner"],
    "active_agents": ["hub", "planner"],
    "model": {
      "resource_name": "Qwen3-4B",
      "model_path": "/root/.../Qwen3-4B"
    }
  }
}
```

检查：

- 至少一个 `trainable=true` Agent。
- 每个 trainable Agent 的有效模型存在。
- 有效模型具备可训练权重。
- 本地路径和 `config.json` 有效。
- 同一训练进程中的 trainable Agent 使用同一模型权重。
- Adapter 过滤包含全部 trainable Agent。
- 冻结 Agent 不进入训练 span。

### Multiple Models

若 trainable Agent 使用不同模型：

```text
hub      Qwen3-4B
planner  Qwen3-8B
```

单个训练进程不能同时优化两套权重。当前必须明确阻断：

```text
多个参与训练的 Agent 使用了不同模型；
当前单个训练进程只能优化一套权重，请统一模型后启动。
```

后续顺序训练调度器应生成：

```text
阶段 1：hub · Qwen3-4B
阶段 2：planner · Qwen3-8B
```

每个阶段独立 Preflight、日志、checkpoint 和 snapshot。

## Compatibility

- 旧 `agent.model=inherit` 继续读取。
- 旧自由字符串若不是资源 ID，保留并显示“不可用/旧配置”，不静默覆盖。
- 旧 training binding 继续作为默认本地训练模型。
- 旧 `rl.model_path` 在没有 binding 时继续回退。
- 旧 inference binding 和 `llm.yaml` 继续提供普通执行默认模型。
- 新 UI 不删除旧字段，不破坏历史 Workflow。

## Non-regression Boundary

本次只修改：

- Agent 模型选择。
- Trainable 能力约束。
- 左侧模型与训练范围总览。
- 普通执行按 Agent 解析 LLM。
- 训练按 trainable Agent 过滤 span。

保持不变：

- 画布拖拽、缩放、节点和连线。
- Agent role、prompt、skills、tools 和 memory。
- Router 行为。
- Sampling Sites、Gate、ARPO/GRPO。
- 数据集、GPU 和 Runner。
- Monitor、日志、Rollout Tree、Harness。
- 历史 run 和 snapshot 读取。

## Files

| 文件 | 调整 |
| --- | --- |
| `TrainingSettings.tsx` | 实验默认模型与 Agent 模型/冻结状态总览 |
| `NodeInspector.tsx` | 所有 Agent 的模型选择与 Trainable 开关 |
| `AgentNode.tsx` | 模型和训练状态摘要 |
| `AgentModelSelect.tsx` | 统一模型选项、能力过滤、本地自动登记 |
| `workflowGraph.ts` | `model` 无损序列化 |
| `mas/workflow/runtime.py` | 每个 Agent 使用对应 LLM |
| `science_infra/control/training.py` | 按 trainable Agent 解析训练范围和模型 |
| `mas/tir_agent.py` | `agent:<id>` trace 节点 |
| `mas/train_tir_agent.py` | 多 Agent span 过滤 |

## Acceptance

- 现有画布布局不变。
- 页面不再出现“本次训练 Agent”。
- 页面不再出现“统一训练模型 / 按 Agent 指定”。
- 每个 Agent 始终显示模型选择。
- Trainable 只控制优化/冻结。
- 冻结 Agent 可以使用 API 模型并继续执行。
- Trainable Agent 只能使用可训练本地模型。
- 继承表示实验默认模型，不是默认训练模型。
- 左侧展示所有 Agent 的模型和训练状态。
- 普通执行真实按 Agent 使用模型配置。
- 训练不会再静默选择第一个 Agent。
- 多个 Agent 使用同一模型时按真实 Agent span 训练。
- 多个 trainable Agent 使用不同权重时明确阻断。
- 其他画布、Sampling、RL、数据、GPU 和监控功能保持不变。
