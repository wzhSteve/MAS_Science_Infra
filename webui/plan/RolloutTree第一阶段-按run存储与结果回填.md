# Plan

本阶段把现有“分支计划树”补为按训练 run 归档的事实树：记录题目、初始 rollout、实际入队的分支及其完成状态与 outcome。继续使用既有 Store/ForkPlan 执行训练，树是这些事实的结构化投影，不把展示 JSON 反过来当成训练输入。

依赖[采样编排第一阶段：可执行站点与续跑前缀](采样编排第一阶段-可执行站点与续跑前缀.md)的事件/快照身份；采样 UI 可按[第二阶段](采样编排第二阶段-画布与策略编辑.md)并行接入。展示和实时更新留给[下一阶段](RolloutTree第二阶段-节点详情与实时更新.md)。

## Position in the Project

[new_framework 设计](../../docs/NEW_FRAMEWORK_DESIGN.md) §2.4 定义“query 是根、分支路径构成树、叶子有 outcome”。当前 `RolloutTree` 合同和 Daemon 写树是起点，不是完整实现；节点级 reward/RAE verdict 必须来自真实训练结果而非浏览器重算。

```text
本次 run 的任务/组 → query root
                 ├─ 初始 rollout A → 从站点 S 续跑的 child → outcome
                 └─ 初始 rollout B → 独立 outcome

Store 的真实 rollout / reward + ForkPlan 的 parent/site/snapshot
                 → 每次状态变更更新同一棵 run 级树
```

## Requirements

- 树身份由 `experiment_id + run_id + task/data_id + group` 确定；跨实验、跨运行的同名题目不合并，不能用用户输入直接作文件名。
- query root 与初始 rollout 是不同节点；parent/child 关联真实 Store rollout ID，计划中尚未入队的分支不得冒充完成节点。
- 节点区分 `planned / enqueued / running / succeeded / failed` 等已知状态；没有真实 outcome 时保持缺失，不以默认负分或空答案填充。
- 结果来自实际 rollout 的最终答案/奖励；RAE verdict 与节点 credit 仅在实际算出后回填，保留 rollout 级 reward 与节点级 credit 的不同含义。
- 节点记录 `site_id`、边界事件身份、可恢复前缀的引用和可用的 Gate 指标；不在公开树中写原始消息、密钥或完整 Archive 快照。
- 新增字段向后兼容现有 `RolloutTree` JSON；旧 `.local_expansion` 的 `{tree, plans}` 与平铺计划读法保持训练兼容。
- 写失败必须留下可诊断错误，不能使训练在树展示失败时被误标成功或无声丢树；同时不因 UI 投影失败中断已有训练计算。

## Scope

- In：树身份/节点状态、实际入队与完成事件的归属、run 级存储与读取合同、旧产物兼容。
- In：没有发生 branch 的题目也可有 query 与独立 rollout；明确标为“无分支”，不把 Collect 的独立采样视为训练分支。
- Out：变更 Store/VERL 的执行策略、让训练从 JSON 树恢复或计算 advantage、重新训练六种算法。
- Out：大规模索引数据库、跨服务器共享文件系统和一口气加载所有树的 UI。

## Current State

- `mas/workflow/contracts.py` 已有 `RolloutTree`/Node/Event，但 `tree_from_plans()` 当前以父 rollout 作为 root；`expansion_payload_from_result()` 未传 task，query 通常为空（`active_set.py`）。
- Runner 写 `plans + tree` 到 `.local_expansion`；Daemon 入队后替换合成子节点 ID，并写全局 `tree_<tree_id>.json`。写入点仅覆盖计划/入队，不会在完成后持续回填树的 outcomes、rewards 和 verdicts（`rl/hooks/daemon.py`）。
- `rl/hooks/rae_advantage.py` 有纯函数 `apply_verdicts_to_tree()`，并不等于当前训练路径已调用它持久回填。
- `/api/mas/rollout-trees` 扫描共享目录；参数 `experiment_id` 仅回显，未用于隔离数据。运行本身已有 `artifacts/runs/<run_id>/status.json`、日志和配置快照。

## Proposed Architecture

### 一份树合同，两种职责

保留 `RolloutTree` 模型，增量加入版本/身份和节点状态。query root 用稳定的合成 ID，初始及分支 rollout 节点使用实际 Store ID；入队前如需展示计划，则用独立计划身份，入队后显式关联真实节点，不能直接把合成 ID 当最终结果。`tree_id` 由受控的 run/task/group 身份生成，与 API 和文件路径一致；旧格式解析只作兼容，不能无依据补上 run ID。

父子边表示“从父 rollout 的 Site/窗口前缀续跑”，而不是把两个相似答案误判为 branch。叶子 outcome 仅在已完成时给出；节点记录 provenance（实际事件、`site_id`、Archive 引用）、指标和判定来源。若此阶段尚无节点 credit，保留 `null` 并显示“未计算”。树投影供 UI/Harness 读取；后续若要用于训练 credit，必须另行核对与 Store 奖励的一致性并明确迁移，不能先让两个版本各算一套。

### 归档和读取

使用现有运行目录下的受控路径：

```text
experiments/<id>/artifacts/runs/<run_id>/rollout-trees/<tree_id>.json
```

由 Control 为启动的子进程传入只属于本次 run 的目录/身份，Daemon 在计划、入队、完成时更新快照；旧 `.local_expansion` 继续供 ForkPlan/旧工具读取，二者职责分开。树文件采用原子替换和同 run 内的更新序列，防止并发更新覆盖或读到半写 JSON；投影失败记录明确日志及状态供重试，不用裸 `except: pass`。不要把全量题目回答/消息写到独立公共索引。

新增受验证的按 run 列表及单树只读 API，例如：

```text
GET /api/rl/runs/<run_id>/rollout-trees?experiment_id=<id>&limit=...
GET /api/rl/runs/<run_id>/rollout-trees/<tree_id>?experiment_id=<id>
```

沿用 `training_run()` 的 experiment/run 归属校验；列表分页且只返回摘要。旧 `/api/mas/rollout-trees` 暂保留为 legacy 兼容，不用它支撑新页面，也不能再把混合历史结果标成当前 run。

## Files and Entry Points

| 区域 | 调整职责 |
| --- | --- |
| `mas/workflow/contracts.py`、`active_set.py` | 增量定义 query root、节点状态及 ForkPlan 来源 |
| `mas/workflow/archive.py` | 提供真正可恢复的引用，不在树中复制私有快照 |
| `rl/hooks/daemon.py` | 计划/入队/完成/判定各阶段回填实际树，归入本次 run |
| `science_infra/control/training.py`、`process_manager.py` | 给子进程传递受控 run 身份和目录，不改训练入口 |
| `science_infra/control/app.py` 或独立树路由 | 归属校验、分页索引和按树读取 |
| `mas/tests/test_rollout_tree.py` | 两个 run 同名题、真实 parent ID、空 outcome、终态和兼容读取 |

## Action Items

- [ ] 定义版本化的 query/task/group/run 身份与节点状态，兼容旧树格式。
- [ ] 为初始 rollout、计划分支、真实入队和完成结果建立不混淆的映射。
- [ ] 回填实际 outcome/reward/verdict，保留缺失值与其来源。
- [ ] 原子、可诊断地写入 run 专属树，不搬走训练仍读取的 expansion 文件。
- [ ] 提供经过实验/run 归属校验的分页列表与单树读取 API。

## Lightweight Validation

用无 GPU 的 Store/Daemon 替身或纯数据 fixture 覆盖“两个独立初始 rollout、一个真实 branch、失败 child、最终 reward 回填、另一 run 同名题、重复事件与进程重启后读取”。只运行受影响 Python 契约检查；真实训练由用户明确触发。

## Acceptance and Delivery Boundary

- 指定 experiment/run 的树列表只返回本次运行，旧全局目录里的树不混入。
- root 是 query，初始节点与分支节点可区分；子节点 parent/site/snapshot 指向同一次实际入队。
- 没有结束的节点不显示 outcome；完成后真实 reward/verdict 可回填并在刷新后持久存在。
- 训练仍读取原有快照、expansion 与 Store；本阶段不宣称树已驱动训练 credit，也不宣称 UI 实时。

## Risks and Edge Cases

- 同一个 `data_id` 可在不同 run 或组中出现；树键必须包含组和 run，而不是仅用父 rollout ID。
- 多个 Worker 的结果可能晚于 enqueue，也可能乱序；更新需要幂等和防覆盖机制。
- 旧文件没有可信 run 身份，不能通过目录扫描或时间猜测归属。
- 已入队但进程中断的 child 保留最后已知状态，不假装成功或丢弃。

## Next Phase

[第四阶段：树详情与实时更新](RolloutTree第二阶段-节点详情与实时更新.md)。
