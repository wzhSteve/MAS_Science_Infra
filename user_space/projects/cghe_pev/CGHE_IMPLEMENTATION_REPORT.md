# CGHE 实现报告

## 做了什么

Step-1 的 Coverage 改为 ACTIVE 路径条数：计划的静态首节点是 v 才计数。`out_degree` 仍只表示直接后继个数，不再用来选节点。

`planning_graph.py` 增加 `compute_step1_coverage`、`select_step1`、`coverage_saturated`、`regeneration_reasons` 和每 DAG 的 `attempt_count`。平局顺序是尝试次数少、从未尝试优先、稳定 `node_id`。构图后 Coverage 全是 1 且还没有真实调用时不重生成。

`cghe_loop.py` 持有全局 Step-1 次数。新 DAG 把本图 `attempt_count` 归零，不归零全局次数。两个触发原因只安装一张新图。Planner 或 schema 失败不记 attempt。未验证失败在重生成提示里标明不是硬约束。

现场失败不再调用 `prune_unique_downstream`。只有 `HypothesisPruner` 准入的 `VERIFIED_COUNTEREXAMPLE` 才把显式依赖的计划标成 `PRUNED`。不再被任何 ACTIVE 计划引用的 PENDING 节点标成 `ORPHANED`。成功节点保留。

Step-1 成功不拆图，也不把这一步当成任务结束。剩余步骤交给普通 PEV。整题停止时才把相关计划标成 `COMPLETED`。CGHE 关闭时不进入这套选择。

`contracts/runtime.yaml` 的 `cghe` 段增加 `max_step1_attempts: 3`、`max_regenerations: 2`、`max_total_step1_attempts: 12` 和 `selection.strategy: max_step1_path_coverage`。因果边写在现有 `EvidenceMemory` 上。`OBSERVED_TO_CAUSE` 没有直接证据时不写入。

## 修改的文件

- `adapted/cghe_types.py`
- `adapted/planning_graph.py`
- `adapted/cghe_loop.py`（新）
- `adapted/evidence_memory.py`
- `adapted/context.py`
- `adapted/recovery_controller.py`
- `adapted/sft_prompts.py`
- `adapted/test_cghe_spec.py`（新）
- `adapted/test_cghe_core.py`
- `adapted/test_planner_graph.py`
- `contracts/runtime.yaml`

## 测试

在 `adapted/` 下执行：

```text
python -m unittest test_cghe_spec.py test_planner_graph.py test_cghe_core.py test_cghe_recovery.py test_cghe_integration.py v2.test_v2
```

结果：81 项通过。

本地工具冒烟：`run_python_locally("print(2 + 2)")` 返回 `4`。没有把这次调用写成任务成功率。

## 未实现或仍偏离的部分

- v2 的 EPV 模块还在，但现场 `_select_node` 不再用它打分。
- `PlanStatus.RETIRED` 仍保留给旧的实例级剪枝写入。新的 Step-1 反例路径写 `PRUNED`。
- 集成夹具 `test_explicit_no_prunes_then_continues_on_the_selected_node` 仍期望 Bing 计划离开活跃集后再走 Web_Fetch。该夹具的工具结果是 `TOOL_NOT_FOUND`，走的是恢复控制器里已经准入的反例，不是图上的 `prune_unique_downstream`。
- 因果边只存在进程内的 `EvidenceMemory._causal_edges`，不另建数据库表。重启后边本身不在 sqlite 里，证据记录仍在。
