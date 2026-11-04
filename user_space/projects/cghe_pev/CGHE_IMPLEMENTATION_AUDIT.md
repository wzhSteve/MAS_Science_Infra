# CGHE 实现审计

审计对象是改动前的 `user_space/projects/cghe_pev/adapted`。合同是 `docs/ACL2027/CGHE_DAG_Implementation_Specification.md`。Coverage 必须是 ACTIVE 计划里静态首节点为 v 的路径条数。直接后继个数不是这个数。

## 调用链

入口是 `EpisodeContext.plan_window` / `exec_window` / `verify_window`（`context.py`）。

CGHE 打开时，`plan_window` 先锁已有的 `continue_execution`，否则 `_open_layered_plan`：`build_outlines_oneshot` 建 `PlanningGraph`，再用 `_graph_select_policy()` 选节点。失败才走 `_plan_cghe_step` → `RecoveryController.begin_round`。

执行经 `_execute_authorized`。验证经 `_cghe_on_verdict`。图上带 `cghe_node_id` 时，`_switch_after_graph_failure` 在 `handle_failure` 之前返回。

## 与规格的差异

1. 现场选择键是 `(progress_rank, -open_url_score, -out_degree, -sharing_score, cost, node_id)`。`out_degree` 是 PENDING 直接后继个数。`sharing_score` 是路径上任意位置包含该节点的计划数。候选是当前前沿，不是静态 Step-1。`test_max_out_degree_prefers_bushier_step1` 里 3 个后继恰好等于 3 条路径，不能当成 Coverage。
2. `RecoveryController._select_node` 在 v2 用 EPV 分数，在 v1 用 frontier 或 first_plan。
3. 超时、执行错误、extract 的 NO、空检索调用 `prune_unique_downstream`，不经过 `HypothesisPruner` 的验证反例门。分析器本身会把普通失败停在 `UNVERIFIED_HYPOTHESIS`。
4. 验证器 YES 时 `_advance_after_success` 把 `controller.graph` 置空，下一轮不带证据重建图。
5. 没有 `attempt_count`、覆盖饱和、`COMPLETED` / `ABANDONED`。重开图受 `recovery_rounds`、`replans`、`tool_calls` 约束，而且分层构图不走 `begin_round`。
6. 可保留的部分：按执行语义合并节点、`FailureAnalyzer`、`HypothesisPruner._global_admission`、`EvidenceMemory` 的成功 / 反例 / 诊断三分、CGHE 关闭后的纯 PEV。

## 计划状态

改动前 `PlanStatus` 只有 `ACTIVE | PRUNED | RETIRED`。覆盖度没有只读 ACTIVE 首节点的函数。
