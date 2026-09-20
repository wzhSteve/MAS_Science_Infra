# Branch Site 设计：可视化站点 + RAE 子轨迹选取与 Reward

日期：2026-09-16

对照：[ROLLOUT_SAMPLING.md](./ROLLOUT_SAMPLING.md)、[SAMPLING_ARPO_APPO.md](./SAMPLING_ARPO_APPO.md)。

**UI 验收**：[BRANCH_ROLLOUT_UI_TEST.md](./BRANCH_ROLLOUT_UI_TEST.md)（`./run.sh branch-ui-test`）。

## 1. 目标

1. MAS UI 在 workflow 图上展示可挂 Branch 的候选，用户勾选站点与门控（熵 / verifier 成败等）。
2. 运行时按 `sampling.sites` 在多屏障处分叉，而非写死「首次 tool」。
3. 子轨迹选取对齐 RAE：探针 $K$、结局熵 $\bar H^B$、$U=\bar H^\pi\bar H^B$。
4. Reward 匹配父/子：R0 前缀置零 → R1 validate/invalidate → $A^{\mathrm{RAE}}$。

## 2. 合同（`SamplePolicy.sites`）

见 [`mas/workflow/contracts.py`](../mas/workflow/contracts.py)：`BranchAnchor` / `BranchGate` / `BranchForkSpec` / `BranchSiteReward` / `BranchSite`。

旧 `barriers: ["after_tool"]` 在无 `sites` 时自动派生默认站点。

## 3. UI

- **主交互**：MAS 页 **Rollout Sampling** 小窗（[`webui/src/features/sampling/`](../webui/src/features/sampling/)）——由 workflow 推导简化轨迹（start→agents→verify→end），在 agent / tool / verifier 屏障勾选 branch 与 gate，写回 `sampling.sites`。
- 主画布只编辑 MAS 拓扑；Inspector「Branch Rollout 站点」为 Advanced 同步列表。
- SamplePolicy 的 mode / group_n / beam 并入小窗顶栏。

验收：[BRANCH_ROLLOUT_UI_TEST.md](./BRANCH_ROLLOUT_UI_TEST.md)（`./run.sh branch-ui-test`）。

## 4. 运行时

```text
事件 → 匹配 site → Gate →（dual_entropy / rae_adjudicate 则探针）→ ForkPlan
  → Daemon meta 透传 → R0 / verdict / A^RAE
```

- Gate：`entropy_delta` / `dual_entropy` / `verifier_*` / `tool_*` / `always` / `contradiction`
- Reward：`scalar_grpo`（R0）| `rae_adjudicate`（R1-lite 或 `tir.rae_full_tgt` 完整 π_tgt）
- Ready-batch：`ActiveSetScheduler` + 并行 tool；Daemon 在 `ready_batch` 下增量 enqueue

## 5. 阶段

| 阶段 | 状态 |
|------|------|
| 合同 + gates/probes | ✅ |
| UI 角标 + Inspector | ✅ |
| **Rollout Sampling 轨迹小窗** | ✅ |
| Barrier 多站点分叉（ForkPlanner） | ✅ |
| R0 + `tir_algo=rae` + meta/verdict 闭环 | ✅ |
| Runner ActiveSet ready-batch（异步 tool + ready 聚合） | ✅ |
| Token 站点 + 完整 π_tgt KL | ✅（`on_token` / `token_ids` / `rae_full_tgt`） |

说明：训练热路径仍保持「一 Store 任务一条轨迹」；ForkPlanner 产 plan，Scheduler 用于 Collect/local；真 vLLM Worker 共置不在本仓范围。
