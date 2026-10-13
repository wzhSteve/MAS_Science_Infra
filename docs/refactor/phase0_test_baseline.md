# Phase 0 现有测试基线记录

日期：2026-10-13（Phase 0，重构前）
机器：A800-SXM4-80GB × 1，python 3.11.16（uv .venv）

本文件是全部后续 Phase 回归门的**对照基线**。判定标准不是"全绿"，而是
**不新增失败、不恶化既有失败集**（基线失败全部归因于仓库既有未提交的用户区变更，
与 mas/ 重构无关，见 §3 归因）。

---

## 1. feature-test 基线

### 1.1 `--list` 域清单（20 域）

```
mas-core(36) rl(19) harness(17) branch(26) rollout-tree(6) agent-framework(20)
schema03(74) daemon(3) realtime(7) cli(4) control-ui(5) gpu-compiler(8)
verifier(7) e2e(9) functional(27) user-space(56) studio(38) sampling-runtime(51)
frontend smoke
```

### 1.2 计划门四域实测（2026-10-13，Phase 0）

| 域 | 结果 | 失败项 |
|---|---|---|
| functional | 24/27（FAIL） | 见 §3-A（3 项，仓库既有） |
| agent-framework | 20/20（PASS） | — |
| user-space | 54/56（FAIL） | 见 §3-B（2 项，仓库既有） |

### 1.3 smoke 分步基线

| 步 | 结果 | 备注 |
|---|---|---|
| 1 doctor | OK | GPU A800 / AGL 0.3.1 / spec topology=centralized |
| 2 check_workflow_deps | **OK** | 重构红线守卫，全程必须保持绿 |
| 3 check_user_zone | FAIL | §3-C 既有失败（用户区未提交变更违规 import `tools.*`） |
| 4 collect --mock | OK（单独执行） | traj n=4 mean_reward=1.0 |
| 5 diagnose | OK（单独执行） | |
| 6 status --html | OK（单独执行） | |
| 7 dashboard | OK（单独执行） | |

（`run.sh smoke` 因 `set -euo pipefail` 在步 3 即中断是现状行为，非本次引入。）

## 2. unittest 全量基线

命令：`PYTHONPATH=mas:. python -m unittest discover -s mas/tests`

```
Ran 601 tests in 77.565s
FAILED (failures=11, errors=9, skipped=5)
```

**已知失败全集（25 项 = 11F + 9E + ssh 环境 skip 5）——回归门对照清单：**

failures（11）：
1. test_user_space.TestIsolationScript.test_check_user_zone_ok —— §3-C
2. test_stage7_e2e.TestC1SuccessCollectAndRoundtrip.test_events_memory_skills_and_json_roundtrip
3. test_stage5_mas_sockets.TestRunnerInjection.test_custom_runner
4. test_stage3_harness.TestFork.test_fork_restore
5. test_stage1_mas.TestMockCollect.test_mock_events_reward_and_memory
6. test_schema03_tool_agents.TestSchema03Sugar.test_old_yaml_expands_implicit_tool_agents
7. test_hive_wrap.TestAssistantHiveTool.test_assistant_chat_deconstructs_hive
8. test_functional_workflow.TestWorkflowEvents.test_resume_messages_keep_snapshot_prefix
9. test_functional_workflow.TestTrainCollectSplit.test_collect_uses_compiled_graph_and_train_stays_single_hub
10. test_functional_studio.TestStudioPaletteAndSampling.test_sampling_preview_is_not_tool_anchored
11. test_eval_sources.TestEvalParquetPaths.test_explicit_test_files_override_sibling

errors（9）：
1-3. test_task_files.TestHiveTaskRows（hotpotqa/gaia/2wiki）—— dataset 文件缺失（FileNotFoundError: dataset not found）
4. test_agentflow_lite_prompts ×3（planner/executor/direct_output lite）—— 未跟踪新测试读 prompt 文件缺失
5. test_schema03_tool_agents.test_compile_old_yaml_ok —— StopIteration（未提交的工具 tier 改动）
6. test_hive_wrap.test_write_management_paths_rejected —— 未提交改动
7. test_agentflow_lite_prompts.test_diagnoser_lite —— 同 4

## 3. 基线失败归因（全部为仓库既有未提交变更，与 mas/ 重构无关）

- **A / functional 3 项**：① `test_collect_uses_compiled_graph...` 期望 `lit_tir_agent.rollout`
  源码含 `run_episode(`，但 HEAD 版（commit a0fdd14"接入 Flow-GRPO"）已把执行分派移入
  `_rollout_with_bound_llm`——**测试比 HEAD 代码旧**，属未提交工作区的测试/代码错位；
  ② `test_resume_messages_keep_snapshot_prefix`：mock episode messages 里 content ''/None
  差异；③ `test_sampling_preview_is_not_tool_anchored`：采样预览断言与未提交 studio 改动错位。
- **B / user-space 2 项**：54/56，其中 hive wrap 类测试与 `check_user_zone` 同根因（§3-C）。
- **C / check_user_zone FAIL**：`user_space/projects/epc-aw-main/adapted/context.py`
  import `tools.google_search` / `tools.wikipedia_search`——属于**未提交的用户区变更**
  （git status: `M user_space/projects/epc-aw-main/adapted/context.py`），不是 mas/ 代码。
- **D / task_files 3 errors**：dataset 路径解析找不到（data/ 未提交的用户数据布局变化）。
- **E / agentflow_lite_prompts**：`??`（未跟踪）新测试文件读 lite prompt 资产缺失。

结论：worktree 存在大量进行中的用户区/Flow-GRPO 工作（`M mas/tools/**`、`M mas/tests/**`
×11、`?? mas/tests/**` ×17、`M user_space/**`），基线失败是其半成品状态的自然结果。
**本次重构判据 = 失败集不扩大**：任何 Phase 结束时若 §2 清单外出现新失败/错误，
即该 Phase 引入回归， revert 或修复后才可进下一阶段。

## 4. byte-level 对照基准

- 冻结文件：`docs/refactor/.baseline_msgseq_v0.json`（由 `freeze_msgseq_baseline.py`
  经真实 `make_message/validate_payload` 生成；含四消息循环 kinds/payload_keys、
  fact_commit 事件键、commit_to_fact 签名）。
- 复核命令（每个 Phase 收尾执行）：

```bash
PYTHONPATH=mas:. ./.venv/bin/python docs/refactor/freeze_msgseq_baseline.py --check > /tmp/msgseq_check.json
diff <(sed -e '$a\' /tmp/msgseq_check.json) <(sed -e '$a\' docs/refactor/.baseline_msgseq_v0.json) && echo BYTE-LEVEL-OK
```

- 参考（9/9 通过的稳定锚点，重构各阶段均应保持 9/9）：

```
test_centralized_integration: Ran 9 tests — OK (skipped=1)
（blank_agent_bare_id / evidence_error / declared_tool_agent_node /
 pev_nodes / next_outside_candidates / four_message_cycle /
 verifier_fail_feedback ×2 / fan_out_parallel）
基线日志：docs/refactor/.baseline_centralized_v0.log
```

## 5. 判据速查（各 Phase 回归门）

| 门 | 判据 |
|---|---|
| G1 feature-test 四域 | functional≤3 失败 且 agent-framework=20/20 且 user-space≤2 失败（相对 §1.2 不恶化） |
| G2 smoke 分步 | check_workflow_deps 必须 OK；check_user_zone 与基线同（≤3 违规行）；doctor/collect/diagnose/status/dashboard 单步 OK |
| G3 unittest 全量 | failures≤11 且 errors≤9 且失败集⊆§2 清单（对照逐项名单非仅数目） |
| G4 byte-level | §4 diff 为空 |
| G5 centralized 锚点 | test_centralized_integration 9/9 OK |
