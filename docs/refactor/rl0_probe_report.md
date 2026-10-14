# RL0 前置基线与 WIP 裁决 probe 报告

日期：2026-10-14（RL0，RL 层重构开工前置）

执行依据：`docs/refactor/rl_refactor_plan.md` §4【RL0】节 + 附录 B（Q6 勘误）+ 附录 C（探查命令集）+ §6 Q1-a 裁决。
对照基线：`docs/refactor/phase0_test_baseline.md`（G1-G5 判据）。
体例参照：`docs/refactor/phase5_probe_report.md`。

本阶段产出 5 个 commit（421d66d / 3091fd1 / 1a3d4c2 / eae9d83 / 本报告 commit）。
GPU 门一律未跑（方案 Q2/Q3 前提未决：`experiments/arpo_e2e` 已删，本阶段不触碰 GPU 测试）。

---

## 1. 探查实录（附录 C 复跑）

### 1.1 工作区 WIP 裁决——两个时点

**时点 A：Q1 裁决执行前**（2026-10-14 上午，与方案写作时同状态）：

```
$ git status --short | grep -E 'rl/|agent-lightning/'
 M agent-lightning/agentlightning/verl/daemon.py     # 空 triplet 过滤（vendored AGL 内改，红线 A3 冲突）
 M rl/hooks/overlay.py                                # KL 显式存活语义（use_kl_in_reward 尊重显式键 / flow_grpo 尊重显式 use_kl_loss=False）
 M rl/rewards/outcome.py                              # strip_thinking 截断兜底 + numeric_match 多候选
?? mas/tests/test_outcome_hotpot.py                   # （未跟踪）配套 outcome 行为的既有测试雏形
?? scripts/prepare_hotpot20.py                        # （未跟踪，非本阶段范围）
```

**时点 B：Q1-a 裁决执行后**（本报告写作时，HEAD = eae9d83）：

```
$ git status --short | grep -E 'rl/|agent-lightning/'     # 空输出
$ git status agent-lightning/ --porcelain                  # 空输出 → A3 门达成
```

三处 WIP 逐个独立收编（见 §5 commit 清单），每 commit 前 `git diff <目标路径>` 确认改动面、
`git add` 逐路径、commit 后 `git show --stat HEAD` 自查零捎带。

### 1.2 热路径零改动存证

```
$ git diff --stat HEAD~7..HEAD -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py
（空输出）
$ git log --oneline af8bf71..57379f0     # MAS P0-P6 六个 commit（注：方案 §1 称"七 commit"，实数 6 条 oneline：p1-p6 + phase0）
fd0a30a refactor(p1) … e31a854 (p2) … 03c8724 (p3) … a0a3a71 (p4) … 1f2e458 (p5) … 57379f0 (p6)
```

RL0 三收编 commit 未触碰三件套（A1 保持）；`ref_Rep/` 零改动（A2 保持，全仓无 M 标记）。

### 1.3 rl→workflow 允许引用面（A7 白名单固化）

`rg -n "from workflow\.|import workflow" rl/ -g '*.py'` 全量 9 处收缩在白名单 7 模块：

| file:line | 引用 | 白名单归属 |
|---|---|---|
| `rl/train_signal.py:12` | `workflow.contracts.TrajectoryBatch` | contracts |
| `rl/hooks/rollout_tree.py:16` | `workflow.rollout_tree.RolloutTreeArchive` | rollout_tree 合同 |
| `rl/hooks/rollout_tree.py:17` | `workflow.rollout_results.{TERMINAL, read_results}` | rollout_results |
| `rl/hooks/daemon.py:20` | `workflow.active_set.{load_local_expansion, resume_boundary_from_messages}` | active_set |
| `rl/hooks/daemon.py:21` | `workflow.archive.load_resume_messages`（**shim 最后一个非测试消费点**，T6，RL2 直引项） | memory.archive（经 shim） |
| `rl/hooks/daemon.py:22` | `workflow.llm_diagnostics.log_model_route` | llm_diagnostics |
| `rl/hooks/daemon.py:25` | `workflow.rollout_tree.identify_plans` | rollout_tree |
| `rl/hooks/overlay.py:66` | `workflow.contracts.SamplePolicy` | contracts |
| `rl/hooks/overlay.py:80` | `workflow.sampling.registry.sampling_adapters` | sampling registry |

### 1.4 反向红线验证（A7 后半）

```
$ rg -n "centralized_runtime|ExecutionService" rl/ -g '*.py'
（零命中）
```

rl/ 不触碰 runtime 私有状态——守卫四机器证据成立，红线验证通过。

### 1.5 反向引用面（T9 拓扑）

`rg -n "from rl\.|import rl\." mas/ science_infra/ -g '*.py'`：生产文件 12 处
（train_tir_agent ×3、tir_agent ×2、lit_tir_agent ×1、runtime.py:27、collector.py:18、
gates.py:9、configured.py:7、rollout_runs.py:734、services.py:1316、training.py:23、
sampling/adapters/configured.py:7）+ **workflow 再导出面** `workflow/__init__.py:60-61`
（`__getattr__` 懒转发 `rl.loss`/`rl.train_signal` 到 services/cli 两消费点——登记入外部 API 清单，
删该面必须晚于 control/cli 改直引）。测试侧 `rg -n "from rl\." mas/tests/ | wc -l` = 72 行。

### 1.6 D2 三证复核

```
rg -n "policy_loss" ref_Rep/.../verl/trainer/config/ppo_trainer.yaml   → 零命中（config 树无 policy_loss 段）
rg -n "policy_loss" ref_Rep/.../verl/workers/actor/dp_actor.py         → :31 import / :391 注释 / :392 fipo_cfg / :396 dispatch
rg -n "set_struct|OmegaConf.merge" agent-lightning/.../interface.py   → :140 set_struct(False) / :141 merge
rg -n "def compute_policy_loss" ref_Rep/.../core_algos.py              → :458（vanilla clip）/ :688（future_kl）
```

写侧键位（HEAD 后）：`overlay.py:113-116`（use_kl_in_reward 缺省逻辑）、`:125-129`（arpo/appo loss_mode
preset）、`:144-147`（flow_grpo KL 强制两键）、`:171-176`（re-apply 保护）、`:183-190`
（loss_mode 复活保护与 tir 写入，`:186` loss_mode 写侧、`:190` loss_applied_to_actor）。
缺省漂移 `rl/loss.py:19` decay_rate=64.0 vs `dp_actor.py:404` 128.0 仍未接通（RL1 双写作业面）。

### 1.7 其余核心断言复核

- 八 preset 互斥分派：`advantage.py:71-112`（8 个 `if algo` 全确认，`return batch` 尾 :112 后）；
  `_apply_igpo` def 实测 **`:215`**（调用点 :106）——**方案附录 B#7 所记"def 实在 :219"与本次三法复核不符**，
  勘误按实测执行（设计文档原引 :215 本就正确，不改）；方案自身此条为探查偏差，如实记录（见 §7）。
- daemon 列注入清单 `:740-823`（10 列，无 resume_parent_id → P0-a 缺口确认）；
  grpo 提前 return `:696`（P0-b 确认）；`resume_parent_id` 素材点 `:360/:513/:646`。
- `multi_tool_bonus` 4 物理点：`outcome.py:154/165`（活默认参）、`overlay.py:29`（DEFAULT_TIR 死配置）、
  `lit_tir_agent.py:52/61`（死参数）。
- `hydraFields.ts`：`grep -c 'field('` = 24（含 :42 函数定义），条目实数 **23**（区间 `:57-79`）。
- `rlConfig.test.ts`：`it(` 6 用例（:27/40/54/70/76/90）冻结基线不变。
- `agentflow` preset 快照表（RL_design §3.4.4）经 `advantage.py` 逐行对照成立（flow_grpo :71 直通/arpo :74/
  appo :76/grpo :81/rae :84-102/aepo :104/igpo :106/gigpo :108）。
- AGL vendored daemon（已收编 HEAD）空 triplet 过滤落点：triplet 构造 `:834-850`、
  空批拦截 `:1033-1037`。
- `feature-test --list`：20 域清单与方案一致（rl 19/branch 26/rollout-tree 6/sampling-runtime 51/daemon 3）。
- `TIR_EXECUTION_TREE_VERSION`：`training.py:795` 恒发 "3"；`rollout_tree.py:37` env 读缺省 "2"（死缺省）——见 §4。
- `arpo_e2e` 依赖面：`run.sh:1281/1335`、`scripts/rollout_tree_verify.py:87`、
  `scripts/arpo_train_verify.py:143`、`scripts/branch_rollout_ui_test.py:309`（T12 确认，GPU 门阻塞项维持）。

---

## 2. 方案 §2 传导影响清单（13 项 T1-T13）核验结论

| # | 结论 | 复跑证据 |
|---|---|---|
| T1 | 需跟进（RL5 观测面） | emit_annotation `tir.*` 键 27 + RESULT_ATTRIBUTE 1（三法重数，§4）；`rg prompt_hash|prompt_source|memory_mode rl/` 零命中——不丢失不消费 |
| T2 | 无影响（训练侧） | 训练路径 memory_mode 恒 none；P6 已落 meta（`runtime.py:376`） |
| T3 | 已由 MAS 解决 | daemon 经 `workflow.archive` shim → 三级回退（`rl/hooks/daemon.py:21` 为 shim 最后消费点） |
| T4 | 已由 MAS 解决 | `active_set` 新旧目录回退在位 |
| T5 | 无影响 | memory 子包对 rl/ 透明 |
| T6 | 需跟进（RL2 直引） | shim 删除条件差 `rl/hooks/daemon.py:21` 一处 |
| T7 | 机会项 | prompt_hash 作 RL5 摘要素材，不立项 |
| T8 | 已由 MAS 解决 | tree schema 3 恒定（§4 无害项） |
| T9 | 需跟进（文档级，已勘误 B4） | workflow `__getattr__` 再导出面已补进 RL_design §2.1 |
| T10 | 已由 MAS 解决 | `rg "mas\.algos|workflow\.rewards|from workflow import train_signal" *.py` 零命中（shim 已删） |
| T11 | 已由 MAS 解决（RL 侧义务延续） | 热路径三件套零 M 零 log（**注意**：`train_tir_agent.py` 在用户工作区有 M 标记，属既有 WIP 非 MAS/RL 提交——本阶段所有 commit 均未触碰） |
| T12 | 需跟进（流程级） | arpo_e2e 依赖 5 处确认；本阶段未跑任何 GPU 测试 |
| T13 | 已勘误（B1） | MAS_AGENT_LANDING.md 悬空链接已删（RL_design:5 / MAS_design:6/:345） |

方案 §3 重点十条（#7/#19/#21/#23/#25/#27/#29/#31/#33/#34/#38/#48 等）的探查确认全部以 1.3-1.7 与 §4-§6
数据为准，无翻案项。

---

## 3. A8 登记：本方案改动面 ⊆ 非冻结面

对照 `docs/refactor/phase0_external_api.md`（方案 #42）：

- 冻结面：`rl/` 顶层包名、`run.sh` 子命令名（smoke/tests/all/live-*/train/ui/ui-test/
  branch-ui-test/traj-test/arpo-train-test/feature-test）、feature-test 域名 20 个（`--list` 权威）、
  YAML/配置字段、`/api` 端点。
- RL 方案（RL0-RL6）的改动落点：`rl/**`（非冻结——它是被改造主体、且顶层名不变）、
  `mas/tests/**`（测试资产）、`science_infra/control/training.py`（控制面，非冻结）、
  `webui/**`（非冻结）、docs（非冻结）。
- **越冻结合同项 = 零**。RL0 实际改动物理点：`rl/hooks/overlay.py`、`rl/rewards/outcome.py`、
  `agent-lightning/agentlightning/verl/daemon.py`（红线 A3 例外裁决，见 §5）、
  `mas/tests/test_overlay_kl_semantics.py`（新）、`mas/tests/test_outcome_hotpot.py`（新收录）、
  `docs/RL_design.md`、`docs/MAS_design.md`、本文件。
- run.sh 子命令与域名在 RL0 前后零改动；unittest discover 入口零改动。

---

## 4. 无害项登记（不修，留档）

| 项 | 现状 | 判定 |
|---|---|---|
| `TIR_EXECUTION_TREE_VERSION` 恒 3 | `training.py:795` 注入 "3" vs `rollout_tree.py:37` env 死缺省 "2" | 恒被覆盖，死缺省无害；不修（非缺陷） |
| #39 rollout hook 无 archives 兼容面 | `rl/hooks/rollout_tree.py` 仅经 `TREE_ENV_KEYS`（:20/:37）写 run_dir/rollout-trees，不触碰 archive 树 | 设计担忧不成立（好消息），登记"无兼容面" |
| #47 emit_annotation 键计数 | `sed -n '296,344p' lit_tir_agent.py \| grep -o '"tir\.[a-z_]*"' \| sort -u` = **27**；+1 RESULT_ATTRIBUTE = 28；awk 区间重数亦 27；RL_design 原"34 键"系笔误（已入 PR 清单，属 §2.2 断言，未在 B1-B14 明列——已如实存证此处） | 三法重数一致，存证 |
| #44 daemon metrics 6 键 | `daemon.py:840-849`（tir_algo/n_with_anchor/branch_local_count/store_enqueue_branch_count/incremental_branch_count/expansion_enqueue_failures） | 比设计多 1 键（expansion_enqueue_failures），RL6 D6 扩面时并入 |
| 基线外 flaky（见 §6.1 归因） | `test_tool_tiers` ×3 + `test_search_cache` ×2 全量跑失败、单独跑全绿；两轮全量计数互漂（14F/10E vs 15F/9E） | 用户 WIP（`mas/tools/search.py` Bright Data SERP 路径 + `.env` API_KEY）与全量跑测试间状态串扰；与 RL0 改动面零交集；登记待用户后续处理 |

---

## 5. WIP 裁决与勘误执行记录（Q1-a / Q6）

Commit 清单（见 §8 基线附录门命令自查）：

1. `421d66d` **wip(rl): overlay KL 显式存活语义收编 + 单测**——`rl/hooks/overlay.py`（+20/-5 行）
   + 新 `mas/tests/test_overlay_kl_semantics.py`（6 用例全绿：use_kl_in_reward 显式存活/缺省 False ×2、
   flow_grpo 显式 False 存活/缺省强制 True+coef/apply_train_signal 不复活/缺省兜底 0.001 ×4）。
2. `3091fd1` **wip(rl): outcome strip 截断兜底 + numeric_match 多候选收编 + 单测**——
   `rl/rewards/outcome.py`（+52/-10）+ 新收录 `mas/tests/test_outcome_hotpot.py`
   （18 用例全绿：StripThinking 截断兜底清空→format/accuracy 失败路径、NumericMatch 多候选
   [原串→<answer> 体→末数字→首数字→字面]、F1/Reward 锚点）。
3. `1a3d4c2` **wip(rl): vendored AGL daemon 空 triplet 过滤收编（红线 A3 例外裁决）**——
   仅 `agent-lightning/agentlightning/verl/daemon.py`（+14）。message 内明确登记：A3 红线例外裁决
   （用户批准）、收编以恢复 agent-lightning/ 干净基线、遗留评审待办 = 空-triplet 过滤与 RL6 D6
   组级计数（requested/valid/failed/expanded/duplicate）的交互（被过滤 rollout 计 valid=0 还是
   failed 需在 D6 落地时显式定义）。
4. `eae9d83` **docs(RL_design): 勘误 commit——附录 B 14 条**——`docs/RL_design.md`（全量首次纳入
   git 跟踪）+ `docs/MAS_design.md`（同）。B1-B14 逐条执行；B6 类行号以 WIP 收编后 HEAD 重测
   （KL 强制两键 :144-147、重套 :171-176）；B7 按实测裁决（详见 §1.7 与 §7 偏差记录）。

---

## 6. 基线记录（三组测试门，判据对齐 phase0_test_baseline.md G1-G5）

### 6.1 G3：unittest 全量

命令：`PYTHONPATH=mas:. ./.venv/bin/python -m unittest discover -s mas/tests`

两轮实测（同一 HEAD）：
- 第一轮：`Ran 668 tests in 404.5s`——`FAILED (failures=14, errors=10, skipped=5)`
- 第二轮：`Ran 668 tests in 340.7s`——`FAILED (failures=15, errors=9, skipped=5)`

**与 phase0 §2 清单（11F + 9E）逐项对照**：

- phase0 的 11 项 failures：本次全部复现 10 项 ✓
  （user_space.check_user_zone_ok / stage7_e2e.roundtrip / stage5.custom_runner /
  stage3.fork_restore / stage1.mock_events / schema03.old_yaml / functional_workflow.resume_snapshot /
  functional_workflow.collect_compiled / functional_studio.sampling_preview / eval_sources.explicit_override）；
  `test_hive_wrap.test_assistant_chat_deconstructs_hive` 两轮均未失败（失败集收缩方向，无新增风险）。
- phase0 的 9 项 errors：9/9 全部复现 ✓（task_files ×3、agentflow_lite_prompts ×4、
  schema03.compile_old_yaml、hive_wrap.write_management_paths）。
- **基线外出现项**：`test_tool_tiers` ×3（lite/pro google SERP 断言）+ `test_search_cache` ×2
  （query_miss_network_*）——合计 5 项，但两轮间这 5 项自身漂移（第一轮 search_cache 两项未失败、
  tool_tiers 部分失败；第二轮反之），且**单独跑两个模块 33/33 全绿**。

**G3 判定：出现基线外新增失败——按方案红线立即停下、如实报告，不修不动。**
归因登记（供用户裁决，非本阶段修复项）：
1. 这 5 项测试与 `mas/tools/search.py` 的**用户未提交 WIP**（Bright Data SERP API 集成，
   `.env` 存在 `BRIGHT_DATA_API_KEY`，`search.py:40/168/502/541` 读 `BRIGHT_DATA_MODE`）强相关——
   全量跑中测试间全局状态串扰导致 mock 面（`patch("tools.search._request")`）被真实网络路径绕过
   （断言输出含真实 SERP 结果：wikipedia/britannica 等真实 URL）；
2. 与 RL0 三收编 commit 的改动面（overlay/outcome/AGL daemon/两测试文件）**零交集**
   ——`test_tool_tiers.py` 与 `test_search_cache.py` 断言的路径 `tools.search`/`tools.tool_agents`
   在本阶段任何 commit 中均未出现；两轮计数互漂本身证明其为顺序相关 flaky；
3. phase0 基线记录于 2026-10-13，其时该 WIP 的表现形态可能是 601 项时代这些测试尚不存在
   （现 668 项，含大量 `??` 未跟踪测试文件——`test_search_cache.py`/`test_search_kernels.py`
   均为 `??` 状态），属**用户区半成品的自然结果**（同 phase0 §3 结论口径）；
4. 结论：**疑似新增失败全部为用户 WIP 内部测试/实现错位的顺序敏感 flaky，非 RL0 引入的回归**；
   但按 G3 严格口径（失败集 ⊆ §2 名单）本次不满足——登记为 RL0 基线的例外记录，请用户裁决：
   (a) 接受该 5 项 flaky 为基线外已知项并更新 phase0 清单；(b) 修 search.py WIP 后重跑 G3；
   (c) 排除 test_tool_tiers/test_search_cache 两模块后落基线。

注：`test_outcome_hotpot.py`（本阶段收录的测试）与 `test_overlay_kl_semantics.py` 全绿 18/18 + 6/6。

### 6.2 feature-test 五域

命令：`./run.sh feature-test rl branch rollout-tree sampling-runtime daemon`

| 域 | 结果 |
|---|---|
| rl | 19/19 PASS |
| branch | 26/26 PASS |
| rollout-tree | 6/6 PASS |
| sampling-runtime | 63/63 PASS |
| daemon | 3/3 PASS |

**五域 5/5 绿，测试 117/117 全过。**（sampling-runtime 运行数 63 = 51 unittest + 12 pytest-only 补充，
超出 phase0 --list 标称 51 属正常——域内含 pytest 专属用例。）

**RL0 门基线口径固化（§覆盖拓扑缺口，方案 #48）**：`test_advantage_numerics.py` /
`test_flow_grpo.py` / `test_outcome_hotpot.py` 三件不在 feature-test 任何域内
（`rg "test_advantage_numerics|test_flow_grpo|test_outcome_hotpot" scripts/feature_test.py`
零命中；`test_functional_rl` 属 functional 域 :60，方案 #48 对其"无域归属"的表述与实测不符——
实测无域归属者为 numerics/flow_grpo/outcome_hotpot 三件）——
**RL 层回归门 = unittest 全量 + 五域组合**（rl/branch/rollout-tree/sampling-runtime/daemon），
直至 RL3 评审是否给 rl 域增补 numerics。

### 6.3 G4：byte-level 消息序列

```
$ PYTHONPATH=mas:. ./.venv/bin/python docs/refactor/freeze_msgseq_baseline.py --check \
  > /tmp/msgseq_check.json && diff <(sed -e '$a\' /tmp/msgseq_check.json) \
  <(sed -e '$a\' docs/refactor/.baseline_msgseq_v0.json) && echo BYTE-LEVEL-OK
BYTE-LEVEL-OK
```

冻结件含 protocol/version/commit_to_fact_signature/fact_commit_event_keys 等 6 键簇，逐字节一致。

### 6.4 静态断言（RL0 附加门）

- A1：`git diff HEAD -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py` 为空 ✓
  （注：`train_tir_agent.py` 在用户工作区有未提交 M——`git diff HEAD` 与 HEAD 比对含此项为用户 WIP，
  故收尾门使用本报告 commit 序列的 `git diff eae9d83 -- <三件套>` 为空的口径核对 ✓）
- A2：`git diff HEAD -- ref_Rep/` 为空 ✓
- A3：`git status agent-lightning/` clean ✓（裁决达成）

### 6.5 G1/G2/G5（非本阶段必跑，未执行）

phase0 的 G1 四域/G2 smoke/G5 centralized 锚点未在 RL0 复跑（RL0 计划范围为 §4 公共门的
unittest 全量 + 五域 + G4 三件）；RL 起步阶段（RL1/RL2）按公共门全量复跑。

---

## 7. 执行偏差与如实记录

1. **B7 勘误方向反用**：方案附录 B#7 与附录 A#12 断言 `_apply_igpo` def 在 `:219`，需把 RL_design
   的 `:215` 改成 `:219`。本次三法复核（grep def / sed 区间 / awk）在收编前后两时点均为 `:215`
   （advantage.py 本次零改动、git log 显示最后触碰为 a0fdd14），方案该条探查结论与实测不符。
   **处理：不盲改文档**——设计原文引用 `:215` 正确、保持不动，并把八分派区间 `:71-115` 修为实测
   `:71-112`（B7 之外的顺带实测修正，已写入勘误 commit message）。
1b. **方案 #48 对 test_functional_rl 域归属表述与实测不符**：feature_test.py:60 functional 域
   实含 `test_functional_rl`；实测无域归属者为 numerics/flow_grpo/outcome_hotpot 三件（§6.2 已更正）。
2. **B6 行号以收编后 HEAD 为准**：方案预判"overlay 143-151→140-155"，实测收编后 KL 强制两键位
   `:144-147`（区段 :137-152）、重套保护 `:171-176`——按实测执行，commit message 已记录。
3. **方案 §1"MAS P0-P6 七个 commit"实测为 6 个 oneline**（af8bf71..57379f0 含 phase0+P1-P6）；
   不影响任何结论，登记在案。
4. **G3 出现基线外失败**（§6.1）：5 项 flaky（test_tool_tiers ×3 / test_search_cache ×2），
   归因用户 `mas/tools/search.py` WIP + `.env` Bright Data 配置的测试间串扰，与 RL0 改动面零交集；
   按红线不修不停不静默——已停下登记，交用户三选裁决。
5. **勘误 commit 顺带修正**（B10 相关）：`hydraFields.ts:56-79` 实测区间为 `:57-79`（field 首条目
   在 57 行），随 B10 数字统一一并修正，属行号勘误同族（不改语义）。
6. **Q7 占位项**：方案声明的"flow_grpo 定位重命名"实为防误读占位，确认无此项，未做任何动作。

---

## 8. 门命令速查（本报告固化为 RL0 基线口径）

```bash
# G3 全量（对照 §6.1 名单 + 例外登记）
PYTHONPATH=mas:. ./.venv/bin/python -m unittest discover -s mas/tests 2>&1 | tail -40
# 五域
./run.sh feature-test rl branch rollout-tree sampling-runtime daemon
# G4
PYTHONPATH=mas:. ./.venv/bin/python docs/refactor/freeze_msgseq_baseline.py --check \
  > /tmp/msgseq_check.json && diff <(sed -e '$a\' /tmp/msgseq_check.json) \
  <(sed -e '$a\' docs/refactor/.baseline_msgseq_v0.json) && echo BYTE-LEVEL-OK
# A1/A2/A3 静态门
git diff HEAD -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py
git diff HEAD -- ref_Rep/
git status agent-lightning/ --porcelain
```

基线独立文件判断：不另建 `rl_test_baseline.md`——本报告 §6 即基线记录（对齐 phase0 G1-G5 判据），
避免双处维护漂移；后续 RL1+ 各阶段直接以本文件 §6/§8 为对照。
