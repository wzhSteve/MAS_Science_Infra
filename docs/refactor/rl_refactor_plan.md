# RL 层修改优化方案（rl_refactor_plan）

日期：2026-10-14
状态：方案稿（只读分析产物，未改任何代码）。本文档是 `docs/RL_design.md`（v1，组合式 RL 与 0 代码实验框架设计）落地前的执行计划，
对照对象是 MAS 层 Phase 0-6 重构（af8bf71..57379f0）完成后的代码现状。
纪律对齐 MAS 重构模式：probe-first、每阶段独立 commit、byte-level/等价门、可单独 revert、绝不捎带工作区既有 WIP。
最小改动承诺：本方案全程零代码改动（只新建本文件）；后续各阶段的代码改动均属提案，须逐阶段评审后执行。

探查环境说明：本文档所有 `rg` 命令在仓库根 `/root/autodl-tmp/MAS_Science_Infra` 下执行；
Python 侧结论基于 `uv .venv`（`./run.sh` 同源）；行号引用以工作区现状为准，注意工作区含用户未提交修改（见 §8）。

---

## 0. 阅读指引

| 读者关心 | 直接跳 |
|---|---|
| 为什么做、做什么、多大 | §1 执行摘要 |
| MAS 重构把什么传导给了 RL 层 | §2 传导影响清单 |
| 设计与代码到底差在哪 | §3 重点差距（全量表在附录 A） |
| 分几步做、每步怎么验收 | §4 分阶段实施方案 RL0-RL6 |
| 什么绝对不能动 | §5 红线与不变式 |
| 需要我拍板什么 | §6 待用户确认项 |
| RL_design 本身的勘误 | 附录 B |
| 可复制的探查命令 | 附录 C |

关键文档间关系：`docs/MAS_design.md` v2（747 行）§11 只承诺两个 MAS→RL 合同（Trajectory、SamplePolicy.sites）并附边界声明；
`docs/RL_design.md` §0.2 把"唯一入口 + 守卫四"包装为第三合同——其内容真实存在（MAS v2 §11 边界段原文），但编号归属需校准（附录 A#1）。
本文档不修改 RL_design.md；其行号漂移与悬空链接集中在附录 B，待批准后单独 commit 勘误。

---

## 1. 执行摘要

**为什么现在做。** MAS 层 Phase 0-6 已完成（7 个 commit）：合同冻结、legacy 隔离、memory 子包、shim 删除、prompts 单入口、
产物迁移、planner_ctx 激活。RL 层是被这场重构"半包围"的一层——它消费的 Trajectory 合同新增了
`prompt_hash / prompt_source / memory_mode` 三个 meta 键（`mas/workflow/runtime.py:371-377`），它依赖的
archive/expansion 路径整体迁向 `artifacts/mas/`（`mas/workflow/memory/archive.py:25-42`），它声称唯一正典的
`rl/` 包已成为 Reward/TrainSignal 的唯一物理入口（P3 兑现）。RL_design.md（1355 行）写在 MAS 重构次日，
大部分 file:line 断言经本次逐点核验仍然成立，但它有一个内部矛盾（§5.A 红线 vs 其 §3.6 #6 改 `lit_tir_agent.py`）
和若干与现实漂移的前置假设（已删除的 `experiments/arpo_e2e`、已删除的六份 docs 手册）。
在动任何 RL 代码之前，先把这三件事裁干净，是本方案 RL0 阶段的存在理由。

**方案主体。** RL_design §3-§7 的组合式架构（Advantage Pipeline / Reward Assembly / Loss Mode / rl.yaml v2 /
resolved_algo 审计 / L0-L1 UI / 测试体系）**全部处于"设计已冻结、代码零实现"状态**——全仓 `rg` 无
`expand_preset|AdvantagePipelineSpec|RewardSpec|resolved_algo|composed` 任何实现踪迹；差距 D1-D6 逐条核验全部属实。
本方案把 RL_design 的 P0-P4 五阶段**重排为 RL0-RL6 七阶段**：把"文档对齐与基线"独立成 RL0（原设计把勘误与
写侧双写混在 P0），把 daemon 列注入小修独立成 RL2（原设计埋在 P0 清单里，但它与 D2 双写无依赖、可并行、可提前验收），
每阶段规模控制在 1-3 个文件（原设计 P1 一次动 ±350 行）。裁剪原则：小成本高确定性优先，凡依赖 GPU 验证或
用户裁决的门，全部显式标注而非默默排期。

**预期工作量与风险。** 代码改动总量与 RL_design §6 自估一致（净增 ~1600 行，测试占 1/3，5 个主改动文件），
但按本方案切分后每阶段净增 0-600 行、可独立 revert。三个风险峰：
(a) RL1 的 D2 双写让 future_kl 首次真实生效——APPO 预设名实相符化是行为变化，需 release note + GPU 验证（征询后执行）；
(b) RL3 的 pipeline 重构是字节级等价门最重的一期——8 preset 逐元素对照测试是唯一防线，先于 RL0 修好基线才能起步；
(c) 工作区现有**三处未提交行为修改**（`rl/hooks/overlay.py`、`rl/rewards/outcome.py`、`agent-lightning/agentlightning/verl/daemon.py`），
其中第三处直接触碰"AGL 黑盒不动"红线，必须在 RL0 由用户裁决归属，否则任何"全量测试绿"门都会被误收（RL_design §6 自己同样警告过前两处，
但漏了第三处，这第三处是 MAS 重构后的新事实）。

---

## 2. MAS 重构对 RL 层的传导影响清单

逐项标注三态：已由 MAS 阶段解决 / RL 侧需跟进 / 无影响。依据全部来自对当前代码的实测（file:line）。

| # | 传导项 | 现状实测 | 定性 | RL 侧动作 |
|---|---|---|---|---|
| T1 | Trajectory.meta 新增 `prompt_hash / prompt_source`（P4） | 合同层已进 meta（`mas/workflow/runtime.py:371-375`，`contracts.py` Trajectory meta 通道）；centralized 路径经 `prompt_provenance()` 填充（`centralized_runtime.py:971-983`） | **RL 侧需跟进** | 训练数据流**不丢失也无消费**：`lit_tir_agent.py` emit_annotation 的 27 个 `tir.*` 键（`lit_tir_agent.py:305-343`，精确计数见附录 A#47）不含 prompt 审计键，daemon 列注入（`rl/hooks/daemon.py:740-823`）亦无。方案动作：不强制进训练张量流（守卫四边界），只在 RL5 的 resolved 环境摘要中透出 prompt_hash，供 run 间对照（§1.7 控制变量的 prompt 轴） |
| T2 | Trajectory.meta 新增 `memory_mode`（P6） | `runtime.py:376` 同步进 meta；Mode B（训练路径）`runtime.py` 单 hub ReAct 不接 planner_ctx 注入（planner_ctx 接线只在 `centralized_runtime.py:341-353/985-990`），训练 rollout 的 memory_mode 恒 "none" | **无影响（训练侧）+ 已由 MAS 解决（评估侧）** | 训练热路径无污染风险——评估污染红线的机器证据（memory_mode 进 meta）已由 P6 交付。RL 侧动作：preflight 摘要（RL5）展示该字段即可，不新增校验拦截（latest 在 eval 是否合法属策略问题，留给 Harness 层） |
| T3 | archive 路径迁移 `.tir_archives → artifacts/mas/tir_archives`（P5） | RL 侧仅一处消费 `load_resume_messages`（`rl/hooks/daemon.py:21`），经 `workflow.archive` shim（`mas/workflow/archive.py:1-19`）转发到 `memory/archive.py`；其内部已实现 内存索引 → 新目录 → 旧目录三级回退（`memory/archive.py:250-289`） | **已由 MAS 解决** | 无需防护。verdict：设计文档若预期 rollout hook 会读 `.tir_archives`，实测不成立——`rl/hooks/rollout_tree.py` 完全不触碰 archive 树（只经 `TIR_ROLLOUT_TREE_DIR` 写 `run_dir/rollout-trees`，`training.py:795-800` 注入 env） |
| T4 | local_expansion 迁移（P5） | `rl/hooks/daemon.py:20` `load_local_expansion` 来自 `workflow.active_set`，其内部已做新目录→旧目录读回退（`active_set.py:37-50/481-487`） | **已由 MAS 解决** | 无需动作 |
| T5 | memory 子包化 + Archive RLock/原子写（P2） | 对 rl/ 完全透明（shim `from .memory.archive import *`）；锁修复对 daemon 单线程消费不可见 | **无影响** | 无 |
| T6 | `workflow.archive` shim 仍存活 | shim 删除条件 = "mas/scripts 与 tests 全部直引后"（`phase0_external_api.md §6`）；实测 **`rl/hooks/daemon.py:21` 是 shim 的最后一个非测试消费点**（`rg -l "from workflow.archive" rl/` 唯一命中） | **RL 侧需跟进** | RL2 顺手项：`rl/hooks/daemon.py:21` 改 `from workflow.memory.archive import load_resume_messages`，一步达成 shim 删除条件的最后一块（phase0 底稿 §6 清单当年没列出 rl 消费点，属登记缺口） |
| T7 | prompts.py 单入口（P4）对实验可比性 | 单 hub 训练 prompt 常量在 `mas/tir_agent.py` 原地不动（MAS v2 §7 表）；画布路径 prompt 已可审计（T1） | 无影响 → 机会 | prompt_hash 成为 RL 实验新的可比性锚：同一 yaml 下 prompt 漂移可由 meta 查证。作为 RL5 resolved 摘要素材（同 T1），不独立立项 |
| T8 | Snapshot `schema_version=1`（P2）+ RolloutTree schema 3 | `rl/hooks/rollout_tree.py:37` 默认读 `TIR_EXECUTION_TREE_VERSION` env，控制层恒发 "3"（`training.py:800`）；`contracts.py:337-358` schema 3 分支已实现 | **已由 MAS 解决** | 无需动作；RL_design 未提及此层，属"代码已有但设计未捕获"（附录 A#30） |
| T9 | P3 后 TrainSignal 入口拓扑变化 | 设计 §2.1 说 3 个转换点全部走 `batch_to_train_signal`；实测后两个点改为 `from workflow import Collector, batch_to_train_signal`（`science_infra/control/services.py:820`、`science_infra/ui/cli.py:85`），经 `workflow/__init__.py:60-70` 的 `__getattr__` **懒转发**到 `rl.train_signal` | **RL 侧需跟进（文档级）** | 机制等价（仍是唯一构造器），但消费拓扑从"直引 rl"变"workflow 再导出面"，RL_design §2.1 消费图需补一行（附录 B#4）；且该 `__getattr__` 是 rl→workflow 反向依赖之外新增的 workflow→rl 正向转发面，守卫评估纳入 RL0 |
| T10 | `mas/algos/` 与 `workflow.{rewards,train_signal}` shim 删除（P3） | 实测 `rg "mas\.algos|from workflow import train_signal|workflow\.rewards"` 在 py 中零命中（仅 docs 历史文本）；`rl/rewards/__init__.py` 正常导出全集 | **已由 MAS 解决** | RL_design §0.2 合同 3 的"shim 已删（03c8724）"表述与代码一致，无跟进 |
| T11 | 训练热路径三件套零改动承诺 | `git log af8bf71..57379f0 -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py` 无任何 commit 触碰；当前工作区 git status 三件套无 M 标记 | **已由 MAS 解决** | RL 侧义务：延续该红线（§5.A）。注意 RL_design 自身在此有一处违约设计（§3.6 #6 提改 `lit_tir_agent.py` 与其 §6 红线冲突），本方案已裁决（§4 RL4） |
| T12 | experiments 目录用户清理（arpo_e2e / demo / alpha 已删） | `experiments/` 仅存 `agentflow_4algo`；`run.sh:1281/1334` 与 `scripts/arpo_train_verify.py:10`、`scripts/branch_rollout_ui_test.py:309` 默认 `--experiment arpo_e2e` → GET /api/experiments/arpo_e2e 将失败 | **RL 侧需跟进（流程级）** | 所有含 GPU 门/arpo-train-test/branch-ui-test 的阶段步骤标注此前提变化：验收前要么恢复 arpo_e2e bundle，要么改脚本默认实验（改属 .py 变更，需用户确认，见 §6-Q2） |
| T13 | docs 批量清理（六份手册被删） | `run.sh:552/593/1293/1347/1370` 仍引用 `docs/ARPO_TRAIN_TEST.md`、`docs/MAS_AGENT_FRAMEWORK_TEST.md`、`docs/BRANCH_ROLLOUT_UI_TEST.md` 等（已删除，`git status docs/` 显示 D）；`RL_design.md:5` 配套阅读链接 `MAS_AGENT_LANDING.md` 悬空 | **RL 侧需跟进（文档级）** | RL0 勘误清单收录（附录 B#2）；本方案回归门表述不再引用已删手册，只引 `run.sh` 子命令与 `--list` |

小结：13 项传导中 6 项已被 MAS 阶段解决、4 项无影响、3 项需 RL 侧跟进的实质动作（T1 观测面、T6 shim 直引、T12 门流程），
其余为文档勘误。**没有任何一项要求修改训练热路径或 vendored 代码**——这是本方案可以承诺红线（§5 A1/A2）的实测基础。

### 2.1 热链路九环节重点核验（RL_design §2.2 对照）

RL_design §2.2 用编号 ①-⑨ 描述一个 train step 的九环节。本次对 RL 侧可见的四个环节实测（AGL/vendored 内的三环节不越界核验）：

| 环节 | RL_design 断言 | 实测 | 判定 |
|---|---|---|---|
| ② 波次1 | `daemon._async_set_up :209` 把 train_rollout_n 收缩为 initial_rollouts | `rl/hooks/daemon.py:209` `_async_set_up`；`:226` `self._preinject_expand_fields(data, n_init=...)` aepo 时 n_init=1 | 一致 |
| ③ 波次2 | `:420` 等待 + `:444` runner 扩展 | `:420 _wait_for_training_rollouts`、`:444 _enqueue_from_runner_expansions`、legacy 回退别名 `:684-685 _enqueue_tree_branches` | 一致 |
| ④ rollout→batch | `:688` triplet→DataProto + 扩展列 | `:688 get_train_data_batch`；grpo 提前 return `:696`；列注入区 `:740-823`；组级 metrics `:836-849`；`emit_rollout_tree_event` loss 帧 `:852-861` | 一致（细节行号漂移见附录 A#48） |
| ⑦ advantage | trainer monkeypatch → `apply_tir_advantages :47` | `rl/hooks/trainer.py:16-32` `_train_step` 内换 `compute_advantage`；try/finally 恢复 `:38-40`；`bound_daemon_cls` `:44-54`（Ray 序列化前闭包 tir_algo/tir_config） | 一致 |

另发现一处设计未捕获的写侧行为：`apply_train_signal` 对 appo 有"loss_mode 复活保护"——
`rl/hooks/overlay.py:184-186`：`algo == "appo" and loss_mode == "vanilla"` 时从 tir dict 抢回 `loss_mode`
（缺省 future_kl），即 **LossSpec 显式 vanilla 会被静默改写**为 yaml 里的值或 future_kl。RL1 实施双写时该分支
成为"用户显式 vanilla 也会被镜像为 actor.policy_loss.loss_mode=vanilla"的关键路径，须以测试锁定
（该行为的唯一豁口：yaml `tir.loss_mode: vanilla` 显式存在时存活）。

---

## 3. 重点差距（全量矩阵见附录 A）

按"对方案排序的影响"挑最关键的十条。编号与附录 A 一致。

**#7 / D2 断链属实且代价可精确计量。** 写侧 `apply_train_signal` 把 loss_mode 写进 `algorithm.tir.loss_mode`
（`rl/hooks/overlay.py:186`），vendored 读侧 `dp_actor.py` 读 `self.config.policy_loss`（`ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/workers/actor/dp_actor.py:392-393`，fipo_cfg 缺省 `loss_mode="vanilla"`）；
`ppo_trainer.yaml` 无 `policy_loss` 段 → 两棵 config 树永不相遇，`compute_policy_loss_future_kl`（`core_algos.py:688`）是死代码。
参数缺省漂移同步确认：写侧 `decay_rate=64.0`（`rl/loss.py:19`）vs 读侧 `128.0`（`dp_actor.py:404`）。
**新的关键证据：AGL `interface.py:139-141` 先 `OmegaConf.set_struct(base_cfg, False)` 再 merge 用户 dict——
用户 yaml 里的任意新键（含 `actor_rollout_ref.actor.policy_loss.*`）会无裁剪地存活进 config 树**。
这使 RL_design §3.3 方案 A（纯写侧双写）的可行性从"推测"升格为"有实据"——双写四行即通，vendored 一字不动。

**#23 设计与红线内部冲突（方案裁决点）。** RL_design §3.6 改动清单 #6 要改 `mas/lit_tir_agent.py`（"参数链收编"），
但其 §6 P0-P2 红线清单白纸黑字"训练热路径三件套零移动、零改动"。实测：`lit_tir_agent.py:52/61` 的
`self.multi_tool_bonus` 确实是**死参数**（`:247` 调 `default_reward_fn` 不传该参、也不读 self 属性给 reward）；
`DEFAULT_TIR["multi_tool_bonus"]`（`overlay.py:29`）同样是**死配置**（`collector.py:36-43` 调
`compute_outcome_reward` 不传该参，函数默认参数 0.1 兜底，`outcome.py:154`）。裁决：**不改 lit_tir_agent.py**——
reward 组件的接通走 overlay→collector 通道（RL4），热路径死参数原地标注；RL_design#6 的 lit_tir_agent 部分作废。

**#19 工作区三处未提交修改，其一触碰 AGL 红线（RL0 的存在理由）。**
`git status`：`M rl/hooks/overlay.py`（KL 显式存活语义：use_kl_in_reward 尊重显式键、flow_grpo 尊重显式 False——
恰是 RL_design §2.3/§4.2 描述的"现状"，即 **RL_design 是对着未提交工作区写的**，HEAD 与设计书已经分叉）；
`M rl/rewards/outcome.py`（strip_thinking 截断兜底 + numeric_match 多候选增强，行为级变更）；
`M agent-lightning/agentlightning/verl/daemon.py`（**vendored AGL 内改**：空 triplet 过滤——直接违反"AGL 黑盒不动"，
RL_design §6 只警告了前两处）。三处不裁决，RL0 基线便不可信，任何后续门都会被污染。

**#33 preset 等价基线目前不存在。** RL3 pipeline 重构要求"同一 seed 同一 batch 逐元素一致"，但现实中
`test_advantage_numerics.py` 只锁数值点（其 `TestFutureKlStatusQuo` 在 `:461-475` 锁 future_kl 未接入）、
`test_appo_arpo_credit.py` 锁 appo/arpo 两 preset 的关键行为、**没有任何测试把 8 个 preset 的
advantages/returns 张量当作可跨版本对照的持久基线**。RL0 需要先建"旧 dispatch 冻结为对照实现 + 等价测试"，
否则 RL3 无从验收（这与 MAS 重构 freeze_msgseq_baseline.py 的角色完全同构）。

**#21 RL_design 的行号引用大体准确，但非零漂移。** 逐点核验（附录 B 全表）：`advantage.py:71-115` 八分支互斥（实测 71-112，断言成立）、
`rae_advantage.py:33/85/161/229` 全对、`daemon.py:209/444/688` 全对、`loss.py:16-17` 注释全对、
overlay flow_grpo KL 段 `143-151` 漂到 `140-155`（因未提交改动）、`core_algos.py:503` vanilla clip 实为 `:458`（future_kl `:688` 不漂）、
`rollout_trees.py:84/129/215` 漂到 `85/130/216`、`collector.py:26-46` 实为 `27-43`。结论：**结构级断言全部成立，行号需要一次系统性勘误 commit**（本方案不执行，附录 B 备好）。

**#34 MAS v2 §11 合同表述需校准。** RL_design §0.2 称"三条合同来自 MAS_design §11"；v2 实文（`docs/MAS_design.md:628-636`）
只编号承诺**两个**合同（Trajectory、SamplePolicy.sites），"唯一入口 + 守卫四"是其"边界"段落的散文内容。
语义闭环检查结论：**消费断言全部闭合**——Trajectory 字段 `format_ok/n_search/n_python` 确实同步进 meta
（`contracts.py:210-215` sync_tir_meta）、reward 侧确实以合同字段喂入（`collector.py:36-43`）、
sites 确实同 schema（`overlay.py:66-80` resolved_sites 消费）——但"合同 3"的编号出处应改为"MAS v2 §11 边界段 + §12 守卫四"。

**#31 双轨 schema 的存量前提变化。** RL_design ADR-5 承诺"v1 存量 `experiments/*/rl_*.yaml` 永久可跑、零迁移成本"
——实测 `experiments/` 只剩 `agentflow_4algo`（内含 rl.yaml + rl_{grpo,arpo,appo,flow_grpo}.yaml，全为 v1 形态），
用户已删除其余实验目录。"双轨兼容"的存量义务实际收窄到 5 个文件，RL5 的迁移验证面随之缩小（好事，但设计书前提需更新）。

**#29 训练批构造器消费拓扑新增一跳。** 详见 T9：`batch_to_train_signal` 仍唯一（`rl/train_signal.py:27-30`），
但 services/cli 两点经 `workflow/__init__.py:60-70` `__getattr__` 懒转发。这既是 rl 唯一入口的兑现形态，
也意味着"删 workflow 再导出面"必须永远晚于"control/cli 改直引"——不删无害，但 RL0 需把该面记入外部 API 清单。

**#27 grpo 提前 return 掩盖 zero_prefix（P0-b 实测确认）。** `rl/hooks/daemon.py:696` `if self.tir_algo == "grpo": return`，
列注入（:740-823）被跳过；而 grpo preset 默认 `zero_prefix_adv_for_children: True`（`overlay.py:35`）、
advantage 分支会调 `_maybe_zero_prefix`（`advantage.py:81-82/182-190`），缺 `rollout_role/resume_boundary` 列时静默 no-op。
v1 时代"grpo 不分支"掩盖了它；一旦 sites 显式配给 grpo（合法），分支行的前缀清零将静默失效。RLDesign 的 P0-b 判定属实。

**#25 resume_parent_id 缺列（P0-a 实测确认）。** daemon 把 `resume_parent_id` 写进 sample dict
（`daemon.py:360/513/646`）并带进 annotation（`lit_tir_agent.py:321`），但 **`non_tensor_batch` 从不注入该列**
（:740-823 注入清单无）——`apply_appo_credit` 读 `ntb.get("resume_parent_id")`（`advantage.py:132`）永远落空，
随后走 `(data_id, parent_id)` 全组兜底分组（`advantage.py:129-170`）。一行列注入可根治，RL2 项。

**#38 evaluation/训练双模式对组合面的天然约束未被 RL_design 充分表述。** MAS v2 §8/§0.3 锁死"RL 层只服务 Mode B"；
而组合面板（RL6）的采样区直接耦合 `workflow.sampling`。实测联动键写实：`apply_sample_policy` 的五档映射
（`overlay.py:70-79`）+ group_n 同步（:85-91）。RL6 的联动消解规则必须复用该链路（RLDesign §4.3 规则一二三成立），无需变更。

**#48 测试资产盘点：§7 体系与现状的精确距离。** RL 相关测试现状（`ls mas/tests/`）与 feature-test 域映射（`scripts/feature_test.py:27-90`）：
`rl` 域 = `test_stage2_rl + test_appo_arpo_credit`；`branch` 域 = `test_gates_and_rae + test_branch_policy_activeset + test_phase_abcd_rae_activeset`；
`sampling-runtime` 域 = `test_rl_config_contract + test_sampling_core + ...`（9 件）；`daemon` 域 = `test_daemon_expand`。
**关键差距与既有资产**：`test_advantage_numerics.py`（含 `TestFutureKlStatusQuo :461-475`——RL1 的翻转对象）
**不在任何 feature-test 域中**——`rg -n "test_advantage_numerics" scripts/feature_test.py` 零命中，
它只在 unittest 全量里跑（`pythonpath=[".","mas"]`，`pyproject.toml:53`）。方案动作：RL0 基线以 unittest 全量
+ 三域组合为门（`./run.sh feature-test rl branch sampling-runtime daemon`），并提议 feature_test.py 给 rl 域追加该模块
（属 .py 变更，随 RL3 等价测试落点一起评审——纯测试资产，风险可控）。
`test_daemon_expand.py` 的 rollout_role 锚点、`test_gates_and_rae.py` 的裁决测试、`test_appo_arpo_credit.py` 的
分支行清零/写回父行断言，构成 RL3 算子注册表复用的现成等价受害者；`test_outcome_hotpot.py` 是 RL4 逐值比对的锚点集。

---

## 4. 分阶段实施方案

阶段总览与依赖：

```
RL0 前置基线与 WIP 裁决 ──┬─→ RL1 D2 写侧双写 ──┬─→ RL3 Advantage Pipeline ──┬─→ RL5 schema v2 + resolved ─→ RL6 UI L0/L1 + D6
（文档+清点，无代码）      └─→ RL2 daemon 列注入 ──┘   RL4 Reward Assembly ─────┘                    │
                                                                                                └──────────────→ RL7（可选裁剪项见 §6-Q5）
并行性：RL1 ∥ RL2（文件不相交：overlay/loss vs daemon）；RL3 依赖 RL1+RL2；RL4 依赖 RL0（可与 RL3 并行，
       但 RL5 依赖 RL4 的 RewardSpec 先存在）；RL6 依赖 RL5。
```

每阶段格式：目标 / 前置探查 / 改动文件 / 函数级测试 / 回归门 / commit 信息草案 / 回滚。
全部阶段的公共回归门（每次必跑，不再重复书写）：

```bash
PYTHONPATH=mas:. ./.venv/bin/python -m unittest discover -s mas/tests   # 对照 phase0_test_baseline.md §2 的 25 项失败清单（G3：不新增）
./run.sh feature-test rl branch rollout-tree sampling-runtime daemon      # RL 触碰域
PYTHONPATH=mas:. ./.venv/bin/python docs/refactor/freeze_msgseq_baseline.py --check \
  > /tmp/msgseq_check.json && diff <(sed -e '$a\' /tmp/msgseq_check.json) \
  <(sed -e '$a\' docs/refactor/.baseline_msgseq_v0.json) && echo BYTE-LEVEL-OK   # G4
```

### RL0 前置基线与 WIP 裁决（零代码改动）

- **目标**：让"全量测试绿"从不可信变为可信。产出三份文档件：`docs/refactor/rl0_probe_report.md`（探查实录，
  对齐 phase5_probe_report.md 体例）、RL_design 勘误结论（附录 B 原样提交评审）、RL 基线记录（对齐 phase0_test_baseline.md 判据 G1-G5）。
- **前置探查命令**（全部已在本文档执行过一次，RL0 时复跑固化）：

```bash
git status --short | grep -E 'rl/|agent-lightning/'            # 三处未提交行为修改
git diff --stat HEAD~7..HEAD -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py   # 必须为空（热路径零改动存证）
rg -n "from workflow\.|import workflow" rl/ -g '*.py'         # rl→workflow 全部 6 文件 10 处引用面
rg -n "from rl\.|import rl\." mas/ science_infra/ -g '*.py'    # 反向引用面（8 文件，T9 拓扑）
rg -c 'field\(' webui/src/features/rl/model/hydraFields.ts      # 23（L0 现存面）
rg -n "arpo_e2e" run.sh scripts/                                # T12 门流程依赖
rg -n "policy_loss" ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/config/ppo_trainer.yaml   # 空输出 = D2 之二存在
./run.sh feature-test rl branch rollout-tree sampling-runtime daemon   # 域基线
```

- **改动文件**：无 .py；新增 `docs/refactor/rl0_probe_report.md` 一份（本方案的 §2/§3/附录 A 即其素材）。
- **函数级测试**：无新测试；固化既有 `mas/tests/test_advantage_numerics.py::TestFutureKlStatusQuo`
  （`:461-475`）的当前断言语义为 RL1 的"翻转前基线"。
- **回归门**：G1-G5 全套（见公共门 + phase0_test_baseline.md §5）；另加两条静态断言：
  `git diff HEAD -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py` 为空；
  `git status agent-lightning/` 干净（达成方式 = 用户对 §6-Q1 的裁决被执行）。
- **commit 草案**：`docs(rl0): RL 前置基线与探查报告——WIP 裁决记录 + 设计勘误结论 + 域测试基线`
- **回滚**：纯文档新增，revert 无风险。
- **依赖**：串行阻塞一切（RL1-RL6 开工条件）。

### RL1 D2 写侧双写（`rl/hooks/overlay.py` + `rl/loss.py` + 契约测试翻转）

- **目标**：`loss_mode=future_kl` 首次真实抵达 vendored 分派点；vanilla run 行为字节级不变。消化 D2，消化 RLDesign §3.3 方案 A。
- **前置探查**：RL0 已确认 AGL merge 的 struct=False 存活面（`interface.py:139-141`）与 `ppo_trainer.yaml` 无 policy_loss 段；
  另需单测环境复算 `_deep_merge`（`train_tir_agent.py:278`）不会裁剪 `actor_rollout_ref.actor.policy_loss.*`（unknown-key 透传，探查命令：
  `rg -n "_deep_merge|struct" mas/train_tir_agent.py`）。
- **改动文件**（3 个）：
  1. `rl/hooks/overlay.py`：`apply_train_signal` 尾部（`:180-190` 区段，重套保护覆盖的范围内、`ensure_trainer_horizon` 调用前）
     增镜像四键：`actor_rollout_ref.actor.policy_loss.{loss_mode, future_kl_weight, decay_rate, future_kl_clip_ratio}`，
     可选透传 `chunk_size/safety_thresh/future_kl_clip_high_only`（读侧缺省见 `dp_actor.py:403-409`）。
     位置选 apply_train_signal 而非 apply_algo_overlay 的理由与 RLDesign §3.3.2 相同：受 `train_tir_agent.py:370` 区段重套保护。
  2. `rl/loss.py`：`decay_rate` 缺省 64.0 → 128.0（读侧对齐，`:19`）；删 `:16-17` 过时注释；
     `loss_applied_to_actor` 降级标记（`overlay.py:190`）随双写删除。
  3. `mas/tests/test_advantage_numerics.py`：`TestFutureKlStatusQuo`（`:461-475`）断言反转改写——
     从"future_kl 不进 actor"改为"镜像落点存在且 vanilla 路径无 policy_loss 子树"。
- **函数级测试**（新增用例并入现有文件）：
  `test_future_kl_mirror_written_to_actor_policy_loss`（appo preset + LossSpec(loss_mode="future_kl") → cfg 含四键镜像）；
  `test_vanilla_run_gets_no_policy_loss_subtree`（其它 preset 不产生 actor.policy_loss，保证 vendored 读点缺省行为不变）；
  `test_explicit_user_yaml_override_beats_mirror`（用户 yaml 显式 policy_loss 值优先于镜像——_deep_merge 序语义）。
- **回归门**：公共门 + `./run.sh feature-test rl sampling-runtime`；GPU 验证（**征询后执行**，见 §6-Q3）：
  恢复的 arpo_e2e（或替代实验）跑 `./run.sh arpo-train-test --steps 3`，断言 stderr/metrics 出现
  `actor/fipo_*` 指标族（`core_algos.py:846-853` 的发射清单）即证明分派真实走进 future_kl。
- **commit 草案**：`feat(rl1): future_kl 写侧双写——apply_train_signal 镜像 actor.policy_loss，LossSpec decay_rate 对齐读侧 128`
- **回滚**：revert 单 commit即回未接入现状；契约测试随之回转（同一 commit 内，无跨阶段耦合）。
- **风险登记**：行为变化边界 = 仅 `loss_mode=future_kl` 的 run（现网仅 appo preset 默认命中，`overlay.py:129`）；
  APPO 用户"以为在训 FIPO 实际 vanilla"的现状翻转为名实相符——release note 必须置顶；同一 yaml 重跑历史实验的
  可复现性语义变化（同一 rl.yaml 现在真的训 FIPO），预flight_revision 不变但 git_commit 变——审计链（§RL5）天然区分。

### RL2 daemon 列注入小修（`rl/hooks/daemon.py`，可与 RL1 并行）

- **目标**：根治 P0-a/P0-b 两个数据依赖缺口 + 算子缺列降级可观测。消化 RLDesign §3.1.5 两个缺口声明。
- **前置探查**：

```bash
rg -n "non_tensor_batch\[" rl/hooks/daemon.py        # 现有 10 列注入清单 :740-823
rg -n 'tir_algo == "grpo"' rl/hooks/daemon.py        # :696 提前 return
rg -n "rollout_role" mas/tests/test_daemon_expand.py # 现有测试锚点
```

- **改动文件**（1 个主 + 1 个测试）：`rl/hooks/daemon.py`——(a) `:740` 区段注入第 11 列
  `data_proto.non_tensor_batch["resume_parent_id"]`（素材现成：`parent_ids` 列表 `:765` 已构建）；
  (b) `:696` grpo 提前 return 条件化（pipeline 含 zero_prefix 或 DEFAULT_TIR `zero_prefix_adv_for_children=True` 时仍注入列）；
  (c) 新增 noop 降级路径骨架与 `training/pipeline_noop` 计数位（为 RL3 算子 no-op 预埋，本阶段只埋计数器不接算子）；
  顺带项：`:21` `from workflow.archive import load_resume_messages` → `from workflow.memory.archive import ...`（T6）。
  测试：`mas/tests/test_daemon_expand.py` 增 3 用例。
- **函数级测试**：`test_resume_parent_id_injected_for_branched_algos`；`test_grpo_with_zero_prefix_still_injects_rollout_role`；
  `test_grpo_without_zero_prefix_keeps_early_return`。
- **回归门**：公共门 + `./run.sh feature-test daemon rl branch`。注意：grpo+zero_prefix 注入列后行为差异极小但非零
  （zero_prefix 从 no-op 变真实清零）——feature-test rl 域内 `test_stage2_rl`/`test_appo_arpo_credit` 全绿即对照；
  若 appo 冻结用例受 (a) 影响（精确父子表 vs 全组兜底产出路径不同），属**预期差异**，RL0 基线需先记录该用例现状并在
  commit message 显式声明。
- **commit 草案**：`fix(rl2): daemon 注入 resume_parent_id 列 + grpo 提前 return 条件化 + noop 计数预埋 + archive 直引`
- **回滚**：三小项彼此独立，revert 单 commit 全量回退即可（不拆分 commit，保持可整体恢复）。
- **依赖**：与 RL1 文件不相交可并行；是 RL3 的前置（管线算子的输入完备性）。

### RL3 Advantage Pipeline（`rl/hooks/advantage.py` 重构 + `rl/loss.py` 扩 Spec + 等价测试）

- **目标**：`apply_tir_advantages` 内部改"base + 有序算子栈"管线，签名与 monkeypatch 宿主零变化，8 preset 字节级等价。消化 D1。
- **前置探查**：RL2 完成（列依赖齐全）；固化旧 dispatch 为对照实现的抽取点清单——
  `rg -n "^    if algo|return batch" rl/hooks/advantage.py`（8 分支 :71-112）；
  fixture 复用面 `rg -n "def make|class Test" mas/tests/test_advantage_numerics.py mas/tests/test_appo_arpo_credit.py`。
- **改动文件**（3 个 + 2 个测试）：
  1. `rl/loss.py`：新增 `AdvantagePipelineSpec`（base ∈ grpo|gigpo|igpo + operators 列表，`extra:"forbid"`）；
  2. `rl/hooks/advantage.py`：八 if-return 保留为 `legacy_apply_tir_advantages`（冻结对照实现，单一入口改读 pipeline spec，
     spec 解析兼容 v1 `tir_algo` 单选→查展开表）；算子注册表五值 `zero_prefix/apppo_credit/aepo_entropy/rae_verdict/rae_full_tgt`，
     每算子"纯 DataProto→DataProto + 缺列 no-op + 消费列声明"；rae 两算子签名规范化（内部委托 `rae_advantage.py:85/:161` 原实现，算法内核不动）；
  3. `mas/tests/test_preset_equivalence.py`（新增）：8 用例一组 `test_preset_equivalence_{grpo,arpo,appo,aepo,igpo,gigpo,rae,flow_grpo}`——
     同一 fixture 分别过 legacy 路径与 pipeline 路径 `np.testing.assert_array_equal` 逐元素断言 advantages/response_mask；
     flow_grpo 的 KL 覆写断言在 overlay 层单独用例（其 KL 是配置叠加，不进张量比较域，RLDesign §3.1.6 域外差异声明原样采纳）；
  4. `mas/tests/test_operator_composition.py`（新增）：非法组合断言（operators 重复、rae_verdict XOR rae_full_tgt——
     对齐 `advantage.py:92-99` 现状分叉、base 枚举外值）。
- **回归门**：公共门 + preset 等介试件全绿（**主件**）+ GPU 3-step 冒烟（征询）：核对
  `training/tir_algo` 指标发射正常（`daemon.py:836-843` algo_id 映射新旧两处一致）。
- **commit 草案**：`refactor(rl3): apply_tir_advantages 管线化——base×operators 重组，8 preset 逐元素等价门`
- **回滚**：revert 单文件为主体重构；等价测试文件随之删除（同一 commit）。
- **风险登记**（对齐 RLDesign 风险表 R1）：fixture 边界覆盖需含空批/全零组/孤立 anchor 三类（复用
  zero-prefix、gigpo 退化既有断言素材）；本阶段**不删除**旧 dispatch（双实现并存一个版本周期 pipeline 稳定后另行裁汰，
  比 RLDesign"此后旧路径删除"更保守——等价门须先在生产 run 上验证一轮）。

### RL4 Reward Assembly（`rl/rewards/outcome.py` 组件化 + 死配置接通）

- **目标**：format_penalty / multi_tool_bonus 可开关可配权，默认组件下数值逐值等价。消化 D4。
- **前置探查**：`rg -n "multi_tool_bonus" rl/ mas/ science_infra/ -g '*.py'`（确认仅 4 个物理点：outcome.py:154/165、
  lit_tir_agent.py:52/61（死）、overlay.py:29（死）——死配置接通面收敛）；锚点复跑清单
  `rg -c "def test" mas/tests/test_outcome_hotpot.py`。
- **改动文件**（3 个 + 测试）：
  1. `rl/loss.py` 或新 `rl/rewards/assembly.py`：`RewardComponent / RewardSpec`（RLDesign §3.2.2 结构照抄；
     首期注册 `format_penalty / accuracy_reward(不可关) / multi_tool_bonus`，`cost_penalty` 只留 schema 位 disabled）；
  2. `rl/rewards/outcome.py`：`compute_outcome_reward` 增可选参 `reward_spec: RewardSpec | None = None`——
     None 走现状字面量路径（字节级兼容），提供时按组件逐个施加；
  3. `mas/workflow/collector.py`：`default_reward_fn`（`:27-43`）增一路"reward_spec 存在则透传"分支；
     死配置接通：overlay 侧把 DEFAULT_TIR 的 `multi_tool_bonus` 写入 config 的同路径（algorithm.tir）——
     RewardSpec 经 `algorithm.reward` 下发（RL5 的 v2 段），本阶段先打通 collector→outcome 的可选参链，
     `lit_tir_agent.py` **零改动**（§3 裁决），其死参数 `self.multi_tool_bonus` 原地保留并注释标注"装饰参数，自 RL4 起由 RewardSpec 承载"。
- **函数级测试**：`test_outcome_hotpot.py` 全锚点复跑逐值比对；新增
  `test_reward_spec_none_path_byte_equal`（None vs 现状字面量逐值一致）、`test_component_off_disables_bonus`、
  `test_format_penalty_zero_is_allowed`、`test_cost_penalty_placeholder_disabled`。
- **回归门**：公共门 + `./run.sh feature-test functional mas-core rl`（reward 消费面全域）。
- **commit 草案**：`feat(rl4): RewardSpec 组件化——outcome reward 可选参路径 + multi_tool_bonus 死配置接通（热路径零改动）`
- **回滚**：`reward_spec=None` 即回现状；revert commit 即整路回退。

### RL5 rl.yaml v2 schema + 校验器 + resolved_algo 审计（组合的合同面）

- **目标**：v1/v2 双轨 schema 落地、V1-V6 三层校验、launch.json 增 `resolved_algo`、preflight 摘要展示。消化 D3 的声明半边（dict 收敛继续双轨）。
- **前置探查**：

```bash
rg -n "def build_training_plan|preflight_revision|launch = {" science_infra/control/training.py   # :320 / :614 / :819-835 落点
rg -n "TrainRunBody|preflight_stale" science_infra/control/app.py science_infra/control/training.py
cat experiments/agentflow_4algo/rl.yaml    # v1 存量样本（5 文件迁移验收面）
```

- **改动文件**（3 个 + 测试）：
  1. `rl/loss.py`：`AdvantagePipelineSpec` 补 operators 校验（无重复、XOR）+ `expand_preset(algo) -> AdvantagePipelineSpec`
     （8 preset 展开表实现，以 RLDesign §3.4.4 快照表为唯一权威）；
  2. `rl/hooks/overlay.py`：`apply_algo_overlay` 重构两步"展开→deep-merge 用户覆写"（现有 `tir.update(user_tir)` :121
     即同型逻辑的推广）；展开函数过 V1-V5 校验（V6 future_kl 连通检查在双写后自动翻转文案）；
  3. `science_infra/control/training.py`：`_launch_training`（`:796-820` 三快照写盘区）增 `resolved_algo` 字段一次序列化调用；
     `build_training_plan` preflight 内联校验（非法组合 fail-fast）+ 保存路径 `rl_contract_validation` 400 错误体（字段路径+规则号）。
- **函数级测试**：新增 `mas/tests/test_resolved_audit.py`（快照回读：launch.json 含 resolved_algo 且既有字段不丢；
  旧 run 无该字段时 TrainingHistory 渲染不报错）；`test_expand_preset.py`（8 preset 展开产物与冻结清单精确匹配、
  yaml dump 稳定排序）；V 规则断言组并入 `test_operator_composition.py`。
- **回归门**：公共门 + `./run.sh feature-test sampling-runtime control-ui` + 前置单测组；存量回归：
  5 个 `experiments/agentflow_4algo/rl*.yaml` 逐一 `python mas/train_tir_agent.py fast --rl-yaml <f> --dry-run` 式解析冒烟
  （实际无 dry-run 参数，探查 `rg -n "add_argument" mas/train_tir_agent.py` 后用最小改动方式：借 unittest 构造批训前校验）。
- **commit 草案**：`feat(rl5): rl.yaml v2 双轨 schema + V1-V6 校验器 + launch.json resolved_algo`
- **回滚**：v2 段不出现即全量回 v1 行为；revert 序列化调用与校验器各自独立。

### RL6 UI L0 参数面 + L1 组合面板 + D6 计量（可裁剪，见 §6-Q5）

- **目标**:消化 UI 通用字段差距（附录 A#10，原 D5 表述）与 D6 组级计量。前提：RL5 的 schema 与 resolved 已稳定（"点得出来的组合一定跑得通"）。
- **前置探查**：

```bash
rg -c 'field\(' webui/src/features/rl/model/hydraFields.ts                       # 23 现存
rg -n "SAMPLING_ALGORITHMS|applyWorkflowSampling" webui/src/features/rl/model/normalizeRlPayload.ts
rg -n "it\(" webui/src/features/rl/model/rlConfig.test.ts                         # 现 6 用例（冻结基线只增不删）
rg -n "training/tir_algo|branch_local_count" rl/hooks/daemon.py                    # D6 发射通道现状 :840-849
rg -n "metrics" mas/workflow/active_set.py                                        # :84 ActiveSetResult.metrics Dict
```

- **改动文件**：`webui/src/features/rl/model/hydraFields.ts`（+~40 行声明：表 1 剩余 KL 三态/tir.* 键、表 2 Advantage 组、表 3 Reward 组）、
  `types.ts`（ComposedSpec）、`normalizeRlPayload.ts`（三条归一规则）、`rlPrefills.ts`（expandPreset 前端镜像，等价表经共享常量提供）、
  新渲染容器 ComposedPanel（挂 TrainingSettings 03 区与 RlSettingsForm 各一行实例）、
  `rl/hooks/daemon.py` 组级五计数扩展（requested/valid/failed/expanded/duplicate——素材取 ActiveSetResult.metrics 键约定，
  对齐 MAS v2 §9.3）、webui TrainingConsole/TrainingHistory 列。
- **函数级测试**：`rlConfig.test.ts` 扩展（composed 归一/联动消解/旧 6 例不回退）；后端
  `test_daemon_expand.py` 增组级五计数用例（requested=卷面值 vs valid 断言对账关系）。
- **回归门**：公共门 + `./run.sh feature-test frontend`（vitest）+ webui build + `./run.sh branch-ui-test`（前置 §6-Q2）+ 手动验收脚本
  （RLDesign §0.4 两场景 walkthrough：基线复现者路径零变化、消融研究者展开→改→存→preflight 摘要可见）。
- **commit 草案**：`feat(rl6): hydraFields L0 补全 + ComposedPanel L1 + daemon 组级十计数 + TrainingConsole 列`
- **回滚**：前端独立回滚；D6 计量与 UI 可拆两个 commit（先计量后展示），项间无耦合。

### 阶段预估与裁剪优先级

| Phase | 新增/主改文件 | 净增行 | 优先级 | 可并行 |
|---|---|---|---|---|
| RL0 | 文档 1 份 | ~0 | A（必做，先于一切） | 否 |
| RL1 | overlay/loss + 测试 | ~120 | A（D2 名实相符，行为风险已圈定） | ∥RL2 |
| RL2 | daemon + 测试 | ~80 | A（一次性根治输入缺口） | ∥RL1 |
| RL3 | advantage 重构 + 2 新测 | ~400 | B（组合核心；RL4-6 的地基） | ∥RL4 |
| RL4 | outcome/assembly/collector + 测试 | ~240 | B | ∥RL3 |
| RL5 | loss/overlay/training + 测试 | ~260 | B | 依赖 RL3+RL4 |
| RL6 | webui 前端 + daemon | ~600 | C（可裁剪推迟，见 Q5） | 依赖 RL5 |

"宁少勿多"声明：若研究带宽受限，**RL0-RL2 即是完整可交付的最小闭环**（D2 连通 + 数据缺口根治 + 基线可信），
RL3 及之后每一阶段独立成立、随时可以下一阶段立项而不留残局。

与 RL_design §6 P0-P4 的映射（供评审对照）：RL0 ≈ 其 P0 的"验证与基线"半段 + 新增勘误/WIP 裁决；
RL1+RL2 ≈ 其 P0 的"D2 修复"+埋在其清单里的列注入项；RL3 ≈ 其 P1；RL4 ≈ 其 P2；RL5 ≈ 其 P0 尾段（expand/schema/校验）
+ 其 P4 前半；RL6 ≈ 其 P3 + P4 后半（UI、D6 计量）。差异理由均见 §1/§4 各阶段说明。

---

## 5. 红线与不变式（每条给验证方式）

| # | 红线/不变式 | 验证方式（命令级） |
|---|---|---|
| A1 | **训练热路径三件套零改动**：`mas/{tir_agent,lit_tir_agent,train_tir_agent}.py`。MAS 重构承诺已兑现（`git log af8bf71..57379f0 -- <三件套>` 为空），本方案延续——含 RL4 裁决：lit_tir_agent 的 multi_tool_bonus 死参数不动 | 每阶段收尾：`git diff HEAD -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py` 必须为空；任何含三件套的 diff 即阶段不过门 |
| A2 | **vendored verl 不 fork 不改**（`ref_Rep/APPO-main/APPO/verl_arpo_entropy/`）。D2 双写全部在 rl/ 写侧完成；vendored 的 `dp_actor.py:392-393` 分派点与 `core_algos.py` 双 loss 函数零触碰 | 每阶段收尾：`git diff HEAD -- ref_Rep/` 必须为空；RL1 验收重点看镜像是否只增 config 键 |
| A3 | **AGL agent-lightning 黑盒**。当前工作区存在未提交的 `agent-lightning/agentlightning/verl/daemon.py` 修改（空 triplet 过滤）——属用户 WIP，与红线冲突，RL0 必须裁决（§6-Q1），裁决完成前 A3 视为**红灯** | `git status agent-lightning/` 必须 clean（相对 HEAD 或相对用户裁决后的基线 commit）；`check_workflow_deps.py` 在 smoke 步 2 保持 OK（`run.sh smoke`） |
| A4 | **byte-level 消息序列基线不破**（MAS Phase 0 冻结件） | 每阶段跑 G4 diff（§4 公共门）为空输出 BYTE-LEVEL-OK |
| A5 | **preset 等价门**：RL3 后 8 preset 的张量输出与旧 dispatch 逐元素一致（默认参数下） | `mas/tests/test_preset_equivalence.py` 8 用例全绿；GPU 3-step 冒烟 `training/tir_algo` 发射正常 |
| A6 | **eval 污染红线**（planner_ctx latest）：评估批次默认禁 latest；启用必须显式声明并落 `memory_mode` 进 Trajectory.meta（P6 已交付机器证据，`runtime.py:376`、`centralized_runtime.py:983`） | RL5 的 preflight 摘要透出 workflow memory 段；手动用例：cv 批次 rl+workflow 快照 grep `planner_ctx: latest` 命中时 UI 摘要显著提示（不改语义只做可视化——是否硬拦截归 Harness 层，见 Q4） |
| A7 | **RL 算法只经 Trajectory 合同消费轨迹**（守卫四，MAS v2 §12）：rl/ 不读 runtime 私有状态 | 现状面清点：`rg -n "centralized_runtime|from workflow.runtime import ExecutionService" rl/ -g '*.py'` 必须为空；rl→workflow 允许面 = contracts/rollout_tree/rollout_results/active_set/archive(memory.archive)/llm_diagnostics/sampling registry（`rl/hooks/daemon.py:20-25`、`rollout_tree.py:16-17`、`overlay.py:66/80`、`train_signal.py:12`）——该白名单进 rl0_probe_report 固化 |
| A8 | **绝对不捎带 WIP**：每阶段逐路径 `git add`；工作区现存的 mas/workflow/user_gateway/*、specs 模板、docs 批量 D 状态、agentflow_4algo 相关未跟踪物，一概不进 RL commits | 每阶段 `git show --stat HEAD` 审查仅含本阶段文件；commit 前后 `git status` 差异仅允许本阶段清单内的路径变化 |
| A9 | **v1 algo 单选永久保留**（双轨）：CLI `--algo`、`applyWorkflowSampling` 五档映射、`experiments/*/rl*.yaml` 存量零迁移 | RL5 门：5 个存量 yaml 全部照常展开（探查命令见 RL5 节）；`rlConfig.test.ts` 旧行为用例不回退 |
| A10 | **不动 RL_design.md 原文除非走勘误 commit**：本方案附录 B 是勘误底稿，执行须用户批准后独立 commit（本方案自身零改动该文件） | git status docs/RL_design.md 保持 `??`（未跟踪新增状态）或用户批准后的专用勘误 commit，不与其他阶段混合 |

---

## 6. 待用户确认项

| # | 事项 | 需要的决策 | 影响范围 |
|---|---|---|---|
| Q1 | **三处未提交行为修改的归属**：`rl/hooks/overlay.py`（KL 显式存活）、`rl/rewards/outcome.py`（strip_thinking 截断 + numeric_match 多候选）、`agent-lightning/agentlightning/verl/daemon.py`（vendored AGL 改动）。RL_design 是对工作区状态写的——若按 HEAD 落基线，RL_design 的 KL 行为断言将失真 | 三选一each：(a) 先行独立提交为前置 commit（推荐 overlay/outcome，各带测试）；(b) stash 至对应阶段合入；(c) 放弃回退。**AGL daemon.py 修改强烈建议 (a) 单独 commit 并评审其空-triplet 过滤与 D6 计数的交互**，或回退 | RL0 开工前置；影响 A2/A3 门 |
| Q2 | **GPU 门实验目录前提变化**：`experiments/arpo_e2e`、demo、alpha 已被删除，而 `./run.sh arpo-train-test`（`run.sh:1334`）与 `branch-ui-test`（`run.sh:1281`）及两脚本默认 `--experiment arpo_e2e`。RL1/RL3 的 GPU 验证与 RL6 的 branch-ui-test 全部踩这条路 | (a) 恢复 arpo_e2e bundle（git 历史里有）；(b) 新建最小实验目录并提供脚本 `--experiment` 覆盖参数说明；(c) 改脚本默认值（.py 变更，需评审） | RL1/RL3 GPU 门；RL6 门；本方案不替用户执行删除恢复 |
| Q3 | **GPU 验证窗口**：RL1 双写前后各需一次短步数实跑（前置 Q2 解决）；RL3 管线化后 3-step 冒烟一次 | 排期与卡型（A800×1 或 ×2）；非阻塞项可否降级为 CPU 单测等价论证 | RL1/RL3/RL5 阶段时长 |
| Q4 | **eval 污染红线的机器化程度**：preflight 对 eval 批次 planner_ctx=latest 只提示还是拦截 | 拦截动作落在 Harness 还是 RL preflight（当前方案倾向：只提示不拦截） | RL5 |
| Q5 | **RL6 是否进本轮**：UI 面是研究员直接受益面，但 600 行前端改动是最大单项；RL5 交付后研究员已可用手写 v2 yaml + CLI 完成 A-G 矩阵全部实验位（RLDesign §1.7 零成本支撑声明的前半句） | 本轮范围是否含 RL6，或冻结为下一立项 | 范围与排期 |
| Q6 | **RL_design.md 勘误执行**：附录 B 清单（行号漂移 8 处、悬空链接 2 处、LOSSES 与红线矛盾的 #6 条目、1098→1097 行数、22/23 字段数不一致等） | 是否批准独立勘误 commit（doc-only，马上可做，不影响代码阶段） | 文档可信度 |
| Q7 | **1036 行的 flow_grpo 定位重命名**（RLDesign ADR 系外的溢出项，无）——**无此项**，占位防误读；真实第七项见 Q1-Q6 | — | — |

---

## 附录 A 差距矩阵全量（RL_design 断言 → 代码现状 → 定性 → 修订动作）

定性口径：**一致** = 断言与现状吻合；**偏差** = 机制存在但表述/行号漂移；**缺失** = 设计有、代码无；**过时** = 前提已变化；
**反向缺失** = 代码有、设计未捕获。

| # | RL_design 断言（出处） | 代码现状（file:line） | 定性 | 修订动作 |
|---|---|---|---|---|
| 1 | §0.2 "三合同来自 MAS_design §11" | MAS v2 §11（`docs/MAS_design.md:628-636`）只编号两个合同 + 边界散文 | 偏差 | 勘误：合同 3 改引"§11 边界段 + §12 守卫四" |
| 2 | 头部配套阅读含 MAS_AGENT_LANDING.md（`:5`） | 该文档已删除（git status D）；`MAS_design.md:6` 同悬空 | 过时 | 勘误删链接（Q6） |
| 3 | §0.2 合同 1 消费闭环：format_ok/n_search/n_python → reward | `contracts.py:210-215` sync_tir_meta 三键进 meta；`collector.py:36-43` 喂入 compute_outcome_reward ✓ | 一致 | 无 |
| 4 | §2.1 转换点 3 处"全部走 batch_to_train_signal"（train_tir_agent.py:365-381 / services.py:820 / cli.py:85） | 构造器唯一（`rl/train_signal.py:27-30`）✓；后两点经 `workflow/__init__.py:60-70 __getattr__` 懒转发到 rl.train_signal（P3 新拓扑） | 偏差 | 勘误补"workflow 再导出面"；rl0_probe 固化该转发为外部 API |
| 5 | §2.1 结构表 13 文件行数（loss 52/train_signal 30/outcome 234/overlay 213/advantage 306/rae 371/daemon 877/gigpo 101/rollout_tree 211/trainer 54/branch_policy 106/arpo_rollout 185/training_batch 32） | `wc -l` 逐一对上，13/13 ✓ | 一致 | 无 |
| 6 | §2.1 消费点 8 处（train_tir_agent:24/228/352；tir_agent:21,28；collector:18；gates:9；configured:7；training.py BRANCHING_ALGOS 预检；services:1236；rollout_runs:734；legacy mas_agent:21） | 实测 train_tir_agent:24/228/352 ✓、tir_agent:20-29 ✓、collector:18 ✓、gates:9 ✓、configured:7 ✓、legacy/structagent/mas_agent.py 引 rl ✓；控制层三点实测位置略移（rollout_runs.py:734 区域 accuracy 直引 ✓） | 一致（行号微漂） | 勘误行号对齐（附录 B） |
| 7 | §2.4 D2：写侧 algorithm.tir（overlay.py:186）↔ 读侧 actor.policy_loss（dp_actor.py:393）不通 | overlay.py:186 ✓ 实；dp_actor.py `:392-393 fipo_cfg=self.config.get("policy_loss"...)` ✓ 实；`ppo_trainer.yaml` 无 policy_loss 段 ✓；缺省漂移 decay 64（`loss.py:19`）vs 128（`dp_actor.py:404`）✓ | 一致（差距存在） | RL1 实施双写 |
| 8 | §2.4 D3：DEFAULT_TIR 26 键野 dict 无 schema | `overlay.py:18-43` 恰 26 键 ✓；全消费走 `tir_config.get(..., 默认)` 自由兜底（`daemon.py:583-587` 等）✓ | 一致 | RL5 Spec 承载开始收敛 |
| 9 | §2.4 D4：format 罚 -1 与 bonus 0.1 硬编码 | `outcome.py:160-161` 字面量 -1.0 ✓；`outcome.py:154` 函数默认参数 0.1（call site 不传）✓ | 一致 | RL4 |
| 10 | §2.4 D6 前半：UI 仅通用字段，DEFAULT_TIR 26 键 / KL 三态 / loss_mode 不可见 | `webui/src/features/rl/model/hydraFields.ts:57-79` 恰 **23 项 field()**（第一处 42 行是 `function field(` 定义非条目；grep -c 'field(' = 24 含定义），条目键全为通用 verl/data/trainer 键，无 tir.* ✓（RL_design 文内"22 个"与"23 项"自相矛盾，实测条目 23） | 一致（文内数字不一致） | 勘误统一 23 |
| 11 | §2.4 D6：组级 requested/valid/failed/expanded/duplicate 未落盘 | daemon 现 6 项 metrics（`daemon.py:840-849`）无组级五计数；`ActiveSetResult.metrics` 为自由 Dict（`active_set.py:84`）无键约定 ✓ | 一致（缺失存在） | RL6 |
| 12 | §2.3 关键事实 1-4（八预设真相表） | 逐行核验：flow_grpo 直通 :71-73、arpo return :74、appo credit :76→118、grpo zero_prefix :81/182、rae :84-102（full_tgt 分叉 :92-99 ✓）、aepo :104→193、igpo :106→215（def 实在 :219，行号微漂）、gigpo :108→gigpo_core:19 | 一致 | 勘误 igpo 行号 |
| 13 | §2.5 亮点：preflight sha256（training.py:614）/ 三快照（:795 区段） | training.py:612-618 哈希构造 ✓、:758-765+819-835 写盘 ✓（launch.json schema_version=1 现字段清单实测：run_id/request_id/experiment_id/git_commit/created_at/preflight_revision/algorithm/algorithm_source/profile/gpu_ids/model/argv/snapshot） | 一致 | RL5 的 resolved_algo 由此落点扩展 |
| 14 | §2.5 亮点：_deep_merge 透传（train_tir_agent.py:278）+ ensure_trainer_horizon（overlay.py:198） | `:278 def _deep_merge` ✓、overlay `:196-212` ✓ | 一致 | 无 |
| 15 | §2.5 亮点：arpo/appo budget 守卫（train_tir_agent.py:430-444） | `:433-450` 区段实测（tir_algo in arpo/appo → sampling_budget≤1 解释退出）✓ | 一致 | 无 |
| 16 | §2.2 波次结构 daemon.py:209（波1）/420（等待）/444（runner 扩展）/688（batch） | `:209 _async_set_up`、`:420 _wait_for_training_rollouts`、`:444 _enqueue_from_runner_expansions`、`:688 get_train_data_batch` 全 ✓ | 一致 | 无 |
| 17 | §3.1.5 列依赖表（生产者行号） | anchor_obs :740 ✓、ig_logprob_delta :741 ✓、rollout_role :819 ✓、site_id_list :820 ✓、verdict_list :821 ✓、resume_boundary :822 ✓、action_key_list :823 ✓ | 一致 | 无 |
| 18 | §3.1.5 resume_parent_id "AGL 未注入 non_tensor_batch" | daemon 从不写该列（:740-823 无）✓；span meta 有（lit_tir_agent.py:321）✓ | 一致（缺口存在） | RL2 |
| 19 | （设计未声明）工作区三处未提交行为修改 | overlay.py（KL 存活语义）、outcome.py（strip/numeric 增强）、**AGL vendored daemon.py（红线冲突，设计漏登）** | **反向缺失** | RL0 裁决（§6-Q1） |
| 20 | §3.3 方案 A 可行性"低验证成本——纯 dict 加四行" | AGL `interface.py:139-141` struct=False merge 实证新键存活——可行性从推测升实据 | 一致+证据加强 | RL1 按此实施 |
| 21 | 全文 file:line 引用精确性 | 8 处漂移（附录 B 逐条）：overlay 143-151→140-155（因未提交改动）、core_algos 503→458、rollout_trees 84/129/215→85/130/216、collector 26-46→27-43、igpo 215→219、训练清点若干 | 偏差（系统性小漂） | 勘误 commit（Q6） |
| 22 | §4.2 附注二：entropy_weight 双缺省 0.5（DEFAULT_TIR）vs 0.2（daemon 读侧） | `overlay.py:23` 0.5 ✓；`daemon.py:586` get 缺省 0.2 ✓；`:284` 预注入读 0.5（第三处差异，设计未列） | 一致（+新证据） | RL5 合同冻结时三处统一；勘误补第三点 |
| 23 | §3.6 #6 "mas/lit_tir_agent.py（参数链收编）" vs §6 红线"三件套零改动" | lit_tir_agent multi_tool_bonus :52/:61 死参数实测确认（:247 不消费）；红线两处文本互斥 | 缺陷（设计内部矛盾） | 方案裁决：不改热路径（§3/#23）；勘误修正 #6 |
| 24 | §3.2.2 "collector.py:33 调用不传 multi_tool_bonus 属死配置" | 实测 `:36-43` 调用确不传 ✓（行号微漂 33→36） | 一致 | RL4 接通 |
| 25 | §2.3 P0-b：grpo 提前 return 使 grpo+zero_prefix 拿不到列 | `daemon.py:696-704` ✓；grpo 默认 zero_prefix=True（overlay.py:35）+ advantage.py:81 调 _maybe_zero_prefix ✓ | 一致 | RL2 |
| 26 | §3.7 roll 未来算子三预留位（mu_ref_list / regime_list / adv_batch_var 列） | 代码无任何预留 ✓（rg 全仓无） | 缺失（按设计属"不实现"） | 保持不实现；预留位仅登记 |
| 27 | §4.3 组合面板 / §4.2 表 1-3 的 UI 声明 | webui 无 ComposedPanel/composed types；hydraFields 23 项全通用 ✓ | 缺失 | RL6 |
| 28 | §4.4 preflight 摘要/resolved 字段、TrainingConsole/History 列 | 均无（launch.json 无 resolved_algo；TrainingConsole 现显 meta.algo） | 缺失 | RL5/RL6 |
| 29 | §5 V1-V6 校验器三层挂点（保存/preflight/启动） | 无任何组合校验代码；现有 rbudget 守卫与 VALID_ALGOS 校验是其最小前体（train_tir_agent.py:365/435） | 缺失 | RL5 |
| 30 | （设计未捕获）TIR_EXECUTION_TREE_VERSION=3 恒定 | training.py:800 发"3"；rollout_tree.py:37 读 env 缺省"2"（实际恒 3，缺省值死代码） | 反向缺失（无害） | 归档 rl0_probe；不修（非缺陷） |
| 31 | ADR-5 "v1 存量 experiments/*/rl_*.yaml 永久可跑" | experiments/ 仅存 agentflow_4algo（5 个 rl yaml，全 v1）→ 存量义务收窄 | 过时（前提变化） | 勘误 + RL5 验收面按 5 文件计 |
| 32 | §7 测试设施断言（pytest pythonpath [".","mas"] pyproject.toml:53；rl 相关测试 14+ 清单；rlConfig.test.ts 6 用例） | pyproject:53 ✓ 精确；测试清单逐个在 ✓；rlConfig.test.ts 恰 6 it()（`:27/40/54/70/76/90`）✓ | 一致 | 无 |
| 33 | §7 计划新增 test_preset_equivalence.py | 不存在 ✓（rg 无） | 缺失（计划内） | RL3 建件，此为等价门主件 |
| 34 | §0.2 合同 2 消费闭环：sites 同 schema、resolved_sites 显式优先 | `overlay.py:66-80` SamplePolicy.model_validate + resolved_sites ✓；`overlay.py:91-100` sites dump 进 tir ✓ | 一致 | 无 |
| 35 | §1.1 表 "compute_outcome_reward（outcome.py:145）" / "apply_tir_advantages（:47）" / "gigpo_core:36" | outcome.py:145 def ✓；advantage.py:47 def ✓；gigpo_core.py:19 def（:36 是函数体内 A^E 构造区，断言的"双级优势"正文即在此区）✓ | 一致（gigpo 锚点措辞需细化为 :19 def/:36 权重区） | 勘误微调 |
| 36 | §1.3 表 anchored core_algos.py:113 GRPO | 未逐行验证（vendored 大文件；上下游断言链不依赖此精确行） | 未核 | RL0 复验（列入 probe 清单） |
| 37 | §6 P0 "expand_preset 8 preset 人工核对语义"指向 §3.4.4 快照表 | 快照表内容与 §2.3 真相表/§3.1.6 展开表已三方互证（本方案 §3 重点#12 的核验覆盖关键行） | 一致 | RL3 展开表以此为准 |
| 38 | §4.3 联动消解依赖 applyWorkflowSampling 五档映射 | `overlay.py:70-79` 六行映射表（grpo_n/grpo/arpo/aepo/appo/rae→algo）实测；webui 侧 SAMPLING_ALGORITHMS 前件待 RL6 再验 | 一致（后端侧） | RL6 验前端侧 |
| 39 | rolout hook 对 .tir_archives 的兼容担忧（Task 设问） | hook 不读 archives 树：`rl/hooks/rollout_tree.py` 仅经 TREE_ENV_KEYS（:22/:29-37）写 run_dir/rollout-trees（training.py:797 注入） | 反向缺失（担忧不成立——好消息） | rl0_probe 记录"无兼容面" |
| 40 | §2.1 "rl/rewards/outcome.py 是唯一 reward 实现（TIR 主栈 + legacy 栈）" | `rg "compute_outcome_reward" mas/ science_infra/` 全走 rl.rewards ✓（tir_agent:28、collector:18、runtime.py:27 分键直引）——P3 后消费形态实测：runtime.py 只引 `has_answer_format`、collector.py 引 `compute_outcome_reward + parse_alias_field`、tir_agent.py 经 `rl.rewards` 包面引 5 函数 | 一致（消费键有细分） | rl0_probe 固化三处消费键清单 |
| 41 | （Task 设问 6）prompt_hash/prompt_source/memory_mode 是否被 rl/ 消费 | rl/ 侧零消费零校验（`rg -n "prompt_hash|prompt_source|memory_mode" rl/ -g '*.py'` 零命中）；训练数据流 emit_annotation 27 键（lit_tir_agent.py:305-343，见 #47）不含 → **不丢失也不消费**（设计未捕获，见 §2 T1/T2） | 反向缺失 | 观测面透出（RL5 摘要），不入张量流 |
| 42 | （Task 设问 9）phase0_external_api.md 的 RL 调用面冻结项对照 | 冻结面实测有效："rl/" 为 repo 根顶层名、run.sh 子命令不改名（§4 第 50-58 行）、feature-test 域名清单（:54-58 权威）；**RL 方案越合同项 = 零**（本方案所有改动均落 rl/、mas/tests/、science_infra/control/training.py、webui——training.py 属控制面非冻结面，MAS v2 §11 边界段允许） | 一致（未越冻结合同） | rl0_probe 登记本方案改动面 ⊆ 非冻结面 |
| 43 | §4.1 图中"实线链路全部现状已有"（saveExperiment saveOrder :9-31、preflight/409 :471、三快照写盘 :795-823、观测通道） | 后端链实测全有（app.py:471 create_train_run、training.py:758-765+819-835、daemon metrics）；webui 侧 saveOrder/normalizeRlPayload/useRlDraft 存在但本轮未逐行验测，余给 RL6 前置探查复核（对 MAS 重构零依赖，可后置） | 一致（后端）; 待验（前端） | RL6 前置探查清单 |
| 44 | daemon metrics 发射断言 ":843-847 六项" | 实测 `:840-849` 共 6 键（tir_algo/n_with_anchor/branch_local_count/store_enqueue_branch_count/incremental_branch_count/expansion_enqueue_failures）——比设计书多一项 `expansion_enqueue_failures`（新增失败计数） | 一致（+1 键） | D6 组级五计数扩在同一 dict；勘误补记 |
| 45 | `TIR_EXECUTION_TREE_VERSION=3` 由 `_launch_training` 注入 | training.py:800 恒 "3" ✓；rollout_tree.py:37 读 env 缺省 "2" 实为死缺省（恒被覆盖）——无害 | 一致 | 不修；rl0_probe 登记 |
| 46 | trainer.py "18-41 monkeypatch + 44-54 bound_daemon_cls" | `:16-32` `_train_step`、`:38-40` try/finally、`:44` bound_daemon_cls（文件仅 54 行） | 一致（微小行位差） | 勘误可选 |
| 47 | lit_tir_agent "34 键 tir meta"（RL_design §2.2 断言） | 三法独立重数一致 = **27 个 `tir.*` 键 + 1 个 RESULT_ATTRIBUTE = 28 键**（`sed ':296,344p' | grep -o '"tir\.[a-z_]*"' | sort -u` = 27，无重复键；awk 区间计数亦 27），非 34 | 偏差（数字勘误） | rl0_probe 存证；勘误改 27+1 |
| 48 | §7 专测资产与 feature-test 域映射 | rl 域= stage2_rl+appo_arpo_credit（feature_test.py:27）；branch 域三件（:29-31）；sampling-runtime 九件（:81-90）；**test_advantage_numerics.py 不在任何域**、test_flow_grpo/test_outcome_hotpot/test_functional_rl 亦无域归属（仅 unittest 全量跑） | 偏差（覆盖拓扑遗漏） | RL0 基线记录；随 RL3 评审 rl 域增补 |

---

## 附录 B RL_design.md 勘误清单（Q6 批准后一个 doc-only commit）

| # | 位置 | 现文 | 实况 | 建议改法 |
|---|---|---|---|---|
| B1 | 头部:5 | 配套阅读含 MAS_AGENT_LANDING.md | 已删除 | 删该链接（MAS_design.md:6 同样处理） |
| B2 | 头部:3 | 素材"1098 行" | 实测 1097 行（wc -l） | 改 1097 |
| B3 | §0.2 合同 3 行 | "来自 MAS_design §11" | §11 两个编号合同+边界散文 | 改引"§11 边界段 + §12 守卫四" |
| B4 | §2.1 消费图 | services.py:820 / cli.py:85 直引 | 经 workflow/__init__ __getattr__ 懒转发（:60-70） | 补"第 9 消费面：workflow 再导出面（P3 拓扑）" |
| B5 | §1.1/§2.2 | core_algos.py:503 vanilla clip | 实 :458（future_kl :688 不变） | 503→458（涉及 2 处） |
| B6 | §2.3 | overlay.py:137-152 flow_grpo KL | 实 :140-155（未提交 KL 存活改动后） | 更新区间（若 Q1 裁决先落 commit，以落定后行号为准重测一次） |
| B7 | §2.3 | aepo advantage.py:193 ✓ 保留 / igpo :215 | _apply_igpo def 实 :219 | 215→219 |
| B8 | §4.4 | rollout_trees.py:84/129/215 | 实 :85/:130/:216 | +1 修 |
| B9 | §3.2.2 | collector.py:33 | 实 :36-43 | 33→36 |
| B10 | §2.4 D5 | "22 个通用 hydra 字段…23 项" | 实 23 项一处口径 | 统一 23 |
| B11 | §3.6 #6 | 改 mas/lit_tir_agent.py 参数链收编 | 与 §6 红线互斥；死参数实测 | 改"lit_tir_agent 零改动；参数链收编经 overlay→collector（见 rl_refactor_plan §4 RL4）" |
| B12 | §6 P0 前置 | "工作区 2 个未提交" | 实 3 个（+AGL vendored daemon.py） | 补登第三处并标注红线冲突 |
| B13 | §4.2 附注二 | entropy_weight 双缺省两处 | 第三处：daemon.py:284 预注入读 0.5 | 补第三点 |
| B14 | ADR-5 | "存量 experiments/*/rl_*.yaml" | 现仅 agentflow_4algo 5 件 | 存量面按实况注明 + arpo_e2e 前置变化脚注 |

---

## 附录 C 探查命令集（真实 rg 模式，RL0 复跑用）

```bash
# —— rl/ 双向引用面 ——
rg -n "from workflow\.|import workflow" rl/ -g '*.py'                 # rl→workflow：6 文件 10 处
rg -n "from rl\.|import rl\." mas/ science_infra/ -g '*.py'           # →rl：8 生产文件 + 测试若干（rg -l 计数）
rg -n "from rl\." mas/tests/ -g '*.py' | wc -l                        # 测试侧消费量

# —— 核心断言复核 ——
rg -n "loss_mode|future_kl" rl/hooks/overlay.py rl/loss.py            # D2 写侧全键位
rg -n "policy_loss" ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/workers/actor/dp_actor.py   # D2 读侧 :392-412
rg -n "policy_loss" ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/config/ppo_trainer.yaml  # 空 = 树不通三证
rg -n "def compute_policy_loss" ref_Rep/APPO-main/APPO/verl_arpo_entropy/verl/trainer/ppo/core_algos.py  # 458/688
rg -n "set_struct|OmegaConf.merge" agent-lightning/agentlightning/algorithm/verl/interface.py  # :139-141 struct=False
rg -n 'if algo' rl/hooks/advantage.py                                  # 八互斥分支 :71-112
rg -n "-resume_parent_id|non_tensor_batch\[" rl/hooks/daemon.py       # 列注入清单 :740-823（无 resume_parent_id 即 P0-a）
rg -n 'tir_algo == "grpo"' rl/hooks/daemon.py                          # :696 提前 return（P0-b）
rg -n "multi_tool_bonus" rl/ mas/ science_infra/ -g '*.py'            # 死配置 4 物理点
rg -c 'field\(' webui/src/features/rl/model/hydraFields.ts            # 23
rg -n "it\(" webui/src/features/rl/model/rlConfig.test.ts             # 6 用例基线

# —— 工作区与门 ——
git status --short | grep -E 'rl/|agent-lightning/|docs/'              # 三处未提交 + docs 批量 D
git log --oneline af8bf71..57379f0                                     # MAS P0-P6 七 commit
git diff HEAD -- mas/tir_agent.py mas/lit_tir_agent.py mas/train_tir_agent.py   # 热路径零改动（须空）
./run.sh feature-test --list                                           # 域清单权威
rg -n "arpo_e2e" run.sh scripts/                                       # GPU 门实验前提
rg -n "TIR_EXECUTION_TREE_VERSION" science_infra/control/training.py rl/hooks/rollout_tree.py  # 树 schema 3 恒定
PYTHONPATH=mas:. ./.venv/bin/python docs/refactor/freeze_msgseq_baseline.py --check   # G4
```

## 附录 D 本方案自身的红线自查

- 本方案执行过程：新建 `docs/refactor/rl_refactor_plan.md` 一份，未触碰任何 `.py` / rl/ / mas/ / webui/ / ref_Rep/ / agent-lightning/ 文件；
  未修改 `docs/RL_design.md` 与 `docs/MAS_design.md`。
- 文中全部代码断言的核验时间为 2026-10-14 上午，基于工作区现状（含用户未提交项——已在 §3#19 显式标注其影响），
  未读任何 `.pyc`；所有行号均为实测行号（非引用自设计文档自述）。
- 与 MAS v2 章节对齐口径：记忆 §4、ToolAgent/tier §5、planner_ctx §6、Prompt §7、两模式 §8、分支采样 §9.3、
  contracts §10、接口 §11、守卫 §12、Phase §13、测试 §14、ADR §16——本文档所有 "MAS v2 §x" 引用已按此实文重核。
