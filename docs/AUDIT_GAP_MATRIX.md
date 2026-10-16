# 审计差距矩阵（MAS / RL / user_space）

日期：2026-10-15
对照：[MAS_design.md](./MAS_design.md) v2 · [RL_design.md](./RL_design.md) v1 · 官方参考 `ref_Rep/AgentFlow-main`、`user_space/projects/epc-aw-main/upload/EPC-AW-main`

每行一个差距：现象、证据、处理、验收。状态取值：`fixed`（本轮已修，测试锁定）/ `deferred`（明确推迟，见触发条件）/ `known`（诚实限制，文档声明）。

## 1. 测试与环境基线

| # | 差距 | 证据 | 处理 | 状态 | 验收 |
|---|---|---|---|---|---|
| T1 | 34 个测试模块未挂入 feature-test（含 Flow-GRPO、AgentFlow、EPC-AW、8 preset 等价） | `scripts/feature_test.py` 原域表 | 新增 `rl-algos` / `user-projects` / `infra` / `pytest-extra` 四个域，并入 `--all` | fixed | `./run.sh feature-test --list` |
| T2 | 测试依赖已删的 `experiments/arpo_e2e/workflow.yaml` | `test_schema03_tool_agents.py` | 从 git 历史取回为 `mas/tests/fixtures/centralized_router.yaml`（本地 1.7B 端点） | fixed | `feature-test schema03` |
| T3 | `ref_Rep/HIVE` 已删，数据集测试与写保护测试报错 | `test_task_files.py`、`test_eval_sources.py`、`test_hive_wrap.py` | 数据集改指同源的 `ref_Rep/EPC-AW/test`；HIVE 源码写保护在源缺失时 skip | fixed | `feature-test studio user-space` |
| T4 | registry 列出 `hive` 为 ready，但项目树与上游源码都不存在 | `user_space/registry.yaml` | 移除孤儿条目；scaffold 能力仍由 `user_gateway/hive.py` 提供 | fixed | UI 项目列表不再出现不可运行项 |
| T5 | `_normalise_chat_messages` 丢掉 `tool_calls` / `tool_call_id` | `mas/workflow/runtime.py`；`test_resume_messages_keep_snapshot_prefix` | 保留 OpenAI 角色消息上的工具调用字段 | fixed | `feature-test functional` |
| T6 | 搜索缓存测试在 `.env` 先加载时绕过 mock 打真实 Bright Data | `test_search_cache.py` | wiring 测试 setUp 内关闭 Bright Data key | fixed | `feature-test infra` 与顺序无关 |
| T7 | EPC-AW `adapted/context.py` 直接 `from tools...`，可能误命中管理面 `mas/tools` | `check_user_zone` FAIL | `_upstream_tool_modules()`：按两种导入名加载并校验文件落在 EPC-AW 源树 | fixed | `mas/scripts/check_user_zone.py` OK |
| T8 | 默认训练 / serving 模型是 4B；1.7B 只在 `rl_gpu_gate` | `train_tir_agent.py`、`serve_local_llm.sh` | 门禁与新实验一律 1.7B；4B 保留为可选 | fixed | `scripts/gate_1p7b.sh` |

本地资源：2×A800-80GB；`LLM/Qwen3-1.7B`（Qwen3ForCausalLM，28 层，40960 ctx）；vLLM 0.9.2 / torch 2.7.0；`data/train.parquet` 20 条 + `val.parquet` 5 条 HotpotQA；`data/search_cache.json` 可离线回放搜索。

## 2. AgentFlow（官方身份）

| # | 差距 | 证据 | 处理 | 状态 |
|---|---|---|---|---|
| A1 | 源树指向不存在的 `ref_Rep/agentflow` | `manifest.yaml`、`adapted/context.py` | 改指 `ref_Rep/AgentFlow-main/agentflow` | fixed |
| A2 | 窗口调用 EPC-AW API（Diagnoser / outline / BTS / 三份 memory） | `context.py` 原 `verify_window` | 重写为官方 Planner–Executor–Verifier：共享 `Memory`、`Verifier.verificate_context`、`planner.generate_direct_output` | fixed |
| A3 | 冻结模块与可训 planner 共用同一引擎，且 tracer 只记 planner 以外也会进 batch | `_patch_engines_for_local`、`train_tir_agent.py` 未设 `agent_match` | 只有 `planner.llm_engine`（`generate_next_step`）发 `agent.planner` token span；其余调用不进 tracer | fixed |
| A4 | lite prompt 读 EPC-AW 的 `prompts/*.txt` | `OPTIMIZATION.md` | 官方提示写在 `.py` 内，lite 改为窗口层的上下文截断，不再替换模板文件 | fixed |
| A5 | 工具默认集与论文不一致 | `runtime.yaml` | 默认 `Base_Generator / Python_Coder / Google_Search / Wikipedia_Search`；`Web_Fetch` / `Bing` 为可选扩展 | fixed |

## 3. EPC-AW（论文身份）

| # | 差距 | 证据 | 处理 | 状态 |
|---|---|---|---|---|
| E1 | `n: 1` 让 BTS / belief / `epc_aw_diagnosis` 永远不跑 | `contracts/runtime.yaml` | 双 profile：`cheap`（n=1）与 `paper`（n≥2）；`SCIENCE_EPC_AW_PROFILE` 选择 | fixed |
| E2 | belief / report / diagnosis 只有长原版模板 | `adapted/prompts/` | 补 lite 模板，`n>1` 时不打满 1.7B 上下文 | fixed |
| E3 | 没有绑定 EPC-AW 的实验 | `experiments/` | 新增 `experiments/epc_aw_1p7b/` | fixed |
| E4 | BTS 6 路串行 | `_select_plan` | 不阻塞门禁 | deferred（触发：paper profile 用于大批量 eval） |

## 4. RL / Flow-GRPO

| # | 差距 | 证据 | 处理 | 状态 |
|---|---|---|---|---|
| R1 | sibling `workflow.sampling.mode` 改写 rl.yaml 的 `algo` | `train_tir_agent.py` → `apply_sample_policy` | `apply_sample_policy(..., algo=)`：rl.yaml 显式算法优先，sampling 只调 group_n / sites / beam | fixed |
| R2 | `flow_grpo` 不是任何 sampling adapter 的别名 | `sampling/adapters/independent.py` | 注册为 independent 别名 | fixed |
| R3 | Flow-GRPO 用层级 outcome（-1 / F1 / +0.1） | `rl/rewards/outcome.py` | `RewardSpec.outcome="binary"`：答对 1 否则 0；`flow_grpo` preset 默认 | fixed |
| R4 | planner-only 过滤关闭 | `train_tir_agent.py` | user_space 窗口图下启用 `agent_match`，正则对齐 `agent.planner` span | fixed |
| R5 | RL_design 把 Flow-GRPO 写成 flow-matching SDE | §1.3 | 改为 AgentFlow 论文语义（in-the-flow planner 优化） | fixed |
| R6 | RL_design 仍写 D2（future_kl）未连通 | §2.3/§2.4 | 以代码为准：`apply_train_signal` 已双写 `actor.policy_loss` | fixed |
| R7 | token ids 为 tokenizer 重编码，不是 vLLM 原值 | `_install_tracer_token_recorder` | 发 `retokenize_roundtrip_ok` 指标 | known |
| R8 | 无 GPT-4o 裁判 / NQ+DeepMath 数据 / 7B 分数 | — | 1.7B + 本地 HotpotQA 只验证链路，不复现论文数字 | known |

## 5. 设计稿与代码对齐

| # | 差距 | 处理 | 状态 |
|---|---|---|---|
| M1 | MAS_design §8.3 写 user_space 不进训练，代码已经进（`agent_window_rollout`） | 新增 Mode B-window 合同（§8.5） | fixed |
| M2 | TECHNICAL_FRAMEWORK / README 算法表缺 `flow_grpo`、APPO 写成 ARPO 别名 | 文档修订 | fixed |
| M3 | `mas/docs/DESIGN.md` 指向已删 `mas/algos/` | 改指 `rl/` | fixed |
| M4 | `topology: decentralized` 无实现 | — | deferred（MAS ADR-12） |
| M5 | `CreditAssignmentSpec` 无调用点；archive 6 行 shim | — | deferred |
| M6 | L1 组合面板之外的 L3 研究方向（RL_design §1.6） | — | deferred（RL_design §3.7 立项门槛） |
