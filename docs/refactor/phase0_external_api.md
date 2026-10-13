# Phase 0 外部兼容接口清单

日期：2026-10-13
状态：Phase 0 产出。本清单枚举"重构期间不许破"的全部外部可见接口。
方法：静态 grep + `mas/tests` 引用统计（412 处 `from workflow.*`）。

---

## 1. 顶层 import 名（sys.path=mas/ 与 sys.path=repo/ 双根）

以 `sys.path.insert(0, mas)` 拉起的顶层名（测试与控制层依赖）：

| 顶层名 | 物理位置 | 引用量级 | 重构动作 |
|---|---|---|---|
| `workflow` | mas/workflow/ | mas/tests 412 处 | 不动（平铺保持，见 ADR-1） |
| `tools` | mas/tools/ | tests/collector | 不动 |
| `tir_agent` | mas/tir_agent.py | 训练入口 | 零移动红线 |
| `lit_tir_agent` | mas/lit_tir_agent.py | 训练入口 | 零移动红线 |
| `train_tir_agent` | mas/train_tir_agent.py | control/training.py:591 文件路径拉起 | 零移动红线 |
| `mas_agent` | mas/mas_agent.py | lit_tir_agent.py:122 顶层名懒加载 import | **Phase 1 特例：移 legacy/ 后在 mas/ 留 6 行 re-export shim** |
| `MAS`（动态注册名） | bootstrap_mas_package 注册 | vendored 包内部 `from MAS.epc_aw...` | 注册名不变，`__path__` 改指 legacy/structagent/epc_aw/ |
| `algos` | mas/algos/（6 文件 shim → rl/） | tests（stage1/2/7） | Phase 3 删除，同步改测试 |

以 `sys.path=repo/` 根拉起的名：`rl.*` / `science_infra.*` / `webui` / `legacy.*`（新增）。

## 2. 事件 payload 契约（archive 消费方 schema）

| 事件 | payload 键 | 消费方 |
|---|---|---|
| MEMORY_WRITE (fact_commit) | `phase="fact_commit", turn, ok` | harness 归因 / rollout_resume / UI |
| MEMORY_WRITE (hub close) | `{task_id, final_answer, n_messages}`（episode 收尾） | 相同消费方 |
| MEMORY_READ | 现状 runtime 侧发射结构 | harness |
| AGENT_MESSAGE (window_close) | input/output_msg_id + 消息链 | 执行追踪/审计 |
| SNAPSHOT | reason / rollout_id / 恢复字段 | ARPO 第二波 resume |

禁止重构期间：改键名、改枚举值、把事件从 events.jsonl 挪走。允许：append-only 追加键
（如 fact_commit+`fact`）。

## 3. 落盘契约

| 物 | 位置 | 变更窗口 |
|---|---|---|
| Archive events.jsonl / snapshots / rollout_index | `mas/.tir_archives/` | Phase 5 迁 `artifacts/mas/tir_archives/`，旧目录只读 fallback |
| 采样树 JSON | `mas/.local_expansion/` | Phase 5 迁 `artifacts/mas/local_expansion/`，同上 |
| Snapshot 字段 | contracts.py Snapshot model | Phase 2 仅**追加** `schema_version: int = 1` |
| `TIR_ARCHIVE_DIR` env 优先级 | archive.py | 一直保留，优先级不变 |
| checkpoints / tensorboard_log / mas/data | 原地 | Phase 5 probe 后标注，不强制迁 |

## 4. 入口脚本调用面

- `run.sh` 子命令：smoke / tests / all / live-* / train / ui / ui-test / branch-ui-test /
  traj-test / arpo-train-test / feature-test——全部不随重构改名。
- `science_infra/control/` 对 mas/ 的调用：`train_tir_agent.py` 文件路径拉起（行 591 附近）、
  `user_projects.py:184` 的 `cfg["source_root"] + "/prompts_rel"`（manifest source_root 均指向
  user_space/projects/<id>/upload/...，已核查三个 manifest，无 mas/MAS_structagent 引用）。
- feature-test 域名：mas-core / rl / harness / branch / rollout-tree / agent-framework /
  schema03 / daemon / realtime / cli / control-ui / gpu-compiler / verifier / e2e /
  functional / user-space / studio / sampling-runtime / frontend / smoke（`--list` 权威）。

## 5. YAML / 配置字段

- MASSpec（spec.py）：`extra` 容忍度按 model_config；旧 YAML 经 `_normalize_schema03`
  sugar 展开（tests 依赖）。Phase 6 仅给 MemorySpec **追加** `planner_ctx: "none"|"latest"`
  默认 none；不改任何既有字段名。
- `experiments/<id>/workflow.yaml`：`hub.system_prompt` + `agents[].system_prompt` 为
  权威 prompt 存储（prompts.py 收口后解析链不变）。

## 6. 新增的内部兼容 shim 登记表（含删除条件）

| shim | 建立于 | 删除条件 |
|---|---|---|
| mas/workflow/archive.py → memory/archive.py 6 行 re-export | Phase 2 | mas/scripts 与 tests 全部改直引后（CONCEPT_MAP.md 记录预计删除版本） |
| mas/mas_agent.py → legacy/structagent/mas_agent.py re-export | Phase 1 | lit_tir_agent.py 改为显式 legacy 路径注入（或训练入口下线）时 |
| legacy/structagent/ 自身 | Phase 1 建立 | 研究栈整体下线时（无期限） |

## 7. Phase 1 前置探查记录（移动前补录）

- `MAS_structagent` 代码引用全集（除 docs 与 Phase 0 底稿自身）：
  - `mas/tools/tool_agents.py`：**5 处懒加载**（python_coder/wikipedia_search/google_search/web_search/base_generator → `python coder.tool`，行 107/115/123/131/139）——plan 所列"think"实际是 base_generator。
  - `mas/mas_agent.py`：`bootstrap_mas_package`（HERE/MAS_structagent）+ docstring。
  - `mas/train_mas_agent.py`：docstring×2。
  - `mas/MAS_structagent/epc_aw/engine/runtime_override.py`:3：vendored 包内部自指 docstring，随目录走无需改。
  - `rl/rewards/outcome.py`:174：`compute_mas_outcome_reward` docstring 提及（计划已列）。
  - `mas/workflow/plugins.py`:44、`mas/tests/test_stage8_verifier.py`:1、`design.md`：**注释/文档**，语义为"不是 MAS_structagent"——不动。
- `mas_agent` 顶层名引用（Phase 1 保 shim 的依据）：
  - `mas/lit_tir_agent.py:122`（零移动红线的训练热路径，lazy import `bind_training_llm_env`）
  - `scripts/agentflow_gate_a_mas_live.py:42`（`from mas.mas_agent import`）
  - `mas/train_mas_agent.py:169`（`from mas_agent import LitMASAgent, bootstrap_mas_package`）
- **现状 sys.path 事实**：`lit_tir_agent.py` 与 `train_mas_agent.py` 均以 `sys.path` 顶层名解析 `mas_agent`；运行期控制层把 repo 根加入 sys.path（repo/ 在 mas/ 前），因此静态上 `'mas'` 包与 `'mas_agent'` 顶层名并存的现状最保真的做法阅读见 Board.md。

**移动时 sys.path 布局**（Phase 1 引用更新的关键约束）：
- 控制层（control/training.py 等）拉起训练时：`sys.path = [repo根, ...其他, ${PYTHONPATH}]`，未含 mas/，`import mas_agent` 依赖 shell PYTHONPATH=mas:. 注入（run.sh/workflow scripts 由 `PYTHONPATH="${ROOT}:${TIR}"` 提供）。产品脚本 `check_workflow_deps.py` 用 `PYTHONPATH=mas` 单根；`mas_agent` 在 mas/ 单根下可解析。

### 7.1 静态 import 派生的 sys.path 结论

1. 工具懒加载 `from MAS_structagent...` 现状依赖 sys.path 含 **mas/**（包根）。移动后需改为在 mas/ 的父目录（repo 根）可见的 `legacy.structagent.epc_aw...`，且 `legacy/` 需含 `__init__.py`。
2. `train_mas_agent.py` 移动后自身位于 `legacy/structagent/`，内部 `from mas_agent import ...`（legacy/structagent/ 目录在 sys.path 时可解析）；`bootstrap_mas_package` 改为 HERE/epc_aw 相对解析。
3. vendored 包内 `from MAS.epc_aw...` 依赖 bootstrap 注册（sys.modules["MAS"].__path__ 指向新物理路径），零改动。
4. `mas/mas_agent.py` shim 保 `lit_tir_agent.py:122` 的顶层名 lazy import（PYTHONPATH=mas 下）。shim 内部将 sys.path 补 repo 根 + import legacy.structagent.mas_agent 并重导出，保持三公开符号签名不变。
