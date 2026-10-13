# Phase 0 接口约束清单（合同冻结）

日期：2026-10-13
状态：Phase 0 产出（评审基线，Phase 1 起每个移动 commit 前对照自查"只移不改"）。
本清单按 docs/MAS_design.md v2 §13 Phase 0 要求产出。任何 Phase 的代码改动若触碰
下列合同的外部可见行为，必须在 commit message 中显式声明并重新评审。

---

## 1. AgentMessage 信封合同（mas/workflow/protocol.py）

冻结字段（`model_config = {"extra": "forbid"}`）：

| 字段 | 类型 | 语义 |
|---|---|---|
| msg_id | str (uuid) | 因果链节点 |
| task_id | str | 逻辑任务标识 |
| turn | int | 逻辑决策轮次（runtime 维护，agent 不写；永不复用于 LLM 调用计数） |
| src / dst | str | 发送/接收 agent id |
| kind | 8 值枚举 | plan_step / route_decision / tool_invoke / tool_result / verify / feedback / final_answer / error |
| payload | Dict | 按 kind schema 校验（make_message 非法即 raise ValueError） |
| trace_ref | Optional[str] | 单一直接因果引用（指向上一条 msg_id） |

冻结 per-kind required 键（PAYLOAD_SCHEMAS，行 66-119）：

- `plan_step`: required `next, args, sub_goal, done`；`next/args` 允许 str|list（多 router fan-out）
- `route_decision`: required `selected, args, candidates, strategy`
- `tool_invoke`: required 无（args 由 tool-agent 专测）
- `tool_result`: required `output, ok, evidence_type`；evidence_type ∈ {DIRECT, ABSENCE, ERROR, EMPTY}
- `verify`: required `ok, reason, step_conclusion, slot_updates`；step_conclusion ∈ {COMPLETE, INCOMPLETE}
- `feedback`: required `reason, attributed_to`
- `final_answer`: required `answer`
- `error`: required `agent_id, error`

冻结域值校验（validate_payload 行 148-170）：未知 kind / 非 dict payload / 缺 required /
类型不匹配 / 枚举越界 → (False, reason)，make_message 转 ValueError——非法 plan_step
不得进入执行阶段（现状行为，测试锁定）。

冻结辅助方法：`chain_to(child)`（设 trace_ref）、`payload_digest()`（sha256 前 16 位，sort_keys）。

## 2. commit_to_fact 签名与 FactLedger 判定矩阵

冻结签名（protocol.py 行 273，tests 依赖）：

```
commit_to_fact(verify_payload: Dict, tool_result_payload: Dict) -> bool
```

冻结判定逻辑（= 未来 FactLedger.classify 的权威矩阵）：

| # | step_conclusion | evidence_type（verifier 优先，缺省回退 tool_result） | slot_updates | 返回 |
|---|---|---|---|---|
| 1 | COMPLETE | DIRECT | 无 slots | True (COMMIT_FACT) |
| 2 | COMPLETE | DIRECT | 全部 filled=true | True (COMMIT_FACT) |
| 3 | COMPLETE | DIRECT | 存在 filled=false | False (KEEP_WINDOW_ONLY) |
| 4 | COMPLETE | ABSENCE/EMPTY/ERROR | — | False (REJECT) |
| 5 | INCOMPLETE | 任意 | — | False (KEEP_WINDOW_ONLY) |

注意现状 evidence_type 的回退语义：verifier payload 无 evidence_type 时**回退查
tool_result_payload 的 evidence_type**（protocol.py 行 279-280）——FactLedger 实现时保留。

冻结 fact_commit 事件（centralized_runtime.py 行 897-906，外部消费方 schema）：

```json
{"kind": "MEMORY_WRITE", "agent_id": "<verifier_id or 'verifier'>", "payload": {"phase": "fact_commit", "turn": <int>, "ok": true}}
```

Phase 2 FactLedger 收口后：payload 三键 `phase/turn/ok` 不变，仅**追加** `fact` 字段
（append-only，不改既有键）。

## 3. MemoryStore 三作用域（mas/workflow/memory.py，90 行）

冻结接口：`read(scope, owner='hub')` / `write(item)` / `latest(scope, owner)` /
`read_agent(agent_id, memory_scope='shared', policy=None)` / `write_agent(agent_id, content, memory_scope='shared')`。

冻结语义：

- 存储形态：`_agent[owner]` 字典 + `_system` 独立列表；shared = `_agent["hub"]` 特殊键。
- `write_agent` append-only：不覆盖，冲突两条并存。
- `_apply_policy`：仅 `policy="append_latest" + max_items` 生效，其余原样返回。
- episode 生命周期：MemoryStore 进程内对象，每 rollout 新建（lit_tir_agent 现状，训练隔离特性）。

## 4. Archive 逻辑四区（mas/workflow/archive.py）

冻结接口：`Archive(archive_id?, root_dir?)` / `append(event)` / `snapshot(messages, meta?, event_id?, rollout_id?, token_ids?)` / `restore(snapshot_id)` / `ref()` / `to_branch_point(...)` / `Archive.load(archive_id, root_dir?)`；
模块级：`register_archive` / `get_archive` / `get_default_archive` / `dump_resume_with_archive` / `load_resume_messages(rollout_id)` / `_load_legacy_resume_cache` / `branch_point_to_resume_task_fields(bp)`。

冻结落盘布局（Phase 5 迁移前）：

```
mas/.tir_archives/<archive_id>/events.jsonl        # ExecutionEvent 逐行 JSONL
mas/.tir_archives/<archive_id>/snapshots/<id>.json  # Snapshot
mas/.tir_archives/rollout_index.json               # rollout_id -> (archive_id, snapshot_id, root_dir)
.resume_cache/（legacy）                            # 只读 fallback，永不写
```

冻结事件类型（EventKind，contracts.py）：TASK_START / AGENT_MESSAGE / TOOL_CALL /
TOOL_RESULT / FINAL_ANSWER / MEMORY_READ / MEMORY_WRITE / SNAPSHOT / FEEDBACK / ERROR。

冻结恢复语义：`load_resume_messages` 查找顺序 = 内存索引 `_ROLLOUT_INDEX` → 磁盘
`rollout_index.json` → legacy `.resume_cache`；前缀-only 恢复（router 计数器等状态不恢复）。

已知缺口（登记，Phase 2 修复）：`Archive.append` 无锁——lit_tir_agent ActiveSetScheduler
线程池共享同一 `arch` 并发 append 时 events.jsonl 有交错风险；`snapshot()` 非原子直写。
exclusive 注意：centralized_runtime fan-out 现状安全（主线程串行 append）。

## 5. Router 四契约（mas/workflow/router.py + centralized_runtime）

1. `llm_choice` ≡ `from_plan` 别名（现状 registry 归一，router 不调 LLM）。
2. `score` 策略未显式 scorer 时回退候选自报分（`scorer_id = _rs.scorer or cid`）——设计
   合同：score 可比前提是同一评分合同；现状差距登记为文档合同，不改回退行为。
3. `round_robin` 计数器是模块级 dict（进程生命期）；训练场景使用前必须显式
   `reset_round_robin`（文档合同）。
4. 并行 fan-out 部分失败不丢成功结果：失败 hop 产出 ok=false 的 tool_result 消息，
   不中断其余 hop（现状已满足，测试锁定）。

## 6. 两执行路径行为差别（训练 vs Workflow）

| 维度 | TirAgent 路径（runtime.run_episode） | 中心化路径（centralized_runtime.run_centralized_episode） |
|---|---|---|
| 协议 | plain chat messages（无 AgentMessage） | AgentMessage 信封全程 |
| memory 接线 | `_open_construct` 读 / `_close_construct` 写 + MEMORY_READ/WRITE 事件 | 收 `memory` 参数但现状未使用（Phase 6 opt-in） |
| 终止 | ReAct 循环判定 | runtime 判定（done+ok / ready_to_stop / answer） |
| 采样树 | BranchSite + Archive 快照 resume | 无 |

红线：训练热路径三件套（tir_agent.py / lit_tir_agent.py / train_tir_agent.py）零移动、
零行为变化；`lit_tir_agent.py:122` 的 `from mas_agent import bind_training_llm_env`
顶层名 import **必须继续可解析**（Phase 1 在 mas/ 留 mas_agent shim 保住）。

## 7. 架构守卫既有边界

- `check_workflow_deps.py`：workflow/*.py 不得 import agentlightning/verl/ray/lit_tir_agent。
- `check_user_zone.py`：workflow/（除 user_gateway）不得 import user_space；用户区 adapted/*.py
  不得 import science_infra/rl/agentlightning/verl/ray/lit_tir_agent/tools 顶层名。
- 本 Phase 基线中 check_user_zone 因 `user_space/projects/epc-aw-main/adapted/context.py`
  （既有未提交用户区变更）FAIL——登记为既有失败，见 test_baseline.md §3。

## 8. 复核方式

任何 Phase 完成后，执行：

```bash
PYTHONPATH=mas:. ./.venv/bin/python docs/refactor/freeze_msgseq_baseline.py --check \
  | diff - docs/refactor/.baseline_msgseq_v0.json && echo BYTE-LEVEL-OK
```

diff 输出非空即协议合同已破坏，该 Phase 需回滚或显式升级本清单加评审。
