# CGHE 低准确率诊断

数据来源：

- CGHE Full：`artifacts/full_eval_20261104_205650/`（`AGENTFLOW_MAX_STEPS=4`）
- pure_pev：`../pure_pev/artifacts/full_qa_20261104_205608/`（`MAX_STEPS=10`，评测时仍在跑）

## 总表

| 系统 | Hotpot | Bamboogle |
|---|---|---|
| CGHE Full EM | 16/100 = **0.160** | 32/125 = **0.256** |
| pure_pev correct（中途） | 38/96 ≈ **0.396** | 47/100 ≈ **0.470** |

## 空答案是主因

| | Hotpot | Bamboogle |
|---|---|---|
| 空答案 | **73/100** | **89/125** |
| 有答案时命中率 | 16/27 = **0.593** | 32/36 = **0.889** |

有答案时模型表现不差；多数题在停步时没有写出最终答案。

## 与 pure 同题配对（Hotpot 97）

| | 题数 |
|---|---|
| 双方都对 | 14 |
| 仅 CGHE 对 | 2 |
| 仅 pure 对 | **24**（其中 CGHE 空答 **23**） |
| 双方都错 | 57 |

## 机制证据

1. **步数不公平**：CGHE 默认 4 步，pure 10 步。CGHE 平均 `step1_exec≈3.5`，预算几乎耗在失败搜索上。
2. **过剪**：Hotpot 空答题平均 `pruned_plan_count=7.05`，有答案题 3.48；`plans_pruned` 事件 307 / `counterexample_validated` 311。
3. **停止路径**：`continue_pev` 且空答 60 题；`budget_empty_stop` 在无 fetch 证据时强制 `answer=""`。
4. **次要**：有答案仍 miss 仅 11 题（别名/粒度），不是主因。

## 修复与消融方向

- 对齐 `MAX_STEPS=10`；有观测时禁止 `force_empty`。
- `SNIPPETS_INSUFFICIENT` 改为只淘汰当前执行计划（实例级），避免共享 Step-1 一次剪光多路。
- **Soft-Bing（2026-11-06）**：Bing 有 URL/非空摘要但 Verifier 不满足 → `SUBGOAL_NOT_SATISFIED`（UNVERIFIED），沿同 plan 进 Fetch；仅空搜才硬 `SNIPPETS_INSUFFICIENT`。Fetch 失败后再延后剪上游 query；每题最多 2 次 snippet 硬剪；重生继承 CE 截断为 32。
- 消融臂：Full / NoPrune / NoMemory / SelectOnly / ForceRegen（`CGHE_FORCE_REGEN_AFTER_HARD_PRUNE`），与 pure 同协议对比。

## 消融结果

目录：`/root/autodl-tmp/MAS_Science_Infra/user_space/projects/cghe_pev/artifacts/smoke20_20261105_093728`

| 臂 | 数据集 | n | EM | accuracy | nonempty | pruned | regen |
|---|---|---|---|---|---|---|---|
| full | hotpot | 20 | 0.500 | 0.500 | 1.000 | 195 | 61 |
| noprune | hotpot | 20 | 0.350 | 0.350 | 1.000 | 0 | 19 |

## 消融结果

目录：`/root/autodl-tmp/MAS_Science_Infra/user_space/projects/cghe_pev/artifacts/smoke20_softbing_20261106_153500`

| 臂 | 数据集 | n | EM | accuracy | nonempty | pruned | regen |
|---|---|---|---|---|---|---|---|
| full | hotpot | 20 | 0.450 | 0.450 | 1.000 | 0 | 0 |
| full | bamboogle | 20 | 0.600 | 0.550 | 0.950 | 0 | 2 |
| forceregen | hotpot | 20 | 0.500 | 0.500 | 1.000 | 0 | 2 |
| forceregen | bamboogle | 20 | 0.750 | 0.750 | 1.000 | 4 | 3 |

## 消融结果

目录：`/root/autodl-tmp/MAS_Science_Infra/user_space/projects/cghe_pev/artifacts/smoke20_deepen_20261108_181647`

| 臂 | 数据集 | n | EM | accuracy | nonempty | pruned | regen |
|---|---|---|---|---|---|---|---|
| full | hotpot | 20 | 0.450 | 0.450 | 1.000 | 19 | 6 |
| full | bamboogle | 20 | 0.600 | 0.600 | 1.000 | 25 | 10 |

## 消融结果

目录：`/root/autodl-tmp/MAS_Science_Infra/user_space/projects/cghe_pev/artifacts/smoke20_dedup_20261109_114003`

| 臂 | 数据集 | n | EM | accuracy | nonempty | pruned | regen |
|---|---|---|---|---|---|---|---|
| full | hotpot | 20 | 0.400 | 0.400 | 1.000 | 0 | 8 |
| full | bamboogle | 20 | 0.550 | 0.500 | 0.950 | 0 | 6 |

## Dedup-refund 实验结论（2026-11-09，已回滚禁用）

方案：命中缓存的重复调用退步（`max_dedup_refunds_per_episode=3`）。
smoke20 同题对照（vs smoke20_deepen_20261108_181647）：

| 数据集 | dedup acc | deepen acc | dedup llm/题 | deepen llm/题 | 配对 gained/lost |
|---|---|---|---|---|---|
| hotpot | 0.40 | 0.45 | 17.9 | 14.7 | 0 / 1 |
| bamboogle | 0.50 | 0.60 | 16.2 | 14.5 | 2 / 4 |

机制触发正常（70-75% 题有 refund、非empty 保持），但**假设被证伪**：退步没有转化为
新的有效搜索（exec/题反而 2.0<2.3），而是让 agent 在 EXHAUST 后的同一批节点上
免费空转（24 题出现 EXHAUST→dedup 空转），多烧 LLM 调用、acc 下降、并出现 1 例空答案。
处理：runtime.yaml/RUNTIME_DEFAULTS 将 cap 置 0（代码保留、门控关闭）；单测与回归 155 项通过。
推论：full 数据里"高重复→高兜底率"是相关性而非因果——重复调用是"无路可走"的症状，
不是预算浪费的原因；真正的瓶颈是 DAG 在证据不足时缺少更聪明的前进方式。
