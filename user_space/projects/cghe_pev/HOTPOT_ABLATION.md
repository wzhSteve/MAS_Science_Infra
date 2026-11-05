# HotpotQA 消融

评测集是 `data/hotpotqa.json` 的 100 条。每题最多 4 步。命中指标准答案字符串出现在模型答案里。三个进程同时跑：

| 变体 | 开关 | 输出 |
|---|---|---|
| Full | 一次回复分层 DAG、覆盖度选择、反例剪枝、记忆指导重生成 | `/tmp/cghe_hotpot_full.jsonl` |
| 无剪枝 | `CGHE_NO_PRUNE=1` | `/tmp/cghe_hotpot_noprune.jsonl` |
| 无记忆重生成 | `CGHE_NO_MEMORY=1` | `/tmp/cghe_hotpot_nomemory.jsonl` |

三组都已跑完 100 条（一次 Planner 回复构图后）。命中按答案字符串包含标准答案计算。

| 变体 | 命中 | EM | 报错 | 重生成次数合计 | 实际剪掉计划合计 | LLM 调用合计 |
|---|---|---|---|---|---|---|
| Full | 36 | 0.360 | 2 | 87 | 0 | 957 |
| 无剪枝 | 37 | 0.370 | 1 | 84 | 0 | 945 |
| 无记忆重生成 | 34 | 0.340 | 3 | 83 | 0 | 937 |

相对上一轮 Coverage 恒为 1 时的 0.230 / 0.230 / 0.250，本轮一次分层构图后三组 EM 都明显更高。本轮无剪枝略高于 Full（+1），无记忆略低于 Full（-2）。

三组 `pruned_plan_count` 合计均为 0，原因已定位并修正：当时用「Bing 有链接 / Fetch 正文够长」覆盖 Verifier，把工具调用成功当成子目标满足；失败路径又绑不上计划 query，82 次所谓反例全是 `UNVERIFIED_HYPOTHESIS`，从未进入剪枝。现已改为以 Verifier 对 Planner 子目标的判断为准，搜索不满足时绑定计划 `argument` 生成 `SnippetsSufficient` 反例。上表数字是修正前的跑次，需重跑后才能反映剪枝效果。
