# CGHE 模块优化记录

模型是 `.env` 中的 `qwen3.5-27b`。检索使用第二套 Bing 账号。每次只保留最后一版过关结果。

## 一次回复的分层 DAG

`build_outlines_oneshot` 只发一次 Planner 调用。提示要求同一次回复里先写最多 2 条起点搜索（恰好一行 `primary`），立刻为 primary 写 3 条不同的 `Web_Fetch`、为其他起点写 1 条，最后一行共享抽取。解析仍用 `parse_step1_lines` / `parse_step2_lines` / `plans_from_layered_steps` 读同一段文本；解析失败不发第二次模型调用，也不本地编造 `facts` / `relation` 查询。Step 1 解析只收 Bing 行；抓取行容忍缺 `|`。计划上下文写入 `environment=cghe_pev`，以便查询级反例能对上 scope。

固定 5 条 HotpotQA，每条恰好 1 次 Planner 调用，最大 Step 1 Coverage 均 ≥ 2，选中的就是 primary。

## 模块 II：执行与剪枝

高覆盖搜索成功后沿路径进入抓取子节点；`protect_children_of` 避免子节点覆盖度全是 1 时立刻齐平重生成。Bing 与已绑定 URL 的抓取不调模型；抽取是该步唯一的 Executor 调用。Verifier 一次调用。剪枝门只接受直接、完整 scope、且与计划显式 `argument` 相同的反例。人工构造「同一查询被证伪」：声明该查询的计划被删，共享抽取节点仍 pending；超时和空检索不删计划。

1 条题（Oliver Reed）日志：第一步 Coverage 为 3 的搜索，下一步是它的 `web_fetch_tool` 子节点，全程没有 `dag_regenerated`。

## 模块 III / IV：记忆与重生成

成功写入正向记忆；可信反例写入 assumption、scope 和被剪计划。未验证失败不进入 `constraints_for_pruning`。重生成的那一次 Planner 调用通过 `context_guidance` 带上已验证记录；`drop_contradicted_plans` 丢掉重复同一 `argument` 的起点。资源 X 的否定不禁用资源 Y。`CGHE_NO_MEMORY=1` 时重生成不读因果记忆；`CGHE_NO_PRUNE=1` 时只记录反例不删计划。

## 全量与消融

100 条 HotpotQA、每题最多 4 步。三个进程同时跑 Full、无剪枝、无记忆重生成。结果见 `HOTPOT_ABLATION.md`。上一轮 Coverage 恒为 1 时的 0.230 / 0.230 / 0.250 不作本轮结论。
