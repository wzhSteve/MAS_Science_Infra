# agentflow — Phase 2.2 (opt-af)

## 改了什么
- **Prompt lite**：`adapted/prompts/{planner_generate_next_step,diagnoser_verificate_context,executor_generate_tool_command,executor_direct_output}.lite.txt`；`contracts/runtime.yaml: prompts_lite: true`。
- **注入**：`context.py` 在 `agentflow.models.{planner,executor,diagnoser}` 上装幂等 `module.open` shim（标记 `_agentflow_lite`），**不改** `builtins.open` / `ref_Rep`。
- **JSON 容错**：fence → 平衡 JSON 子串 → error payload；挂引擎 `_generate_text`。
- **Memory 去重**：三方 `get_actions()` 按 `(tool_name, tool_input)` 去重，上限 `max_steps*2`。

## 为何
原版 planner/diagnoser/executor 模板含大量 few-shot（planner 约 6.4k chars），占满 4B 上下文；窗口 JSON 常带 fence 导致 `ValueError` 穿透；重复 tool 调用污染后续 prompt。

## 证据
| 模板 | 原版 | lite | 上限 |
|------|------|------|------|
| planner/generate_next_step | 6371 | 2233 | 2400 |
| diagnoser/verificate_context | 4116 | 1798 | 1800 |
| executor/generate_tool_command | 2327 | 1497 | 1600 |
| executor/direct_output | 2155 | 991 | 1000 |

- 占位符集合与原版相等（`string.Formatter().parse`）。
- `mas/tests/test_agentflow_lite_prompts.py` + `test_agentflow_pure_path.py` + `test_agentflow_query_gate.py` 全绿。
- 隔离子进程：shim 后 `builtins.open` 不变，ref 路径读到 lite 正文。
