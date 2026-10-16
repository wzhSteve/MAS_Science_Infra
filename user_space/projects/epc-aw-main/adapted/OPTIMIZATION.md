# epc-aw-main — Phase 2.2 (opt-epc)

## 改了什么
- **Prompt lite**：`adapted/prompts/{planner,diagnoser,executor}/*.lite.txt` 主链 4 模板；`runtime.yaml` / `manifest.yaml: prompts_lite: true`。
- **注入修复**：旧实现把 lite 路径指到 `ref_Rep/.../*.lite.txt`（文件不存在 → 从未生效），且全局 patch `builtins.open`（评估器隔离曾炸）。现改为：
  - 映射到 `adapted/prompts/**/*.lite.txt`
  - 仅在 `MAS.epc_aw.models.{planner,diagnoser,executor}` 装 `module.open` shim（`_science_prompt_lite`）
  - 启动时若发现遗留 builtins hijack 则还原
- **JSON 容错**：`_generate_text` 包装 + `json.loads` hook；垃圾输入在 `response_format` 路径降级为 `{"error":"json_parse_failed",...}`。
- **Memory 去重**：`Memory.get_actions` 按 `(tool, command)` 去重限条。

## 为何
与 agentflow 同构减负，但 EPC-AW engine/models 分叉独立实现；必须修映射与 builtins 风险，lite 才真正进窗口且评估器不炸。

## 证据
| 模板 | 原版 | lite | 上限 |
|------|------|------|------|
| planner/generate_next_step | 3540 | 1889 | 2000 |
| diagnoser/verificate_context | 3580 | 1799 | 1800 |
| executor/generate_tool_command | 2327 | 1548 | 1600 |
| executor/direct_output | 1434 | 981 | 1000 |

- 占位符集合与原版相等。
- `mas/tests/test_epc_aw_lite_prompts.py`（新建）+ `test_epc_aw_query_gate.py` 全绿。
- 隔离子进程：`builtins.open` 未变；`planner.open(ref_path)` 读出 adapted lite 正文。

## 与 agentflow 同构 / 分叉
| 点 | 对齐 | 分叉原因 |
|----|------|----------|
| lite 4 主链 + runtime 开关 | ✓ | — |
| 模块级 open shim / 幂等标记 | ✓ | 包名 `MAS.epc_aw.models.*` vs `agentflow.models.*` |
| JSON 容错三步 | ✓ | epc 另挂全局 `json.loads`（无 response_format 调用多） |
| memory 去重 | ✓ | 类路径与 marker 名不同 |
