# single-ppo — Phase 2.1

## 改了什么
- 新建 `user_space/projects/single-ppo/`：`manifest.yaml`、`adapted/{__init__,entry,context}.py`、`contracts/{runtime,workflow,io_contract}.yaml`。
- 单窗口 ReAct 循环（APPO/ARPO 范式）：`<search>/<python>` → `<result>`（800 字截断）→ `<answer>`；`max_steps=6`。
- 解析三板斧：strip code fence → 首个平衡 JSON → 尾部 `<answer>`（含未闭合容错）；`validate_format` 拒双 answer / 错误嵌套。
- 工具走 `mas.tools` 内核（Wikipedia / Bing / Web_Fetch / Python）；协议 payload 对齐 `plan_step` / `tool_result` / `verify`。

## 为何
三项目评估需要可对比的「单 LLM + 工具」基线；集中式 multi-agent 窗口过重，single topology 用 planner 包装 solver 即可挂上 user_space 网关。

## 证据
- `mas/tests/test_single_sppo_parsing.py`：格式校验、解析、ReAct 闭环、max_steps 强制作答、协议 schema 全绿。
- SYSTEM_PROMPT ≤1200 chars；runtime 默认与测试断言一致。
