# Agent 与 Tool 产品模型

后端可以统一执行抽象，但 UI 应保持用户熟悉的 Agent、Tool、Router 三类对象。

| 类型 | 示例 | 是否需要 Agent 化 |
| --- | --- | --- |
| 确定性工具 | 计算器、文件读取、HTTP 请求、Python 执行器 | 不需要 |
| 智能工具 | 搜索后总结、代码生成与执行、文档分析 | 可以在运行时 Agent 化 |

## 架构层与 UI 层

架构层允许智能工具使用 `agents[].kind: tool`，从而获得模型、Memory、训练、工作窗口和 Branch 能力；确定性工具继续使用顶层 `tools` 与标准 `tool_call`。两种形式都由运行时兼容。

UI 层统一把它们放在 Tool 分类中：

- 普通工具显示为“内置工具”。
- `kind: tool` 显示为“智能工具”，按能力提供模型、提示词、Memory、训练和扩展配置。
- `kind: blank` 显示为“自定义 Agent”，不向用户暴露 blank 术语。
- Router 独立显示，候选项按 Agent 与 Tool 分组。

因此，UI 不要求用户理解后端的 Tool Agent 适配器；保存时仍准确输出 legacy Tool 或 `kind: tool`，不改变 main 的执行协议。
