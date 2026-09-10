# math_gsm：用 LangChain / LangGraph 构建 Agent Workflow

本文专门拆 `math_agent.py` 里 **LangChain 消息/工具/ChatModel** 与 **LangGraph 状态机** 如何拼成一条可训练的数学 Agent。入口是训练时的这段 `invoke`：

```python
# math_agent.py LitMathAgent.rollout
handler = self.tracer.get_langchain_handler()
initial: AgentState = {
    "question": question,
    "num_turns": 0,
    "asked_finalize": False,
    "messages": [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ],
}
result_state = agent.graph().invoke(
    initial,
    {"callbacks": [handler] if handler else [], "recursion_limit": 40},
)
```

对应源码：`[math_agent.py](../math_agent.py)` `LitMathAgent.rollout`（约 628–641 行）。

系统级训练环路（VERL / GRPO / Adapter）见 [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)；本文只把 **Agent 图本身** 讲透。

---



## 1. 先记住三层分工


| 层                   | 库                                             | 在本项目里干什么                                                                                         | 主要代码                                                                                     |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| 消息 / 工具 / ChatModel | **LangChain**（`langchain` + `langchain_core`） | 定义 `System/Human/AI/Tool` 消息、把 Python 函数包成 Tool、用 OpenAI 兼容 API 调模型、`bind_tools` 让模型能发 tool-call | `math_agent.py` 顶部 import、`execute_python_tool`、`MathAgent.__init__`                     |
| 状态机 / 循环            | **LangGraph**（`langgraph.graph.StateGraph`）   | 把「调模型 → 跑工具 → 再调模型 → 强制收口」编成有向图，用条件边决定下一步                                                        | `MathAgent.graph` / `call_model` / `call_tools` / `request_finalize` / `should_continue` |
| 训练封装                | **Agent-Lightning**                           | 注入当前策略的 LLM endpoint、挂 tracer、算 reward、上报 span                                                   | `LitMathAgent.rollout`、`agl.LitAgent`、`Tracer.get_langchain_handler`                     |


一句话：**LangChain 提供积木，LangGraph 把积木连成循环，Lightning 把循环接到 RL。**

本仓库 **没有** 用 `langchain.agents.create_agent` / `AgentExecutor` 那种「黑盒 ReAct」。workflow 是手写 LangGraph 节点，tool 循环由 `should_continue` 显式控制。这和 Spider SQL 示例同一模式，见 [SPIDER_WALKTHROUGH.md](SPIDER_WALKTHROUGH.md)。

```text
LitMathAgent.rollout
        │
        │  组装 AgentState + callbacks
        ▼
MathAgent.graph().invoke(...)     ← LangGraph CompiledStateGraph
        │
        ├─ node "agent"     call_model()      ← LangChain ChatModel.invoke
        ├─ node "tools"     call_tools()      ← LangChain Tool.invoke
        └─ node "finalize"  request_finalize() ← 往 messages 追加 HumanMessage
```

---



## 2. 从 628–641 往外读：这段代码在整次 rollout 里的位置

`invoke` 不是孤立的。它夹在「拿 LLM 资源」和「解析答案 / 打分」之间：

```text
LitMathAgent.rollout(task, resources, rollout)
  │
  ├─ 1. 从 task 取 question / answer
  ├─ 2. resources["main_llm"] → 按 train/val 选温度
  ├─ 3. llm.get_base_url(rollout_id, attempt_id)  → 本次尝试专用 OpenAI base
  ├─ 4. 构造 MathAgent(endpoint, model, temperature, max_turns, max_tokens)
  ├─ 5. ★ graph().invoke(initial, {callbacks, recursion_limit})   ← 本文焦点
  ├─ 6. extract_answer_from_messages(result_state["messages"])
  ├─ 7. compute_reward(...) → agl.emit_reward(reward)
  └─ 8. return None   （奖励只 emit，不 return float，防双计）
```

对应函数：


| 步骤             | 函数 / 字段                                       | 文件                                                                                                                                 |
| -------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| rollout 入口     | `LitMathAgent.rollout`                        | `[math_agent.py](../math_agent.py)` ~565                                                                                           |
| LLM 资源         | `resources["main_llm"]`，类型 `agl.LLM`          | 同上 ~591；框架侧 `agentlightning/types`                                                                                                 |
| 专用 URL         | `llm.get_base_url(rollout_id, attempt_id)`    | 同上 ~605                                                                                                                            |
| 构图 + 跑         | `MathAgent.graph().invoke`                    | 同上 ~477 / ~638                                                                                                                     |
| 抽答案            | `extract_answer_from_messages`                | 同上 ~182                                                                                                                            |
| 打分             | `compute_reward`                              | 同上 ~260                                                                                                                            |
| 上报             | `agl.emit_reward`                             | `[agentlightning/emitter/reward.py](../../../agentlightning/emitter/reward.py)`                                                    |
| 基类 / tracer 属性 | `LitAgent.tracer` → `get_langchain_handler()` | `[litagent.py](../../../agentlightning/litagent/litagent.py)` ~114；`[tracer/base.py](../../../agentlightning/tracer/base.py)` ~193 |


训练时真正调用 `rollout` 的是 Runner：它先进入 `tracer.trace_context(...)`，再调 `training_rollout` / `validation_rollout`（默认都转到 `rollout`）。见 `[agentlightning/runner/agent.py](../../../agentlightning/runner/agent.py)` ~663。所以 628 行拿到的 `handler` 一定落在一次已打开的 trace 里。

---



## 3. LangChain 积木：消息、工具、ChatModel



### 3.1 消息类型（轨迹的载体）


| LangChain 类     | 角色                                    | 谁写入                         |
| --------------- | ------------------------------------- | --------------------------- |
| `SystemMessage` | 系统规则：可调工具、最终必须 `### NUMBER ###`       | `initial["messages"]` 第一句   |
| `HumanMessage`  | 题目；以及 finalize 时注入的 `FINALIZE_PROMPT` | 初始状态 / `request_finalize`   |
| `AIMessage`     | 模型回复；可能带 `tool_calls`                 | `call_model` 里 `llm.invoke` |
| `ToolMessage`   | 工具执行结果，`tool_call_id` 对齐某次 call       | `call_tools`                |


这些类型来自 `langchain_core.messages`。整个 workflow **不另建对话历史对象**：LangGraph 状态里的 `messages: List[AnyMessage]` 就是完整 transcript，也是事后抽答案、算长度惩罚、以及 tracer 还原 LLM span 的依据。

初始状态（628–637）固定两句：

1. `SystemMessage(SYSTEM_PROMPT)` — 教模型怎么用工具、怎么收口。
2. `HumanMessage(question)` — 当前 GSM8K 题干。

`question` 字段再存一份原始题目，方便日志；推理只看 `messages`。

### 3.2 工具：`@tool` 包装沙箱

```python
@tool
def execute_python_tool(code: str) -> str:
    return execute_python(code)

TOOLS = [execute_python_tool]
TOOL_MAP = {t.name: t for t in TOOLS}   # 默认 name = "execute_python_tool"
```


| 概念               | 说明                                                                                             | 代码                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| `@tool`          | LangChain 把函数变成可 `bind_tools` / `invoke(args)` 的 Tool；函数 docstring 会进 tool schema，模型据此填 `code` | `[math_agent.py](../math_agent.py)` ~83                |
| `execute_python` | 真正沙箱：AST 禁 import/危险名、白名单 builtins、注入 `math`/`operator`、超时线程                                   | `[python_tool.py](../python_tool.py)` `execute_python` |
| `TOOL_MAP`       | `call_tools` 按 `tool_calls[].name` 分发                                                          | `[math_agent.py](../math_agent.py)` ~96                |


约定：模型应把终值赋给变量 `result`；沙箱返回形如 `result = 180` 的字符串。抽答案时若 assistant 忘了写 `### 180 ###`，仍可从 `ToolMessage` 里的 `result=` 兜底（`extract_answer_from_messages`）。这是一条潜在 reward 泄漏路径，面试可当 hacking 例子。

**没有** 使用 LangGraph 自带的 `ToolNode`。`call_tools` 手写循环，是为了：

- 兼容 `tool_calls` 为 dict 或对象两种形态（不同 langchain 版本）；
- 未知工具 / 执行异常都写成错误字符串回传，让模型有机会改代码，而不是把整张图打崩。



### 3.3 ChatModel：两套客户端

`MathAgent.__init__` 用 LangChain 的 `init_chat_model(..., model_provider="openai")` 连 OpenAI 兼容服务（训练时是 VERL/vLLM，debug 时是任意 `OPENAI_API_BASE`）。


| 客户端                 | 绑定工具？                | 温度 / max_tokens                          | 用途                   |
| ------------------- | -------------------- | ---------------------------------------- | -------------------- |
| `self.llm`          | `.bind_tools(TOOLS)` | 训练默认 ~0.7，`max_tokens` 默认 1024           | 主循环：推理 + 发 tool-call |
| `self.llm_finalize` | **不绑**               | `min(temperature, 0.3)`，`max_tokens≤256` | 收口轮：只输出 `### N ###`  |


`bind_tools` 会把 tool JSON schema 塞进 Chat Completions 的 `tools` 字段。模型若决定调用工具，返回的 `AIMessage` 带 `tool_calls`（name / args / id），**不一定**带自然语言 `content`。

`openai_api_key` 用环境变量或 `"dummy"`：本地 vLLM 通常不校验 key，但 OpenAI 客户端库要求字段非空。

对应代码：`[math_agent.py](../math_agent.py)` `MathAgent.__init__` ~338–385。

---



## 4. LangGraph：状态、节点、条件边



### 4.1 状态 `AgentState`

```python
class AgentState(TypedDict):
    messages: List[AnyMessage]   # 对话，也是 RL 轨迹
    question: str                # 原题，日志用
    num_turns: int               # 已调用模型次数（含 finalize）
    asked_finalize: bool         # 是否已注入 FINALIZE_PROMPT
```

LangGraph 的 `StateGraph(AgentState)` 约定：每个节点返回 **部分或完整 state**；本实现每次 `return {**state, ...}`，即整表替换。`messages` 采用追加，不覆盖历史。

### 4.2 三个节点

```text
START ──► agent ──┬──► tools ──► agent ──► ...
                  ├──► finalize ──► agent ──► END
                  └──► END
```


| 节点名          | 绑定方法               | 做什么                                                                            | 代码   |
| ------------ | ------------------ | ------------------------------------------------------------------------------ | ---- |
| `"agent"`    | `call_model`       | `llm.invoke(messages)` 或 `llm_finalize.invoke`；追加 `AIMessage`；`num_turns += 1` | ~387 |
| `"tools"`    | `call_tools`       | 读上一条 AIMessage 的 `tool_calls`，`TOOL_MAP[name].invoke(args)`，追加 `ToolMessage`   | ~425 |
| `"finalize"` | `request_finalize` | 追加 `HumanMessage(FINALIZE_PROMPT)`，置 `asked_finalize=True`                     | ~413 |


`call_model` 的切换逻辑：

- `asked_finalize == False` → `self.llm`（带 tools）
- `asked_finalize == True` → `self.llm_finalize`（无 tools、更短、更低温度）

LLM 调用失败时写入占位 `AIMessage("### None ###")`，保证图能走到打分（通常 reward=0），而不是整进程崩溃。

### 4.3 条件边 `should_continue`

`agent` 节点之后 **不是** 固定下一跳，而是 `add_conditional_edges`：

```python
builder.add_conditional_edges(
    "agent",
    self.should_continue,
    {"tools": "tools", "finalize": "finalize", "end": END},
)
```

路由优先级（`[should_continue](../math_agent.py)` ~451）：


| 优先级 | 条件                                                       | 返回           | 下一跳          |
| --- | -------------------------------------------------------- | ------------ | ------------ |
| 1   | 上一条有 `tool_calls`，且 `num_turns < max_turns`，且还没 finalize | `"tools"`    | 执行 Python    |
| 2   | `extract_answer_from_messages` 已能解析出答案                   | `"end"`      | `END`        |
| 3   | 还没 finalize，且还有轮次                                        | `"finalize"` | 注入收口 prompt  |
| 4   | 否则                                                       | `"end"`      | `END`（可能无答案） |


另外两条 **固定边**：

- `tools → agent`：工具结果回来必须再让模型看一眼。
- `finalize → agent`：注入 prompt 后立刻用 `llm_finalize` 再生成一次。

`max_turns` 默认 8（`LitMathAgent` / `MathAgent` 构造参数）。它数的是 **模型调用次数**，不是 tool 次数。

### 4.4 `graph()` 编译

```python
def graph(self) -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("agent", self.call_model)
    builder.add_node("tools", self.call_tools)
    builder.add_node("finalize", self.request_finalize)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", self.should_continue, {...})
    builder.add_edge("tools", "agent")
    builder.add_edge("finalize", "agent")
    return builder.compile()
```

`compile()` 得到 `CompiledStateGraph`。每次 `rollout` / `run` 都 **重新** `graph()` **一次**（无 checkpoint、无跨题记忆）。

`invoke` 的第二个参数是 LangGraph 的 **config dict**（不是 state）：


| config 键          | 本项目取值              | 作用                                                                         |
| ----------------- | ------------------ | -------------------------------------------------------------------------- |
| `callbacks`       | `[handler]` 或 `[]` | 把 LangChain 回调挂进本次图执行，LLM/chain/tool 事件进 tracer                            |
| `recursion_limit` | `40`               | LangGraph 内部步数上限，防止 tool 死循环把进程卡死；应大于 `2 * max_turns` 量级（agent+tools 各算一步） |


训练入口用 40；`MathAgent.run`（本地手工试跑）同样 40，但不挂 callbacks。

---



## 5. 一次题目的逐步执行（对照 628–641）

以「What is 12 multiplied by 15?」为例，假设模型先调工具再收口。

**T0 —** `invoke` **之前（628–637）**

```text
state.messages = [
  SystemMessage("You are a careful math assistant... ### NUMBER ###"),
  HumanMessage("What is 12 multiplied by 15?"),
]
num_turns = 0, asked_finalize = False
```

**T1 — START → agent（**`call_model`**）**

`self.llm.invoke(messages)`。模型返回带 tool_call 的 `AIMessage`，例如：

```text
AIMessage(
  content="",
  tool_calls=[{
    "name": "execute_python_tool",
    "id": "call_abc",
    "args": {"code": "result = 12 * 15"},
  }],
)
num_turns = 1
```

**T2 —** `should_continue` **→** `"tools"`

有 tool_calls、未超轮、未 finalize。

**T3 — tools（**`call_tools`**）**

`execute_python_tool.invoke({"code": "..."})` → 沙箱返回 `"result = 180"` → 追加：

```text
ToolMessage(content="result = 180", tool_call_id="call_abc")
```

固定边回到 `agent`。

**T4 — agent 第二轮**

模型看到 tool 结果，输出：

```text
AIMessage(content="### 180 ###")   # 无 tool_calls
num_turns = 2
```

**T5 —** `should_continue` **→** `"end"`

`extract_answer_from_messages` 从最后一条 AIMessage 解析出 `"180"`，图结束。

**T6 — 回到** `rollout`**（642 行之后）**

```text
prediction = "180"
raw_text   = "### 180 ###"
n_chars    = len(raw_text) 等
reward     = compute_reward(...)   # binary 下为 1.0
agl.emit_reward(reward)
```

若模型一直调工具却不写答案：轮次将尽时走 `finalize` → 注入 `FINALIZE_PROMPT` → `llm_finalize` 强制短格式。若仍解析失败，reward=0。

---



## 6. callbacks：LangChain 如何把轨迹交给训练框架

628 行：

```python
handler = self.tracer.get_langchain_handler()
```


| 问题                 | 答案                                                                                      | 代码                                                                           |
| ------------------ | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `self.tracer` 从哪来？ | `LitAgent.tracer` 属性，优先 `runner.tracer`，否则 `trainer.tracer`                             | `[litagent.py](../../../agentlightning/litagent/litagent.py)` ~114–124       |
| 默认实现？              | `AgentOpsTracer.get_langchain_handler` 返回 `agentops` 的 `LangchainCallbackHandler`       | `[tracer/agentops.py](../../../agentlightning/tracer/agentops.py)` ~194      |
| 基类默认？              | 打 warning，返回 `None`；628 行因此写成 `[handler] if handler else []`                            | `[tracer/base.py](../../../agentlightning/tracer/base.py)` ~193              |
| 何时生效？              | Runner 已 `async with tracer.trace_context(rollout_id=...)`                              | `[runner/agent.py](../../../agentlightning/runner/agent.py)` ~663            |
| 下游谁消费 span？        | Adapter 读 `langchain.chain.type` / `langchain.Chain.*` 识别 agent 名，把 LLM call 编成 Triplet | `[adapter/triplet.py](../../../agentlightning/adapter/triplet.py)` ~344、~362 |


LangGraph `invoke(..., {"callbacks": [...]})` 会把 handler 传到内部每次 `ChatModel.invoke` / tool 调用。AgentOps handler 把这些事件写成 OTEL span；`emit_reward` 再写一条 reward span。Adapter（`TracerTraceToTriplet`）把「LLM 输入 token / 输出 token / 终局 reward」拼成 VERL 要的 `(state, action, reward)`。

**直觉 mask**：tool 返回文本是环境观测，通常不对策略算 logprob/loss；真正进 GRPO 的是模型自己生成的 token（含 tool-call 那一段 assistant 输出）。

本地 `MathAgent.run` / `debug_math_agent` 可以不挂 handler；训练路径必须挂，否则 Adapter 看不到 LLM span。

---



## 7. 两套入口：debug vs 训练


| 入口                        | 谁 `invoke`                                          | callbacks                 | endpoint 从哪来                              |
| ------------------------- | --------------------------------------------------- | ------------------------- | ----------------------------------------- |
| `MathAgent.run(question)` | 纯图，返回最后一条消息文本                                       | 无                         | 构造 `MathAgent` 时传入                        |
| `LitMathAgent.rollout`    | 训练/验证                                               | `get_langchain_handler()` | `resources["main_llm"].get_base_url(...)` |
| `debug_math_agent()`      | `Trainer.dev(LitMathAgent(), records)`，仍走 `rollout` | 有 tracer（dev 模式）          | 环境变量 `OPENAI_API_BASE`                    |


`debug_math_agent`（~675）不启 VERL：用 `agl.Trainer.dev` + 外部 OpenAI 兼容服务冒烟整张图。数据优先 `data/val.parquet` 前 4 条。

---



## 8. Prompt 如何约束 workflow

LangGraph 管「走哪条边」；LangChain 消息管「模型被允许说什么」。两套 prompt：


| 常量                | 注入时机                                  | 作用                                                                       |
| ----------------- | ------------------------------------- | ------------------------------------------------------------------------ |
| `SYSTEM_PROMPT`   | 初始 `SystemMessage`                    | 允许 `execute_python_tool`；禁止乱 import；终值放 `result`；最终格式 `### NUMBER ###`   |
| `FINALIZE_PROMPT` | `request_finalize` 追加为 `HumanMessage` | 「Stop using tools… output ONLY ### NUMBER ###」；配合 `llm_finalize` 无 tools |


只靠 prompt **不能**保证模型停手，所以图上还有 `max_turns` + `asked_finalize` 硬开关。这是「prompt 软约束 + 图硬约束」的典型 Agent 写法。

---



## 9. 代码模块定位总表



### 9.1 本示例（应先读）


| 模块          | 路径                                                                | 职责                                         |
| ----------- | ----------------------------------------------------------------- | ------------------------------------------ |
| 图 + rollout | `[examples/math_gsm/math_agent.py](../math_agent.py)`             | 本文主体                                       |
| Python 沙箱   | `[examples/math_gsm/python_tool.py](../python_tool.py)`           | tool 真实执行环境                                |
| 训练入口        | `[examples/math_gsm/train_math_agent.py](../train_math_agent.py)` | VERL 超参、`Trainer.fit`                      |
| 数据          | `[examples/math_gsm/prepare_data.py](../prepare_data.py)`         | GSM8K → `{id, question, answer}` parquet   |
| 对照 SQL 图    | `[examples/spider/sql_agent.py](../../spider/sql_agent.py)`       | 同一套 LangGraph + `get_langchain_handler` 模式 |


`math_agent.py` 内锚点：


| 符号                                  | 约行号     | 角色                               |
| ----------------------------------- | ------- | -------------------------------- |
| `SYSTEM_PROMPT` / `FINALIZE_PROMPT` | 41 / 54 | 软约束                              |
| `AgentState`                        | 68      | LangGraph 状态                     |
| `execute_python_tool` / `TOOLS`     | 83 / 97 | LangChain Tool                   |
| `MathAgent.__init__`                | 338     | `init_chat_model` + `bind_tools` |
| `call_model`                        | 387     | 图节点 agent                        |
| `request_finalize`                  | 413     | 图节点 finalize                     |
| `call_tools`                        | 425     | 图节点 tools                        |
| `should_continue`                   | 451     | 条件边                              |
| `graph`                             | 477     | `StateGraph` 编译                  |
| `run`                               | 500     | 无训练的 invoke                      |
| `LitMathAgent.rollout`              | 565     | 训练 invoke（含 628–641）             |
| `debug_math_agent`                  | 675     | `Trainer.dev` 冒烟                 |




### 9.2 框架侧（invoke 之后）


| 模块                   | 路径                                                                                                                      | 和 workflow 的关系                          |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------- | --------------------------------------- |
| LitAgent 基类          | `[agentlightning/litagent/litagent.py](../../../agentlightning/litagent/litagent.py)`                                   | `rollout` 接口、`tracer` 属性                |
| Runner               | `[agentlightning/runner/agent.py](../../../agentlightning/runner/agent.py)`                                             | `trace_context` 包住整次 rollout            |
| Tracer 接口            | `[agentlightning/tracer/base.py](../../../agentlightning/tracer/base.py)`                                               | `get_langchain_handler`                 |
| AgentOps Tracer      | `[agentlightning/tracer/agentops.py](../../../agentlightning/tracer/agentops.py)`                                       | 真正的 LangChain callback                  |
| AgentOps 插桩          | `[agentlightning/instrumentation/agentops_langchain.py](../../../agentlightning/instrumentation/agentops_langchain.py)` | 进程级 langchain 集成                        |
| Reward span          | `[agentlightning/emitter/reward.py](../../../agentlightning/emitter/reward.py)`                                         | `emit_reward`                           |
| Triplet Adapter      | `[agentlightning/adapter/triplet.py](../../../agentlightning/adapter/triplet.py)`                                       | span → GRPO 训练样本                        |
| 官方 SQL 教程（同类 invoke） | `[docs/how-to/train-sql-agent.md](../../../docs/how-to/train-sql-agent.md)`                                             | 文档里的 `callbacks` + `recursion_limit` 范例 |




### 9.3 第三方库（本文件 import）

```python
from langchain.chat_models import init_chat_model          # ChatOpenAI 工厂
from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool                      # @tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
```

依赖版本与 Spider 一致：仓库示例倾向 `langgraph<1.0`、`langchain[openai]<1.0`（见 `examples/spider/README.md`）。

---



## 10. 和 Spider SQL Agent 的同构点


|               | math_gsm                               | spider                              |
| ------------- | -------------------------------------- | ----------------------------------- |
| 图库            | LangGraph `StateGraph`                 | 同左                                  |
| LLM           | `init_chat_model` + OpenAI 兼容 endpoint | 同左                                  |
| 训练包装          | `LitMathAgent.rollout`                 | `LitSQLAgent.rollout`               |
| invoke config | `callbacks` + `recursion_limit=40`     | `callbacks` + `recursion_limit=100` |
| 环境            | 受限 Python                              | SQLite `QuerySQLDatabaseTool`       |
| 收口            | `finalize` 节点 + `### N ###`            | `check_query` 判断 CORRECT/INCORRECT  |
| reward        | 答案匹配（默认 binary）                        | SQL 执行匹配                            |


数学侧把「工具循环」收成 **3 节点**（agent / tools / finalize），SQL 侧是 write → exec → check → rewrite。模式相同，任务图不同。

---



## 11. 读代码建议顺序

1. `AgentState` + `graph()` — 先建立状态机心理模型。
2. `call_model` / `call_tools` / `should_continue` — 看一轮 tool 怎么闭环。
3. `LitMathAgent.rollout` 628–641 — 看训练如何把题塞进图、如何挂 tracer。
4. `python_tool.py` — 环境边界。
5. `tracer/agentops.py` + `adapter/triplet.py` — 图执行如何变成 GRPO 样本。

若只想口头讲清 628–641：

> 这里不是「调一次 LLM」，而是把一道题编成 LangGraph 初始状态（system + human），`compile` 后的图从 `agent` 节点起步，按 tool / finalize / end 循环；`callbacks` 把每次 LangChain 调用记成 span，供 Adapter 转 triplet；`recursion_limit=40` 防止死循环。图跑完后的 `messages` 才拿去抽答案、算 reward。

