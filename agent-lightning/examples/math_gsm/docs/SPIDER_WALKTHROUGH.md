# Spider SQL Agent 代码详解

本文说明 `examples/spider` 中 text-to-SQL 训练示例的架构与代码对应关系，作为数学 GSM 示例的对照基线。

## 1. 目标与文件

Spider 示例用 Agent-Lightning + VERL（GRPO）训练一个可多轮改写 SQL 的 LangGraph agent。

| 文件 | 职责 |
|------|------|
| `sql_agent.py` | SQL 图逻辑 + `LitSQLAgent.rollout` + 调试入口 |
| `train_sql_agent.py` | VERL 超参、配置变体、`Trainer.fit` 入口 |
| `data/` | Spider parquet 与 SQLite 数据库 |
| `spider_eval/` | 执行匹配评估（`eval_exec_match`） |

## 2. 整体数据流

```text
parquet 任务 ──► Trainer.fit
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
   agl.VERL(config)        n_runners × LitSQLAgent
        │                       │
   vLLM OpenAI API ◄── main_llm 资源 ──┘
   FSDP actor/ref               │
        ▲                       ▼
        │              LangGraph: write→exec→check→rewrite
        │                       │
   GRPO 更新 ◄── Triplet Adapter ◄── reward (0/1)
```

要点：

1. **Algorithm 侧**（VERL）：启动 vLLM、FSDP，把当前策略挂成 `NamedResources["main_llm"]`。
2. **Runner 侧**（`LitSQLAgent`）：拉任务 → 调 LLM endpoint 跑 agent → 返回 float reward。
3. **Adapter**：把 traces 转成 GRPO 需要的 triplets，再更新权重。

## 3. `SQLAgent`（LangGraph）

类定义在 `sql_agent.py`。状态字段包括 `question`、`query`、`execution`、`feedback`、`num_turns`、`messages`。

节点循环：

1. **`write_query`**：按 schema 生成 SQL（markdown 代码块）。
2. **`execute_query`**：`QuerySQLDatabaseTool` 在 SQLite 上执行。
3. **`check_query`**：让模型判断 `THE QUERY IS CORRECT/INCORRECT`。
4. **`should_continue`**：正确或达到 `max_turns` 则结束，否则 **`rewrite_query`** 再回到执行。

LLM 初始化有两条路径：

- **训练/验证（verl_replacement）**：`model_provider="openai"`，`openai_api_base=endpoint`（VERL daemon 提供的 base URL）。
- **独立调试**：读环境变量 `OPENAI_API_BASE` / `OPENAI_API_KEY` / `MODEL`。

## 4. `LitSQLAgent.rollout`

继承 `agl.LitAgent`，核心步骤：

1. 从 `task` 取 `question`、`db_id`、金标 `query`。
2. 按 `rollout.mode` 选择 `database/` 或 `test_database/`，拷贝到临时目录再跑，避免污染原库。
3. `resources["main_llm"]` → `llm.get_base_url(rollout_id, attempt_id)`。
4. 编译 LangGraph 并 `invoke`；可选挂 LangChain tracer。
5. `evaluate_query`（`spider_eval.exec_eval.eval_exec_match`）得到 **1.0 / 0.0**。
6. **直接 `return reward`**（也可 `agl.emit_reward`，二者勿混用）。

`trained_agents` 默认匹配含 `"write"` 的 span，用于只训练写查询相关轨迹。

## 5. `train_sql_agent.py` 配置

`RL_TRAINING_CONFIG` 是 Hydra/VERL 风格 dict，关键字段：

| 配置块 | 含义（spider 默认） |
|--------|---------------------|
| `algorithm.adv_estimator` | `"grpo"` |
| `data.train_files` / `val_files` | parquet 路径 |
| `data.train_batch_size` | 32 |
| `actor_rollout_ref.rollout.name` | `"vllm"` |
| `rollout.n` | GRPO group size = 4 |
| `rollout.multi_turn.format` | `"hermes"`（Qwen tool-call） |
| `engine_kwargs.vllm.tool_call_parser` | `"hermes"` |
| `model.path` | HF 或本地模型路径 |
| `trainer.n_gpus_per_node` | 1 |
| `trainer.total_epochs` | 2 |

配置变体：

- `fast`：小模型、1 step，CI 用。
- `qwen`：默认 Qwen2.5-Coder-1.5B。
- `llama`：换 llama3_json tool format。
- `npu`：NPU 设备与部分 vLLM 开关关闭。

训练入口：

```python
agent = LitSQLAgent()
algorithm = agl.VERL(config)
trainer = agl.Trainer(n_runners=10, algorithm=algorithm, adapter={"agent_match": active_agent})
trainer.fit(agent, train_dataset=..., val_dataset=...)
```

## 6. 调试

```bash
export OPENAI_API_BASE=...
export OPENAI_API_KEY=...
python sql_agent.py
```

`debug_sql_agent()` 用 `Trainer.dev` + 固定 `initial_resources`，在小样本上跑通图，不启动 VERL。

## 7. 与数学 GSM 示例的对应关系

| Spider | Math GSM |
|--------|----------|
| `SQLAgent` + SQL DB 工具 | `MathAgent` + `execute_python` |
| 执行匹配 reward | 答案数值/字符串匹配 reward |
| Spider parquet | GSM8K parquet |
| `train_sql_agent.py` | `train_math_agent.py` |
| Qwen2.5-Coder 等 | 本地 Qwen3-4B + A800 配置 |

数学示例复用同一套 **LitAgent + VERL + Trainer** 模式，只替换任务图、工具与 reward。
