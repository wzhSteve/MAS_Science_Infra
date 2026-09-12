# TIR Agent

单 Agent、LangGraph、Hermes tool-call，训练外壳对齐 [`examples/math_gsm`](../math_gsm)。工具是 **web_search / wikipedia_search / execute_python**。数据默认 **GSM8K + HotpotQA 各 50%**。

`--algo grpo|arpo|aepo|igpo|gigpo` 只改本目录的 trainer/daemon。Hydra 里 **`algorithm.adv_estimator` 永远是 `grpo`**（关掉 Critic）；真实算法名在 `algorithm.tir_algo`。

这不是论文引擎的 100% 复现。IGPO / GiGPO 落在 triplet + custom advantage 上更接近原文；ARPO / AEPO 用两阶段 enqueue + tool 后前缀续写近似树采样。细节见 [docs/DESIGN.md](docs/DESIGN.md)；与官方 ARPO 仓库的架构 / tool / 熵 fork 对照见 [docs/ARPO_VS_TIR.md](docs/ARPO_VS_TIR.md)。对照笔记 [math_gsm/docs/interview/10_agent_rl_arpo_aepo_igpo_gigpo.md](../math_gsm/docs/interview/10_agent_rl_arpo_aepo_igpo_gigpo.md)。

## 准备数据

```bash
# 默认走 HF_ENDPOINT=https://hf-mirror.com；先跑通可用 1024 条
python prepare_data.py --train-limit 1024 --val-limit 200

# 无 Hub / 无网（仅算术 + 合成 QA，search 意义不大）
python prepare_data.py --offline --train-limit 32 --val-limit 8

TRAIN_LIMIT=1024 VAL_LIMIT=200 bash scripts/prepare_data.sh
```

检索默认走 `WEB_SEARCH_PROXY=http://127.0.0.1:7890`（与 epc_aw Clash 一致）。先验证三工具：

```bash
python scripts/check_tools.py
bash scripts/debug_tir_agent.sh   # 1 GPU 起 vLLM，确认会调 python / search
```

## 训练

Python 建议：`/root/autodl-tmp/AgentFlow/.venv/bin/python`。`--algo grpo` 走与 math_gsm 相同的股票 `agl.VERL`（不挂自定义 trainer/daemon）。

```bash
bash scripts/train_2gpu.sh                    # 2×A800 vanilla GRPO
python train_tir_agent.py a800_2gpu --algo grpo
```

`--n-runners` / `--model` 与 math_gsm 相同。配置档：`fast` / `a800` / `a800_2gpu`。正式训练 `max_prompt_length=8192`（检索 snippet 更长）。

## MAS_structagent RL（独立入口）

用 Planner→Actor→Verifier 多智能体做 RL，**不改**上方 TirAgent 路径。训练 LLM 走 VERL `main_llm`（默认本地 Qwen3-4B），**不读** `.env` 的 dmxapi。详见 [docs/DESIGN.md §7](docs/DESIGN.md)。

```bash
bash scripts/train_mas_2gpu.sh
python train_mas_agent.py fast
OPENAI_API_BASE=http://127.0.0.1:8000/v1 python mas_agent.py   # 冒烟，不启 VERL
```

## Agent 冒烟（不启 VERL）

需要 `OPENAI_API_BASE`（OpenAI-compatible，例如 vLLM）：

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
python tir_agent.py
```

## 奖励

所有 algo 共用层次 outcome，方便对比：

- 格式坏（无 `<answer>...</answer>`）→ `-1`
- 格式好但 Acc=0 → `0`
- Acc>0 → Acc（GSM8K 0/1，QA 为词级 F1∈[0,1]）；若本轨迹 **search 与 python 都用过** → `+0.1`

## 诚实边界

| 算法 | 本目录实际做了什么 |
| ---- | ------------------- |
| GRPO | 与 math_gsm 相同：`rollout.n` 条独立轨迹，轨迹级 outcome Â |
| IGPO | 采样同 GRPO。过程奖励 MVP 用「工具观测是否新出现 GT」作 IG 代理，再组内 z-norm + 折现。不是 actor teacher-forcing `log π(GT)` |
| GiGPO | `A = A^E + ω A^S`，anchor = `hash(tool + obs[:512])`。数学题无重复状态时 `A^S=0`，退化为 GRPO |
| ARPO | 先 enqueue `initial_rollouts` 条完整轨迹，高熵 tool 后带 `resume_messages` 再采；M 条完整轨迹进 soft GRPO。**不改 vLLM 内核**。与官方实现差距见 [docs/ARPO_VS_TIR.md](docs/ARPO_VS_TIR.md) |
| AEPO-lite | 预监控 1 条轨迹分预算 + 连续高熵分支惩罚 + Â 乘熵项。**没有**论文里 clip 的 stop-grad（二期） |

## Workflow collect (no RL / no training GPU)

本仓根目录可装薄 CLI：`pip install -e .` 后 `science-infra collect --mock --out /tmp/traj.json`。`science-infra dashboard` 写 HTML；AGL Dashboard `/science` 可打开同一 JSON。

```bash
PYTHONPATH=. python scripts/collect_rollouts.py --mock --out /tmp/traj.json
PYTHONPATH=. python scripts/check_workflow_deps.py
PYTHONPATH=. python -m unittest discover -s tests -v
```

See `docs/DESIGN.md` §8 and the `workflow/` package.

## 模型配置与最小执行契约

Control 的模型就绪接口位于仓库根目录 `science_infra\control`，执行契约仍位于本目录的 `workflow`，不依赖 HTTP 或 Agent-Lightning：

- `LLMConfig` 支持显式 `api_key`、模型来源、可选策略版本和 tokenizer 身份。未传 Key 时保留原 CLI / 训练环境解析；传空字符串时不继承环境密钥，可用于无鉴权兼容服务。
- `Collector(llm_config=...)` 复用请求级模型配置，不修改全局环境；不传时保持现有 endpoint/model 调用方式。
- `ModelIdentity` 与 `ModelCapabilities` 是可公开的记录契约，不含凭据；缺失能力默认 `unknown`，不通过模型名推断。
- `AgentExecutionContext` 记录 Agent 进入时的 Prompt、工具定义、Skills 和模型归属；实际裁剪后的模型输入以 `model_call` 为准，不能把进入时上下文当成最终请求。
- `ExecutionRecorder.append()` 与 Archive 的已有事件记录接口兼容。
- `ExecutionEvent` 增加可选的执行片段 / 模型 / 工具关联 ID 和实际发生时间；旧记录不填时保持未知，不能用默认 hub 身份推断完整归属。
- `Snapshot.coverage` 标记当前快照实际覆盖的状态，现有默认仅包含消息，不承诺全系统恢复。

执行上下文不包含参考答案等评估数据，Skill 沿用代码注册接口，不自动解释为提示词文件。MemoryStore 仍是运行内存储，不提供向量库或跨任务长期记忆。

## 统一执行与单题 Rollout（第二阶段）

`ExecutionService(mock=..., spec=..., llm=..., archive_root=..., run_id=...).run(task)`
是 Control、Collector 与 `LitTirAgent` 的共同 MAS 执行入口，不计算 reward。
Collector / 训练适配继续负责评分；训练 trace callbacks、rollout / attempt 身份和采样参数仍传入真实模型调用。
`last_raw` 和 `last_archive` 是本次执行的训练兼容数据，不是第二套执行路径。
显式 `run_id` 适用于一次提交；省略时每次 `run` 都生成新 ID。每次调用都清空运行内 MemoryStore。
Control 可提前调用 `workflow.runtime.validate_execution_spec(spec)` 校验同一运行时支持范围；
该函数不执行、不创建 Archive、不加载模型，服务构造时也会执行相同校验。

- 编译失败直接拒绝，不回退到示例运行。`hub_react` / `single` 和 `graph_hub_subset`
  使用 hub + 可选 verifier 兼容语义；一般 `graph_compiled` 按显式 route / message / feedback 行走。
  不支持分叉派发、不可达的已配置 Agent、Agent 独立模型绑定或其他 memory 后端。
- 每个 Agent 使用自己的 Prompt、Skill 和工具绑定。显式 `Agent.tools=[]` 禁用工具，
  `tool_call` 边可显式增加工具；仅未声明 hub 节点的旧配置继承顶层 tools。
  `LLMConfig.enabled_tools` 是额外的允许列表；`TirAgent(enabled_tools=None)` 仅在旧直接入口继承默认工具。
  无工具的 Agent 不绑定工具，模型请求省略 `tools` 字段，而不是发送部分兼容服务拒绝的 `tools: []`。
- Skill 使用现有注册表，在 Agent 进入时执行，结果进入当前 Agent 上下文；verifier / critic
  无 Prompt 和工具时仅运行验证 Skill，有 Prompt / 工具时先执行模型、再验证。
  Skill 的 route 元数据保留，但一般图的派发以已编译边为准，不动态创建边。
- Memory 按 Agent owner 隔离，声明 `memory_scope=system` 才读取共享 memory；
  启用 system=kv 时成功输出写入共享层。模型输入只含问题、上游输出、Skill 结果和明确读取的 memory，
  不含 task.answer / answers 等评分数据。
- 模型异常不伪造 `<answer>None</answer>`；工具异常与已有 `Error:` / `search_unavailable:`
  返回记录为可恢复失败。轮次、token、反馈和图步数上限均有独立终止原因。
- 不生成熵或 token 概率。旧 `h_root` / `h_tool` 只在取得 logprobs 时保存
  **选中 token surprisal 的近似指标**；缺失时中性零值带 `uncertainty_evidence=unavailable_neutral_zero`，
  不能解释成测得熵为零。provider 的原始 logprobs / usage / response metadata 单独保留。

### 持久记录契约

轨迹 `schema_version="2"`，保留旧字段。`meta` 包含 `run_id`、`status=succeeded|failed`、
`termination_reason`、`trace_status=complete|partial`、起止时间、Workflow 快照、实际执行模式、
请求模型身份与 provider 返回的 `actual_models`；失败另有脱敏的 `error`、`error_stage`、`error_code`。
`complete` 只表示本阶段的采集完整，不表示满足任意 RL 算法或严格可复现。
Mock 合成轨迹、只有消息前缀的历史续跑及发生凭据脱敏的轨迹标为 `partial`；
`trace_coverage` 区分本次边界记录、合成调用和缺失的历史调用证据，不补造历史调用计数。

| 事件 kind | 记录内容 |
| --- | --- |
| `agent_enter` / `agent_exit` | Agent 执行片段、角色、进入配置、结果与终止原因 |
| `model_call` | 在 invoke 前记录实际裁剪后 messages、工具定义、实际 sampling_parameters、请求模型、重试关联、context_processing |
| `model_result` | 同一 model_call_id 的状态、完整可见 message、finish_reason、实际模型名（若返回）、原始 usage/logprobs |
| `tool_call` / `tool_result` | 同一 tool_call_id 的名称、完整 args / content、状态和可恢复失败；关联所属 model_call_id |
| `skill_call` / `skill_result` | Skill 输入与完整结果，由 parent_id 关联 |
| `handoff` / `error` / `termination` | 上下游 Agent、输出 / 反馈、错误阶段与最终终止 |

所有新事件均有 run_id、trajectory_id、occurred_at；Agent 内事件有 agent_execution_id，
模型 / 工具事件带对应调用 ID。计数分别取 `kind=model_call` / `kind=tool_call`，
不从消息条数推断。Mock 工具事件明确 `mock=true`，**不会伪造 model_call**。
OpenAI adapter 可提供请求构造器时 `request_format=provider_payload`；兼容 adapter 则明确
`langchain_messages`，不声称记录了不可见的 HTTP headers。

Archive 保存完整事件、消息快照及 `trajectory.json`，不使用日志摘要替代训练记录。
模型请求的 context fitting 不会删掉档案中的原始工具返回；凭据脱敏范围记录在 `meta.redaction`。
执行失败尽量保留已完成调用与部分轨迹；任何档案写入失败直接抛出，不报告采集成功。
已有 `resume_messages` / BranchPoint 仅支持 hub 消息前缀续写，不恢复工具外部状态、MAS 游标或 MemoryStore；
一般图明确拒绝此类续跑。快照 coverage 不承诺任意位置分支。

## Included Files

| 文件 / 目录 | 职责 |
| --- | --- |
| `tir_agent.py`、`tools` | 模型与工具交互 |
| `workflow\runtime.py`、`collector.py` | 执行及轨迹采集 |
| `workflow\contracts.py` | 轨迹与最小执行契约 |
| `workflow\recorder.py` | 调用边界关联、完整消息序列化与凭据脱敏 |
| `workflow\memory.py`、`plugins.py` | 运行内记忆和 Skill 注册 |
| `workflow\archive.py`、`spec.py`、`compiler.py` | 记录、快照、Workflow 定义与编译 |
| `lit_tir_agent.py`、`train_tir_agent.py`、`algos` | 训练适配与研究算法 |
| `prepare_data.py`、`scripts`、`tests`、`docs` | 数据准备、脚本、既有测试与说明 |
