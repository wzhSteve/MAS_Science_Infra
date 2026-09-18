🔥🔥🔥🔥MAS infra
目的是：让研究人员简单配置 MAS workflow并进行研究，然后直接进行 rollout / RL / evaluation等等
每层之间由单一的接口连接，也就是说每层单独可以正常工作。比如MAS workflow就可以完全不需要RL和Harness层

制定方案，必须实现的原则：
1）所有位置都要留出可扩展的接口，要分层进行高度抽象。且层与层之间的接口交互的信息要严格格式化。2）每层内部高度模块化，开放自由度，也就是支持众多模块插件
3）

前端UI界面（优先级别第四）（负责给MAS层、RL层、Harness层传递参数来帮助他们初始化以及部分在执行中可以动态调整的参数，主要目的是传参和可视化）：
支持用户：提供默认版本agent，通过拖拽来搭建MAS
自动检测本地显卡版本，进行环境配置（本地提供环境版本，通过uv进行配置）
挑选RL算法进行选择
跳转dashboard界面，显示loss、reward曲线等

MAS workflow层（优先级别第二）（数据产生、采集与传输，分成两个小层，MAS构建和数据层，MAS构建包含了MAS如何编排。数据采集，指的是如rollout、reward等数据如何采集与整理：
（PS：可以参考静态的Framework和动态的Runtime这两个概念来进行设计）
MAS构建层：
提供subagent、LLM、Tool、Memory、档案库（记录系统状态，用来支持执行过程的回退）5层（是否建议langGraph，是否有更优的选择，以及是否需要扩充）。
一定有一个核心结构的agent作为中枢（担任角色近似orchestrator/planner）
可扩充subagent，赋予级别和角色（可以先建立参考的是executor/verifier等），切角色技能是可扩充的（可以是转化为skill等），比如在原始的planner-executor-verifier的结构上，我要对verifier赋予因果分析，然后选择给executor一个反馈还是给planner反馈，这些在结构上和subagent技能上的扩充都要留出接口，是重点抽象的point）
LLM提供第三方API调用、本地LLM和RL层返回的LLM endpoint三种类型
Tool优先包含web search和python coder
memory提供每个subagent独有的memory和系统MAS memory（两层memory）
档案库存放一定窗口内的MAS历史snapshot，支持整个系统状态会退（也为RL中返回历史某一状态采样trajectory、harness中失败归因后错误修复回退等提供接口）
数据层：
rollout采样
advantage设计： reward - critial
loss设计：
提供对应接口传递给RL和Harness

RL层（优先级别最高）（从MAS数据层接收数据，用来训练MAS构建层中的LLM）：
从三个结构，来进行抽象，提供接口函数，精简用户需要修改的地方（基于agentlightning/verl的框架进行修改）：
接收MAS层传递过来的rollout、advantage、loss等来进行RL


Harness层（优先级别第三）（自动化分析+可视化窗口等）（会接收从MAS层和RL层传递过来的数据，并且会返回对应的反馈，整体Harness按照插件的逻辑来构思，也就是需要哪个，即插即用，所以需要MAS层和RL层留出来接口：
对于MAS的harness：
基于日志的错误检测
对于planner后的多专家评估（EPC-AW）
Agent 认知是否收敛：对比同类型错误，认知对齐则是学会举一反三，认知高偏差则是相同的错误反复失败
等等（优先留出接口，后面再实现）

对于RL的harness（自动化分析+可视化窗口等）：
Reward hacking：对比真实日志信息，调用外部三方强LLM监督，对比是否真的reward
reward不上升等
归因loss剧烈波动、不下降等


---

## Phase 0 / 初版本（单 Agent / 对齐 tir_agent）

> **实现位置（2026-09）**：编排与数据层在 `mas/workflow/`；训练仍走 `LitTirAgent` + `algos/` + AGL/VERL。薄 CLI 在本仓 `science_infra.ui.cli`。对照分析见 [docs/infra-gap-analysis.md](docs/infra-gap-analysis.md)。

第一期**不实现多智能体**。默认拓扑 `specs/hub_react.yaml`（单 hub ReAct：`web_search` + `wikipedia_search` + `execute_python`）。`ExecutionService` 默认 adapter 为 LangGraph `TirAgent`。

### 层职责（初版本已落地的部分）

| 层 | 入口 | 无 RL 可否独立 | 初版本有 |
|----|------|----------------|----------|
| MAS 构建 | `workflow.spec` + `ExecutionService` | 是 | YAML spec、Skill.run 热路、verifier 回边、双层 MemoryStore、Archive、`run`/`fork` |
| MAS 数据 | `Collector` + `workflow.rewards` | 是 | 单路径 episode、canonical RewardFn、TrainSignal |
| RL | `apply_train_signal` + `train_tir_agent.py` | 否（需 AGL/GPU） | TrainSignal→Hydra；五算法仍走 AGL |
| Harness | `workflow.harness` | 是 | `log_error`、`loss_volatility`、薄 `cognitive_convergence` / `reward_hacking`；`epc_aw_consensus` stub |
| UI | `science-infra` + AGL Dashboard `/science` | collect 不依赖 RL | collect / diagnose / status HTML / dashboard / doctor；AGL 页加载 collect JSON |

### 无 RL 采集（不占训练显卡）

```bash
pip install -e .
science-infra collect --mock --n 2 --out /tmp/traj_batch.json
# 或
cd mas
PYTHONPATH=. python scripts/check_workflow_deps.py
PYTHONPATH=. python scripts/collect_rollouts.py --mock --n 2 --out /tmp/traj_batch.json
```

`workflow/` **不** import `agentlightning`，**不**启动 VERL。

### 与 tir 三钩子的对应

- rollout 采样 → `Collector.collect(..., n=)` / `TirAgentModeDaemon` enqueue
- reward → `workflow.rewards.compute_outcome_reward`（`algos.rewards` 再导出；Collector 与 LitTirAgent 共用）
- advantage / loss → `TrainSignal` → `apply_train_signal`；token-level advantage 仍在 VERL `compute_advantage`

### v1.1 MAS 套接（构建层）

Skill / Memory 不再只是空 Protocol：

- `hub.skills` 在每次 `ExecutionService.run` 上真正 `Skill.run`；未知 skill 仍 `KeyError`（不偷偷 import MAS_structagent）
- `MemoryStore`：默认只写 `scope=agent, owner=hub`；YAML `memory.system` 不是 `none` 时才写系统层
- 改 `specs/hub_react.yaml` 的 `tools` 会反映到 mock 事件和 `TirAgent.from_spec`（关掉 wikipedia 不必改图拓扑）
- 仍不做 PEV、拖拽、第二套产品 Runtime

### v1.2 薄 Harness（认知收敛 + 反 hacking）

- `cognitive_convergence`：把 ERROR 文案归一成 signature；同一类错误出现在 **≥2 条轨迹且后半段仍在重复** → `high deviation`。前半段失败、后半段不再重复 → 视为对齐，返回 `[]`。不是 EPC-AW。
- `reward_hacking`：`reward≥0.9` 但缺答案/format，或同轨迹带 ERROR → 假设 hacking。不调用外部 Judge LLM。
- `epc_aw_consensus` 仍为 stub。Harness 反馈仍只经 `fork`，插件本身不改图。

### v1.3 最小 verifier 回边

- `hub.verify: verifier` 为**事后**检查（不要放进 `hub.skills` 当预执行）。失败则 `SkillResult.route=hub`，`ExecutionService` 最多再跑 `max_feedback_hops` 次（默认 1）。
- 默认 `specs/hub_react.yaml` **不**开启 verify，行为与 v1.2 相同。
- 档案落 `EventKind.FEEDBACK`。不是 PEV 拓扑，也不 import `MAS_structagent`。
- mock：第 0 hop 的 `_mock_error` 在 hop≥1 时默认清除（除非 `_mock_error_persist`），以便测回边。

### v1.4 Science MAS UI（AGL Dashboard）

- `science-infra status` / `science-infra dashboard` 写出单文件 HTML：mean_reward 卡片、SVG reward 曲线、轨迹点选事件、Harness 过滤、TrainSignal。不依赖 AGL 进程。
- Agent-lightning Dashboard 增加 **Science MAS**（`/science`）：打开或粘贴 `collect` JSON，客户端做薄 harness，画 episode reward。不新增 Store API，也不做拖拽搭图。
- 训练 live 曲线仍在 `/metrics`；可选填 TensorBoard URL。