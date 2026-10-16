MAS Infra 重构方案：架构审查、设计风险与最终优化建议

一、总体结论

我认为，这份 MAS 层重构方案已经具备相当不错的工程设计基础。它不是简单地移动文件，而是试图在不破坏现有训练链路的前提下，把 Agent 定义、运行时执行、通信协议、记忆管理、分支采样、遗留研究代码和运行时产物的边界重新明确下来。

但是，如果站在一个准备长期维护、开源发布，并且希望让其他研究者能够在其上开发 MAS 和 Agent RL 算法的基础设施项目的架构师角度，我认为仍然存在一些需要在实施前解决的问题。

我的核心判断是：可以进入渐进式重构阶段，但不建议直接按照当前设计稿完整实施。应该先冻结架构边界，修正几个关键语义问题，再开始移动文件。

原因不是目录设计有明显缺陷，而是部分模块的职责、协议语义和运行时边界还没有完全闭合。

尤其需要注意以下六个问题：

1. Agent、ToolAgent、Router 的抽象层级仍然不够统一。 当前统一了节点声明，但没有完全统一节点的执行语义。
2. MemoryStore、FactLedger、Archive 虽然实现了物理分层，但逻辑上仍然存在职责交叉。 尤其是事实提交、事件持久化与记忆读取之间的关系。
3. AgentMessage 协议还不足以成为长期稳定的运行时协议。 缺少对消息关联、并行执行、幂等性和失败语义的明确规定。
4. MAS 设计态与 Agent RL 训练态之间仍然存在结构性断层。 目前这是整个项目最重要的架构限制之一。
5. 当前的测试策略主要验证代码没有被重构破坏，但不足以证明重构后的架构语义正确。
6. 文件组织优化已经接近合理，但还缺少一套清晰的依赖规则、扩展接口和兼容性策略。

下面按照架构定位、目录结构、Agent 抽象、通信协议、记忆、Prompt、运行时、RL 接口、测试和实施顺序逐层分析。

⸻

二、首先明确：你的 MAS Infra 到底应该是什么？

这是审查所有文件结构之前，最重要的前提。

结合你此前的 AgentOS 设计方向，你的目标并不是单纯实现一个能够调用多个 LLM 的工作流引擎，而是希望构建一个支持：

* 专业化 Agent 的创建和组合。
* 多智能体协作与复杂任务执行。
* Agent 执行过程的记录、分析与评估。
* Agent RL 的 rollout、分支采样和训练。
* 研究算法的快速接入与比较。
* 用户自定义 Agent 和工具的扩展。
* 后续面向不同学科与企业场景的应用。

因此，MAS 层不能只被理解成一个 Workflow Engine。

它至少承担三个不同层面的职责。

层次	核心问题	典型组件
MAS Definition	用户如何定义一个多智能体系统？	AgentNodeSpec、RouterSpec、MASSpec、Prompt、Schema
MAS Execution	系统如何实际运行？	AgentRegistry、Runtime、Router、Protocol、Memory
MAS Observability & Learning	如何记录、评估和改进执行过程？	Trajectory、Archive、BranchSite、Harness、RL Adapter

这三层相互关联，但不能混成一个层次。

例如：

* spec.py 应该描述 Agent 的配置，而不是负责执行 Agent。
* agents.py 应该负责 Agent 实例的注册和解析，而不是直接承担整个执行流程。
* centralized_runtime.py 应该负责执行协调，而不是同时成为消息协议、事实提交、记忆管理和 Archive 写入的主要实现位置。
* contracts.py 应该描述稳定的数据合同，而不是逐渐成为所有模块共享的杂物文件。
* active_set.py 应该负责分支展开策略，而不是重新实现 Runtime 的执行逻辑。

你当前的方案已经朝着这个方向发展，但还没有完全做到。

2.1 我建议采用的最终架构原则

你的项目应当遵循一个基本的依赖方向：

                User / UI / CLI / API
                         |
                         v
                 MAS Definition
            Spec / Schema / Prompts
                         |
                         v
                 MAS Execution
          Runtime / Router / Agents
                         |
              +----------+----------+
              |          |          |
              v          v          v
           Memory     Protocol    Tools
              |          |          |
              +----------+----------+
                         |
                         v
                Execution Records
             Events / Trajectory
                         |
                 +-------+-------+
                 |               |
                 v               v
              Harness        RL Adapter
             Evaluation      Rollout/Train

这里的图表示逻辑依赖，而不是要求你立刻按照这些层次创建新的目录。

我赞成你暂时不创建 mas/engine/，但不赞成因为暂时不拆目录，就放弃对模块职责的严格约束。

目录结构可以保持平铺，架构边界不能因此变得模糊。

⸻

三、目录结构审查：哪些应该移动，哪些不应该移动？

3.1 顶层目录总体合理

你设计的顶层目录大致为：

MAS_Science_Infra/
├── mas/
├── legacy/
├── rl/
├── science_infra/
├── webui/
├── experiments/
├── artifacts/
├── user_space/
└── agent-lightning/

这个结构整体合理。

它已经体现出四个重要区分：

* mas/：Agent 系统定义和执行。
* rl/：训练算法与优化逻辑。
* science_infra/：控制层和服务入口。
* legacy/：保留的历史研究实现。

与此同时，experiments/ 与 artifacts/ 分离，也符合研究基础设施的需求。

不过，这里需要特别区分两个概念：

物理目录隔离，不等于运行时隔离；代码归属清晰，也不等于依赖关系清晰。

例如，即使把 StructAgent 移入 legacy/structagent/，如果产品主栈的 tool_agents.py 仍然通过动态导入依赖它，那么 legacy 代码仍然是产品运行时依赖的一部分。

这并不一定错误，但需要明确。

建议：将 legacy 依赖分为三种

依赖类型	示例	建议
产品必需的运行时依赖	某个 ToolAgent 必须调用的页面分析器	可以保留，但必须明确声明
可选的高级能力	tier: pro 才需要的遗留 LLM 工具	使用可选适配器，并提供明确的缺失依赖错误
独立研究栈	train_mas_agent.py、完整 EPC-AW 实现	不应被产品主栈默认加载

这里尤其要注意：

如果 legacy/structagent/epc_aw/tools/ 被产品主栈的五个 ToolAgent 调用，那么它实际上不是完全独立的遗留代码，而是一个被产品主栈消费的可选后端。

建议在文档中明确写出这个事实。

更长期的方向是将真正需要复用的通用能力抽象成稳定接口，而不是让产品主栈直接依赖整个研究栈。

但这不需要在本轮重构中完成。

⸻

3.2 mas/workflow/：暂时不拆目录是正确的，但应限制继续膨胀

你目前决定不创建：

mas/engine/

我支持这个决策。

原因有三个。

第一，现有的：

from workflow.runtime import ...
from workflow.spec import ...
from workflow.memory import ...

已经形成稳定的导入路径。

第二，训练控制层通过 sys.path.insert(0, mas) 拉起模块，意味着目录变化可能影响运行入口、测试和运行环境。

第三，你当前的目标是低成本重构，而不是大规模包结构迁移。

为了清晰而增加一层目录，如果需要同步改动数百个 import，收益不一定超过风险。

但是，我建议增加一个约束：

允许 workflow/ 保持平铺，但新增模块必须遵守明确的职责边界，禁止将其当作默认的公共代码存放区。

当前的文件分布已经说明这一点。

例如：

workflow/
├── spec.py
├── contracts.py
├── compiler.py
├── protocol.py
├── runtime.py
├── centralized_runtime.py
├── router.py
├── agents.py
├── prompts.py
├── active_set.py
└── memory/

这个规模还可以接受。

真正需要警惕的不是 31 个还是 33 个文件，而是未来出现：

workflow/
├── memory.py
├── memory_utils.py
├── memory_manager.py
├── memory_service.py
├── memory_adapter.py
├── memory_helpers.py
└── memory_v2.py

这类结构。

因此，我建议新增一份：

mas/docs/CONCEPT_MAP.md

并明确三个规则：

1. 每个稳定概念只有一个权威实现。
2. 兼容性 shim 必须有明确的删除条件。
3. 新模块必须声明其职责、依赖方向和公共接口。

这比现在就创建新的顶层目录更有价值。

⸻

3.3 将 MAS_structagent 移入 legacy/structagent/：赞成，但要补充 Python 包隔离规则

这项调整有必要。

当前的：

mas/MAS_structagent/

容易让开发者误以为它是产品 MAS 主栈的一部分。

移动后：

legacy/
└── structagent/
    ├── epc_aw/
    ├── mas_agent.py
    ├── train_mas_agent.py
    └── scripts/

语义上会清晰很多。

但有一个需要重点验证的问题：

sys.modules["MAS"] = ...

这种注册方式存在潜在的全局命名空间冲突。

如果产品主栈、测试框架或其他 Python 包已经加载了同名的 MAS，那么导入结果可能受到模块加载顺序影响。

因此，不能仅仅通过更新八处 import，就认为隔离完成了。

建议实施三个验证

第一，验证干净环境中的独立导入。

在全新 Python 进程中分别执行：

* 产品主栈导入。
* StructAgent 导入。
* 产品主栈与 StructAgent 同进程导入。

重点观察第三种情况。

第二，验证模块身份。

确认：

import MAS

是否指向预期的 vendored 包，而不是其他同名模块。

第三，验证失败隔离。

如果用户没有安装 StructAgent 的可选依赖：

* 产品主栈的 lite 工具应该仍然可以工作。
* pro 工具应该返回明确的依赖缺失错误。
* 不能在 import 阶段直接导致整个产品不可用。

如果短期无法消除 sys.modules["MAS"] 的全局注册，至少应将它限定在独立 bootstrap 中，并明确禁止产品主栈直接依赖这个注册名。

这部分我会列为重构阶段一的必测项。

⸻

四、Agent 抽象：这是当前设计最值得深入优化的部分

你现在的节点定义是：

class AgentNodeSpec(BaseModel):
    id: str
    kind: Literal["planner", "tool", "verifier", "blank"] = "blank"
    role: str = "agent"
    skills: List[str] = []
    tools: List[str] = []
    memory_scope: str = "none"
    system_prompt: str = ""
    model: str = "inherit"
    trainable: bool = True
    profile: Dict[str, Any] = {}

这个设计足够支持当前产品。

但如果希望未来支持不同的 MAS 拓扑、不同类型的 Agent、不同的执行后端以及 Agent RL，这个模型还需要做一次概念上的整理。

4.1 当前最大的问题：Agent 类型与执行职责混在一起

你现在有四种类型：

planner
tool
verifier
blank

它们并不完全属于同一分类维度。

例如：

* planner 描述的是决策职责。
* verifier 描述的是验证职责。
* tool 描述的是执行方式或执行对象。
* blank 只是尚未指定语义的通用节点。

这会产生一个问题：

一个负责分析图像的 Agent，究竟属于什么类型？

如果它需要调用工具并验证结果，又应该属于什么类型？

如果用户希望定义一个 Planner-Executor-Verifier 架构中的 Executor，但 Executor 自身也是一个可以自主推理、使用多个工具的 LLM Agent，那么它究竟是 tool 还是 blank？

目前的设计没有给出统一答案。

我的建议：短期保留四类，长期区分两个维度

短期内不要为了理论上的抽象完整性重写整个系统。

可以继续保留：

kind: Literal["planner", "tool", "verifier", "blank"]

但是，在文档和接口层面，明确区分：

概念	说明
Agent Role	节点承担的逻辑职责
Execution Type	节点如何执行
Backend	节点使用什么模型或执行环境
Capability	节点具备哪些工具和技能

例如：

id: medical_reasoner
kind: blank
role: diagnostic_reasoner
profile:
  backend: llm
  capabilities:
    - image_analysis
    - evidence_synthesis

这里不建议现在就把这些字段全部正式加入 Pydantic Schema。

首先要确认这些维度在当前代码中确实存在不同的执行行为。

只有当两个字段能够导致不同的运行时行为时，它们才有必要成为独立配置项。

否则，只是增加了配置复杂度。

⸻

4.2 trainable 字段存在语义歧义

当前：

trainable: bool = True

看起来很简单，但在 Agent RL 系统中，它可能代表至少三种不同含义：

1. 该 Agent 是否参与梯度更新。
2. 该 Agent 的输出是否参与训练样本构造。
3. 该 Agent 的行为是否可以被策略优化算法改变。

这三件事并不等价。

例如：

* 某个 Agent 使用冻结的开源 LLM，但其调用过程产生的消息可以进入训练轨迹。
* 某个 ToolAgent 完全不训练，但它的输出会影响 Planner 的奖励。
* 某个 Agent 的输出参与轨迹记录，但其参数不属于当前策略的优化范围。

因此，我建议在文档中明确：

trainable = 当前节点是否属于可优化策略的候选范围

同时明确它不代表：

该节点一定产生梯度。

真正的训练参与范围，应该由训练配置、策略绑定和 RL Adapter 共同确定。

在你当前的架构中，trainable 暂时可以保留，但不能直接把它当成训练调度器的唯一依据。

⸻

4.3 tools: List[str] 不应同时承担工具授权与路由候选两个职责

当前：

tools: List[str] = []

注释为：

planner 可路由的 tool-agent id

这说明该字段实际存储的是可路由的 Agent 集合，而不一定是工具本身。

这会造成概念歧义。

因为你的项目中同时存在：

mas/tools/

和：

AgentNodeSpec.tools

前者是 ToolAgent 实现目录，后者却是 Agent 节点 ID 列表。

我建议把它的语义明确为：

tools: List[str]

继续保持兼容，但在文档和内部变量中使用更精确的术语：

available_agent_ids

或者：

routable_agent_ids

短期不必改字段名，以免影响 YAML 和 UI。

长期可以通过 Schema 版本迁移，把“可路由的 Agent”和“Agent 可调用的原子工具”分开。

这会使未来的工具权限控制、Agent 组合和拓扑编译更加清晰。

⸻

五、ToolAgent 设计：五个内置工具是否应该全部保持当前形态？

你目前定义的五个内置 ToolAgent：

* wikipedia_search
* google_search
* web_fetch
* python_coder
* think

同时采用：

纯函数内核 + 可选 LLM 页面分析

这是一种合理的设计。

它允许相同的工具接口在不同的成本和能力等级之间切换。

例如：

ToolAgent
   |
   v
Tool Kernel
   |
   +---- lite：直接执行
   |
   +---- pro：执行 + LLM 分析

这个结构的优点是：

* 工具逻辑与 LLM 推理分离。
* 同一个工具可以用于普通推理和复杂 Agent 工作流。
* 更容易测试工具的确定性行为。
* 更容易分析 LLM 带来的额外成本。

但是，我认为需要明确三个问题。

5.1 lite/pro 不应只是一个隐式开关

当前：

profile.backend
tier: lite | pro
SCIENCE_INFRA_TOOL_KERNEL=1

这些配置可能分别影响：

* 使用什么执行内核。
* 是否调用 LLM。
* 是否加载 StructAgent。
* 是否生成额外分析结果。

建议在设计文档中定义明确的解析优先级。

例如：

实验显式配置
    >
工具实例配置
    >
环境变量强制覆盖
    >
默认配置

环境变量的作用应该特别明确。

如果：

SCIENCE_INFRA_TOOL_KERNEL=1

是安全性或运行成本方面的强制降级开关，那么它可以覆盖 pro 配置。

但如果它只是普通默认项，就不应该静默覆盖用户的实验设置。

建议启动时记录最终解析结果：

{
  "tool": "web_fetch",
  "configured_tier": "pro",
  "effective_tier": "lite",
  "override_reason": "SCIENCE_INFRA_TOOL_KERNEL"
}

这样 Harness 才能判断实验差异来自算法，还是工具配置。

5.2 工具调用必须具备可重放性约束

对于 Agent RL，工具输出会影响后续决策和奖励。

如果同一个 rollout 在训练和评估时调用实时搜索，结果可能不同。

因此建议将以下信息记录进 Tool Call 事件：

tool_id
tool_version
input_digest
output_digest
execution_status
latency
cache_status
backend

如果工具支持缓存，还应记录是否命中缓存。

这里不要求把所有工具输出都保存到同一份 JSON 文件中，但需要确保能够区分：

* 工具执行失败。
* 工具返回空结果。
* 工具执行成功但没有获得有效证据。
* 工具返回的内容发生变化。

这对于后续的失败归因和 Agent RL 都非常重要。

5.3 think 不应被误认为外部工具

think 与搜索和代码执行的性质不同。

它可能只是一个额外的推理窗口，也可能是一个需要单独统计成本的 LLM 调用。

建议在工具接口层明确区分：

External Tool
Reasoning Agent
LLM-in-Tool

不一定需要立刻创建三个不同的 Python 基类。

但是，至少要让日志、成本统计和训练轨迹能够识别这些执行类型。

否则，后续统计 ToolAgent 的成本时，很容易把搜索调用和额外 LLM 推理混为一谈。

⸻

六、Router：总体方向正确，但需要补上四个关键细节

你当前的 Router 设计：

class RouterSpec(BaseModel):
    id: str
    candidates: List[str]
    strategy: Literal["from_plan", "llm_choice", "score", "round_robin"]
    scorer: Optional[str]
    output_contract: str = "json"

总体上是合理的。

尤其是你决定让：

from_plan / llm_choice

直接解析 Planner 的输出，不再让 Router 额外调用一次 LLM。

这有利于减少额外调用和不必要的推理成本。

但是，这里存在一个命名问题和三个运行时问题。

6.1 from_plan 和 llm_choice 当前存在语义重叠

如果二者都只解析：

{
  "next": "wikipedia_search"
}

那么从 Runtime 的角度，它们没有区别。

因为：

* from_plan：Planner 决定下一步。
* llm_choice：名称暗示 Router 自己使用 LLM 决策。

但你明确规定 Router 不再调用 LLM。

因此，当前 llm_choice 容易造成误解。

建议：

* 保留现有枚举，避免立即破坏配置兼容。
* 明确它是历史兼容名称，实际执行与 from_plan 一致。
* 新建实验默认使用 from_plan。
* 只有在未来实现真正独立的 LLM Router 时，才赋予 llm_choice 独立语义。

不要让两个名称不同的策略长期拥有完全相同的实现，却没有任何文档说明。

6.2 并行 fan-out 必须定义部分失败语义

你目前支持：

Planner
   |
   v
Router
   |
   +---- Tool A
   |
   +---- Tool B
   |
   +---- Tool C

通过：

ThreadPoolExecutor

并行执行多个 hop。

但你还没有明确：

假设三个 Agent 中：

* A 成功。
* B 超时。
* C 返回错误。

Runtime 应该怎么办？

我建议采用明确的逐节点结果模型：

Hop A → SUCCESS
Hop B → TIMEOUT
Hop C → ERROR

然后由 Verifier 根据全部结果决定是否接受当前子目标。

不要让其中一个工具失败就直接抛出异常，导致其他已经成功的结果丢失。

同时，Runtime 必须保证：

1. 每个被选中的节点都有对应的终态记录。
2. 每个成功的工具结果都能够被 Archive 保存。
3. 部分失败不会导致整轮事件流不完整。
4. Verifier 能够识别部分成功与整体成功的区别。

这会显著提高复杂 MAS 的可调试性。

6.3 score 策略必须防止比较标准漂移

当前设计为：

每个候选 blank agent 对同一输入打分，scorer 取最高。

这里有一个重要问题：

如果候选 Agent 各自使用不同的 Prompt、不同的模型，或者不同的评分尺度，那么分数可能不可直接比较。

例如：

Agent A：0.92
Agent B：0.84

这不意味着 A 的结果一定更好。

因此，建议将 score 的语义明确为：

只有当所有候选返回同一评分合同，并且评分方向、范围和缺失值处理一致时，才允许比较。

如果评分是由独立的 LLM 完成的，那么最好由同一个 Scorer 对各候选输出进行评估，而不是默认相信各 Agent 自报的分数具有可比性。

这也是为什么 scorer 应该在 score 策略中成为强约束，而不只是可选配置。

6.4 round_robin 必须定义计数器生命周期

你计划将其用于训练中固定形状的组采样。

但是计数器究竟属于：

* 全局 Runtime？
* 当前 Episode？
* 当前 Rollout？
* 当前分支？

不同答案会导致不同的行为。

我建议默认：

round_robin counter scope = episode

并明确分支恢复时计数器如何处理。

如果第二波分支任务从某个快照恢复，而计数器没有恢复，那么相同前缀可能选择不同的 Agent。

这会让训练结果难以复现。

如果当前并不需要在分支恢复后继续严格复现路由行为，可以先将这一限制写入文档，但不要把它留成未定义行为。

⸻

七、通信协议：AgentMessage 是正确方向，但目前还不够稳定

这是整个 MAS 执行架构中，我认为最值得提前加强的部分。

你定义了：

class AgentMessage(BaseModel):
    msg_id: str
    task_id: str
    turn: int
    src: str
    dst: str
    kind: Literal[
        "plan_step",
        "route_decision",
        "tool_invoke",
        "tool_result",
        "verify",
        "feedback",
        "final_answer",
        "error"
    ]
    payload: Dict[str, Any]
    trace_ref: Optional[str]

它已经解决了一个核心问题：

MAS 内部不再直接传递 LangChain 的私有消息对象，而是通过自己的协议传递结构化数据。

这对于独立于特定 Agent 框架、支持多种执行后端非常重要。

但需要注意：

AgentMessage 不应该只是把原来的消息对象换一个名字。

它应该成为稳定的内部协议。

7.1 缺少 run_id、episode_id 和 parent 级别的关联

目前只有：

task_id
turn
trace_ref

这对简单的串行执行够用，但对并行执行、分支采样和跨进程恢复还不够明确。

建议区分：

字段	作用
task_id	逻辑任务标识
run_id	一次实验或运行标识
episode_id	一次任务执行实例
msg_id	当前消息唯一标识
causation_id	直接导致当前消息产生的消息
correlation_id	将同一轮并行调用或请求关联起来
parent_episode_id	分支任务的父执行实例，可选

其中，run_id 可以在 Archive 外层元数据中维护，不一定要在第一版消息对象中强制加入。

但至少应该明确：

消息关联标识的权威来源在哪里，以及跨进程恢复时如何保持关联。

你当前的 trace_ref 可以继续作为兼容字段，但建议未来统一其语义为直接因果引用，而不是模糊地指向“上一条消息”。

因为并行执行中不存在唯一的上一条消息。

例如：

                    Planner Message
                    /             \
                   v               v
             Tool Invoke A    Tool Invoke B
                   |               |
                   v               v
             Tool Result A    Tool Result B
                    \             /
                     v           v
                        Verifier

Verifier 的输入有两个直接来源。

它不应该只能通过一条 trace_ref 表达完整的因果关系。

可以让每条消息具有单一的直接因果引用，同时由执行事件记录完整的多父节点关系。

这是比较低成本、也比较容易扩展的方案。

7.2 turn 不能同时代表所有执行层级

你目前将 turn 定义为 Runtime 维护。

需要明确的是：

* Planner 第三次调用。
* 当前 Episode 的第三轮。
* 第三次工具调用。
* 分支任务恢复后的第三轮。

这些不是同一个计数器。

建议在内部统一采用：

turn_id / turn_index

表示逻辑决策轮次。

具体的 LLM 调用次数、工具调用次数和事件序号由执行记录独立统计。

不需要现在就把所有计数器放进 AgentMessage，但不能让同一个 turn 字段在不同模块中拥有不同含义。

7.3 payload: Dict[str, Any] 不应该是最终验证边界

你现在设计了：

payload: Dict[str, Any]

并在每种 kind 上进行 schema 校验。

这是可以接受的。

不过，我建议在实现层面明确：

AgentMessage envelope validation
          +
MessageKind payload validation

两层校验分别负责：

* 信封字段的结构和合法性。
* 不同消息类型的业务合同。

例如：

plan_step
  ├── next
  ├── args
  ├── sub_goal
  └── done
tool_result
  ├── output
  ├── ok
  └── evidence_type
verify
  ├── ok
  ├── reason
  ├── step_conclusion
  └── slot_updates

对于 plan_step，还应检查 next 与 done 之间的约束。

例如，若 done=True，是否允许同时指定工具？如果允许，这表示什么？

这种跨字段约束不能只靠 JSON Schema 的基本类型验证解决。

另外，如果希望未来允许用户在画布中定义自定义 Agent，那么 AgentMessage 需要能够携带版本明确的自定义 Payload Schema。

否则，用户虽然可以定义新 Agent，却无法可靠地扩展通信协议。

7.4 消息持久化不能等同于把消息写入训练对话

你目前的链路是：

AgentMessage
    ↓
EpisodeRaw.messages
    ↓
_normalise_chat_messages()
    ↓
Trajectory.messages

这在工程上有一定合理性，但需要严格区分：

1. 执行事件。
2. 协议消息。
3. 训练对话。
4. 分支恢复上下文。

它们不是同一种数据。

例如：

route_decision
MEMORY_WRITE
SNAPSHOT

这些记录对审计和训练分析很有价值，但不一定应该直接进入 LLM 的对话序列。

反过来，训练所需的某些信息，也可能不能简单从原始消息恢复。

建议明确建立四个数据视图：

数据视图	用途	典型内容
Raw Protocol	协议审计	原始 AgentMessage
Execution Events	执行追踪	Tool Call、Memory Write、Error
Chat Trajectory	模型训练	OpenAI 四角色消息
Branch Snapshot	分支恢复	对话前缀、运行状态、必要元数据

这些视图可以从同一份事件流派生，不要求建立四套数据库。

最重要的是，不要让 Trajectory.messages 变成既用于训练、又用于审计、还用于分支恢复的万能容器。

⸻

八、Memory：这是本次重构的核心，也是最容易发生概念混淆的地方

你提出：

MemoryStore
FactLedger
Archive

分别负责：

* 进程内缓冲。
* 事实提交门控。
* 持久事件和快照。

这个方向是对的。

但是，我认为当前设计仍然需要做一次更严格的语义区分。

8.1 先回答一个关键问题：什么才是真正的 Memory？

在 MAS 系统中，至少存在四种经常被混称为 Memory 的对象。

类型	含义	生命周期
Working Context	当前 Agent 推理所需的上下文	当前窗口或 Episode
Agent Memory	某个 Agent 积累的历史信息	Episode 或跨 Episode
Shared Knowledge	多个 Agent 可以共同访问的事实	可跨 Agent、跨 Episode
Execution History	系统过去发生了什么	跨 Episode、可持久化

你的设计实际上已经覆盖了这些概念，但使用了不同的实现方式。

其中：

* MemoryStore 管理部分 Working Context 和 Agent Memory。
* FactLedger 管理 Shared Knowledge 的提交资格。
* Archive 管理 Execution History。
* resume_messages 用于恢复特定分支的对话前缀。

因此，最重要的不是再增加一个 MemoryManager，而是明确这些组件之间的职责边界。

建议采用以下正式定义

MemoryStore：管理当前执行上下文中的可读写记忆。

它决定 Agent 在运行时能够读取什么、写入什么，以及这些内容何时失效。

FactLedger：管理经过验证、符合提交条件的事实记录。

它不负责让事实自动成为所有 Agent 的上下文，而是负责事实是否有资格进入共享知识区域。

Archive：记录系统实际发生了什么，并提供可审计的持久化与恢复能力。

Archive 可以存储 MemoryStore 和 FactLedger 的操作事件，但不应因此自动成为 Agent 可读取的共享记忆。

这是你整个记忆体系中最重要的边界。

⸻

8.2 MemoryStore：结构合理，但默认关闭与三层记忆的说法需要调整

你现在的设计为：

_agent[agent_id]
_agent["hub"]
_system[]

这是一种简单而有效的进程内数据结构。

但我认为，将它称为“三层记忆”并不完全准确。

因为：

_agent["hub"]

本质上仍然是 _agent 字典中的一个特殊键。

而 _system 则是一个单独的列表。

从存储结构上看，它们并不完全对称。

建议在文档中使用：

三种记忆作用域，而不是严格意义上的三层记忆架构。

分别为：

agent
shared
system

这不会改变任何代码，但能够避免未来开发者误以为它们具有严格的层级继承关系。

还需要明确四个问题

第一，作用域权限。

哪些 Agent 可以读取 shared？

所有 Agent 默认都可以写入吗？

如果是，那么一个不可信的用户自定义 Agent 就可能污染共享上下文。

建议：

* 默认不允许所有 Agent 写入共享记忆。
* 根据执行策略决定哪些节点可以写入。
* 对共享写入保留来源、时间和关联事件。

第二，容量限制。

当前：

profile.memory = {
    "policy": "append_latest",
    "max_items": N
}

只解决了条目数量限制，没有解决 Token 预算。

如果每条记忆长度差异很大，限制条目数不等于限制上下文成本。

短期可以继续使用 max_items，但建议在接口中区分：

max_items
max_tokens

后者可以在未来需要时实现，不必立刻加入执行逻辑。

第三，记忆覆盖规则。

当两个 Agent 对同一事实写入不同内容时，系统是：

* 追加两条记录？
* 覆盖旧记录？
* 保留冲突并交由 Verifier 解决？

我建议默认采用追加式记录，不要静默覆盖冲突事实。

第四，Episode 边界。

必须明确 MemoryStore 在何时创建、何时销毁，以及失败退出时是否清理。

你当前的训练路径已经明确每个 rollout 创建新的 MemoryStore。

这一点应当继续保留。

⸻

8.3 FactLedger：应该保留，但不能让它成为另一个隐式事件系统

你新增：

class FactLedger:
    @staticmethod
    def classify(
        verify_payload: dict,
        tool_result_payload: dict
    ) -> str:
        ...

并采用：

COMPLETE + DIRECT + slot filled
    → fact_crit
COMPLETE + ABSENCE / EMPTY / ERROR
    → rejected
INCOMPLETE
    → rejected

这套门控逻辑具有清晰的研究动机。

它可以避免将不完整结果、空结果或者工具错误错误地提交为可靠事实。

但我建议进一步明确它的输出语义。

目前：

fact_crit
window_only
rejected

这三个名称并不完全平行。

例如：

* fact_crit 看起来像一个事实分类或重要性等级。
* window_only 表示保存位置。
* rejected 表示提交决策。

建议改成明确的内部决策结构：

COMMIT_FACT
KEEP_WINDOW_ONLY
REJECT

或者使用对应的 Enum。

不过，这里要区分两个层面：

内部实现可以更严格，对外兼容性不应该因此被破坏。

如果已有测试依赖：

protocol.commit_to_fact(...)

就保留原签名。

如果已有 Archive 消费者依赖：

{
  "phase": "fact_commit",
  "turn": 3,
  "ok": true
}

就继续保留这些字段。

不要在本次结构重构中同时修改事件协议。

还需要注意：DIRECT 不等于事实必然正确

evidence_type=DIRECT 表示证据类型，而不自动证明证据正确、无冲突或可信。

因此，FactLedger 的职责应当是：

根据已定义的验证结果和证据合同，判断某个结果是否满足事实提交条件。

它不应该在没有额外依据的情况下，将提交成功等同于事实被证明为真。

建议在文档中把：

live fact

定义为：

通过当前提交门控的事实记录

而不是系统已经保证绝对正确的知识。

⸻

8.4 最重要的问题：FactLedger 提交后，其他 Agent 能否读到这个事实？

这是当前设计中最值得进一步明确的地方。

你写道：

事实区即 Archive 事件流中带 phase: fact_commit 标记的 MEMORY_WRITE 事件，不是独立存储。

这个设计避免了第二份真相源，是合理的。

但需要注意：

事件记录不自动等于可检索的共享记忆。

例如：

ToolAgent A
    ↓
Verifier
    ↓
FactLedger 提交事实
    ↓
Archive 记录 MEMORY_WRITE

下一轮 Planner 或 ToolAgent B 能否读取这个事实？

如果它们只能读取：

MemoryStore._agent["hub"]

而 FactLedger 只向 Archive 写入事件，那么事实可能已经被记录，却没有进入下一轮 Agent 的可读上下文。

这会产生一个重要的架构缺口：

事实已经提交，但没有定义事实如何进入共享知识的读取路径。

这不意味着你一定需要新建独立数据库。

你可以采用两种方案。

方案 A：当前阶段的最小实现

继续将 Archive 作为持久化记录的权威来源。

FactLedger 只负责门控与提交事件。

如果当前产品不需要跨窗口读取已提交事实，就明确将其定位为：

Auditable Fact Commit

而不是完整的 Shared Knowledge Store。

这种方案最符合你当前的低成本重构目标。

方案 B：未来需要真正的跨 Agent 共享知识时

增加一个可选的事实读取接口：

FactLedger
    ├── classify()
    ├── commit()
    └── query()

其持久化记录仍然可以以 Archive 为权威来源。

query() 可以通过索引或者按需扫描实现，而不一定需要另一份独立的真相源。

但需要定义：

* 事实如何被检索。
* 如何区分不同 Episode。
* 如何处理过期事实。
* 如何处理冲突事实。
* 哪些 Agent 可以读取哪些事实。
* 如何防止一个 Episode 的信息污染另一个独立训练样本。

我的建议是当前先实施方案 A，把 query() 留作明确的后续扩展点。

不要因为未来可能需要共享知识，就在当前重构中提前实现一个通用知识库。

⸻

8.5 Archive：整体合理，但事件流、快照和恢复必须进一步解耦

你目前的 Archive 同时负责：

1. 事件流追加。
2. 快照持久化。
3. Rollout 索引。
4. 历史目录回退。
5. 恢复消息加载。

对于当前规模，这样的实现并非不合理。

但需要防止它继续演变为整个 MAS 系统的万能存储层。

建议在内部逻辑上区分：

Archive
├── Event Writer
├── Snapshot Store
├── Rollout Index
└── Resume Resolver

这不意味着立刻拆成四个文件。

可以继续留在：

mas/workflow/memory/archive.py

中。

但是，每部分应当有独立的接口和测试。

特别需要补充的三个问题

第一，事件追加的原子性。

如果多个并行 ToolAgent 同时写入同一个 events.jsonl，如何保证事件不会交错或损坏？

需要明确写入是：

* 由单一 Runtime 串行化。
* 由进程级锁保护。
* 还是通过独立事件队列写入。

在当前以单进程执行为主的情况下，最简单的方案通常足够。

但不要假设 ThreadPoolExecutor 中的并行任务不会引发并发写入问题。

第二，快照与索引一致性。

假设：

1. 快照写入成功。
2. 更新 rollout_index.json 失败。

恢复逻辑应该如何找到最新快照？

反过来，如果索引已经更新，但快照文件尚未完整写入，又怎么办？

建议采用：

write temporary snapshot
        ↓
atomic rename
        ↓
update index

恢复时不能完全相信索引，应当在索引失效时扫描快照目录。

第三，恢复快照必须具有版本。

建议快照至少携带：

{
  "schema_version": 1,
  "rollout_id": "...",
  "snapshot_id": "...",
  "messages": [],
  "meta": {}
}

未来如果调整消息协议或 Trajectory Schema，就能够识别旧快照。

这比现在增加复杂的迁移框架更重要。

⸻

九、planner_ctx: latest：建议保留，但需要限制它的使用语义

你计划：

class MemorySpec(BaseModel):
    agent: str = "messages"
    system: str = "none"
    planner_ctx: Literal["none", "latest"] = "none"

并在中心化执行中实现：

episode 开始
    ↓
读取最近一条 shared memory
    ↓
拼入 Planner 输入
    ↓
执行 Episode
    ↓
保存 final_answer

这个设计作为体验路径的最小实验是可以接受的。

尤其是你把默认值设为：

planner_ctx = "none"

有利于保证现有行为不变。

但这里有一个容易被忽略的问题：

跨 Episode 的 Planner Memory 可能导致评估污染和训练数据泄漏。

例如：

Episode A：问题与答案
        ↓
MemoryStore
        ↓
Episode B：读取 Episode A 的最终答案

如果 A、B 原本应该独立执行，那么 B 的结果就不再是独立样本。

这对于普通产品体验可能是合理的，对于算法比较、基准测试和 Agent RL 则可能是严重问题。

因此，我建议：

* planner_ctx: latest 只在显式启用的体验或实验配置中使用。
* 默认禁止它进入正式基准评估。
* 对使用该配置的实验，在元数据中记录记忆模式。
* 确保同一评估批次中的样本不会无意间共享历史答案。
* 如果未来需要跨 Episode 共享记忆，增加明确的 Session 或 Memory Namespace。

最后一点尤其重要。

你现在的 task_id、Episode 生命周期和 MemoryStore 生命周期并不完全等价。

如果未来一个用户 Session 包含多个任务，就不能简单地把 Session Memory 当作单个 Episode Memory。

当前不需要实现 Session Memory，但应该避免将两者的概念混为一谈。

⸻

十、Prompt 管理：单入口解析是正确的，但不要追求所有 Prompt 物理集中

你目前保留了六处 Prompt 存储，并新增：

mas/workflow/prompts.py

作为解析入口。

我支持这个方向。

Prompt 的存储位置并不一定需要统一。

因为不同 Prompt 的生命周期和用途不同：

Prompt 类型	适合的归属
训练热路径 ReAct Prompt	tir_agent.py
默认 MAS Planner Prompt	默认 Spec 或默认常量
用户自定义 Agent Prompt	实验 YAML
画布模板 Prompt	模板 YAML
StructAgent 角色 Prompt	Legacy 研究栈
user_space 包装窗口 Prompt	Gateway

把所有 Prompt 强行迁入一个目录，可能只是将物理分散变成逻辑耦合。

但你当前的设计还需要解决三个问题。

10.1 Prompt 优先级必须统一

你定义：

planner_node.system_prompt
    >
spec.hub.system_prompt
    >
DEFAULT_PLANNER_PROMPT

这个优先级应该成为唯一的正式规则。

但是，还需要明确：

* 空字符串是否表示未配置？
* 纯空白字符串是否视为未配置？
* 显式配置为空是否允许覆盖默认 Prompt？
* 模板 Prompt 与实验 Prompt 的关系是什么？

我建议默认规则为：

None / 缺失 / 空白字符串
    → 继续查找下一级默认值
非空字符串
    → 使用该值

如果确实需要显式禁用 Prompt，应该使用独立的配置语义，而不是依靠空字符串隐式表达。

10.2 Prompt 应当有可审计的版本信息

对于 Agent RL，Prompt 变化会直接影响 rollout。

因此建议在运行元数据中保存：

prompt_source
prompt_hash
resolved_prompt_version

这里不一定需要复杂的 Prompt Registry。

只要能够回答：

这条 Trajectory 实际使用的是哪个 Prompt？

就已经具有很大价值。

10.3 候选菜单属于运行时上下文，不属于 System Prompt

你目前已经将：

candidate_menu(...)

与：

planner_prompt(...)

分开。

建议继续保持。

原因是：

* System Prompt 负责定义 Agent 的稳定角色和行为约束。
* Candidate Menu 描述当前运行时可用的节点。
* Verifier Feedback 描述本轮执行结果。

三者应当在输入构造阶段组合，而不应该被当成同一种 Prompt 配置。

这样既有利于测试，也更容易分析行为差异。

⸻

十一、三条运行时：当前的边界描述是诚实的，但还没有形成统一的执行抽象

这是整份设计稿中最重要的系统级问题。

你明确列出了：

路径	执行方式
Collect / 批量测试	centralized_runtime
单题调试	centralized_runtime
训练	TirAgent + BranchSite
user_space	centralized_runtime + gateway

你也明确承认：

画布 PEV 图在 Collect/eval 执行，训练热路径不原样重放。

这是非常重要的架构事实。

我认为，你现在不应该为了追求统一，立即将训练路径重构成完整的多 Agent Runtime。

但必须清楚认识到：

目前 MAS 画布描述的系统，与 Agent RL 实际优化的策略，并不是完全相同的对象。

这会带来三个后果。

后果一：用户可能误以为画布就是训练结构

如果 UI 允许用户定义：

Planner → ToolAgent → Verifier

但训练时实际执行的是：

TirAgent + BranchSite

那么用户可能会认为自己训练的是完整的 PEV 工作流。

实际上，训练可能只是在单 Agent 轨迹上进行分支采样。

这不是简单的文档问题，而是产品语义问题。

后果二：评估收益不一定能够归因到某个 Agent

假设 Collect 中：

Planner → Search Agent → Verifier

表现很好。

但训练时优化的是单个 TirAgent。

此时，训练前后的收益变化并不能简单解释为整个 MAS 工作流的改进。

后果三：不同 Agent 的策略边界尚未与训练边界统一

如果未来你想训练多个可优化 Agent，就必须明确：

* 哪些消息属于哪个策略。
* 哪些 Trajectory 片段用于更新哪个策略。
* 奖励如何归属到不同 Agent。
* 不同 Agent 的采样如何组织成 GRPO 组。
* 不同分支是否允许共享历史信息。

这些问题目前还没有通过统一的执行协议解决。

我的建议：当前先明确两种 Execution Mode

不要马上重构训练核心。

而是在产品语义上明确：

Execution Mode A: Workflow Execution

用于：

* Collect。
* 单题调试。
* 批量评估。
* 用户自定义 MAS。

以及：

Execution Mode B: Trainable Policy Rollout

用于：

* TirAgent。
* BranchSite。
* ARPO 等分支采样。
* RL 训练。

两种模式可以共享：

* Trajectory 合同。
* Archive。
* Tool 接口。
* Harness。
* Reward 接口。

但不应该声称它们已经执行同一个计算图。

未来如果要统一，可以考虑引入更明确的策略绑定：

AgentNode
    ↓
PolicyBinding
    ↓
Rollout Backend

但我不建议在本次小成本重构中实现这一层。

当前最重要的是确保用户知道自己定义了什么、评估了什么、训练了什么。

⸻

十二、分支采样：整体合理，但还缺少状态一致性与实验可复现性约束

你当前的采样设计：

第一波独立采样
    ↓
ActiveSetSession.plan_forks_from_raw
    ↓
根据 gate 决定是否分支
    ↓
Archive Snapshot
    ↓
第二波分支任务
    ↓
补齐 group size

从模块分工来看，这套流程是合理的。

尤其是你已经明确：

MemoryStore 不参与跨 runner 恢复，分支恢复只能使用 Archive 快照。

这一条应当保留。

但需要进一步区分三个概念。

12.1 分支快照不是只有 messages

你目前将：

resume_messages

作为分支恢复介质。

这对于当前的单 Agent 对话续写是可行的。

但如果未来的 MAS 分支涉及：

* Router 调用计数器。
* 随机数状态。
* 已完成的 Tool Call。
* Agent 私有 Memory。
* 已提交事实。
* 当前执行图状态。

那么只恢复对话消息可能不足以恢复原有执行状态。

因此建议明确：

当前的分支快照只保证对话前缀恢复，不保证完整 MAS Runtime 状态恢复。

如果后续扩展到多 Agent 分支采样，再引入完整状态快照。

不要在当前阶段为了未来需求实现一个过度复杂的状态序列化框架。

12.2 必须区分前缀一致性与环境一致性

即使两个分支拥有完全相同的：

resume_messages

它们后续的外部工具输出也可能不同。

例如，实时搜索工具可能返回不同内容。

所以：

相同消息前缀

不等于：

相同环境状态

建议在训练元数据中记录：

* 分支父快照。
* 恢复的消息数量。
* 工具结果是否使用缓存。
* 工具后端版本。
* 恢复模式。

这样才有条件判断分支间的性能差异究竟来自策略，还是来自环境变化。

12.3 补齐组大小不能掩盖采样失败

你当前的设计会在第二波补齐组大小。

这对保持 GRPO 训练形状有帮助。

但要明确：

有效采样数量

和：

计划采样数量

并不相同。

假设计划采样 8 条轨迹，但只有 5 条有效。

不能简单地将补齐后的 8 条都视为 8 个独立、有效的策略样本。

建议将以下指标写入采样统计：

requested_group_size
valid_rollout_count
failed_rollout_count
expanded_rollout_count
duplicate_rollout_count

如果系统采用重试、复制或其他补齐策略，还应记录具体原因。

尤其是当多个轨迹最终变成相同结果时，需要防止训练系统错误地将其视为独立探索结果。

⸻

十三、contracts.py：需要防止成为所有模块的公共垃圾场

当前：

mas/workflow/contracts.py

包含：

* Trajectory
* ExecutionEvent
* MemoryItem
* BranchSite
* RolloutTree

从低成本重构角度看，这样做可以接受。

但这些对象实际上属于不同的概念领域。

例如：

* Trajectory 属于执行与训练数据合同。
* MemoryItem 属于记忆。
* BranchSite 属于分支采样。
* ExecutionEvent 属于执行审计。
* RolloutTree 属于分支采样结果。

随着功能增加，这个文件可能逐渐变成一个无法独立演化的共享模块。

当前阶段的建议

不必立刻拆文件，但要明确它的职责：

contracts.py 只存放跨模块共享的数据合同，不存放执行逻辑、存储逻辑或策略实现。

同时建议为关键合同增加：

schema_version

或者在对应的外层记录中保存版本信息。

尤其是：

* Trajectory。
* AgentMessage。
* Archive Snapshot。
* RolloutTree。

不一定每个 Pydantic 对象都要增加版本字段。

但持久化到磁盘、需要跨版本读取的对象，应当具有明确的兼容性策略。

⸻

十四、删除 shim：方向正确，但必须先完成依赖清点

你准备删除：

mas/algos/
workflow/rewards.py
workflow/train_signal.py

并让所有引用直接指向：

rl/

我支持这个方向。

因为如果一个模块只有：

from rl.rewards import ...

却又在 workflow/ 中提供相同名称的入口，就会出现两套看似独立、实际相同的导入路径。

长期来看，这会增加维护成本。

但需要注意：

删除 shim 的前提是确认它们不再承担隐式兼容职责。

建议在删除前完成以下检查。

第一，静态引用检查

覆盖：

* Python 文件。
* 测试文件。
* Shell 脚本。
* YAML 配置。
* 动态 import。
* CLI 启动参数。
* 文档中的实际命令。

第二，运行时导入检查

特别注意：

importlib.import_module(...)

和：

sys.path.insert(...)

这类动态导入方式。

普通 grep 可能无法完整发现运行时依赖。

第三，训练入口检查

训练热路径中：

tir_agent.py
lit_tir_agent.py
train_tir_agent.py

必须确认仍然能够使用唯一的 rl.* 实现。

第四，公共接口检查

如果外部用户可能已经使用：

from workflow.rewards import ...

那么删除它虽然不会影响内部测试，却可能破坏开源 API。

你需要明确项目当前处于什么阶段：

* 如果尚未正式发布稳定 API，可以直接删除并在更新说明中记录。
* 如果已有外部使用者，应当考虑一个过渡版本。

这不意味着必须永久保留 shim，而是要有明确的兼容性策略。

⸻

十五、运行时产物迁移：方向正确，但路径切换需要防止双写和旧数据误读

你准备将：

mas/.tir_archives/
mas/.local_expansion/

迁移到：

artifacts/mas/tir_archives/
artifacts/mas/local_expansion/

我赞成。

源码与运行时产物分离，是基础设施项目应该尽早完成的工作。

但是，迁移过程中有两个常见问题。

15.1 读旧目录不等于可以自动混合两个目录

建议明确：

新目录：唯一写入位置
旧目录：只读兼容来源

并确保：

* 新运行不会继续写入旧目录。
* 不会因为新目录为空就覆盖旧目录。
* 不会因为两个目录同时存在而误选旧快照。
* 同一 rollout 在两个目录中出现时，有明确的优先级。

建议固定：

显式 TIR_ARCHIVE_DIR
    >
新默认目录
    >
旧目录只读 fallback

但要注意，旧目录回退的触发条件也必须明确。

例如，不能因为新目录存在就忽略其中缺失的历史 rollout。

15.2 Archive 路径不能只在一个函数里修改

你已经提出：

archive_root()

作为统一入口。

这是正确的。

但还需要确认以下路径是否全部使用同一个解析函数：

* ExecutionService。
* dump_resume_with_archive。
* Collector。
* Resume Resolver。
* 采样树相关接口。
* Monitor。
* 旧数据展示接口。

建议在重构前先搜索所有写入点和读取点，形成一份路径依赖清单。

这比直接移动目录更重要。

⸻

十六、测试体系：当前回归门必要，但还不足以验证架构正确性

你目前的验证策略：

./run.sh feature-test functional workflow user-space
./run.sh smoke

并在最后增加：

feature-test --all
smoke
GPU arpo-train-test --steps 3

这套策略有一个优点：

每个阶段都有独立的回归门。

但是，现有测试可能更多关注：

重构前后，原有功能是否还能工作？

而你的重构还涉及：

* 模块边界。
* 持久化路径。
* 消息协议。
* 记忆提交语义。
* 并行执行。
* 分支恢复。
* Legacy 隔离。

这些问题需要专门的架构测试。

16.1 建议增加五类测试

测试类型	核心目标	示例
Import Contract Tests	保证公共导入路径稳定	from workflow.memory import MemoryStore
Protocol Contract Tests	保证消息 Schema 与失败语义稳定	非法 plan_step 不得进入执行阶段
Persistence Tests	保证快照和事件流一致	索引损坏后仍能扫描恢复
Runtime Semantics Tests	保证执行语义正确	并行工具部分失败仍能记录全部结果
Isolation Tests	保证不同执行之间不发生信息污染	独立 Episode 不共享 MemoryStore

此外，还应增加：

* 旧 Archive 目录回退测试。
* FactLedger 判定矩阵测试。
* Legacy 缺失可选依赖测试。
* Router 并行结果关联测试。
* 分支恢复后的消息前缀一致性测试。

16.2 byte-level 测试很有价值，但不能过度依赖它

你要求：

planner_ctx: none

时消息序列与现状 byte-level 等价。

这对于证明重构没有改变默认行为非常有价值。

但它只覆盖特定输入和执行路径。

有些变化虽然改变字节序列，却不改变实际语义；反过来，有些行为变化可能没有被固定测试样本捕捉。

因此，我建议：

* 对严格兼容的序列化接口保留 byte-level 测试。
* 对运行时语义使用结构化断言。
* 对持久化和恢复使用端到端不变量测试。

这比所有测试都采用相同的比较方式更稳健。

⸻

十七、我建议额外增加的架构守卫

这是当前设计稿没有充分展开、但对长期维护非常重要的一项工作。

你已经有：

check_workflow_deps

用于约束部分依赖。

建议在此基础上逐渐建立以下架构守卫。

守卫一：MAS 主栈不能隐式依赖 Legacy 训练入口

允许：

ToolAgent → 可选 Legacy Tool Backend

不允许：

centralized_runtime → legacy/train_mas_agent.py

守卫二：Workflow 不应该反向依赖 UI

允许：

webui → science_infra → workflow

不允许：

workflow → webui

守卫三：MemoryStore 不应该依赖 Runtime

允许：

runtime → MemoryStore

不允许：

MemoryStore → centralized_runtime

Archive 的事件写入可以通过 Runtime 注入的接口实现，或者由 Runtime 显式调用，而不是让 MemoryStore 自己去寻找全局 Runtime。

守卫四：RL 算法应当通过稳定合同消费轨迹

允许：

RL Adapter → Trajectory

不建议：

RL Algorithm → centralized_runtime 内部私有状态

守卫五：持久化事件不能直接反向驱动任意业务逻辑

例如，不能让某个模块通过检查：

events.jsonl 中是否出现字符串 "fact_commit"

来决定是否提交事实。

事件应该通过明确的接口和结构化合同被消费，而不是通过对持久化文件的临时解析实现隐式耦合。

这些守卫可以从简单的 AST 检查、导入检查和 grep 检查开始，不需要一开始就引入复杂的架构治理框架。

⸻

十八、针对当前设计稿的逐项决策

下面给出我建议的最终处理意见。

项目	当前方案	我的判断	建议
顶层目录划分	mas/rl/science_infra/legacy/artifacts	保留	不需要增加 mas/engine/
workflow/ 平铺结构	保留大部分文件	保留	以模块职责和架构守卫控制增长
Legacy StructAgent 迁移	移至 legacy/structagent/	必须做	补充同进程导入和缺失依赖测试
AgentNodeSpec	四类节点	短期保留	明确 Role、Execution Type 和 Backend 的概念区别
trainable	Boolean	暂时保留	明确它不等于实际梯度更新
tools	保存可路由 Agent ID	暂时保留	文档中明确其不是原子工具注册表
Router	四种策略	需要修订	处理 llm_choice 重叠、并行部分失败和评分标准
AgentMessage	统一消息信封	保留并加强	明确关联、版本和并行执行语义
Payload Schema	按消息类型校验	必须做	增加跨字段业务约束
MemoryStore	三种作用域	保留	规范命名、权限和生命周期
FactLedger	新增	保留	先做门控，不急于实现通用知识库
Archive	事件、快照、恢复	保留	增加原子写入、版本和恢复一致性测试
planner_ctx	默认关闭	保留	体验与正式评估严格区分
Prompt 单入口	prompts.py	必须做	保留物理分散，统一解析规则和 Prompt Hash
三条运行时	继续并存	保留	明确 Workflow Execution 与 Trainable Policy Rollout 的差别
分支采样	两波展开	保留	记录有效样本数和环境恢复语义
contracts.py	多类合同集中	暂时保留	只存数据合同，不存执行逻辑
删除 shim	改为直引 rl.*	必须做	先检查外部 API 兼容性
Archive 目录迁移	移至 artifacts/mas/	必须做	统一路径解析和旧目录只读回退
checkpoints 等目录	暂时不动	正确	查明所有读写方后再处理
测试策略	分阶段回归	需要加强	增加协议、持久化、隔离和并发语义测试

⸻

十九、实施顺序：我建议调整现有六阶段计划

你原来的实施顺序基本合理，但我建议将部分设计验证提前，并把目录迁移放在核心接口稳定之后。

Phase 0：冻结架构合同

这一阶段不移动文件。

先完成以下工作：

1. 明确 Agent、Router、ToolAgent 的职责边界。
2. 明确 AgentMessage 的关联与校验语义。
3. 明确 MemoryStore、FactLedger、Archive 的边界。
4. 明确训练路径与 Workflow 路径的差别。
5. 列出所有外部兼容接口。
6. 建立现有测试基线。

这一阶段的产出不是更多代码，而是一份明确的接口约束清单。

如果这一步不做，后续每次移动文件时都可能顺便修改业务逻辑，导致问题难以定位。

Phase 1：隔离 Legacy StructAgent

按原计划移动：

mas/MAS_structagent/
    ↓
legacy/structagent/epc_aw/

同时迁移：

mas/mas_agent.py
mas/train_mas_agent.py

重点测试：

* 旧训练入口。
* 产品主栈导入。
* 同进程导入。
* 缺失可选依赖时的行为。

这一步应该保持纯结构性变化。

Phase 2：重构 Memory 子包

实施：

workflow/memory.py
    ↓
workflow/memory/store.py

以及：

workflow/archive.py
    ↓
workflow/memory/archive.py

保留兼容 shim。

随后新增 FactLedger。

但是，我建议把 planner_ctx: latest 放到 Phase 2 之后，作为独立的功能变更。

原因是：

Memory 子包迁移和跨 Episode 记忆接通是两个不同的变更。

如果同时实施，一旦测试失败，很难快速判断究竟是 import 路径改变，还是记忆语义改变。

Phase 3：删除 Shim

先完成引用清点，再删除：

mas/algos/
workflow/rewards.py
workflow/train_signal.py

重点检查动态 import、脚本入口和外部兼容性。

Phase 4：Prompt 收口

新增：

workflow/prompts.py

将 Planner、Verifier 和候选菜单的解析逻辑统一起来。

训练热路径的 Prompt 暂时保持原样。

新增 Prompt Hash 和优先级测试。

Phase 5：产物迁移

迁移：

mas/.tir_archives/
    ↓
artifacts/mas/tir_archives/

以及：

mas/.local_expansion/
    ↓
artifacts/mas/local_expansion/

这一步应当重点验证：

* 新目录写入。
* 旧目录读取。
* 索引损坏恢复。
* 历史 rollout 展示。
* 分支恢复。
* 采样树 API。

Phase 6：可选记忆接通与完整回归

最后再启用：

memory:
  planner_ctx: latest

并执行：

* Feature Tests。
* Workflow Tests。
* User-space Tests。
* Smoke Tests。
* Archive 恢复测试。
* GPU ARPO 短训练测试。

如果训练结果需要与历史基线比较，建议固定相同的数据、模型配置和采样设置，并核对有效 rollout 数量，而不是只比较最终训练是否能够完成。

⸻

二十、最终建议：这轮重构应该达到什么标准？

我建议你不要把这轮重构的目标定义为：

目录更漂亮、代码更整齐、Memory 变成了一个文件夹。

而应该定义为：

在不改变既有训练行为的前提下，建立清晰、可审计、可扩展的 MAS 运行时边界。

具体而言，重构完成后应满足以下标准。

标准一：概念有唯一归属

开发者能够快速回答：

* Agent 节点在哪里定义？
* Agent 实例在哪里创建？
* 路由逻辑在哪里实现？
* 消息合同在哪里定义？
* 事实提交逻辑在哪里实现？
* 持久化事件在哪里记录？
* 分支恢复在哪里实现？
* Reward 和 TrainSignal 从哪里导入？

不能出现多个模块分别实现同一套规则。

标准二：执行过程可审计

给定一个失败的 Episode，能够恢复：

Planner 产生了什么计划
    ↓
Router 选择了哪些节点
    ↓
工具实际执行了什么
    ↓
Verifier 为什么接受或拒绝
    ↓
哪些事实被提交
    ↓
为什么系统继续、重试或终止

并且，相关记录能够被 Harness 消费，而不需要解析各模块的私有内部状态。

标准三：训练和评估的语义明确

用户能够清楚地区分：

* 评估的是哪个 MAS 工作流。
* 训练的是哪个策略。
* 分支采样使用什么恢复机制。
* 轨迹中的消息是否与真实执行过程一致。
* 训练配置是否引入了跨样本记忆。

标准四：默认行为可证明不变

对于纯结构性重构：

* 不改变模型调用顺序。
* 不改变默认 Prompt。
* 不改变路由策略。
* 不改变奖励计算。
* 不改变分支采样规则。
* 不改变训练热路径的数据合同。

所有有意引入的行为变化，都应该作为独立变更实施。

标准五：新能力不要求重写核心 Runtime

未来如果你要增加：

* 新的 ToolAgent。
* 新的 Router 策略。
* 新的 Agent Memory 策略。
* 新的分支采样算法。
* 新的用户自定义 Agent。
* 新的 RL 算法。

应该尽量通过已有接口或新增适配器实现，而不是继续向 centralized_runtime.py 中添加大量条件分支。

⸻

二十一、我的最终判断

如果按照基础设施架构评审的标准，我会给这份设计稿如下评价。

维度	当前评价	主要原因
目录结构	良好	产品主栈、训练、Legacy、产物已经开始分离
职责划分	较好	核心概念已有明确归属，但少数边界仍然交叉
运行时设计	中上	中心化执行清晰，但与训练路径存在结构性断层
通信协议	中上	已有显式信封和 Schema，但并行因果关联仍需完善
记忆设计	较好	三类组件职责初步明确，事实提交后的读取路径尚未定义
训练集成	中上	分支采样与 Archive 的边界合理，但策略与工作流尚未统一
可维护性	良好	shim 清理、Prompt 收口、产物迁移均有明确收益
测试与兼容性	中上	分阶段回归很好，但缺少部分架构语义测试
长期扩展性	有潜力	主要取决于是否能避免 Runtime、Archive 和 contracts 持续膨胀

这些是基于设计文档的定性判断，不代表已经验证了代码实现。

我最推荐的决策是：保留整体重构方向，优先解决接口语义和运行时边界问题，然后按阶段实施，不要扩大本轮重构的范围。

其中有三项是我认为真正需要在实施前敲定的：

1. AgentMessage 的并行执行关联、Payload 校验和持久化语义。
2. MemoryStore、FactLedger 与 Archive 的职责边界，以及事实提交后是否需要可检索。
3. Workflow Execution 与 Trainable Policy Rollout 的产品语义差异。

另外三项可以在实施过程中逐步完善：

* Router 的失败处理和计数器恢复。
* Prompt Hash 与配置审计。
* Archive 快照的原子性与版本管理。

最后，关于你整个 MAS Infra 项目的定位，我想强调一点：

你不需要在这次重构中实现一个理论上完美的通用 MAS 框架。你需要建立一个足够稳定的核心，使未来增加新 Agent、新工具、新记忆策略、新分支采样算法和新 RL 算法时，不必反复重写现有基础设施。

这比目录是否完全符合某种经典软件架构模式更重要。