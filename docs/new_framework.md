infra更新构思

UI层
和下面三层进行通过yaml文件来进行交互，完全脱离出来

MAS层（测试状态就是MAS跑任务，训练状态就是MAS要搜集rollout给RL层进行训练）
重新构思MAS层设计
并且MAS中不再出现tool，而是完全封装成了agent，例如本来可以调用wikipedia、google、web search、python coder、思考这五个工具。但是实际上他们中间也需要调用LLM来进行文本处理和分析
所以把他们全部都封装成agent，然后通过agent名字来挑选对应的agent。
例如planner-executor-verifier workflow中，executor就变成可选的tool agent集合了，而不是一个单纯的executor。

Planner->agent路由，根据planner的输出来选择 tool  agent->从tool agent集合中选择对应的agent->执行结束后输出
Memroy的话分为两层，MAS和agent层。默认MAS memory记录历史日志等。agent层都是空的，可由用户自己添加。
Agent之间的通信就可以非常简化，就是Agent的输入输出。然后每个agent对于输入

所以就是两类agent和agent路由器
一类是封装好的agent，比如planner、tool agent、verifier
一类是空白agent，可以自己定义profile
Agent路由器就是帮助在agent集合中挑选一个agent，比如tool agent，也可以是多个profile不同的思考agent，像多专家一样，但是他们功能都是一样的，比如打分评估等

训练状态：
branch rollout
￼

flowchart TB
  subgraph official [Official_ARPO_same_worker]
    g1[batch_generate_active_prefixes]
    t1[tools_concurrent_threadpool]
    w1[wait_all_active_ready]
    fork1[entropy_fork_copy_prefix]
    g1 --> t1 --> w1 --> fork1 --> g1
  end

  subgraph mas [MAS_current]
    w1a[wave1_Store_rollouts]
    ep[LitTirAgent_run_episode_serial]
    plan[ActiveSetSession_plan_only]
    d2[Daemon_enqueue_resume]
    w2a[wave2_separate_rollouts]
    w1a --> ep --> plan --> d2 --> w2a --> ep
  end

维度	官方 ARPO	当前 MAS
前缀形态
token list（同 vLLM worker）
messages（首次 tool 后 branch_messages）
分叉位置
tool 写回后、仍在 generate 循环内
父轨迹整段 episode 跑完后，Daemon/Runner 再开 sibling
谁跑续写
同一 Worker 把新 prefix append 进 active
新的 AGL Store rollout → 另一/同批 Runner 再跑图
ActiveSet
真·活跃前缀集合；EOS 就踢出
ActiveSetSession：对已完成 root 算要不要 fork、分几条；训练路径 execute_local=False，不在本 Runner 内跑 sibling
LLM batch
generate(所有 active prefixes)
每条轨迹自己 llm.invoke；跨轨迹靠 n_runners 并行，不是「ready 动态组 batch」
Tool
同批 active 可并发，再等齐进下一轮 generate
单条 episode 内 LLM→tool→LLM 串行；多条 rollout 之间并行，但无「同前缀组的 tool barrier」


token list->vllm batch，但是需要等待，也就是需要等待当前batch的agent执行（例如tool调用等）完之后，一起送到下个agent中并通过vllm batch一起处理

Vllm batch增加GPU消耗，并且中见会有等待时间。适合路径高度相似（也就是执行时间相近的MAS）

MAS也可以是token list，但是它是入池，然后rollout worker动态分配任务，GPU不会增加消耗，但是时间较长，因为一次只能执行rollout worker数量限制的branch rollout

我建议仍然基于ARPO的逻辑来，灵活性通过下面的设置来实现

我们将每个agent看成是有工作窗口的。其实rollout就是agent执行结束后，也就是要输入下一个agent前。所以在ui中可以从MAS workflow上通过挑选agent来选择branch rollout监测点

Branch rollout的监测点会根据指标来判断是否需要branch，比如ARPO的熵、APPO等方法

MAS层向RL层传递的可以表现为rollout树
对于不同的query，每个query都是一个树的root
对于给定query，如果有branch rollout，那就是树上开的分支路径
树上的叶子节点就是outcome，也就是对应从root到叶子节点的执行路径最后的输出
每个树上的节点包含状态，比如熵、比如是否执行成功
可视化不同query的rollout tree

RL层（训练状态）

Reward设计，分成两个级别
rollout level：树分支-level
credit assignment：节点-level

然后对应的reward从MAS层传下来的对应的状态节点来设计并计算reward
比如树分支-level的话，就是通过叶子节点的outcome来奖励完整的执行路径
节点-level的话，就是该节点状态来奖励，或者截取同branch上的多少个历史节点累积计算奖励（需要额外设计）
loss设计
RL层会向Harness层实时的传输累计的<rollout，reward，loss>状态，来让harness进行监控

Harness层
Harness分为1）作用在MAS层上的测试状态和2）在训练状态的时候作用在RL上。
要具有实时功能，先完成这两个功能
比如最简单的MAS上的错误检测，需要实时的接收MAS传递的rollout，然后进行错误归因
比如RL上监控是否出现了reward hacking，需要的是监控rollout tree上的节点状态
