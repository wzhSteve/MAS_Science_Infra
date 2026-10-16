---
name: experiment_config
description: 实验配置助手：解读与修改画布草稿（agent 结构、采样站点、算法切换），并校验可执行性
triggers: [画布, 配置, workflow, 采样站点, 算法, grpo, arpo, aepo, igpo, gigpo, rae, 改实验]
---
# 实验配置助手

## 何时用
用户想改画布上的 agent / router / 采样站点、切换训练算法、或者问当前实验为什么不可执行。

## 步骤
1. 调 `query_experiment(experiment_id)` 拿五份 YAML 的当前摘要
   （llm / workflow.agents / sampling / rl / harness）。
2. 改动走现有工具，不要凭空写文件：
   - 改用户区代码：`write_file`（仅 adapted/contracts/artifacts）；
   - 合入画布草稿：`apply_to_canvas`；之后用户需点顶栏「保存实验」。
   - 校验：`validate_project`。
3. 涉及资源预估时联动 training_resource 技能（query_gpu_env）。

## 配置要点
- Agent kind 只有 planner / tool / verifier / blank；entry_agent 指向入口。
- 采样站点写在 `workflow.sampling.sites`：
  planner `after_agent_turn`、router `after_agent_turn`、verifier `after_verifier`，
  resume_mode 用 `messages`。站点数量直接影响 rollout 显存（联动 training_resource）。
- 算法（VALID_ALGOS）：grpo / arpo / appo(=arpo 别名) / aepo / igpo / gigpo / rae / flow_grpo。
  workflow.sampling 存在时以它为准并同步进 rl.algorithm.tir_algo。
- 切算法时提醒核对 profile 与 group_n：分支算法（arpo/aepo/igpo/gigpo/rae）依赖站点
  `expand_in_runner`，preflight 会对每个站点跑 site_capability 校验。
- 多个可训练 agent 必须使用同一套训练权重，否则 preflight 会拒绝。

## 回答要求
- 解释「为什么」：字段语义 + preflight 校验逻辑，而不是只给 diff。
- 涉及写画布的动作，最后明确告知：草稿已更新，仍需用户点「保存实验」。
