# Agent RL 面试学习包（基于 math_gsm）

本目录把「跑通训练」升级为「面试能讲清」的材料。按 Week 1→4 顺序阅读；Week 3 对应可运行的 reward 消融代码。

**系统架构、模块逻辑图、全量参数与换 Agent/LLM 指南**以 [技术报告](../TECHNICAL_REPORT.md) 为准；LangChain / LangGraph 如何构图见 [LANGCHAIN_WORKFLOW.md](../LANGCHAIN_WORKFLOW.md)。本目录侧重口述稿、公式追问与 FAQ。超参默认值若有出入，以 `[train_math_agent.py](../../train_math_agent.py)` / 技术报告为准。


| 文档                                                                     | 对应计划   | 产出                       |
| ---------------------------------------------------------------------- | ------ | ------------------------ |
| [../TECHNICAL_REPORT.md](../TECHNICAL_REPORT.md)                       | —      | 技术报告：E2E 流、模块图、参数表、迁移指南  |
| [01_hyperparam_dictionary.md](01_hyperparam_dictionary.md)             | Week 1 | 超参词典：改大/改小会怎样            |
| [02_five_minute_walkthrough.md](02_five_minute_walkthrough.md)         | Week 1 | 5 分钟白板讲解稿                |
| [03_ppo_grpo_dpo.md](03_ppo_grpo_dpo.md)                               | Week 2 | PPO / GRPO / DPO 对比与公式直觉 |
| [04_algorithm_self_qa.md](04_algorithm_self_qa.md)                     | Week 2 | 算法自问自答（模拟追问）             |
| [05_experiment_reward_ablation.md](05_experiment_reward_ablation.md)   | Week 3 | Reward 消融实验协议与 takeaway  |
| [06_architecture_and_star_story.md](06_architecture_and_star_story.md) | Week 4 | 系统架构图 + 星形故事话术           |
| [07_interview_faq.md](07_interview_faq.md)                             | Week 4 | 高频题 2 分钟标准答              |
| [08_gpu_memory_oom_hyperparams.md](08_gpu_memory_oom_hyperparams.md)   | 工程实战   | 显存估算、OOM 决策树、超参实验影响      |
| [09_grpo_logit_backprop.md](09_grpo_logit_backprop.md)                 | 公式深挖   | GRPO 下 logit/logπ、长轨迹聚合、反传到 LLM |
| [10_agent_rl_arpo_aepo_igpo_gigpo.md](10_agent_rl_arpo_aepo_igpo_gigpo.md) | Agent RL | ARPO / AEPO / IGPO / GiGPO：rollout、reward、loss 对照与代码入口 |




## 代码入口（Week 3）

- Reward 变体：`math_agent.py` 中 `compute_reward(..., mode=...)`
- 消融脚本：`scripts/reward_ablation_dryrun.py`（不启 VERL，用轨迹文本离线打分）
- 训练时切换：环境变量 `MATH_REWARD_MODE=binary|shaped|format_only`



## 建议口述检查点

1. 能对着 config 解释 `rollout.n` / clip / KL / `n_runners`
2. 能手推 GRPO 组内 advantage（无 Critic）
3. 能讲一次完整 GRPO step（数据 → rollout → reward → adapter → update）
4. 能举出本项目里一种可能的 reward hacking
5. 能区分启动 OOM vs 训练 OOM，并口述 `gpu_memory_utilization` 按整卡而非剩余显存计算
6. 能用「补 rollout / 补 reward / 补 advantage」三层区分 ARPO、AEPO、IGPO、GiGPO，并说明本仓库仍是 vanilla GRPO、没有 pip 算法包可切

