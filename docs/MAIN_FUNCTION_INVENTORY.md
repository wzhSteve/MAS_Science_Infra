# Main 核心功能清单

`main` 是后端、训练和运行行为的功能基线；`feature/ui` 是新 WebUI 的设计与实现基线。目标是将main的一些功能和ui，在新 WebUI 中同步对接迁移实现，就是新的Webui可以使用之前main的部分的所有功能, 完成合并
（采用本地修改通过git integration/new-webui 服务器同步查看测试）
| 核心功能 | `main` 功能入口 | 新 WebUI 适配目标 |
| --- | --- | --- |
| 实验与配置 | `science_infra/control/experiments.py` | 在实验首页和设置页读取、编辑并无损保存 |
| Workflow 0.3 | `mas/workflow/` | 新画布支持 Agent、Tool、Router、入口和完整字段 |
| GPU 与 RL 参数 | `science_infra/control/services.py` | 使用新设置表单配置 GPU、算法、profile 和 Hydra 参数 |
| 训练启动与停止 | `services.start_train()`、`mas/train_tir_agent.py` | 接入新训练按钮、状态、停止操作和错误展示 |
| RL 算法 | `rl/hooks/`、`rl/loss.py` | 支持 GRPO、ARPO、AEPO、IGPO、GIGPO 和 RAE |
| Sampling 与 Branch Sites | `mas/workflow/`、`rl/hooks/` | 在新画布和设置体系中重新实现配置交互 |
| Rollout Tree | `/api/mas/rollout-trees`、`rl/hooks/daemon.py` | 在新 UI 中重新建设树、节点详情和实时更新 |
| 运行日志与监控 | `/api/runs`、`/api/monitor`、`/api/events` | 接入新运行状态、日志、奖励曲线和 SSE |
| AGL Metrics | `/agl`、`/api/agl/health` | 接入新导航和训练状态区域 |
| 数据采集与诊断 | `/api/mas/collect`、`/api/harness/diagnose` | 在新调试、采集和 Harness 页面中接入 |

## 集成约束

- 不迁移或保留 `main` 的旧 UI 表现层。
- `main` 的后端、训练链路、Workflow schema 和运行行为不得回退。
- `main` 旧 UI 中的功能应在 `feature/ui` 的组件、样式和状态管理体系中重新实现。
- Workflow 读取后直接保存时，不得丢失 Router、Sampling、Branch Site 或 Agent 扩展字段。
- 最终训练必须继续通过 `mas/train_tir_agent.py` 和 `rl/hooks/` 执行。

## 当前实施顺序

服务器暂时不可用，先完成可在本地开发和验证的部分：

1. Sampling 与 Branch Sites 配置
2. 训练启动、停止及错误状态
3. RL 算法参数映射
4. Mock 数据采集与 Harness 诊断
5. Rollout Tree 页面
6. 运行日志、监控和 SSE
7. AGL 离线状态与导航入口

服务器恢复后再验证：

1. GPU 探测、选择及多卡配置
2. 真实训练启动与停止
3. 各 RL 算法、Branch Rollout 和 Rollout Tree
4. 实时日志、监控与 AGL Metrics
