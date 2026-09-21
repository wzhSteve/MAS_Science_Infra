# Main 核心功能清单

`main` 是后端、训练和运行行为的功能基线；`feature/ui` 是新 WebUI 的设计与实现基线。目标是讲main的一些功能ui，在新 WebUI 中同步对接实现

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
