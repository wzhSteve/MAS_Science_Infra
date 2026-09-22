# 合并迁移功能步骤

本地修改 → 经确认后提交推送 → 服务器拉取 → 按改动类型启动或重建 → 验证 GPU 和真实训练。

- **服务器项目目录**：`/root/autodl-tmp/MAS_Science_Infra`
- **分支**：在 `integration/new-webui` 进行迁移合并。

## 服务器操作

1. 在代码同步终端拉取已确认并推送的提交：

   ```bash
   cd /root/autodl-tmp/MAS_Science_Infra
   git pull --ff-only origin integration/new-webui
   ```

2. 根据改动类型启动服务：

   ```bash
   # 仅重新启动已有版本，不重新构建
   ./run.sh ui --daemon

   # Python 后端代码有更新：先停再启动，不重新构建前端
   ./run.sh ui --stop
   ./run.sh ui --daemon

   # webui 前端源码有更新：停止后重新构建并启动
   ./run.sh ui --stop
   ./run.sh ui --rebuild --daemon
   ```

3. 不要每次都重新安装依赖。只有 `node_modules` 缺失、不完整，或 `package.json`
   的依赖发生变化时，才执行一次依赖安装。服务器安装 npm 依赖必须使用公网镜像，
   并忽略锁文件中可能存在的内网镜像地址：

   ```bash
   cd /root/autodl-tmp/MAS_Science_Infra/webui
   npm install --include=dev --package-lock=false \
     --registry=https://registry.npmmirror.com/ \
     --no-audit --no-fund
   ```

   依赖完整且 `webui/dist` 已对应当前前端源码时，直接运行
   `./run.sh ui --daemon`，不要重复安装或重建。

4. 不需要验证


# Main 核心功能清单

`main` 是后端、训练和运行行为的功能基线；`feature/ui` 是新 WebUI 的设计与实现基线。目标是将main的一些功能和ui，在新 WebUI 中同步对接迁移实现，就是新的Webui可以使用之前main的部分的所有功能，不是照搬 `main` 的旧界面，而是以 `main` 的训练功能基线进行接入和增强到新ui界面, 最完成合并
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

## 训练启动、停止与 RL 算法迁移

这两项不能只以“按钮可点击”或“算法出现在下拉框”为完成标准。训练后端入口与六种算法实现已经存在，但新 UI 还需要补齐模型来源、启动前检查、运行快照、安全停止、实时日志和真实训练验收。

当前训练调用链：

```text
POST /api/rl/train
  → services.start_train()
  → 同步 workflow.sampling 到 rl.yaml
  → 生成 mas/train_tir_agent.py 参数
  → ProcessManager 启动独立进程
  → stdout 写入 artifacts/runs/<run_id>/stdout.log
  → /api/runs 查询状态和日志
```

`algorithm.adv_estimator=grpo` 是 VERL 的兼容入口；实际算法由 `algorithm.tir_algo` 决定。现有实现包括 GRPO、ARPO、AEPO、IGPO、GIGPO 和 RAE。Sampling Policy 会覆盖最终算法、`rollout.n` 和 Branch Sites 参数。

服务器已确认本地训练模型存在：

```text
/root/autodl-tmp/MAS_Science_Infra/LLM/Qwen3-4B/config.json
```

当前阻塞不是模型缺失，而是模型资源前端已经接入、对应后端路由尚未迁入当前集成分支，导致 `/api/model-resources` 和实验模型绑定接口返回 404。

迁移按四个阶段实施：

| 阶段 | 核心目标 | 实施文档 |
| --- | --- | --- |
| T1 模型来源与启动前检查 | 已实现；待服务器 WebUI 手动验收本地模型发现、绑定和 Preflight | [训练运行第一阶段-模型来源与启动前检查](../webui/plan/训练运行第一阶段-模型来源与启动前检查.md) |
| T2 运行快照与安全启停 | 已实现；本地假进程验证通过，真实训练留待统一服务器验收 | [训练运行第二阶段-运行快照与安全启停](../webui/plan/训练运行第二阶段-运行快照与安全启停.md) |
| T3 实时控制台与训练工作区 | 本地已接入：配置去重、底部训练日志/真实调试双入口、run 级增量日志与只读快照；独立采集 UI 已移除，后端 Collector 保留；待用户 UI 验收，未部署 | [训练运行第三阶段-实时控制台与训练工作区](../webui/plan/训练运行第三阶段-实时控制台与训练工作区.md) |
| T4 RL 算法收口与真实训练验收 | 统一最终算法解析，依次人工验证六种算法的启动、停止、日志和产物 | [训练运行第四阶段-RL算法收口与真实训练验收](../webui/plan/训练运行第四阶段-RL算法收口与真实训练验收.md) |

开发阶段默认只做快速验证：

```bash
python -m compileall science_infra/control mas rl
cd webui && npm run build
```

真实 GPU、VERL、vLLM 和各算法训练由用户在服务器 WebUI 中明确触发，不作为每次代码修改的自动验证。只有完成 T1–T4，并对某算法完成服务器人工闭环后，才能把对应算法标记为“新 UI 已验证”。

## 集成约束

- 不迁移或保留 `main` 的旧 UI 表现层。
- `main` 的后端、训练链路、Workflow schema 和运行行为不得回退。
- `main` 旧 UI 中的功能应在 `feature/ui` 的组件、样式和状态管理体系中重新实现。
- Workflow 读取后直接保存时，不得丢失 Router、Sampling、Branch Site 或 Agent 扩展字段。
- 最终训练必须继续通过 `mas/train_tir_agent.py` 和 `rl/hooks/` 执行。

## 当前实施顺序

已完成或初步接入：

1. 实验与配置
2. Workflow 0.3
3. GPU 与 RL 参数
4. Sampling 与 Branch Sites

下一步按训练四阶段实施：

1. T1 模型来源与启动前检查
2. T2 运行快照与安全启停
3. T3 实时控制台与训练工作区
4. T4 RL 算法收口与真实训练验收

训练闭环完成后继续：

1. Rollout Tree 页面与 run 级关联
2. Monitor、Harness 和 AGL Metrics 的实时事件接入
3. Mock 数据采集与诊断的产品入口收口
