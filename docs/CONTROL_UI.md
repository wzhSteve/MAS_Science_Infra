# Science Control UI 落地说明

对照设计方案实现的控制面与 WebUI（零代码工作室切片）。

## 启动

总命令（仓库 uv `.venv`，不要用 miniconda）：

```bash
./run.sh ui
./run.sh ui --port 8787 --rebuild
./run.sh ui --daemon          # 后台；日志 artifacts/run_smoke/ui.log
./run.sh ui --stop
./run.sh ui-test             # 测 GPU/RL 控制 API（start 后立刻 stop 训练子进程）
```

浏览器打开 http://127.0.0.1:8787/ ，API 文档 `/docs`。

等价于：

```bash
# 由 ./run.sh ui 自动完成：
#   .venv/bin/science-infra serve --host 0.0.0.0 --port 8787
cd webui && npm install && npm run build && cd ..
```

单卡训练冒烟：

```bash
./run.sh train
./run.sh train fast --gpu 0 --n-runners 1 --algo grpo
./run.sh train --rl-yaml experiments/demo/rl.yaml --gpu 0
```

## 目录


| 路径                       | 职责                                                                |
| ------------------------ | ----------------------------------------------------------------- |
| `science_infra/control/` | FastAPI：实验 YAML、GPU 探测、ProcessManager、REST/SSE                   |
| `webui/`                 | React + React Flow：MAS 画布（Agent/Tool/入口）+ GPU 勾选                 |
| `experiments/<id>/`      | `experiment.yaml` + `llm/workflow/rl/harness.yaml` + `artifacts/` |
| `workflow/compiler.py`   | Topology Compiler：route / message / feedback / tool_call           |


## 层 → YAML → 下层


| UI 面板   | YAML                                | 调用                               |
| ------- | ----------------------------------- | -------------------------------- |
| 顶栏      | `rl.yaml` devices.ids；合同摘要        | Collect / Diagnose / Train；GPU 芯片 |
| LLM     | `llm.yaml`（密钥进 `.secrets.env`）      | health / vLLM start-stop         |
| MAS     | `workflow.yaml`（agents/edges/entry_agent）+ Inspector 写 `rl.yaml` 简参 | `Collector` / Compiler |
| RL      | `rl.yaml`（devices / n_runners / rollout_per_gpu） | `train_tir_agent.py --rl-yaml` + `CUDA_VISIBLE_DEVICES` |
| Harness | `harness.yaml`                      | `HARNESS.diagnose`               |
| Monitor | `artifacts/collect.json` + 训练 stdout | `GET /api/runs` + `build_dashboard_model` |


## 可执行性

- `topology: hub_react`：可 Collect。
- `topology: graph`：边合法即可 Collect（含 Planner-Executor-Verifier）。
- 非法边（例如 Agent↔Agent 的 `tool_call`）保存后 Collect 返回 400。

## GPU / RL 旋钮

| 用户说法 | 字段 |
| --- | --- |
| 选用哪些卡 | `rl.devices.ids` → `CUDA_VISIBLE_DEVICES` |
| 训练占几张卡 | `trainer.n_gpus_per_node = len(ids)` |
| 每题采几条 | `rollout_per_gpu` → `actor_rollout_ref.rollout.n` |
| 并行采集进程 | `n_runners` |

`GET /api/gpus` 返回本机 GPU 列表与推荐档位。1 卡时 `a800_2gpu` 自动降为 `a800`/`fast`。

## live-api-data（MAS 面板）

对齐 `./run.sh live-api-data`：

1. LLM 面板配置 API（或依赖 `.env`）
2. MAS → **Collect from parquet (live)**  
   - 默认 `data/val.parquet` × 5 × `gsm8k`
3. 结果表：id / answer / reward / tool；产物写入 `experiments/<id>/artifacts/collect.json`

顶栏 / MAS mock Collect 的 **采集条数** = 题数（默认 1 条轨迹）。以前写死 demo-1+demo-2，所以 Monitor 总是两行。GRPO 每题采样仍在 RL 页 `rollout_per_gpu`。

训练 Reward 与 AGL Metrics 同源（`GET /v1/agl/metrics`）。Collect 曲线只反映 MAS 采集。训练 stdout 在 `experiments/<id>/artifacts/runs/<run>/stdout.log`；Control 重启后会从磁盘恢复，不再只靠内存里的进程表。

API：

```http
GET  /api/gpus
POST /api/mas/sample-data
POST /api/mas/collect  {"mock":false,"parquet":"data/val.parquet","data_n":5,"source":"gsm8k","sequential":true}
```

## 切页不丢草稿

六个专家面板在 `App` 里 **keep-mounted**（`display:none` 切换，不卸载）。未点保存的 React 草稿（RL 数字、MAS 图、LLM 密钥框、Harness 勾选、Experiment seed）切 Tab 后仍在。

- 仅 **切换 `experiment_id`** 时才用磁盘 yaml 覆盖本地草稿。
- 顶栏 Collect / 勾 GPU 会 `reload()` bundle，但 **不会** 把未保存数字打回 yaml。
- 训练 `run_id` / stdout 提升到 App：轮询 `GET /api/runs?experiment_id=`，底条与 RL / Monitor 共用 log tail。切走 RL 再回来训练仍显示 running，可 Stop。

dirty 时 RL 显示「未保存」，MAS 图 / 训练简参各有 chip。

## AGL Metrics 生命周期

LightningStore（默认 `:4747`）只在 `agl.Trainer.fit()` 期间存在。Control UI（`:8787`）自己 **不起** AGL。

| 状态 | UI |
| --- | --- |
| 无 train / 训练已结束 | Metrics 按钮 **禁用**，文案「训练运行后可用」；不要点 `127.0.0.1:4747`（AutoDL 上那是你的笔记本） |
| 训练 running 且 store 探活成功 | 同源打开 `/agl/metrics` |

后端：

```http
GET /api/agl/health          # 探测 127.0.0.1:4747/v1/agl/health；始终 200，ok=false 表示 offline
GET /agl/...                # 反代理到 LightningStore，并改写 /assets、/v1 前缀
```

环境变量 `AGL_METRICS_ORIGIN`（默认 `http://127.0.0.1:4747`）可改 origin。实验字段 `agl_metrics_url` 现为同源 `/agl/metrics`。

## 采集入口 ≠ 每题采样

三个正交旋钮不要混用「Rollout」一词：

| 说法 | 字段 | 在哪改 |
| --- | --- | --- |
| 采集入口（Episode 从哪个 Agent 开始） | `workflow.entry_agent` | 画布拖「采集入口」或 Inspector「设为采集入口」 |
| 可训练 Agent | `agents[].trainable` → `--active-agent` | Inspector 勾选 |
| 每题几条 / 并行进程 | `rl.rollout_per_gpu` / `n_runners` | Inspector「训练简参」或 RL 页；`PUT .../rl` |

**不要**把 `rollout.n` 画成图节点。顶栏合同摘要：`入口=hub · 可训=[hub] · GPU=0 · n=2 · runners=1`。

```http
GET  /api/gpus
GET  /api/agl/health
GET  /api/runs?experiment_id=demo
GET  /api/runs/{run_id}
```
