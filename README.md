# Agent_Science_Infra

实现主仓在 **agent-lightning** 示例目录，CLI / Control UI 入口在本目录 `science_infra`：

[`agent-lightning/examples/tir_agent`](agent-lightning/examples/tir_agent)

## Science Control UI（分层面板）

```bash
./run.sh ui
# 浏览器 http://127.0.0.1:8787/    API /docs
./run.sh ui --daemon
./run.sh ui-test
```

实验配置目录：`experiments/<id>/{experiment,llm,workflow,rl,harness}.yaml`。UI 按层写回 YAML；Collect / Diagnose / Train 经 Control API 调用下层。

开发热更新：终端 A `science-infra serve --port 8787`，终端 B `cd webui && npm run dev`（Vite 代理 `/api`）。

### MAS 模型接入与就绪

远程 OpenAI 兼容 API 由服务端提供模型算力，本机不需要训练 GPU、VERL 或 AGL 服务。Control 的模型接入与 Rollout 执行核心位于 Python 侧，浏览器只负责配置和展示。

在**启动 Control 的 Python 环境**中安装对应的可选依赖：

```text
python -m pip install -e ".[live]"
python -m pip install -e ".[data]"
```

`live` 用于真实模型执行；`data` 仅用于 parquet 读取。示例任务的 mock 执行不依赖它们。安装或升级后重启 Control，就绪状态会重新检查；页面刷新不会自动安装包、启动模型或发送生成请求。

- `GET /api/mas/readiness?experiment_id=demo`：只检查本地配置和导入依赖，分别返回阻断项与提示项。`ready=true` 表示具备本地尝试条件，不代表已验证模型生成或工具调用。
- `POST /api/llm/health?experiment_id=demo`：用户手动探测 `/models`。401/403、429、网络错误分别展示；404/405 标为不支持探测，不推断生成不可用。不会发起聊天生成。
- 模型密钥来源为实验 `.secrets.env`，其次是服务启动时加载的进程环境 / 仓库 `.env`。探测可以显式使用临时输入；普通 Collect 只使用保存的配置。密钥不回显，实验间不会通过修改进程环境串用。
- 空密钥允许尝试无鉴权的兼容服务；不自动将真实执行降级为 mock。独立真实运行拒绝没有训练资源注入的 `rl_endpoint`。
- 模型配置以 `llm.yaml` 为准，Workflow 保存不会覆盖另一个页面刚保存的模型配置；LLM 保存仍同步 Workflow 中的模型摘要。
- API 端点使用不带用户名、密码、查询参数和片段的基础地址，认证信息放在独立 Key 字段。探测结果会随有效配置变更失效。

执行契约包括 Agent 身份、模型身份、有效上下文、工具 / Skill 声明与快照范围。单次执行、Collector 与训练适配器共用 MAS 执行核心，模型与工具事件在调用边界记录；API 未提供的信息不填充虚假概率或策略版本。

### 单次 Workflow Rollout

MAS 控制台的“单次 Rollout”从 Workflow 入口执行一个问题，默认不计算奖励或生成训练信号。“数据集采集”保留现有示例 / parquet Collect 行为。

| API | 用途 |
| --- | --- |
| `POST /api/mas/rollout-runs?experiment_id=demo` | 提交 `{workflow, task: {question, id?}, execution: "mock" 或 "live"}`，同步采集一次轨迹 |
| `GET /api/mas/rollout-runs/{run_id}?experiment_id=demo` | 读取运行状态与结果摘要 |
| `GET /api/mas/rollout-runs/{run_id}/trajectory?experiment_id=demo` | 读取本次 Workflow / 输入、实际模型摘要及完整轨迹 |

运行记录保存于 `experiments\<id>\artifacts\rollout-runs\<run_id>`，不会覆盖原有 `collect.json`。每次运行使用提交时的 Workflow 快照和解析后的模型配置；后续编辑不会加入正在执行的任务。

- `run.json` 是运行摘要和配置记录，`trajectory.json` 保存统一轨迹，`archives` 保存执行事件与快照。
- 记录采用原子替换；运行期间持有本地文件锁。进程意外退出后，读取遗留 running 记录会标记为 interrupted，不自动重新执行。
- HTTP 200 仍可能返回 `status=failed`，必须读取业务状态。执行成功不等于答案正确，也不等于数据满足任意 RL 算法。
- `trace_status=complete` 只表示本阶段调用级记录完整；缺失 token / logprob / 策略版本时不会伪造。
- 单次执行接口没有取消、自动重试或可靠后台队列；连接丢失且未获得运行 ID 时，结果未知。
- API Key 与认证字段不进入公开结果；因脱敏而改变记录内容时明确标记，不能据此承诺严格复现。

### Rollout 查看与定位

控制台“结果”支持最近运行、答案摘要、执行片段、对话、工具调用和事件查看。首次只加载事件预览，展开后读取完整调用内容；对话和事件分别分页，不将静态工具绑定当作实际调用。

- `GET /api/mas/rollout-runs?experiment_id=demo&limit=15` 返回摘要列表及 `next_cursor`；后续页通过 `cursor` 获取。已有记录首次建立本地索引，新记录自动更新索引。
- 轨迹 URL 增加 `view=preview&offset=0&limit=100` 可读取有截断标记的预览，省略时保持原完整响应。
- `.../{run_id}/context` 读取问题与配置快照；`.../{run_id}/trajectory/events/{event_id}` 读取单个完整事件；`.../{run_id}/trajectory/messages` 分页读取消息。
- `.../{run_id}/export` 下载已有运行的脱敏 JSON，不重新执行任务。
- “定位”按真实记录关联到当前画布实体，保持草稿不变；“使用此问题”只填写任务，不自动调用模型。
- 能力说明区区分记录完整性、模型概率信息和快照范围。缺失信息不补造，消息快照不代表可恢复整个系统。

## 初版本 CLI

```bash
# uv 环境（推荐）
bash scripts/setup_uv_env.sh

./run.sh help
./run.sh smoke                 # mock 冒烟
./run.sh live-vllm             # 本地 vLLM：LLM → tool → 答案 → reward
./run.sh live-api              # .env API 单条演示题
./run.sh live-api-data         # data/val.parquet 抽 5 条 gsm8k 经 API 跑 TirAgent
# ./run.sh live-api-data data/val.parquet 5 gsm8k
```

```bash
pip install -e .

science-infra doctor
science-infra collect --mock --n 2 --out /tmp/traj.json
science-infra diagnose /tmp/traj.json
science-infra status /tmp/traj.json --html /tmp/status.html
science-infra dashboard /tmp/traj.json --html /tmp/science-dashboard.html
```

AGL Dashboard（训练时通常是 `:4747`）侧栏 **Science MAS** 可打开同一份 collect JSON。训练 live 曲线仍在 **Metrics**；Control UI 的 RL 面板可跳转该页。

不安装时：

```bash
cd agent-lightning/examples/tir_agent
PYTHONPATH=. python scripts/check_workflow_deps.py
PYTHONPATH=. python scripts/collect_rollouts.py --mock --out /tmp/traj.json
PYTHONPATH=. python tests/test_infra_v1.py
PYTHONPATH=. python train_tir_agent.py fast --algo grpo --rl-yaml ../../../experiments/demo/rl.yaml
```

- 产品原则：[design.md](design.md)
- 技术框架与实现差距：[docs/TECHNICAL_FRAMEWORK.md](docs/TECHNICAL_FRAMEWORK.md)
- UI 落地说明：[docs/CONTROL_UI.md](docs/CONTROL_UI.md)
- 架构提案：[docs/GPT_analysis.md](docs/GPT_analysis.md)
- 早期差距分析：[docs/infra-gap-analysis.md](docs/infra-gap-analysis.md)
- TIR 算法：[agent-lightning/examples/tir_agent/docs/DESIGN.md](agent-lightning/examples/tir_agent/docs/DESIGN.md)
