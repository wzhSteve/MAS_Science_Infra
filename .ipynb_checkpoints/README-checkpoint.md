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
