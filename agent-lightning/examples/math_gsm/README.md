# Math GSM Agent（GSM8K + Python 工具）

用 Agent-Lightning + VERL（GRPO）训练一个可调用 **Python 代码执行工具** 的数学问答 agent。模型默认使用本地 **Qwen3-4B**。

**正式训练推荐硬件：2×A800-80GB**（完整超参，不为迁就单卡而压 batch / 序列长度）。单卡仅作冒烟或临时回退；若显存吃紧，优先加第二块 A800，而不是继续砍参数。

技术报告（模块逻辑图、参数详解、换 Agent/LLM 指南）：[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)。  
架构对照 spider SQL 示例：[docs/SPIDER_WALKTHROUGH.md](docs/SPIDER_WALKTHROUGH.md)。

## 目录

| 路径 | 说明 |
|------|------|
| `math_agent.py` | LangGraph 数学 agent + `LitMathAgent` |
| `python_tool.py` | 受限 Python 执行工具 |
| `train_math_agent.py` | VERL 训练入口（`a800_2gpu` 推荐 / `a800` 回退 / `fast` 冒烟） |
| `prepare_data.py` | 下载 GSM8K → parquet |
| `scripts/` | 环境、数据、调试、训练 shell |

## 环境

使用 `/root/autodl-tmp/AgentFlow/.venv`：

```bash
bash scripts/setup_env.sh
```

会安装：`vllm==0.9.2`、`verl==0.5.0`、本地 editable `agentlightning`，以及 LangChain/LangGraph/datasets，并钉死：

- `transformers==4.53.3`（避免与 vLLM `aimv2` 注册冲突）
- `fastapi>=0.115,<0.116`（litellm 兼容）

训练脚本默认设置 `VLLM_USE_V1=1`（verl 0.5 + vllm 0.9 异步服务必需）。

数据准备默认使用 `HF_ENDPOINT=https://hf-mirror.com`；若 Hub 不可用，会回退到本地 GSM-hard / 合成数据。

## 数据

```bash
bash scripts/prepare_data.sh
```

生成 `data/train.parquet`、`data/val.parquet`（GSM8K main）。

## 调试 Agent

```bash
bash scripts/debug_math_agent.sh
```

## 训练

```bash
# 单步冒烟（1 卡即可）
bash scripts/smoke_train.sh

# 正式训练：推荐 2×A800（完整超参）
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_2gpu.sh

# 仅当暂时只有 1 卡时的回退（会缩小 batch / vLLM 显存占用）
bash scripts/train_1gpu.sh
```

或直接：

```bash
/root/autodl-tmp/AgentFlow/.venv/bin/python train_math_agent.py a800_2gpu
/root/autodl-tmp/AgentFlow/.venv/bin/python train_math_agent.py a800
/root/autodl-tmp/AgentFlow/.venv/bin/python train_math_agent.py fast
```

## 模型与超参要点

- 模型路径：`/root/autodl-tmp/LLM/Qwen3-4B`
- Tool-call：`hermes`
- **推荐 `a800_2gpu`**：`n_gpus_per_node=2`，`train_batch_size=32`，`max_prompt/response=4096/2048`，`n_runners=8`
- `gpu_memory_utilization=0.35`：VERL 在每张卡上 **colocated FSDP + vLLM**；FSDP 约占 ~49GB，vLLM 只能用剩余显存。该值是相对整卡容量的 KV 预留比例，**不是**削减 batch/序列长度
- 回退 `a800`（1 卡）：仅缩小 batch / micro-batch / vLLM mem；序列长度保持完整
- 默认 logger：`console` + `tensorboard`；曲线见 Agent-Lightning Dashboard `:4747/metrics`（不是 Ray `:8265/#/metrics`）
- 冒烟：`bash scripts/smoke_train.sh`（已验证可跑通 1 个 GRPO step）
- 显存估算 / OOM 排查 / 超参实验影响：见 [docs/interview/08_gpu_memory_oom_hyperparams.md](docs/interview/08_gpu_memory_oom_hyperparams.md)

训练前请确认 GPU 空闲（`train_2gpu.sh` 会尝试清理残留 ray/WorkerDict）。

### 训练曲线怎么看

- Ray `8265/#/metrics`：集群监控（需 Prometheus），**不是** reward/loss 曲线  
- AGL Dashboard：`http://<host>:4747/metrics`（episode reward + step reward/loss）  
- TensorBoard：`tensorboard --logdir checkpoints/AgentLightning/<experiment>/tensorboard`  
- 原始标量：`checkpoints/.../metrics.jsonl`（环境变量 `AGL_METRICS_JSONL`）

## Reward

从对话中解析最终数字（支持 `### ans ###`、`<answer>`、`####`、tool 的 `result=`、末行数字等），与 GSM8K 金标匹配：正确 `1.0`，否则 `0.0`。通过 `agl.emit_reward` 上报。

默认模式 `binary`。面试消融可切换：

```bash
# 离线看三种 reward 差异（无 GPU）
python scripts/reward_ablation_dryrun.py

# 训练时：binary | shaped | format_only
MATH_REWARD_MODE=shaped python train_math_agent.py fast
```

## 技术报告与面试材料

系统级说明（推荐先读）：[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md)

跑通之后如何准备 Agent RL 面试口述：见 [docs/interview/](docs/interview/README.md)

| 文档 | 内容 |
|------|------|
| [超参词典](docs/interview/01_hyperparam_dictionary.md) | 每个 config key 改大/改小会怎样 |
| [显存与 OOM / 超参实战](docs/interview/08_gpu_memory_oom_hyperparams.md) | 显存公式、OOM 决策树、实验影响 |
| [5 分钟讲解稿](docs/interview/02_five_minute_walkthrough.md) | 白板口述：数据 → GRPO step |
| [PPO/GRPO/DPO](docs/interview/03_ppo_grpo_dpo.md) | 算法对比与公式直觉 |
| [算法自问自答](docs/interview/04_algorithm_self_qa.md) | Week 2 模拟追问 |
| [Reward 消融实验](docs/interview/05_experiment_reward_ablation.md) | 可运行的面试项目增量 |
| [架构 + STAR](docs/interview/06_architecture_and_star_story.md) | 系统图与项目故事 |
| [高频 FAQ](docs/interview/07_interview_faq.md) | 10 道必答题标准答 |
