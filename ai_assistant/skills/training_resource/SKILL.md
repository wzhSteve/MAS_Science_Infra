---
name: training_resource
description: 训练资源顾问：按真实机器状态与实验配置，建议 GPU 数量、显存分配、profile 与关键参数
triggers: [GPU, 显存, 几张卡, profile, 训练资源, 训练 agentflow, 资源, 显卡]
---
# 训练资源顾问

## 何时用
用户想知道：训练要几张卡、显存怎么分配、为什么、用哪个 profile（fast / a800 / a800_2gpu）、
rollout_per_gpu / group_n 等参数怎么定。

## 步骤（必须先查数据再回答）
1. 调 `query_gpu_env(experiment_id)`（experiment_id 可为空）：拿真实 nvidia-smi 清单、
   按 GPU 数量的 `recommendation_by_gpu_count`，以及该实验的完整训练计划
   （algorithm / profile / gpu_ids / group_n / checks / launch_preview）。
2. 若用户没有指明实验，用 `query_experiment` 或反问确认是哪个实验。
3. 按「显存账本」逐项给出分配与原因，数字必须来自查询结果。

## 显存账本（以 Qwen3-4B fp16 训练为例）
- Actor 权重 fp16 ≈ 8 GB/份。
- 梯度 fp16 ≈ 8 GB；AdamW 优化器状态（fp32 m+v + fp32 master）≈ 32 GB。
  → 训练态静态占用 ≈ 40-48 GB/卡，单张 24 GB 卡装不下完整训练态。
- vLLM rollout 权重是另一份拷贝，由 `actor_rollout_ref.rollout.gpu_memory_utilization`
  划走固定比例：fast=0.35、a800=0.45（见 mas/train_tir_agent.py:110-153）。
  调大可加速 rollout，但会挤压训练态，OOM 时先调小它。
- Ref model（log_prob 计算）与激活/日志随 micro batch 缩放：
  `log_prob_micro_batch_size_per_gpu`、`ppo_micro_batch_size_per_gpu` 均为 OOM 时的第一旋钮。
- KV cache 属于 vLLM 划分内，max_prompt_length / max_response_length 越大越吃。

## 现有 profile 语义（mas/train_tir_agent.py）
- `fast`：1 卡，gpu_memory_utilization=0.35，train_batch_size=2，rollout.n=2，1 step——冒烟/单卡默认。
- `a800`：1 卡 A800 回退，gpu_memory_utilization=0.45，train_batch_size=8——正式单卡档。
- `a800_2gpu`：2 卡，完整超参（batch 8），GPU<2 时会被自动降级为 fast（见 apply_gpu_selection）。

## 回答要求
- 给出「需要几张卡」时，先看 `recommendation_by_gpu_count`，再对照实验 algo
  （分支采样算法 arpo/aepo/igpo/gigpo/rae 的 group_n 越大越吃 rollout 显存）。
- 给出可写回 rl.yaml 的具体字段（devices.ids / profile / rollout_per_gpu / n_runners），
  并提醒：改完要点「保存实验」，训练走「开始训练」的 preflight。
- 机器不够时的诚实替代：减 group_n、减 max_response_length、用 fast 档、
  或用 `science-infra collect --mock` 先验证流程不占训练 GPU。
