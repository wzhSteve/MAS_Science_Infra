---
name: training_diagnose
description: 训练失败/结果诊断：读 run 状态与日志尾部，定位 OOM、reward 异常、rollout 卡住等常见故障
triggers: [诊断, 失败, 报错, OOM, 卡住, reward, 日志, 排查, 崩了]
---
# 训练失败 / 结果诊断

## 何时用
训练启动失败、跑到一半崩了、reward 一直不动、采样结果异常、用户贴出报错片段。

## 步骤（必须先查数据再回答）
1. 调 `query_run(experiment_id)` 拿最新 run 的状态与 stdout 尾部（可传 run_id 指定）。
2. 对照下面「故障模式」匹配证据；证据不足时调 `read_file` / `grep` 查 run 目录下的
   effective-rl.yaml、launch.json（preflight 快照）。
3. 结论必须引用日志行；没有证据就说「还需要看什么」。

## 故障模式速查
- **CUDA out of memory**：日志含 `torch.cuda.OutOfMemoryError` / `CUDA out of memory`。
  修复顺序：调小 `gpu_memory_utilization` → 减 `log_prob_micro_batch_size_per_gpu` /
  `ppo_micro_batch_size_per_gpu` → 减 group_n / rollout_per_gpu → 换 a800_2gpu 分摊。
- **GPU 被占用**：preflight 出现「本地模型服务正在占用 GPU」或日志停在 vLLM 初始化。
  先停 llm 服务（train 启动时会自动停，需确认），或换 devices.ids。
- **reward 全零 / NaN**：日志里 reward 长时间 0 或出现 NaN。先看数据 train_files 是否为
  小样本；检查 reward 函数与 val_files；确认采样站点 resume_mode=messages。
- **rollout 卡住**：状态 running 但 stdout 长时间无新行。常见是 vLLM 与训练抢卡、
  n_runners 过高、或 SSE/网络挂起；查 PROCS 状态与 tensorboard。
- **启动即失败**：preflight checks 的 blocking_issues 已经给出原因
  （数据文件不存在 / GPU 越界 / workflow 不可执行 / 依赖缺失），直接引用对应 check。
- **进程反复重启**：Ray worker OOM 或 vLLM 崩溃循环；用 `query_gpu_env` 看当前占用。

## 回答要求
- 先说结论（哪类故障、证据行），再给修复步骤；每步标注要改的 YAML 字段。
- 修复建议落到操作层面：哪里改（画布设置区 / rl.yaml）、改完需要「保存实验」再重训。
- 不要建议用户绕过 preflight 或手改 experiments/<id>/ YAML。
