# Qwen3-1.7B · 双卡 A800 · APPO / ARPO 训练

脚本：`ref_Rep/APPO-main/APPO/scripts/APPO_1.7B_2gpu/run.sh`

默认 `ALGO=appo` 跑 APPO 论文形态（token 级分支 + FIPO）。`ALGO=arpo` 跑同一套数据、同一套工具，但用 GRPO 优势、vanilla PPO-clip，并关掉 token 级分支。两张 A800-80GB 用 FSDP 数据并行一起参与 rollout 和策略更新，1.7B 不上 CPU offload。

两个算法各自从一份原始权重复本起训，不共用、也不回写 `LLM/Qwen3-1.7B`：

| 算法 | 权重 | checkpoint |
|---|---|---|
| APPO | `LLM/APPO/Qwen3-1.7B` | `checkpoints/appo_local/` |
| ARPO | `LLM/ARPO/Qwen3-1.7B` | `checkpoints/arpo_local/` |

## 1. 一条命令

```bash
# 先看配置，不占 GPU
bash ref_Rep/APPO-main/APPO/scripts/APPO_1.7B_2gpu/run.sh print

# 2 步冒烟（约几十分钟，确认双卡 + dashboard + dump）
bash ref_Rep/APPO-main/APPO/scripts/APPO_1.7B_2gpu/run.sh smoke

# 正式：1 个 epoch（train_10k / 16 ≈ 625 step）
bash ref_Rep/APPO-main/APPO/scripts/APPO_1.7B_2gpu/run.sh full

# 正式但只跑 40 step（推荐先这样看曲线）
TOTAL_STEPS=40 bash ref_Rep/APPO-main/APPO/scripts/APPO_1.7B_2gpu/run.sh full

# 同一套超参跑 ARPO，从 LLM/ARPO/Qwen3-1.7B 起训
ALGO=arpo bash ref_Rep/APPO-main/APPO/scripts/APPO_1.7B_2gpu/run.sh full
```

`smoke` / `full` 在拉起 Ray 之前会启动 Dashboard。浏览器打开：

- 本机：`http://127.0.0.1:7860`
- 远程：`ssh -L 7860:127.0.0.1:7860 <host>` 后同样打开 `http://127.0.0.1:7860`

页面每 4 秒自动轮询。顶栏会显示当前阶段（工具调用、验证生成、指标已写出）和日志最后写入时间。指标曲线要等该 step 整步结束才出现；在此之前选中本次 run（`appo_local/APPO_1.7B_2gpu_...` 或 `arpo_local/ARPO_1.7B_2gpu_...`）即可，不必手点刷新。

## 2. 两张卡怎么被用满

| 项 | 取值 | 原因 |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | `0,1` | 两张 A800-SXM4-80GB |
| `trainer.n_gpus_per_node` | `2` | Ray 放置 2 个 worker |
| `tensor_model_parallel_size` | `1` | 1.7B 不切张量并行。TP=2 会把已经很小的矩阵再劈开，通信多于计算 |
| FSDP | 2 路数据并行 | rollout 的 log_prob、actor 更新、ref 都分到两卡 |
| `param_offload` / `optimizer_offload` | `False` | 4B 单卡才需要把 Adam 卸到 CPU。1.7B 双卡静态大约 15GB/卡，留在 GPU 上更新更快 |
| `gpu_memory_utilization` | full `0.55` / smoke `0.45` | vLLM 占用约 44GB/卡做 KV cache，其余给 FSDP |
| `use_dynamic_bsz` | `True` | 按 token 预算装箱，避免短轨迹浪费算力 |
| `ppo_max_token_len_per_gpu` | `4 × (prompt + response)` | full 下约 32K token/卡 |

批量约束（`ray_trainer.py` 会断言）：

```
每步序列数 = train_batch_size × rollout.n
必须能被 GPU 数整除
```

| 模式 | prompts / step | rollout.n | initial | 每步序列 | 长度 |
|---|---|---|---|---|---|
| smoke | 4 | 4 | 2 | 16 | 1024 + 4096 |
| full | 16 | 8 | 4 | 128 | 2048 + 6144 |

`initial_rollouts=4`、`rollout.n=8` 是 APPO 的预算拆法：每个 prompt 先采 4 条 init，剩下 4 条由 Branching Score 在高熵、高未来价值的 token 上分叉补满。`initial == n` 时不会出现 branch 子节点。

## 3. 算法开关（完整 APPO）

这些 override 都写在 `run.sh` 里，对应关系如下。

| 开关 | 值 | 作用 |
|---|---|---|
| `algorithm.adv_estimator` | `appo` | 组内 GRPO 优势 + 分支行按几何折扣回写父行 |
| `algorithm.reward_scale_discount` | `0.9` | 分支 outcome 回写父轨迹的折扣 \(M\) |
| `rollout.appo_dynamic_branching` | `True` | token 级过程分支。关掉则退化为 ARPO 式、或纯 init |
| `rollout.reward_scale_discount` | `0.9` | rollout 侧同一折扣，和 algorithm 侧保持一致 |
| `actor.policy_loss.loss_mode` | `future_kl` | FIPO：未来 KL 影响权重 \(A_t(1+b f_t)\) |
| `future_kl_weight` | `1.0` | 上式里的 \(b\) |
| `decay_rate` | `64` | 未来 KL 的 token 半衰期 |
| `chunk_size` | `128` | Future-KL 矩阵分块，控显存 |
| `future_kl_clip_high_only` | `True` | 影响权重只从上侧裁剪 |
| `safety_thresh` | `6.0` | 重要性比率过大时，负样本的影响权重封顶为 1 |
| `actor.use_kl_loss` | `True` | 额外的 low-var KL 惩罚，系数 `0.001` |
| `actor.clip_ratio_low/high` | `0.2 / 0.28` | 非对称 PPO clip |
| `actor.optim.lr` | `1e-6` | 与既有 APPO 脚本一致 |

Reward 仍是 `deep_research.compute_score`：格式坏 \(-1\)，格式对但答错 \(0\)，答对按 F1。工具协议是 `<search>…</search>` 与 `<python>…</python>`，结果以 `<result>` 注入，不进 policy loss 的位置由 rollout mask 标出（dashboard 样本面板把工具结果灰显/绿底）。

## 4. 搜索限流

`serp_api1` 的 Access details 写明限额是 **100 req/min**。两个 GPU 进程共用一个文件窗口，默认 `SEARCH_MAX_RPM=80`，给限流重试留出余量。同一条 query 在飞时，另一个进程只等缓存，不再各打一次。

命中 429/502 时按 8s、16s、32s… 指数冷却，并加最多 25% 的随机抖动，最多 4 次，上限 60s。Key 从仓库根 `.env` 的 `BRIGHT_DATA_API_KEY` 读取，并去掉成对引号（带引号直连会 401）。

缓存文件：`ref_Rep/APPO-main/APPO/search_cache/search_cache.json`。重复 query 不打 API。

环境变量可覆盖，不必改脚本：

```bash
SEARCH_MAX_RPM=60 SEARCH_MAX_API_RETRIES=6 \
  bash ref_Rep/APPO-main/APPO/scripts/APPO_1.7B_2gpu/run.sh full
```

## 5. Dashboard 上看什么

服务由 `run.sh` 后台启动，日志在本次 run 目录的 `dashboard.log`，pid 在 `dashboard.pid`。端口已被占用时直接复用，不会再起一个。

训练写入这些文件，页面零配置读取：

```
checkpoints/<appo_local|arpo_local>/<experiment_name>/
├── metrics.jsonl          # 每 step 的完整指标（dashboard 用）
├── paper_metrics.csv      # 论文对比用的固定列，可直接画图
├── paper_metrics.jsonl    # 与 csv 同一行内容
├── sample_branches.csv    # 每个样本一行的分支对照（父轨迹 + 子分支）
├── branches/{step}.jsonl  # 同上，按 step 拆开，重跑该 step 会覆盖
├── rollout/{step}.jsonl   # uid / rollout_kind / branch_meta / 长度 / 全文
├── run.log                # tee 的完整控制台
└── global_step_*/actor    # save_freq 到点后的 FSDP 权重
```

页面布局：

- 左：Reward、token reward、Advantage、policy loss、entropy/KL/FIPO 权重、response 长度与工具调用
- 右：Rollout Tree。init 横排，branch 挂在父节点下，边上是 branching score，节点上有 `split@token`
- 下：点节点看全文。`<search>` / `<python>` / `<result>` / `<answer>` 分色

来源徽标为 `metrics.jsonl` 表示新 dump 生效。若只有 `run.log`，说明还没跑完第一个 step 的日志刷新。

## 6. 论文对比曲线

每个 step 在写完 `metrics.jsonl` 之后，再写一行固定列的 `paper_metrics.csv` / `paper_metrics.jsonl`。列名不随算法变化：ARPO 没有 FIPO 时 `fipo_iw_mean` 留空，APPO 与 ARPO 可以按 `algo` 列叠在同一张图上。

画图时优先用这些列：

| 列 | 含义 | 图上怎么用 |
|---|---|---|
| `reward_mean` / `reward_max` / `reward_min` | `critic/score`，轨迹级 outcome | 主曲线，横轴 `step` |
| `token_reward_mean` | token 级 reward 均值 | 与 outcome 对照 |
| `advantage_mean` | 优势均值 | 看优势是否塌缩 |
| `pg_loss` | policy gradient loss | 算法更新强度 |
| `entropy` | `actor/entropy_loss` | 探索是否过早塌掉 |
| `ppo_kl` | 当前策略相对 old policy 的 KL | 更新是否过猛 |
| `grad_norm` | actor 梯度范数 | 稳定性 |
| `pg_clipfrac` | PPO clip 比例 | clip 是否长期打满 |
| `fipo_iw_mean` | FIPO 影响权重均值 | 仅 APPO 有值 |
| `branch_count` / `branch_score_mean` / `branch_improve_rate` | 本步分支条数、Branching Score 均值、子分支优于父轨迹的比例 | 分支有没有发生、有没有带来更高 reward |
| `response_length_mean` | 回复长度 | 是否在顶满 `max_response_length` |
| `search_success_rate` / `tools_failed_calls` | 搜索是否被限流拖垮 | 排除“reward 下降其实是工具失败” |
| `val_reward` / `val_score` / `val_f1` | 验证集，只在 `test_freq` 那一步有值 | 论文主表用最后几次非空值 |
| `algo` / `experiment` / `adv_estimator` / `branching` / `loss_mode` | 这次 run 是哪种算法 | 分组 |

把多次 run 合成一张表：

```bash
python ref_Rep/APPO-main/APPO/scripts/export_paper_curves.py \
  --root checkpoints \
  --out checkpoints/paper_compare.csv
```

`paper_compare.csv` 多一列 `run_dir`。用 pandas 按 `algo` 分组、以 `step` 为横轴即可画 reward / loss / val_f1。

### 每个样本的分支对照

`rollout/{step}.jsonl` 里仍有全文和原始 `branch_meta`。另外每个训练 step 会写 `sample_branches.csv`：init 和 branch 各占一行，branch 行已经带上父轨迹，不用再按行号去对。

对照时看这些列：

| 列 | 含义 |
|---|---|
| `uid` / `sample_index` | 同一个 prompt 的 init 与 branch 共享 `uid` |
| `rollout_kind` | `init` 或 `branch` |
| `parent_row` / `split_token_idx` / `split_frac` | 从哪一行、response 的第几个 token 分开，以及该点在父回复中的相对位置 |
| `branch_score` / `m_base` / `split_entropy` | 分裂点的 Branching Score、相对本步最大分数的比例、父轨迹该 token 的熵 |
| `score` / `f1_score` / `answer` | 这条轨迹自己的 outcome |
| `parent_score` / `parent_f1` / `parent_answer` | 被分开的那条父轨迹 |
| `score_delta` / `f1_delta` | 子分支减去父轨迹。大于 0 表示这次分支答得更好 |
| `n_search` / `parent_n_search` | 这条轨迹和父轨迹各调用了几次 `<search>` |
| `advantage_mean` | 该轨迹 response token 上的优势均值 |
| `group_n_branch` / `group_improve_rate` | 这个 prompt 本组有几条分支、其中多少比父轨迹更好 |
| `split_preview` | 父轨迹分裂点前约 16 个 token 的文本，方便看是在搜索、推理还是答案处分开 |

全文仍在 `rollout/{step}.jsonl`，csv 不重复整段生成。同一 step 重跑会覆盖 `branches/{step}.jsonl` 再重建汇总表。

多次 run 合成：

```bash
python ref_Rep/APPO-main/APPO/scripts/export_paper_curves.py \
  --root checkpoints \
  --filename sample_branches.csv \
  --out checkpoints/sample_branches_compare.csv
```

## 7. 产物与续训

正式 run 名：

- APPO：`checkpoints/appo_local/APPO_1.7B_2gpu_Qwen3-1.7B`，起点 `LLM/APPO/Qwen3-1.7B`
- ARPO：`checkpoints/arpo_local/ARPO_1.7B_2gpu_Qwen3-1.7B`，起点 `LLM/ARPO/Qwen3-1.7B`

`LLM/Qwen3-1.7B` 保持不动，只作为再复制的源。训练不会改这份源目录。

`trainer.resume_mode` 在 yaml 里是 `auto`。同一条命令再跑，会从该目录最新的 `global_step_*` 接着训。想从头来，换实验名或挪走该目录。

`save_freq=25`、`test_freq=25`。冒烟 `save_freq=-1`（不存 47GB 级权重）、`test_freq=2`（最后一步做一次验证）。

1 个 epoch 约 \(10000 / 16 = 625\) step。带搜索的一步通常是数分钟到十几分钟，完整 epoch 要以天计。先用 `TOTAL_STEPS=40` 看 reward 是否离开 \(-1\)、树上是否出现 branch，再决定是否放满。

## 8. 启动前检查

```bash
nvidia-smi
# 期望两张 A800，显存接近空闲

test -f LLM/APPO/Qwen3-1.7B/config.json && echo appo_model_ok
test -f LLM/ARPO/Qwen3-1.7B/config.json && echo arpo_model_ok
test -f ref_Rep/APPO-main/APPO/rl_datasets/train_10k.parquet && echo data_ok
test -f ref_Rep/APPO-main/APPO/dashboard/frontend/dist/index.html && echo ui_ok
grep -q '^BRIGHT_DATA_API_KEY=' .env && echo key_ok
```

`print` 模式会打印批量、路径和 dashboard 地址，并确认 key 非空，但不打印 key 本身。

## 9. 出问题时看哪里

| 现象 | 处理 |
|---|---|
| `real_train_batch_size ... must be divisible` | 改 `TRAIN_BATCH_SIZE × ROLLOUT_N`，使它能被 2 整除 |
| `Auth method is not supported` | `.env` 的 key 带了引号。脚本会剥引号；若仍失败，检查值本身 |
| 日志大量 `rate-limited` / `Expecting value` | 把 `SEARCH_MIN_INTERVAL` 调到 4 或 5。失败样本会被 EOS 截断，reward 会塌到 \(-1\) |
| Dashboard 打不开 | 看 `<ckpt>/dashboard.log`。缺 `frontend/dist` 时先 `cd dashboard/frontend && npm run build` |
| 页面有 run 但图是空的 | 第一个 step 还没结束。`metrics.jsonl` 在该 step 的更新之后才出现 |
| 树上全是 init、没有紫边 branch | `initial_rollouts` 必须小于 `rollout.n`，且 `appo_dynamic_branching=True` |
| OOM | 先把 full 的 `GPU_MEMORY_UTILIZATION` 降到 `0.40`，或把 `MAX_RESPONSE_LENGTH` 降到 `4096`。不要打开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`（与当前 vLLM 内存池不兼容） |

## 10. 和单卡 4B 脚本的差别

`APPO_4B_local_1gpu.sh` 是给 Qwen3-4B、一张卡、Adam offload 用的。本脚本专用于 Qwen3-1.7B 两卡：

- GPU 数 1 → 2，offload 关闭
- APPO 读 `LLM/APPO/Qwen3-1.7B`，ARPO 读 `LLM/ARPO/Qwen3-1.7B`，源目录 `LLM/Qwen3-1.7B` 不参与训练
- 批量按 128 序列/step 配，而不是单卡 4B 的 8×8
- 开训前自动起 dashboard
- 搜索节流参数（`min_request_interval` 等）走 hydra `+` 前缀，因为它们不在 `ppo_trainer.yaml` 的结构里
