# Branch paired evaluation (lr=0)

四臂各 4 step，学习率 0，`budget_violation` 均为 0。旧 BR-Strict 的 uplift 不进入本表。

任务级配对没有做成。落盘的 `data_id` 是每次作业新生成的 rollout UUID，四臂之间没有交集；`answer` 字段又全部是 1。因此不能报告同一题上的 \(\Delta\)。下面用每条 rollout 的 `outcome>0` 比例，以及训练日志里的均值 reward，作为未配对的代理。

| 臂 | rollout 数 | outcome>0 | 均值 reward | 均值 branch_count | 均值 uplift |
|---|---:|---:|---:|---:|---:|
| No-branch | 28 | 0.250 | 0.219 | 0 | 0 |
| Failure-only | 32 | 0.656 | 0.475 | 4.00 | +0.102 |
| Failure-confirmed | 32 | 0.562 | 0.447 | 3.75 | +0.040 |
| Random（失败候选里按硬 cap 抽） | 32 | 0.625 | 0.503 | 4.00 | +0.047 |

分叉次数上，Failure-confirmed（3.75）没有超过 Failure-only（4.00）。在这个前提下，它的 outcome 比例、reward 和 uplift 都低于 Failure-only，也低于同预算随机分叉。

当前未校准的 \(p<0.5\) 没有提高分叉边际收益。

事件审计（`EVENT_AUDIT.md`）里执行恢复率只有 0.015。没有可纠正的成功，这里的 Branch uplift 不能解释成认知修正。
