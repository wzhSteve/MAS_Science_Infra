# Round20 summary（已全部收束）

进程：无；GPU 空闲。C2 于 util=0.25 重试后 **8/8 完成**（约 03:26）。

全程在线臂 **`budget_violation=0`**。

## 结论一览

| 假设 | 结果 |
|---|---|
| **H1** A5≫A1 | **成立**（A5 mean_p=0.47 BSS=+0.034；A1 塌到 0.03） |
| **H2** B3 坍塌更晚 | **不成立**（B1/B3 均 step2 mean_p→0、unexpected→0） |
| **H3** C1 repair 可达 | **部分成立**（C1 step2–3 pattern/n_corr_hits>0） |
| **H3** C2 开 P 后 repair 仍可达 | **不成立**（C2 step2 起即塌，hits=0） |

均值 reward：B1 0.345 / B3 0.456 / C1 0.374 / C2 0.484（单 seed，勿过读）。

## H1 离线

| 臂 | mean_p | BSS |
|---|---:|---:|
| A1 | 0.030 | 0.0 |
| A5 | 0.471 | +0.034 |

## H2：B1 vs B3（均 step2 坍塌）

| step | B1 mean_p / unexpected | B3 mean_p / unexpected |
|---:|---|---|
| 1 | 0.43 / 22 | 0.52 / 21 |
| 2–8 | ≈0 / 0 | ≈0 / 0 |

## H3

**C1**（无 P + R_repair）：step2 pattern_hit=19；step3 n_corr_hits=5、mean_p=0.55；后期仍衰减。

**C2**（Full-v2-lite，重试满 8 step）：step1 mean_p=0.51 unexpected=35；**step2 起永久坍塌**，repair hits=0——开 P 后旧病复现。

## 产物

- `experiments/fcr_short/round20_resume.log`
- `mas/checkpoints/AgentLightning/fcr_r20_*/metrics.jsonl`
- 报告 §13：`docs/ICML2027/FCR_PREDICTION_COLLAPSE_DIAGNOSIS_AND_NEXT_STEPS.md`
