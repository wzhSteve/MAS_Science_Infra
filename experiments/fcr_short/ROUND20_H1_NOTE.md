# Round20 H1 offline (A1 vs A5)

合成前缀、pos_rate=0.08、30 steps、seed=42。不经 Branch/Bing。

| 臂 | 终局 mean_p | BSS | pos_recall | 读法 |
|---|---:|---:|---:|---|
| A1 Brier+原始 batch | 0.030 | 0.0 | 0.0 | 向恒 NO 坍塌；acc=1 虚高 |
| A5 BCE+0.1 Brier+replay | 0.471 | +0.034 | 波动 | 维持中间概率与正类曝光；BSS>0 |

门控：CONTINUE_B（非双塌）。支持 H1：极端不平衡下纯 Brier 易塌；BCE+持久平衡 replay 可推迟/缓解。
