# Event audit: fcr_audit

- rollouts: 379
- labeled steps: 3689
- adjacent pairs: 3310
- prediction failure rate: 0.7742 (2856/3689)
- execution failure rate: 0.9756 (3599/3689)
- P(p recovers | p failed): 0.0413 (106/2566)
- P(Y=1 | Y=0): 0.0151 (49/3239)
- P(strict | prev pred-fail and exec-fail): 0.0078 (20/2566)
- adjacent strict count: 20
- window-3 count: 28
- lineage count: 49
- false correction count: 0
- sparsity source: `execution_recovery_near_zero`

执行恢复接近 0。后续 Branch uplift 不能解释成认知修正。
