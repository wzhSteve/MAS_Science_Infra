# Literature Cards Index（搜索推理 / 单 Agent）

> **对象**：RAE（Recovery-Aware Exploration with Counterfactual Policy Elimination）  
> **检索截止**：2026-09-14  
> **相对** `Agent_related_work_baselines.md`（2026-09-06）的增量见文末。

## 四判别问（每张卡强制回答）


| 编号  | 问题            | RAE 要求                                              |
| --- | ------------- | --------------------------------------------------- |
| Q1  | 选点信号？         | 外部结局熵 $\bar H^B$ × 策略熵；+ 失败触发                       |
| Q2  | 失败如何进入学习？     | $\pi^{tgt}=\mathrm{Renorm}(\pi_{old}\setminus a_f)$ |
| Q3  | 有无 $\lambda$？ | 单一 KL，无加权超参                                         |
| Q4  | 学什么能力？        | 错→对恢复片段 + 机制指标                                      |


## P0 清单（必深读 / 已出卡）


| 卡                               | 工作                 | Baseline 角色 | 进入主表？                |
| ------------------------------- | ------------------ | ----------- | -------------------- |
| [ARPO](ARPO.md)                 | 纯熵工具后分支            | B2          | Yes                  |
| [AEPO](AEPO.md)                 | 熵预算/平衡             | B3          | Yes                  |
| [APPO](APPO.md)                 | 内部第二因子乘积门控         | B4          | Yes（优先于 RAIL）        |
| [RAIL](RAIL.md)                 | 可学习 recoverability | 附录 A5       | Appendix（可升主表）       |
| [Search-R1](Search-R1.md)       | Stage-1 GRPO 搜索默认  | B0          | Yes                  |
| [DAPO](DAPO.md)                 | Loss 稳定版           | B1          | Yes                  |
| [IGPO](IGPO.md)                 | 答案 IG 过程奖励         | B5          | Yes                  |
| [Agent-R](Agent-R.md)           | 失败→造正样本            | B6          | Yes                  |
| [Fission-GRPO](Fission-GRPO.md) | 失败裂变恢复采样           | B6 备选 / 附录  | Appendix（优先 Agent-R） |
| [BranPO](BranPO.md)             | 对比分支               | 消融 A9       | Appendix             |




## P1 / 增量（抽样卡或仅索引）


| 工作                     | arXiv / venue          | 相对 RAE                         | 处理                     |
| ---------------------- | ---------------------- | ------------------------------ | ---------------------- |
| Belief-Shift Branching | 2609.11061（2026-09-10） | 用信念转移选 fork，仍非外部 $\bar H^B$+消除 | Cite / 消融近邻            |
| GACA                   | 2609.12424（2026-09-11） | 粒度自适应信用；具身主场                   | Cite-only（搜索主表不做）      |
| IGRPO                  | 2607.06223             | $P(y^*)$ 分配树预算                 | Appendix               |
| NGRPO                  | 2509.18851             | 全错组负标量                         | 消融                     |
| VinePPO                | 2410.01679             | MC 估 $V(s)$                    | 消融：MC 价值 vs $\bar H^B$ |
| InfoReasoner           | 2602.00845             | $\lambda$-IG                   | Cite                   |
| Dark Room              | 2607.21273             | 密奖励进 GRPO 危险                   | 讨论引用                   |
| RFT / STaR             | —                      | 只模仿成功                          | B7 Yes                 |
| Reflexion              | NeurIPS 2023           | 无训练恢复                          | 可选一行                   |




## 相对 2026-09-06 文档的增量

1. **Belief-Shift Branching (2609.11061)**：树 RL 的 fork 放置从熵/结构改为答案信念转移；加强“第二因子仍常是内部信念”叙事，不进主表。
2. **GACA (2609.12424)**：长时程 agent 信用粒度自适应；主场 ALFWorld/WebShop，搜索协议不重跑。
3. **APPO 代码栈确认**：与 ARPO 同 VERL + tool cache（`xuc865/APPO`），搜索主表可共享 tool/数据管线。
4. **RAIL 全文校准**：recoverability = 干预带来的**组奖励对比/方差增益**；AgentBench OS/DB + 含 WebShop 叙事；Loss 仍 GRPO。



## 卡片文件命名

`ARPO.md` … 见上表。模板见 `[_TEMPLATE.md](_TEMPLATE.md)`。