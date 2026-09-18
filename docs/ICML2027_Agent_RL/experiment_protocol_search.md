# RAE 搜索推理实验协议（单 Agent）

> 配套：`idea.md`、`Agent_related_work_baselines.md` §0.5、`literature_cards/`。  
> 状态：**可交给工程开跑的设定锁定稿**；不含实验结果。  
> 日期：2026-09-14  

---

## 0. 一句话协议

在 **veRL + 本地 Wikipedia 检索** 上，以 **Qwen2.5-7B-Instruct** 训练单 agent 搜索策略；主对比 **GRPO → DAPO → ARPO → AEPO → APPO → IGPO → Agent-R/RFT → RAE**；所有分支/探针计入同一预算 $M$；主文必须同时报成功率与恢复机制表。

---

## 1. Agent 形态

```text
User question
    → LLM thinks
    → optional <search>query</search>  → retriever snippets
    → (GAIA only) optional <browse>url</browse> → page summary
    → ... multi-turn ...
    → <answer>...</answer>
```

约束：

- **单 agent**；禁止 planner–worker 多 agent。
- 动作空间开放 token，但语义动作 ∈ {`search`, `browse`, `answer`}。
- 观测（检索片段）token：**loss mask**（Search-R1 / IGPO 惯例）。
- 矛盾信号 $c_t$（RAE 路 2）：空检索、tool 报错、短窗 self-consistency 分歧；**不用** LLM-as-judge 作训练主信号。

---

## 2. Backbone

| 角色 | 模型 | 备注 |
| --- | --- | --- |
| 主表 | **Qwen2.5-7B-Instruct** | 对齐 Search-R1 / ARPO 常见 7B 设定 |
| 泛化 | Llama-3.1-8B-Instruct | 同超参见附录表 |
| 时代校验 | Qwen3-8B（Instruct） | **单独小节**，不与 2.5-7B 混主表 |
| 可选规模 | Qwen3-14B 一组 | 仅当算力允许；对标 AEPO GAIA 叙事 |

冷启动：可选短 SFT（Tool-Star / Search-R1 公开轨迹子集）后再 RL；所有方法共享同一冷启动 checkpoint。

---

## 3. Benchmark

### 3.1 主表（必须）

| 集 | 用途 | 指标 |
| --- | --- | --- |
| HotpotQA | 多跳 ID | EM / F1；pass@1 |
| 2WikiMultiHopQA | 多跳 OOD | EM / F1 |
| Bamboogle | 难多跳 OOD | EM |
| GAIA（文本/搜索子集） | 长时程工具 | Pass@1；Pass@5（可选）；tool/token 成本 |

### 3.2 附录

| 集 | 用途 |
| --- | --- |
| MuSiQue | 额外多跳泛化 |
| NQ / TriviaQA / PopQA | 单跳对照（证明没掉点） |
| WebWalkerQA | 若复用 ARPO deep-search 栈 |

### 3.3 分层（机制叙事必需）

1. **首错位置分桶**：第 1 / 2 / ≥3 次 tool-call 后首次被证伪。  
2. **无成功分支分层**：同前缀 $B$ 条扩展全败的题目比例；对比 Agent-R vs RAE。  
3. **难度代理**：金标跳数或检索命中稀疏度。

**不做主表**：ALFWorld、WebShop、SWE（future work 一句）。

---

## 4. Tool 栈（锁定，禁止中途换引擎刷分）

### 4.1 QA 主栈（对齐 Search-R1）

| 组件 | 选择 |
| --- | --- |
| 语料 | Wikipedia 2018 dump（或 Search-R1 公开索引） |
| 检索器 | BM25 或 E5 稠密检索；**全文实验固定一种**；默认 BM25 top-$k=3$ 或 5 |
| 接口 | `<search>…</search>` → 拼接 top snippets |
| 答案校验 | 归一化字符串 EM（Search-R1）；报告 F1 为辅 |

### 4.2 GAIA 扩展（对齐 ARPO/AEPO）

| 组件 | 选择 |
| --- | --- |
| Search | 与 ARPO 公开 deep-search 包装一致的搜索 API / 缓存 |
| Browser | 只读抓取 + 摘要（ARPO Web Browser Agent） |
| 评分 | GAIA 官方规则；禁止私有 judge 换定义 |
| 缓存 | 强制 tool response cache，保证可复现 |

### 4.3 明确不做

- 训练中途切换 Google/Bing/自研检索引擎。  
- 把 Python 工具作为搜索主表必选（可在附录 math+search 混合，非主文）。  
- 用闭源 DeepResearch 系统数字当同设定基线。

---

## 5. 预算对齐 $M$

定义每题训练预算：

$$
M = \#\text{generated tokens} + \alpha\cdot\#\text{tool calls}
$$

建议 $\alpha$ 取该设定下平均“一次 tool 等价生成 token”（例如检索往返折合常数，实验记录中写死）。

规则：

- RAE 探针 $K$、完整分支 $B$、ARPO/APPO partial rollout、Agent-R MCTS/拼接采样 **全部计入 $M$**。  
- 禁止“免费探针”。  
- 主对比固定两组预算：`M_full` 与 `M_half`（对标 ARPO 半预算叙事）。

默认 rollout 组大小：与 ARPO/APPO 对齐偏好 `n∈{8,16}`（写死一种并全方法共用）。

---

## 6. 最优 Baseline：实现要点

| ID | 方法 | 实现要点 | 证伪 |
| --- | --- | --- | --- |
| B0 | GRPO (Search-R1) | 结局 EM；检索 mask；无分支 | 标量结局 |
| B1 | DAPO | clip-higher + 滤零方差组 | “只是不稳” |
| B2 | ARPO | 工具后高熵 partial rollout；hard/soft 归因择一固定 | 纯熵 |
| B3 | AEPO | 熵预监控预算 + 连续高熵惩罚 + stop-grad | 熵平衡够否 |
| B4 | APPO | `BS=z(H)·z(Ω)`；同 ARPO 数据/tool；`future_kl` | 内部第二因子 |
| B5 | IGPO | turn-level $\Delta\log P(y^*)$ + outcome；**仅 QA** | G2 |
| B6 | Agent-R-lite | 同前缀成功续采拼接 + 反思 token + SFT；完整 MCTS 可选 | 需成功侧 |
| B7 | RFT/STaR | 只 SFT 成功轨迹 | 成功侧 δ |
| — | **RAE** | $U=H^\pi H^B$；失败触发；Renorm；单一 KL / $A^{RAE}$-clip | 本稿 |

附录：

- RAIL（方差 recoverability 控制器）  
- BranPO（对比分支）→ 消融 A9  
- NGRPO、VinePPO、IGRPO  
- Reflexion（无训练一行）

---

## 7. 指标

### 7.1 主表 A：任务性能

- EM / F1 / Pass@1（GAIA）  
- 每成功一次的 token 成本、tool-call 成本  
- 训练曲线（reward、熵均值）

### 7.2 主表 B：机制（novelty 载体）

| 指标 | 预言 |
| --- | --- |
| Recovery Rate (RR) | RAE 的 ΔRR > ΔSuccess |
| 熵生命周期（证伪事件对齐） | low→high→low；GRPO/ARPO 无结构 |
| Invalidation Precision | 被 eliminate 动作在更大 $K$ 下仍不可恢复 |
| 重复错误率 | 低于负 advantage 基线 |
| 无成功分支有效更新比例 | Agent-R/Fission 分层崩溃；RAE 仍更新 |
| 探针占比–收益曲线 | 少量探针 > 等量盲目分支 |

---

## 8. 消融（搜索域）

| # | 消融 | 主张 |
| --- | --- | --- |
| A1 | $U$ → 仅 $H^\pi$ / 仅 $H^B$ | 双熵必要 |
| A2 | Renorm → 均匀负 / 负 advantage | 比例保持 |
| A3 | 单一 KL → $R_{out}+\lambda_1 R_{IG}+\lambda_2 R_{ent}$ | 无 λ |
| A4 | 去掉失败/矛盾触发 | 覆盖灾难自信 |
| A5 | 乘积 → min / 加权和 | 组合形式 |
| A6 | 全序列 KL → 首分歧 token | 实现权衡 |
| A7 | 去掉 dead-end 回传 | 非对称归因 |
| A8 | $H^B$ → APPO-$\Omega$ / RAIL-方差 | 外部 vs 内部第二因子 |
| A9 | Renorm → Fission 再采 / BranPO 对比 | 支持集 vs 造正样本 |

超参敏感性：$K\in\{2,4,8\}$，$B\in\{4,8\}$，$n\in\{4,8,16\}$，$p_+\in\{0.6,0.8,1.0\}$。

---

## 9. 训练超参见（默认起点；最终以 sweep 日志为准）

| 项 | 默认 |
| --- | --- |
| 框架 | veRL |
| 优化器族 | GRPO / DAPO clip 管线 |
| 学习率 | 1e-6 量级（随 7B 惯例） |
| 组大小 | 8 或 16（全方法锁定） |
| 最大 tool turns | QA: 4–8；GAIA: 按 ARPO 配置 |
| 温度 | 训练采样 1.0；评测 0.0 或官方 |
| KL 到 ref | 按 Search-R1 默认；RAE 主目标为 KL($\pi^{tgt}\Vert\pi_\theta$)，ref-KL 仅作可选稳定项并消融 |

---

## 10. 实现优先级（工程排期）

1. **Week 0**：Search-R1 数据/索引/评测脚本跑通 B0。  
2. **Week 1**：B1 DAPO；B2 ARPO（官方或同栈复现）。  
3. **Week 2**：B3 AEPO；B4 APPO（同 ARPO 缓存）。  
4. **Week 3**：B5 IGPO；B7 RFT；B6 Agent-R-lite。  
5. **Week 4–5**：RAE 核心（探针 + Renorm + $A^{RAE}$）；机制指标仪表盘。  
6. **Week 6**：消融 A1–A9；半预算曲线；写主表。  
7. **Buffer**：RAIL 附录；Qwen3-8B 校验。

---

## 11. 算力粗估（规划用）

| 项 | 粗估 |
| --- | --- |
| 单方法 7B 全量 QA RL | 视组大小与步数；按 ARPO 1K–数 K 样本叙事可先做 **1K 样本诊断跑** |
| GAIA 扩展 | 工具贵；优先用 ARPO 缓存；限制评测子集 |
| 主表 8 方法 × 2 模型 | 建议先 7B 全方法，再挑 top-3 跑 Llama |

---

## 12. 复现 Checklist

- [ ] 固定 Wikipedia 索引哈希 / 版本号  
- [ ] 固定随机种子与评测 split 文件  
- [ ] 所有方法共享冷启动 checkpoint  
- [ ] $M$ 记账代码单元测试（探针计入）  
- [ ] GAIA 子集清单入库  
- [ ] 机制指标脚本与成功率脚本分离  
- [ ] 每个 baseline 一张 config yaml  
- [ ] 禁止评测期换检索器  

---

## 13. 成功标准（实验签字）

1. 同 $M$ 下 RAE 成功率不低于最强公开可复现基线（APPO/AEPO 同设定）。  
2. 机制表上 RR / 熵生命周期 / 无成功侧分层至少两项显著支持主张。  
3. A8 显示外部 $H^B$ 优于 $\Omega$/方差包装；A2/A9 显示 Renorm 优于负标量与造正样本。  
4. Related work 能逐条指向本协议的 B0–B7，无“未比最强近邻”空洞。
