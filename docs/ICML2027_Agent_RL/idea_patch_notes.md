# 对 `idea.md` 的最小修订建议（不直接大改正文）

> 日期：2026-09-14  
> 原则：保持 method-only draft 结构；只列**必改/建议改**条目，待你确认后再改 `idea.md`。  
> 依据：`Agent_related_work_baselines.md` v1.1、`literature_cards/`、`experiment_protocol_search.md`。

---

## P0（强烈建议，影响 novelty 防御）

1. **§2.7 定位表过时**  
   - 加入：RAIL、APPO、IGRPO、Fission-GRPO、BranPO、BPO、NGRPO、AEPO。  
   - 原句“尚无方法把失败定义为删除被证伪动作后重归一化”**可保留**，但前句须承认“选点侧已被 RAIL/APPO 逼近”。

2. **Related work 开篇强制 RAIL 段**  
   - 使用 [`literature_cards/RAIL.md`](literature_cards/RAIL.md) 中的强制区分句（方差 recoverability ≈ $\bar H^B$，但停在 Rollout / 仍 GRPO）。  
   - 避免审稿人因缩写撞车 desk-level 误读。

3. **§2.2 增补 APPO 乘积门控**  
   - 写明：$\mathrm{BS}=z(H)\cdot z(\Omega)$ 与 $U=\bar H^\pi\bar H^B$ 形式同构；差别在第二因子（内部似然增益 vs 外部探针结局熵）。

4. **§2.4 失败利用：Fission-GRPO 升为第一对照**  
   - Agent-R 保留为经典代表；强调“同诊断、反处方”。

5. **§7.2 基线列表替换为搜索定稿**  
   - 采用 `experiment_protocol_search.md` 的 B0–B7（Search-R1/GRPO、DAPO、ARPO、AEPO、APPO、IGPO、Agent-R-lite、RFT）。  
   - 删除或降级主文中的 GiGPO/ALFWorld 必跑表述 → future work。

6. **§7.1 环境表：主场改为搜索**  
   - 主：HotpotQA / 2Wiki / Bamboogle / GAIA 子集。  
   - 具身与 SWE：一句“可扩展性验证，非 ICML 主表”。

7. **命名提示（摘要首次出现）**  
   - 全称强调 **Elimination**（counterfactual policy Elimination），降低与 RAIL Recoverability 的混淆。不强制改缩写 RAE。

---

## P1（建议，增强可审稿性）

8. **§7.4 机制指标**  
   - 增加“无成功分支时的有效更新比例”分层（已在 baselines §4.4）。

9. **§7.5 消融**  
   - 新增 A8（$H^B$ → $\Omega$ / 组方差）、A9（Renorm → Fission / BranPO）。

10. **§6.1 / 讨论引用 Dark Room (2607.21273)**  
    - 支撑“$U$ / IG 不进 GRPO reward channel、走 KL 到 $\pi^{tgt}$”。

11. **模型表述**  
    - 主文 Qwen2.5-7B 对齐 ARPO/Search-R1；另加一句将跑 Qwen3-8B 校验（避免“过时骨干”攻击）。

12. **增量文献一笔带过**  
    - Belief-Shift Branching (2609.11061)：又一例内部信念选 fork → 归入 §2.2 第二因子段 cite。

---

## P2（可选）

13. 摘要中“覆盖具身/网页/搜索三类环境”→ 改为“以搜索/多跳工具推理为主，具身与代码为扩展”。  
14. 参考文献块与 `Agent_related_work_baselines.md` §8 对齐，补 2026-09 增量。  
15. 在 §7 末链到 `experiment_protocol_search.md`（若投稿附录允许放协议摘要）。

---

## 明确不改（除非你另指示）

- §6 方法主体（双熵、Renorm、单一 KL、定理 1）——深读后**无需改主张**，只需加强与 APPO/RAIL 的对照措辞。  
- 不把 Revolution 文档的 Responsibility-Aware 叙事混入本 idea。

---

## 建议执行顺序

1. 你确认本清单 P0 条目 → 再改 `idea.md` §2 / §7。  
2. Related work 成文按 baselines §9 提纲。  
3. 实验工程严格按 `experiment_protocol_search.md`，避免与 idea 旧三类环境表打架。
