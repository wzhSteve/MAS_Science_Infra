# APPO — Agentic Procedural Policy Optimization

## 1. 出处

- Venue / arXiv / 日期：arXiv:2606.12384（2026-06）
- 代码：[https://github.com/xuc865/APPO（基于](https://github.com/xuc865/APPO（基于) ARPO/VERL 栈）

## 2. Abstract 要点

- 粗粒度单位（tool 边界/workflow）难定位影响结局的中间决策。
- Pilot：关键决策遍布 thinking span；**token 熵不能可靠反映对结局的影响**。
- Branching Score：$\mathrm{BS}=z(H)\cdot z(\Omega)$，$\Omega$ 为后续相对 $\pi_{old}$ 的折扣似然增益。
- Procedure-level advantage scaling；13 benchmark 相对强基线约 +4 / +3 点。

## 3. Intro 诊断的社区盲区

“在哪里分支”若只看工具边界或纯熵，会采到假高熵、漏掉真决策点。

## 4. 方法三层定位


| 层         | 内容                                   |
| --------- | ------------------------------------ |
| Rollout   | 全序列 BS 选点；非仅 tool 后                  |
| Advantage | 双组相对 + future-aware scaling（含超参 $b$） |
| Loss      | GRPO 类 + `future_kl` 等模式             |
| 失败用法      | 组相对标量                                |




## 5. 实验设定线索

- 模型：Qwen2.5-7B-Instruct、Qwen3-8B/14B 等
- Benchmark：与 ARPO 重叠的 13（含 HotpotQA/2Wiki/MuSiQue/Bamboogle/WebWalker/数学）
- Tool：与 ARPO 相同 search/browser/python + tool cache
- 关键参线索：`n=16`, `initial_rollouts=8`, `reward_scale_discount=0.9`



## 6. 对 RAE 四问


| Q   | 判定  | 说明                            |
| --- | --- | ----------------------------- |
| Q1  | ◐   | 乘积门控形式同构；第二因子是**内部** $\Omega$ |
| Q2  | ○   |                               |
| Q3  | ○   | 有 scaling 超参 $b$              |
| Q4  | ○   |                               |




## 7. 与 RAE：同 / 异 / 优 / 劣

- **相似**：明确写“熵 ≠ 对结局的影响”；$H\times(\cdot)$ 乘积；同搜索设定可公平比。
- **不同**：$\Omega$ 是策略内部续写似然比——错误自信同样可抬高；不重构 $\pi^{tgt}$；无失败触发。
- **对方优势**：与 ARPO 同管线，工程成本最低的 P0 第二因子对照；主表数字已在搜索域。
- **对方劣势**：正中 G2——内部增益未经外部 $Y$ 校验。



## 8. Baseline 决策

- **Yes（B4，优先于 RAIL）**
- 理由：搜索域设定对齐；直接证伪“内部第二因子就够”。

