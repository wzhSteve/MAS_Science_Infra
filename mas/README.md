# TIR Agent

单 Agent、LangGraph、Hermes tool-call，训练外壳对齐 [`examples/math_gsm`](../agent-lightning/examples/math_gsm)。工具是 **web_search / wikipedia_search / execute_python**。数据默认 **GSM8K + HotpotQA 各 50%**。

`--algo grpo|arpo|aepo|igpo|gigpo` 只改本目录的 trainer/daemon。Hydra 里 **`algorithm.adv_estimator` 永远是 `grpo`**（关掉 Critic）；真实算法名在 `algorithm.tir_algo`。

这不是论文引擎的 100% 复现。IGPO / GiGPO 落在 triplet + custom advantage 上更接近原文；ARPO / AEPO 用两阶段 enqueue + tool 后前缀续写近似树采样。细节见 [docs/DESIGN.md](docs/DESIGN.md)；与官方 ARPO 仓库的架构 / tool / 熵 fork 对照见 [docs/ARPO_VS_TIR.md](docs/ARPO_VS_TIR.md)。对照笔记 [math_gsm/docs/interview/10_agent_rl_arpo_aepo_igpo_gigpo.md](../agent-lightning/examples/math_gsm/docs/interview/10_agent_rl_arpo_aepo_igpo_gigpo.md)。

## 准备数据

```bash
# 默认走 HF_ENDPOINT=https://hf-mirror.com；先跑通可用 1024 条
python prepare_data.py --train-limit 1024 --val-limit 200

# 无 Hub / 无网（仅算术 + 合成 QA，search 意义不大）
python prepare_data.py --offline --train-limit 32 --val-limit 8

TRAIN_LIMIT=1024 VAL_LIMIT=200 bash scripts/prepare_data.sh
```

检索默认走 `WEB_SEARCH_PROXY=http://127.0.0.1:7890`（与 epc_aw Clash 一致）。先验证三工具：

```bash
python scripts/check_tools.py
bash scripts/debug_tir_agent.sh   # 1 GPU 起 vLLM，确认会调 python / search
```

## 训练

Python 建议：`/root/autodl-tmp/AgentFlow/.venv/bin/python`。`--algo grpo` 走与 math_gsm 相同的股票 `agl.VERL`（不挂自定义 trainer/daemon）。

```bash
bash scripts/train_2gpu.sh                    # 2×A800 vanilla GRPO
python train_tir_agent.py a800_2gpu --algo grpo
```

`--n-runners` / `--model` 与 math_gsm 相同。配置档：`fast` / `a800` / `a800_2gpu`。正式训练 `max_prompt_length=8192`（检索 snippet 更长）。

## MAS_structagent RL（独立入口）

用 Planner→Actor→Verifier 多智能体做 RL，**不改**上方 TirAgent 路径。训练 LLM 走 VERL `main_llm`（默认本地 Qwen3-4B），**不读** `.env` 的 dmxapi。详见 [docs/DESIGN.md §7](docs/DESIGN.md)。

```bash
bash scripts/train_mas_2gpu.sh
python train_mas_agent.py fast
OPENAI_API_BASE=http://127.0.0.1:8000/v1 python mas_agent.py   # 冒烟，不启 VERL
```

## Agent 冒烟（不启 VERL）

需要 `OPENAI_API_BASE`（OpenAI-compatible，例如 vLLM）：

```bash
export OPENAI_API_BASE=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
python tir_agent.py
```

## 奖励

所有 algo 共用层次 outcome，方便对比：

- 格式坏（无 `<answer>...</answer>`）→ `-1`
- 格式好但 Acc=0 → `0`
- Acc>0 → Acc（GSM8K 0/1，QA 为词级 F1∈[0,1]）；若本轨迹 **search 与 python 都用过** → `+0.1`

## 诚实边界

| 算法 | 本目录实际做了什么 |
| ---- | ------------------- |
| GRPO | 与 math_gsm 相同：`rollout.n` 条独立轨迹，轨迹级 outcome Â |
| IGPO | 采样同 GRPO。过程奖励 MVP 用「工具观测是否新出现 GT」作 IG 代理，再组内 z-norm + 折现。不是 actor teacher-forcing `log π(GT)` |
| GiGPO | `A = A^E + ω A^S`，anchor = `hash(tool + obs[:512])`。数学题无重复状态时 `A^S=0`，退化为 GRPO |
| ARPO | 先 enqueue `initial_rollouts` 条完整轨迹，高熵 tool 后带 `resume_messages` 再采；M 条完整轨迹进 soft GRPO。**不改 vLLM 内核**。与官方实现差距见 [docs/ARPO_VS_TIR.md](docs/ARPO_VS_TIR.md) |
| AEPO-lite | 预监控 1 条轨迹分预算 + 连续高熵分支惩罚 + Â 乘熵项。**没有**论文里 clip 的 stop-grad（二期） |

## Workflow collect (no RL / no training GPU)

本仓根目录可装薄 CLI：`pip install -e .` 后 `science-infra collect --mock --out /tmp/traj.json`。`science-infra dashboard` 写 HTML；AGL Dashboard `/science` 可打开同一 JSON。

```bash
PYTHONPATH=. python scripts/collect_rollouts.py --mock --out /tmp/traj.json
PYTHONPATH=. python scripts/check_workflow_deps.py
PYTHONPATH=. python -m unittest discover -s tests -v
```

See `docs/DESIGN.md` §8 and the `workflow/` package.
