# Plan

本阶段收口 GRPO、ARPO、AEPO、IGPO、GIGPO 和 RAE 的最终配置解析、运行展示与服务器人工验收。算法实现继续位于 `rl/hooks/` 和 `rl/loss.py`，新 UI 只编辑配置、展示最终有效值并驱动既有训练入口。

依赖前三个训练运行阶段。真实 GPU 训练有成本且耗时，默认自动验证只做编译和纯配置检查；所有会占用 A800、启动 VERL 或写入 checkpoint 的步骤由用户从 WebUI 明确触发。

## Position in the Project

算法配置不应由 UI、Control 和训练脚本各自解释一遍。必须形成唯一优先级并固化到本次运行快照：

```text
profile 基线
  → 实验 rl.yaml
  → Workflow Sampling / Branch Sites
  → GPU 选择
  → 训练模型来源
  → algorithm overlay
  → effective-rl.yaml
  → mas/train_tir_agent.py
  → Agent-Lightning / VERL / rl/hooks
```

`algorithm.adv_estimator=grpo` 是 VERL 无 Critic 入口的兼容设置；真实算法是 `algorithm.tir_algo`。UI 不得因为底层字段为 `grpo` 而把 ARPO、AEPO、IGPO、GIGPO 或 RAE 显示成 GRPO。

## Requirements

- 六种算法通过统一解析入口产生有效配置。
- Sampling 指定算法时，它是最终算法来源，RL 控件只读并说明来源。
- `rollout_per_gpu`、`actor_rollout_ref.rollout.n` 与 `sampling.group_n` 不得相互漂移。
- Branch Sites、Gate、Reward Scheme 和算法专属参数无损进入快照。
- 启动检查、运行页标题、日志和快照展示同一个最终算法。
- 不支持的参数组合在启动前阻断，不依赖训练运行数分钟后失败。
- 算法实现仍由 `rl/hooks/` 承担，Control 和前端不复制 advantage/loss 逻辑。
- ARPO 先恢复 main 已验证的服务器闭环，再按顺序验证其他算法。
- 每次真实训练必须由用户明确启动；自动化不偷偷占用 GPU。

## Scope

- In：算法解析优先级、UI 来源说明、参数矩阵、快照字段和人工 GPU 验收清单。
- In：GRPO、ARPO、AEPO、IGPO、GIGPO、RAE 的已有 hooks 接线核对。
- In：启动、停止、失败、刷新恢复、日志和产物的算法级闭环。
- Out：发明新的 RL 算法、改写 VERL、自动超参搜索和性能基准平台。
- Out：仅凭配置保存成功宣称算法训练正确。

## Current State

- `VALID_ALGOS` 已包含六种算法。
- `apply_algo_overlay()` 固定 VERL `adv_estimator=grpo`，使用 `tir_algo` 分派真实行为。
- `apply_sample_policy()` 已映射 Sampling mode、`group_n`、beam、初始轨迹、深度和 sites。
- `services.start_train()` 会先同步 Workflow Sampling 到 RL，再调用 `train_tir_agent.py`。
- `train_tir_agent.py` 仍会读取相邻 `workflow.yaml` 再应用一次 Sampling；第二阶段快照化后应改为读取本次快照，保证两次解析结果一致。
- ARPO 已在服务器真实跑通过；其他算法已有实现或测试，但不能据此宣称当前新 UI 的完整真实闭环已经通过。
- 当前 UI 已在 Sampling 决定算法时禁用 RL Algorithm，但尚未在启动检查和运行详情中完整解释最终来源。

## Canonical Algorithm Resolution

建议将纯函数集中在训练配置模块，输入 bundle、模型来源和 GPU 选择，输出 `EffectiveTrainingConfig`：

```text
algorithm
algorithm_source: workflow.sampling | rl
profile
gpu_ids
model_source
rollout_n
tir
data
trainer
warnings
```

解析规则：

1. 读取 profile 基线。
2. 深合并已保存 `rl.yaml`。
3. 应用 Workflow Sampling；其 mode、group 和 sites 优先。
4. 应用选中 GPU，覆盖 `trainer.n_gpus_per_node`。
5. 固定模型来源到 `actor_rollout_ref.model.path`。
6. 最后调用一次算法 overlay，补齐算法默认值但不覆盖用户显式值。
7. 写入快照并生成校验值。

Control Preflight、启动快照和 `train_tir_agent.py` 应复用该结果。训练脚本保留 CLI 兼容，但通过 Control 启动时不再自行从活动实验目录重新推导另一份配置。

## Algorithm Matrix

| 算法 | 核心路径 | 启动前重点 | 运行证据 |
| --- | --- | --- | --- |
| GRPO | Stock AGL/VERL | `group_n`、batch 可整除、无 branch 强依赖 | `tir_algo=grpo`，无二波 branch |
| ARPO | `TirAgentModeDaemon` + branch/resume | sites、初始轨迹、beam、Gate、ready batch | `tir_algo=arpo`、enqueue 和 branch artifact |
| AEPO | ARPO 扩展 + entropy reshape | 全局预算、`aepo_beta`、分支规模 | `tir_algo=aepo`、预算与 entropy 指标 |
| IGPO | turn-level advantage | turn/step 记录完整、`gamma` | `tir_algo=igpo`、turn advantage |
| GIGPO | episode + step advantage | step group、`step_advantage_w`、mode | `tir_algo=gigpo`、episode/step 指标 |
| RAE | verdict + tree credit | reward scheme、verdict、dead-end 参数 | `tir_algo=rae`、verdict 与回溯指标 |

ARPO、AEPO、RAE 依赖 Branch/Tree 语义；IGPO、GIGPO 依赖步骤级记录。缺少必要事件或字段时应在 Preflight 中警告或阻断，不能静默退化成 GRPO。

## UI Behavior

启动检查固定展示：

```text
最终算法：ARPO
来源：Workflow Sampling Policy
VERL advantage estimator：GRPO（兼容层）
group_n：4
初始轨迹：2
Branch Sites：1
Gate：Official ARPO Gate
```

- 算法来源不是 Sampling 时，显示“来自 RL 配置”。
- Sampling 与 RL 字段冲突时显示已采用值和被覆盖值。
- 训练工作区标题、运行历史和配置快照均展示最终算法。
- 算法专属字段只在相关算法下出现；共享字段保持同一控件和保存路径。
- 不用“支持”标签替代真实验收状态，算法可标记为“配置已接入 / 服务器已验证”。

## Manual GPU Acceptance

真实训练按以下顺序由用户操作：

1. 在训练工作区查看 Preflight，全绿后确认启动。
2. 控制台确认模型路径、GPU、`tir_algo` 和有效 Sampling 摘要。
3. 观察至少一个训练 step 或算法特定事件。
4. 对支持分支的算法查看 Rollout Tree/branch artifact。
5. 中途停止一次，确认无残留子进程。
6. 再次启动并允许完成，确认终态、日志与指标可恢复。

优先顺序：

1. GRPO 基线。
2. ARPO 主闭环。
3. AEPO。
4. IGPO。
5. GIGPO。
6. RAE。

每次验证记录 run ID、Git commit、有效配置校验值和结果，不把一次算法的成功推断到其他算法。

## Files and Entry Points

| 文件或区域 | 调整职责 |
| --- | --- |
| `science_infra/control/training.py` | 唯一有效训练配置解析 |
| `science_infra/control/services.py` | 调用解析结果并启动快照 |
| `mas/train_tir_agent.py` | 消费有效快照，保留 CLI 兼容 |
| `rl/hooks/overlay.py` | 六算法默认与 Sampling 映射 |
| `rl/hooks/trainer.py`、`daemon.py` | 算法分派、Branch/Resume |
| `rl/hooks/advantage.py`、`rae_advantage.py` | advantage 与 RAE 行为 |
| `rl/loss.py` | loss/credit 合同 |
| `features/rl` | 编辑算法参数与来源说明 |
| `features/sampling` | 最终 Sampling 与 Branch Sites |
| `features/training` | Preflight、有效配置和验收状态展示 |

## Action Items

- [ ] 建立唯一的有效训练配置解析顺序。
- [ ] 消除 Control 与训练脚本对 Sampling 的双重活动文件读取。
- [ ] 在 Preflight、快照、日志和 UI 中统一最终算法及来源。
- [ ] 核对六算法所需字段并在启动前报告缺失。
- [ ] 保留 `adv_estimator=grpo` 兼容语义并在 UI 解释。
- [ ] 完成 GRPO 和 ARPO 的服务器人工闭环。
- [ ] 依次完成人工 AEPO、IGPO、GIGPO、RAE 验收。
- [ ] 将每种算法的“配置已接入”和“服务器已验证”分开记录。

## Lightweight Validation

默认开发循环只运行快速检查：

```bash
python -m compileall science_infra/control mas rl
cd webui && npm run build
```

算法映射可使用纯配置测试验证，不导入 GPU 训练栈。任何 `train_tir_agent.py`、VERL、vLLM、Ray 或 A800 训练命令均不作为默认自动验证，由用户在 WebUI 中明确启动。

## Acceptance and Delivery Boundary

每种算法至少满足：

- UI 保存后快照中的最终配置正确。
- 控制台日志显示对应 `tir_algo`。
- 启动、停止、失败和完成状态正确。
- 页面刷新后仍能恢复日志、状态和配置快照。
- 非零退出显示真实错误阶段。
- 中途停止不遗留训练子进程。
- 算法特有的分支、advantage、verdict 或指标证据可查看。

只有完成某算法的服务器人工闭环后，才能把它标记为“新 UI 已验证”。完成本阶段后才可在 `MAIN_FUNCTION_INVENTORY.md` 将“训练启动与停止”和“RL 算法”标为完成。

## Risks and Edge Cases

- `sampling.mode` 与 `rl.algo` 双源会产生误导，必须展示最终来源并冻结快照。
- 单卡 A800 的 profile、batch 和显存参数不能直接套用双卡结果。
- 训练日志出现算法名不等于算法行为正确，还需对应分支或 advantage 证据。
- 真实训练失败可能来自环境、模型、数据或算法，UI 应按阶段分类而不是统一显示“启动失败”。

## Next Phase

完成训练闭环后，继续把 run 级 Rollout Tree、实时指标和 Harness 事件接入同一训练工作区；这些能力仍遵循 [NEW_FRAMEWORK_DESIGN.md](../../docs/NEW_FRAMEWORK_DESIGN.md) 的跨层合同。
