# 合并迁移功能步骤

本地修改 → 经确认后提交推送 → 服务器拉取 → 按改动类型启动或重建 → 验证 GPU 和真实训练。

- **服务器项目目录**：`/root/autodl-tmp/MAS_Science_Infra`
- **分支**：在 `integration/new-webui` 进行迁移合并。

## 服务器操作

1. 在代码同步终端拉取已确认并推送的提交：

   ```bash
   cd /root/autodl-tmp/MAS_Science_Infra
   git pull --ff-only origin integration/new-webui
   ```

2. 根据改动类型启动服务：

   ```bash
   # 仅重新启动已有版本，不重新构建
   ./run.sh ui --daemon

   # Python 后端代码有更新：先停再启动，不重新构建前端
   ./run.sh ui --stop
   ./run.sh ui --daemon

   # webui 前端源码有更新：停止后重新构建并启动
   ./run.sh ui --stop
   ./run.sh ui --rebuild --daemon
   ```

3. 不要每次都重新安装依赖。只有 `node_modules` 缺失、不完整，或 `package.json`
   的依赖发生变化时，才执行一次依赖安装。服务器安装 npm 依赖必须使用公网镜像，
   并忽略锁文件中可能存在的内网镜像地址：

   ```bash
   cd /root/autodl-tmp/MAS_Science_Infra/webui
   npm install --include=dev --package-lock=false \
     --registry=https://registry.npmmirror.com/ \
     --no-audit --no-fund
   ```

   依赖完整且 `webui/dist` 已对应当前前端源码时，直接运行
   `./run.sh ui --daemon`，不要重复安装或重建。

---

# 实施方案：恢复 main 的 arpo_e2e 在新版入口下正常训练

更新日期：2026-09-22。

**功能基准固定为 main 的 `735b1140f6b80cdac04f9678288b3921a3fc7a3b`。目标是在新 UI 中保留这版 main 的训练行为，不在此轮建设完整新框架。**

本方案依据该基准与当前 `integration/new-webui` 的代码差异直接安排修改，不设置“先收集现场”的前置阶段。只修改接线和兼容性缺口，保留已经完成的新 UI、资源选择、运行快照和日志。

## 一、已经查清的接线变化

| 改动 | 与 main 的差异 | 本轮处理 |
| --- | --- | --- |
| `3400cf5`：模型配置解析 | 原 `_llm_env()` 只条件覆盖实验配置；现在使用 `resolve_llm_config().subprocess_env()`，默认推理资源参与训练环境，空密钥也会覆盖原环境 | 训练启动恢复 main 的条件覆盖规则，与独立推理资源解析分开 |
| `95fb48d`：训练入口替换 | main 的 `services.start_train()` 实现被改为调用新 `training.py`；新 UI 改用 `/api/rl/runs`。旧 URL 虽保留，内部也不再是原实现 | 恢复原启动参数生成语义，让新旧 API 共享同一个启动核心；仅把按钮改回旧 URL 不足以恢复 |
| 配置同步路径分叉 | main 使用 `_sync_workflow_sampling_into_rl()` 同步顶层 `algo`、`tir_algo` 与采样数；新计划构造直接调用 `apply_sample_policy()`，没有完全复用原同步步骤 | 复用原同步逻辑，避免快照、CLI 参数和训练脚本各自决定算法 |
| `95fb48d`：重启后停止能力缩减 | main 的 `ProcessManager.stop()` 可从磁盘运行记录查找存活 PID；当前旧活动记录统一显示 `interrupted`，停止只查内存进程 | 恢复可验证的运行识别与定向停止，不让遗留训练/模型服务被当作不存在 |
| 模型来源扩展 | main 读取 RL 模型路径；当前 training resource 优先 | 保留资源选择，但只把所选权重转换成原训练入口使用的模型路径 |
| Gate 参数扩展 | `rl/hooks/daemon.py`、`lit_tir_agent.py` 增加 entropy threshold 传递，`ActiveSetSession._sites()` 增加参数继承 | 不继续扩大采样语义；按 main 基准核对本次 ARPO 的实际站点参数 |

以下部分没有缺失，不重写：Agent-Lightning 整个目录、ARPO trainer/advantage/loss 主体、模型训练主体。仓库 `arpo_e2e` 的 RL、LLM 和 Sampling 配置与 main 语义一致；服务器数据路径也已经恢复。

**确定的问题是：迁移改变了训练入口的行为，而不只是 UI 对接。上述差异应直接修复；但不能把其中某一项未经运行确认就写成 503 的唯一原因。**

## 二、阶段一：恢复 main 的模型与训练启动接线

目标：相同实验配置得到与 main 等价的训练命令和环境，资源库只提供明确选择的训练权重。

### 1. 恢复训练专用环境构造

修改 `science_infra/control/training.py` 的 `_launch_training()`：

- 不再用 `resolve_llm_config(...).subprocess_env()` 直接构造训练环境。
- 用局部训练环境函数复用 main `_llm_env()` 的规则：读取原实验 LLM 配置与实验密钥；只有非空值才覆盖，未设置的值继续继承启动进程环境。
- 读取原实验 LLM section，而不是 `load_bundle()` 中已经被 inference resource 替换的展示配置。
- 保持 main 的 `PYTHONPATH`、`CUDA_VISIBLE_DEVICES`、`VLLM_USE_V1`、Python 可执行文件和工作目录行为。
- 不改通用推理资源的凭据策略；单题调试与模型资源管理继续使用当前解析方式。
- 训练主模型的实际服务端点和服务模型名仍由 AGL `main_llm` 注入，不能被页面的默认推理资源替换。

### 2. 恢复原参数同步与启动核心

修改 `training.build_training_plan()`、`services._sync_workflow_sampling_into_rl()` 和 `services.start_train()`：

- 新计划构造复用 main 的 GPU 应用、服务器路径处理、Sampling 同步和算法/profile 校验。
- 统一写入一致的 `algo`、`algorithm.tir_algo`、`rollout_per_gpu` 和 `actor_rollout_ref.rollout.n`。
- 没有 training resource 时保留原 `rl.model_path` / `actor_rollout_ref.model.path` 语义；有绑定时仅明确覆盖这两个权重字段及 `--model`。
- 保留 `--n-runners`、`--active-agent` 和 main 的 profile 行为，不擅自调整 batch、显存比例、Runner 数、模型名称或分支参数。
- 保留当前运行快照；`--rl-yaml` 指向本次 RL 快照，`--workflow-yaml` 指向本次 Sampling 来源。改变文件位置，不改变配置含义。
- `/api/rl/train` 与 `/api/rl/runs` 最终调用同一个启动核心，不增加 main 模式、新模式两套实现。

### 3. 冻结算法执行层

`mas/train_tir_agent.py` 保留 main 的 profile 合并、AGL/VERL 构造与 `trainer.fit()` 主体；不改 Agent-Lightning。

此次不把完整 Workflow spec 接入训练，不扩大多 Agent 训练范围，不把 Gate 继承或其他框架演进混进恢复补丁。对基准实验，将新增参数映射与 main 的实际取值对齐即可，不整体回退新 UI 或已保存 schema。

**本阶段交付：新版入口使用 main 等价的启动核心，模型资源不再隐式改变训练推理环境；不需要先完成其他框架阶段。**

### 阶段一实施记录

代码已接入，服务器真实训练结果待确认：

- `_launch_training()` 改用训练专用 `_training_env()`，读取原始实验 LLM section 和实验密钥；只有非空值覆盖父进程环境。独立 inference resource 不再参与训练环境构造，通用推理凭据逻辑不变。
- `build_training_plan()` 复用 main 的 `_sync_workflow_sampling_into_rl()`。有 Sampling 时由它决定算法；无 Sampling 时按 main 的顶层 `algo` 决定，同时对齐 `tir_algo` 与两处采样数。
- 补回算法与 profile 检查。保留单 GPU 的原 profile 降档行为并显示警告；内部 `_profile_downgraded` 标记不再进入训练快照，避免成为 Hydra 未知配置。
- 模型权重、GPU、Runner、active Agent 和快照传参保留；Preflight 命令预览补齐 Workflow 快照和 active Agent，与真实命令一致。
- 新旧 API 仍共用 `build_training_plan()` / `launch_training()`。没有增加入口或兼容开关，没有修改 AGL/VERL、算法、Gate、UI 或进程管理。

服务器更新本轮 Python 代码后，按上方流程停止并重启 Control 即可，无需重建前端或安装依赖。使用未改参数的 `arpo_e2e`，预期启动摘要为 `fast / arpo`、GPU `0`、Runner `1`、候选数 `4`、初始 Rollout `2`，模型使用原 Qwen3-4B 路径（若显式绑定了其他训练模型，以绑定为准）。

阶段一已消除上述启动接线差异，但不等于已经证明 503 消失。成功运行应有真实响应及有效 token，并越过 `compute_log_prob` 进入参数更新；如果仍持续 503，则仍未达到 ARPO 恢复判据，不能把空响应任务的 Completed 计数当成成功。

本轮附加少量 `[TIR-DIAG]` 检查点，不修改请求、代理环境或重试行为：

- `daemon-to-vllm` / `proxy-config`：每批次记录训练后端地址和实际代理配置的模型映射。
- `runner`：每个 Runner 首次采样记录 AGL 注入的请求端点。
- `failure`：每进程首次模型异常记录实际请求 URL、状态码、限定响应头和最多 1000 字符的脱敏响应摘要，不输出请求正文或认证头。
- `env_proxies` 为脱敏代理地址，`env_bypass_hint` 只是标准库对环境代理绕过规则的判断，不是 HTTP 客户端实际走向的证明。浏览器端口转发不参与这条服务器内部模型链路。

## 三、阶段二：补回 main 的运行识别与停止对接

目标：保留当前按 run 停止和子进程清理，同时恢复 main 原有的重启后运行识别能力。

修改 `science_infra/control/process_manager.py` 和 `training.stop_training_run()`：

- 新运行持久化 PID、进程创建时间、进程组和必要的命令身份，关联 experiment ID 与 run ID。
- 读取磁盘活动记录时，校验实际进程身份；不能不作判断就把所有旧活动运行变成 `running=False`。
- 对确认仍存活、属于本应用的训练和模型服务，恢复活动冲突识别及按 run 定向停止。
- 复用已有的中断、终止、强杀和已识别后代清理，不新增进程管理框架。
- 无法验证身份的旧记录明确提示处理，不能仅凭历史 PID 杀进程；禁止按名称全局结束 Ray、vLLM 或 Python。
- 启动前发现旧训练仍活动时阻止重复启动；需要释放本地模型服务时，仅停止已确认的目标服务，不影响新训练内部的 vLLM。

**本阶段交付：重启 Control 后不会把可识别的旧运行当作不存在，也不会因新 UI 的停止入口改成 run ID 而丢失原有停止能力。**

### 阶段二实施记录

代码已接入，服务器需用“训练中重启 Control”场景验收：

- 新运行在 `status.json` 持久化 PID、创建时间、进程组、完整 argv 的 SHA-256 和工作目录，不保存环境变量或凭据。
- 磁盘中的活动状态只有在 PID、创建时间、命令指纹、工作目录和进程组全部匹配时才恢复为 `running`；旧格式或不匹配记录继续显示 `interrupted`，不会仅凭 PID 停止进程。
- `active()`、活动训练查询、启动冲突检查和本地模型服务冲突检查统一识别已恢复进程。
- 按 run 停止时按需采用磁盘记录，随后复用现有 SIGINT → SIGTERM → SIGKILL 和后代进程清理；停止的 experiment ID 与 run ID 必须匹配。
- 已恢复进程自然结束后标记为 `interrupted`，因为新的 Control 无法取得原进程退出码；日志仍可继续读取。
- 未修改训练参数、算法、WebUI、Agent-Lightning 或其他功能模块。

## 四、阶段三：新 UI 薄适配与 ARPO 恢复确认

目标：新 UI 只保存配置、调用恢复后的后端函数、显示对应 run，不另行推导训练行为。

### 接线收口

- `RuntimeProvider.startTrain()` 保留“保存 → Preflight → 确认 → 创建 run”，但后端检查与启动必须使用阶段一的同一套参数生成逻辑。
- 创建成功后只使用服务端返回的 run ID 打开日志；停止同一个 run，不重新寻找或替换训练目标。
- 顶栏“已启动”只表示进程启动，不代表已经完成有效采样或参数更新。
- 失败通过已有 run 状态和完整日志显示，不再扩展页面布局或增加管理层级。
- 启动日志只补必要的实际模型、算法、GPU、Runner、端点摘要和原始异常，不建立额外诊断系统，不输出密钥。

### 完成判据

本地只编译受影响代码；按开头流程同步服务器，由用户在新 UI 显式运行 `arpo_e2e`：

1. 使用与 main 基准一致的权重、数据、GPU 和 ARPO 参数。
2. 模型实际返回有效响应及 token，不是持续 503 后以 `None` 答案完成任务。
3. 至少完成一个真实训练 step，越过 `compute_log_prob` 并执行参数更新，正常运行最终退出成功。
4. 停止当前运行后，目标训练及已识别子进程退出，可以再次启动。

运行记录、快照和日志沿用现有产物，不把另行收集材料设为实施前置任务。若按上述接线恢复后仍有 503，则针对已保存的原始异常修复具体请求环节，不能宣称接口接通就已完成恢复。

**本阶段交付：main 已能完成的 ARPO 训练可从新版入口启动、更新、结束和停止。**

### 阶段三实施记录

代码接线已收口，服务器保留主动停止验收：

- RuntimeProvider 只表示当前活动训练，不再把历史选择混入全局训练状态；历史记录和控制台继续以 URL 中的 run ID 读取指定运行。
- 启动成功或恢复幂等请求后，只使用服务端返回的 run ID 打开控制台；启动提示明确为“训练进程已启动”，不表达采样或更新已经成功。
- 停止命令在 TypeScript 接口上强制传入 run ID；训练控制台停止当前显示且服务端确认仍在运行的同一 run，不回退到猜测的活动训练。
- 运行摘要补充服务端固定的 Runner 数、候选数和训练对象；算法、GPU、快照、日志和错误仍来自同一 run。
- 已有服务器运行已确认 Qwen3-4B 有效采样、ARPO branch/resume、`compute_log_prob`、`actor/pg_loss`、`global_step=1` 和正常退出。剩余验收是从 UI 主动停止一次当前运行，确认 cancelled、显存释放且可再次启动。

## 五、本轮不混入的工作

- P0–P3、完整 Workflow 快照执行、多 Agent 独立训练、两级 reward、实时 Harness、树页面与其他算法留到 ARPO 恢复之后。
- 模型异常转 `None`、空 token 仅告警是 main 已存在的异常处理缺陷，不是已经确认的迁移新增原因。不能用“更早拒绝空样本”代替恢复模型服务；需要处理时限定在我方训练适配层，不重写 AGL/VERL。
- 不重新安装整个训练环境，不自动跑真实 GPU 训练，不为 Windows 的 GPU 或服务器路径报错追加适配。

**执行顺序：先改回训练入口的兼容行为，再补运行管理对接，最后由新 UI 完成 ARPO 闭环。三个阶段代码均已接入；阶段二的重启恢复/停止与阶段三的 UI 主动停止仍待服务器验收。**

## 六、ARPO 恢复之后的实施顺序

本方案只恢复 `main` 的 ARPO 训练基线；不能据此把 Sampling 的任意站点、完整 Rollout Tree UI 或六种算法的服务器闭环标为完成。下一轮按以下独立方案推进，同时保留上文尚待服务器验收的停止/恢复场景：

1. [采样编排第一阶段：可执行站点与续跑前缀](../webui/plan/采样编排第一阶段-可执行站点与续跑前缀.md)
2. [采样编排第二阶段：画布与策略编辑](../webui/plan/采样编排第二阶段-画布与策略编辑.md)
3. [Rollout Tree 第一阶段：按 run 存储与结果回填](../webui/plan/RolloutTree第一阶段-按run存储与结果回填.md)
4. [Rollout Tree 第二阶段：节点详情与实时更新](../webui/plan/RolloutTree第二阶段-节点详情与实时更新.md)
5. [训练 T4：RL 算法收口与真实训练验收](../webui/plan/训练运行第四阶段-RL算法收口与真实训练验收.md)

以 [Main 核心功能清单](./MAIN_FUNCTION_INVENTORY.md) 记录各项的实际完成状态；[new_framework 设计](./NEW_FRAMEWORK_DESIGN.md) 是目标合同，未有对应事件、快照和运行产物时不视为代码已实现。
