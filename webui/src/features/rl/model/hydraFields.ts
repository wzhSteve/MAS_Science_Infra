import type { RlConfig } from '../../../shared/api/types';

export type HydraGroup = 'Optimization' | 'Rollout' | 'Data' | 'Trainer' | 'Advantage' | 'Reward';

export interface HydraField {
  path: string;
  label: string;
  description: string;
  group: HydraGroup;
  type: 'number' | 'text' | 'toggle' | 'select';
  options?: string[];
  visible?: (rl: RlConfig) => boolean;
  hint?: string | ((rl: RlConfig) => string | undefined);
  read: (rl: RlConfig) => number | string | boolean | undefined;
  update: (rl: RlConfig, value: string) => RlConfig;
}

export const HYDRA_GROUPS: Array<{ id: HydraGroup; label: string; description: string }> = [
  { id: 'Optimization', label: '优化参数', description: 'Actor 更新步长、PPO Batch 和 Loss 约束。' },
  { id: 'Rollout', label: 'Rollout 资源', description: 'vLLM 显存、Tensor Parallel 和 Log-prob Batch。' },
  { id: 'Data', label: '数据', description: '训练/验证数据路径与 Context Length。' },
  { id: 'Trainer', label: '训练过程', description: '训练轮次、步数和日志标识。' },
  { id: 'Advantage', label: 'Advantage 参数', description: '基础优势估计器与各算子的 tir 参数（RLDesign §4.2 表 2）。' },
  { id: 'Reward', label: 'Reward 组件', description: 'Outcome reward 组件权重与开关（RLDesign §4.2 表 3）。' },
];

function readPath(rl: RlConfig, path: string): unknown {
  return path.split('.').reduce<unknown>((value, key) =>
    value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined, rl);
}

function updatePath(rl: RlConfig, path: string, value: number | string | boolean | undefined): RlConfig {
  const keys = path.split('.');
  const root: Record<string, unknown> = { ...rl };
  let cursor = root;
  for (const key of keys.slice(0, -1)) {
    const child = cursor[key];
    cursor[key] = child && typeof child === 'object' && !Array.isArray(child)
      ? { ...(child as Record<string, unknown>) } : {};
    cursor = cursor[key] as Record<string, unknown>;
  }
  if (value === undefined) delete cursor[keys[keys.length - 1]];
  else cursor[keys[keys.length - 1]] = value;
  return root;
}

function field(
  path: string,
  label: string,
  group: HydraGroup,
  description: string,
  type: HydraField['type'] = 'number',
  extras: Pick<HydraField, 'visible' | 'hint' | 'options'> = {},
): HydraField {
  return {
    path, label, group, description, type, ...extras,
    read: (rl) => {
      const value = readPath(rl, path);
      if (type === 'toggle') return typeof value === 'boolean' ? value : undefined;
      return value as number | string | undefined;
    },
    update: (rl, value) => updatePath(rl, path, value === '' ? undefined
      : type === 'number' ? Number(value)
      : type === 'toggle' ? value === 'true' || value === 'on'
      : value),
    visible: extras.visible,
    hint: extras.hint,
    options: extras.options,
  };
}

export const HYDRA_FIELDS: HydraField[] = [
  field('actor_rollout_ref.actor.optim.lr', '学习率', 'Optimization', 'Actor Optimizer 的 Learning Rate。'),
  field('actor_rollout_ref.actor.ppo_mini_batch_size', 'PPO Mini-batch', 'Optimization', '每次 PPO Update 使用的样本数。'),
  field('actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu', 'Actor 单卡 Micro-batch', 'Optimization', '单张 GPU 一次 Forward / Backward 的样本数。'),
  field('actor_rollout_ref.actor.clip_ratio_low', 'PPO Clip 下限', 'Optimization', 'Policy Ratio 的下侧 Clip 范围。'),
  field('actor_rollout_ref.actor.clip_ratio_high', 'PPO Clip 上限', 'Optimization', 'Policy Ratio 的上侧 Clip 范围。'),
  field('actor_rollout_ref.actor.entropy_coeff', 'Entropy 系数', 'Optimization', 'Entropy Bonus 在 Loss 中的权重。'),
  field('actor_rollout_ref.actor.kl_loss_coef', 'KL Loss 系数', 'Optimization', 'KL Loss 在总 Loss 中的权重。'),
  field('actor_rollout_ref.rollout.tensor_model_parallel_size', 'Tensor Parallel 数', 'Rollout', '单个 vLLM 实例使用的 GPU 数。'),
  field('actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu', 'Rollout 单卡 Log-prob Batch', 'Rollout', 'Rollout Policy 单卡计算 Log-prob 的 Micro Batch。'),
  field('actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu', 'Reference 单卡 Log-prob Batch', 'Rollout', 'Reference Policy 单卡计算 Log-prob 的 Micro Batch。'),
  field('actor_rollout_ref.rollout.gpu_memory_utilization', 'vLLM 显存占比', 'Rollout', 'vLLM 可使用的单卡显存比例。'),
  field('data.train_files', '训练数据', 'Data', '训练数据的 Parquet 路径。', 'text'),
  field('data.val_files', '验证数据', 'Data', '验证数据的 Parquet 路径。', 'text'),
  field('data.train_batch_size', 'Training Batch Size', 'Data', '每个 Training Batch 包含的样本数。'),
  field('data.max_prompt_length', 'Prompt 最大长度', 'Data', 'Prompt Token 上限。'),
  field('data.max_response_length', 'Response 最大长度', 'Data', 'Response Token 上限。'),
  field('data.truncation', '截断策略', 'Data', '超过长度限制时使用 error、left 或 right。', 'text'),
  field('trainer.total_epochs', 'Epochs', 'Trainer', '完整遍历训练数据的次数。'),
  field('trainer.total_training_steps', 'Training Steps', 'Trainer', '显式限制 Optimization Step 数。'),
  field('trainer.test_freq', '评估间隔', 'Trainer', '每隔多少 Training Step 执行验证。'),
  field('trainer.nnodes', '训练节点数', 'Trainer', '分布式训练使用的机器数量。'),
  field('trainer.project_name', 'Project 名称', 'Trainer', 'Logger 中的 Project 标识。', 'text'),
  field('trainer.experiment_name', 'Run 名称', 'Trainer', 'Logger 中的本次 Run 标识。', 'text'),
  // ---- RL6 L0 补全（RLDesign §4.2 表 1 剩余：KL 三态 + tir.* loss 键）----
  field('actor_rollout_ref.actor.use_kl_loss', 'KL 进损失开关', 'Optimization',
    '开启后以 KL Loss 约束策略偏离参考分布；flow_grpo 预设会强制 True/0.001（显式 False 存活）。', 'toggle',
    { hint: (rl) => (Number(rl.actor_rollout_ref?.actor?.kl_loss_coef ?? 0) > 0 && rl.actor_rollout_ref?.actor?.use_kl_loss !== true)
      ? 'KL Loss 系数已设置但 use_kl_loss 未开启，该系数不会生效。'
      : undefined }),
  field('algorithm.use_kl_in_reward', 'KL 进 Reward 开关', 'Optimization',
    '以 KL 惩罚项进入 reward；开启后每 step 额外 ref log_prob forward。', 'toggle',
    { hint: '开启后显存与耗时上升（ref forward）。' }),
  field('actor_rollout_ref.actor.use_kl_in_reward', 'KL 进 Reward（旧位）', 'Optimization',
    '部分旧配置把该键放在 actor 下；与 algorithm.use_kl_in_reward 只保留一个写入位。', 'toggle'),
  field('algorithm.tir.loss_mode', 'Loss 形态', 'Optimization',
    'vanilla clip 或 future_kl（FIPO）。future_kl 需 D2 双写连通后生效。', 'select',
    { options: ['vanilla', 'future_kl'],
      visible: (rl) => String(readPath(rl, 'algorithm.tir.loss_mode') || '') === 'future_kl',
      hint: '阶段标注：future_kl 需 D2 双写连通后生效（RLDesign §3.3）。' }),
  field('algorithm.tir.future_kl_weight', 'FIPO 调制幅度 b', 'Optimization',
    'future_kl 优势调制幅度；仅 loss_mode=future_kl 时显示。', 'number',
    { visible: (rl) => String(readPath(rl, 'algorithm.tir.loss_mode') || 'vanilla') === 'future_kl' }),
  field('algorithm.tir.decay_rate', 'FIPO 未来折扣尺度', 'Optimization',
    'future_kl 计算中 future 项的折扣衰减；仅 loss_mode=future_kl 时显示。', 'number',
    { visible: (rl) => String(readPath(rl, 'algorithm.tir.loss_mode') || 'vanilla') === 'future_kl' }),
  field('algorithm.tir.future_kl_clip_ratio', 'FIPO 调制截断 ε', 'Optimization',
    'future_kl 调制因子的 clip 范围；仅 loss_mode=future_kl 时显示。', 'number',
    { visible: (rl) => String(readPath(rl, 'algorithm.tir.loss_mode') || 'vanilla') === 'future_kl' }),
  field('algorithm.tir.reward_scale_discount', '分支结局折扣', 'Optimization',
    'appo_credit 分支行结局回写父轨迹的折扣强度（默认 0.9）。', 'number',
    { visible: (rl) => {
      const composedOps = (rl.composed?.operators || []).map(op => op.id);
      return composedOps.includes('appo_credit') || String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'appo';
    }, hint: '值越大父轨迹优势越被分支结局影响。' }),
  // ---- RL6 L0 补全（RLDesign §4.2 表 2：Advantage 参数组）----
  field('algorithm.adv_estimator', '基础优势估计器', 'Advantage',
    '组内基线估计器：grpo | igpo | gigpo（composed 形态下由 composed.base 承载）。', 'select',
    { options: ['grpo', 'igpo', 'gigpo'], hint: 'igpo/gigpo 需要 anchor_obs / IG 列支持。' }),
  field('algorithm.tir.gamma', '折扣因子 γ', 'Advantage',
    'igpo turn 回报 / gigpo step 折扣的折扣因子（默认 1.0）。', 'number', { hint: '<1 时 credit 衰减加大。' }),
  field('algorithm.tir.step_advantage_w', 'GiGPO 局部优势权重 w', 'Advantage',
    'GiGPO 双级优势中局部 step 组优势的权重（默认 1.0）。', 'number',
    { hint: '0 = 退化为 GRPO；孤立 anchor 场景本身就退化。', visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'gigpo' || readPath(rl, 'algorithm.tir.step_advantage_w') != null }),
  field('algorithm.tir.gigpo_mode', 'GiGPO step 组归一方式', 'Advantage',
    'step 组归一化方式；当前仅 mean_std 一种实现。', 'select',
    { options: ['mean_std'], visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'gigpo' || readPath(rl, 'algorithm.tir.gigpo_mode') != null }),
  field('algorithm.tir.aepo_beta', 'AEPO 门控行为增益', 'Advantage',
    'AEPO 波次 2 熵门控预算分配行为参数（默认 1.0，daemon 读取）。', 'number',
    { visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'aepo' || readPath(rl, 'algorithm.tir.aepo_beta') != null }),
  field('algorithm.tir.entropy_adv_coef', '熵优势系数', 'Advantage',
    'AEPO 优势重塑幅度（默认 0.1，advantage 分派读取）。', 'number',
    { visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'aepo' || readPath(rl, 'algorithm.tir.entropy_adv_coef') != null }),
  field('algorithm.tir.entropy_threshold', '熵门控阈值 τ', 'Advantage',
    'ARPO 分支熵门控阈值（默认 0.15，daemon 预注入）。', 'number',
    { visible: (rl) => ['arpo', 'aepo'].includes(String(rl.algorithm?.tir_algo || rl.algo || 'grpo')) || readPath(rl, 'algorithm.tir.entropy_threshold') != null }),
  field('algorithm.tir.info_gain_type', 'IGPO 信息增益类型', 'Advantage',
    'IGPO 信息增益计算方式；当前仅 log_prob_diff 一种实现。', 'select',
    { options: ['log_prob_diff'], visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'igpo' || readPath(rl, 'algorithm.tir.info_gain_type') != null }),
  field('algorithm.tir.zero_prefix_adv_for_children', '分支前缀优势清零', 'Advantage',
    '分支前缀 token 的优势置 0（zero_prefix 算子的 tir 开关，默认 true）。', 'toggle',
    { hint: '分支前缀 token 不参与训练；与 appo_credit 同开时前缀信号丢弃更彻底。' }),
  field('algorithm.tir.rae_p_plus', 'RAE validate 偏置 p+', 'Advantage',
    'RAE validate 裁决的正向偏置强度（默认 0.8）。', 'number',
    { visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'rae' || readPath(rl, 'algorithm.tir.rae_p_plus') != null }),
  field('algorithm.tir.rae_k_min', 'RAE 裁决最少表决数', 'Advantage',
    'RAE 裁决生效所需的最少表决数 k_min（默认 2）。', 'number',
    { visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'rae' || readPath(rl, 'algorithm.tir.rae_k_min') != null, hint: '过大导致 abstain 比例上升。' }),
  field('algorithm.tir.rae_epsilon_f', 'RAE invalidate 强负系数', 'Advantage',
    'RAE invalidate 强负奖励的 log(ε_f)（默认 1e-3）。', 'number',
    { visible: (rl) => String(rl.algorithm?.tir_algo || rl.algo || 'grpo') === 'rae' || readPath(rl, 'algorithm.tir.rae_epsilon_f') != null, hint: '过小造成极端负梯度。' }),
  // ---- RL6 L0 补全（RLDesign §4.2 表 3：Reward 组件参数组）----
  field('algorithm.tir.multi_tool_bonus', '双工具奖励加成', 'Reward',
    '双工具（search+python）同时使用时的额外加成（默认 0.1；weight 归 0 即关闭）。', 'number',
    { hint: '与未来 process 组件叠加时防止双计（RLDesign §1.5 坑 1）。' }),
  field('algorithm.tir.format_penalty', '格式坏罚分', 'Reward',
    'format_ok=False 时的罚分值（现状硬编码 -1，RL4 组件化后经 RewardSpec 可配）。', 'number',
    { hint: '通常保持 ≤0；设 0 即关闭格式罚分。' }),
  field('algorithm.reward.components.cost_penalty.enabled', '成本惩罚开关（预留）', 'Reward',
    'cost_penalty 组件：每工具调用 -λc。后端实现前仅作占位。', 'toggle',
    { hint: '§3.2 组件实现后开放；与 multi_tool_bonus 方向相反，同启需注意信号对冲。' }),
  field('algorithm.reward.components.cost_penalty.weight', '成本惩罚系数 λc', 'Reward',
    'cost_penalty 每工具调用成本系数（预留位）。', 'number',
    { visible: (rl) => readPath(rl, 'algorithm.reward.components.cost_penalty.enabled') === true }),
];
