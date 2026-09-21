import type { RlConfig } from '../../../shared/api/types';

export type HydraGroup = 'Optimization' | 'Rollout' | 'Data' | 'Trainer';

export interface HydraField {
  path: string;
  label: string;
  description: string;
  group: HydraGroup;
  type: 'number' | 'text';
  read: (rl: RlConfig) => number | string | undefined;
  update: (rl: RlConfig, value: string) => RlConfig;
}

export const HYDRA_GROUPS: Array<{ id: HydraGroup; label: string; description: string }> = [
  { id: 'Optimization', label: '优化参数', description: 'Actor 更新步长、PPO Batch 和 Loss 约束。' },
  { id: 'Rollout', label: 'Rollout 资源', description: 'vLLM 显存、Tensor Parallel 和 Log-prob Batch。' },
  { id: 'Data', label: '数据', description: '训练/验证数据路径与 Context Length。' },
  { id: 'Trainer', label: '训练过程', description: '训练轮次、步数和日志标识。' },
];

function readPath(rl: RlConfig, path: string): unknown {
  return path.split('.').reduce<unknown>((value, key) =>
    value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined, rl);
}

function updatePath(rl: RlConfig, path: string, value: number | string): RlConfig {
  const keys = path.split('.');
  const root = structuredClone(rl) as Record<string, unknown>;
  let cursor = root;
  for (const key of keys.slice(0, -1)) {
    const child = cursor[key];
    cursor[key] = child && typeof child === 'object' && !Array.isArray(child)
      ? { ...(child as Record<string, unknown>) } : {};
    cursor = cursor[key] as Record<string, unknown>;
  }
  cursor[keys[keys.length - 1]] = value;
  return root as RlConfig;
}

function field(
  path: string,
  label: string,
  group: HydraGroup,
  description: string,
  type: HydraField['type'] = 'number',
): HydraField {
  return {
    path, label, group, description, type,
    read: (rl) => readPath(rl, path) as number | string | undefined,
    update: (rl, value) => updatePath(rl, path, type === 'number' ? Number(value) : value),
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
];
