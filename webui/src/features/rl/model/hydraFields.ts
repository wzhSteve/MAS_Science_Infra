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
  { id: 'Optimization', label: '优化参数', description: '更新步长、批次和损失约束。' },
  { id: 'Rollout', label: '推理与采样资源', description: '控制 vLLM 显存、张量并行和 log-prob 批次。' },
  { id: 'Data', label: '训练数据', description: '训练/验证数据路径与上下文长度。' },
  { id: 'Trainer', label: '训练运行', description: '训练轮次、步数和日志标识。' },
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
  field('actor_rollout_ref.actor.optim.lr', '学习率', 'Optimization', 'Actor 优化器步长。'),
  field('actor_rollout_ref.actor.ppo_mini_batch_size', 'PPO mini batch', 'Optimization', '每次 PPO 更新使用的样本数。'),
  field('actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu', 'Actor micro batch / GPU', 'Optimization', '单卡一次前反向处理的样本数。'),
  field('actor_rollout_ref.actor.clip_ratio_low', '裁剪下界', 'Optimization', '策略比率的下侧裁剪范围。'),
  field('actor_rollout_ref.actor.clip_ratio_high', '裁剪上界', 'Optimization', '策略比率的上侧裁剪范围。'),
  field('actor_rollout_ref.actor.entropy_coeff', '熵系数', 'Optimization', '鼓励策略探索的损失权重。'),
  field('actor_rollout_ref.actor.kl_loss_coef', 'KL 损失系数', 'Optimization', '约束策略偏移的损失权重。'),
  field('actor_rollout_ref.rollout.tensor_model_parallel_size', '张量并行数', 'Rollout', '单个推理实例使用的 GPU 数。'),
  field('actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu', 'Rollout log-prob micro batch / GPU', 'Rollout', '单卡计算 log-prob 的微批次。'),
  field('actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu', 'Reference log-prob micro batch / GPU', 'Rollout', '参考模型单卡 log-prob 微批次。'),
  field('actor_rollout_ref.rollout.gpu_memory_utilization', '推理显存占比', 'Rollout', 'vLLM 可使用的单卡显存比例。'),
  field('data.train_files', '训练数据', 'Data', 'Parquet 文件路径。', 'text'),
  field('data.val_files', '验证数据', 'Data', 'Parquet 文件路径。', 'text'),
  field('data.train_batch_size', '训练批次', 'Data', '每个训练批次包含的样本数。'),
  field('data.max_prompt_length', '最大输入长度', 'Data', 'Prompt token 上限。'),
  field('data.max_response_length', '最大输出长度', 'Data', 'Response token 上限。'),
  field('data.truncation', '超长处理', 'Data', '例如 error、left 或 right。', 'text'),
  field('trainer.total_epochs', '训练轮次', 'Trainer', '完整遍历训练数据的次数。'),
  field('trainer.total_training_steps', '训练步数', 'Trainer', '显式限制优化步数。'),
  field('trainer.test_freq', '验证频率', 'Trainer', '每隔多少训练步执行验证。'),
  field('trainer.nnodes', '训练节点数', 'Trainer', '分布式训练使用的机器数量。'),
  field('trainer.project_name', '项目名称', 'Trainer', '日志系统中的项目标识。', 'text'),
  field('trainer.experiment_name', '运行名称', 'Trainer', '日志系统中的本次训练标识。', 'text'),
];
