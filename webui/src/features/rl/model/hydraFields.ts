import type { RlConfig } from '../../../shared/api/types';

type ActorLossKey = 'clip_ratio_low' | 'clip_ratio_high' | 'entropy_coeff' | 'kl_loss_coef';

export interface HydraField {
  path: string;
  label: string;
  group: 'Actor' | 'Rollout' | 'Data' | 'Trainer';
  type: 'number' | 'text';
  read: (rl: RlConfig) => number | string | undefined;
  update: (rl: RlConfig, value: string) => RlConfig;
}

function actorLossField(key: ActorLossKey): HydraField {
  return {
    path: `actor_rollout_ref.actor.${key}`, label: key, group: 'Actor', type: 'number',
    read: (rl) => rl.actor_rollout_ref?.actor?.[key],
    update: (rl, value) => ({
      ...rl, actor_rollout_ref: {
        ...rl.actor_rollout_ref, actor: { ...rl.actor_rollout_ref?.actor, [key]: Number(value) },
      },
    }),
  };
}

export const HYDRA_FIELDS: HydraField[] = [
  {
    path: 'actor_rollout_ref.actor.optim.lr', label: 'actor.optim.lr', group: 'Actor', type: 'number',
    read: (rl) => rl.actor_rollout_ref?.actor?.optim?.lr,
    update: (rl, value) => ({
      ...rl, actor_rollout_ref: {
        ...rl.actor_rollout_ref, actor: {
          ...rl.actor_rollout_ref?.actor, optim: { ...rl.actor_rollout_ref?.actor?.optim, lr: Number(value) },
        },
      },
    }),
  },
  actorLossField('clip_ratio_low'),
  actorLossField('clip_ratio_high'),
  actorLossField('entropy_coeff'),
  actorLossField('kl_loss_coef'),
  {
    path: 'data.train_batch_size', label: 'train_batch_size', group: 'Data', type: 'number',
    read: (rl) => rl.data?.train_batch_size,
    update: (rl, value) => ({ ...rl, data: { ...rl.data, train_batch_size: Number(value) } }),
  },
  {
    path: 'actor_rollout_ref.rollout.n', label: 'rollout.n', group: 'Rollout', type: 'number',
    read: (rl) => rl.actor_rollout_ref?.rollout?.n,
    update: (rl, value) => ({
      ...rl, actor_rollout_ref: { ...rl.actor_rollout_ref, rollout: { ...rl.actor_rollout_ref?.rollout, n: Number(value) } },
    }),
  },
  {
    path: 'actor_rollout_ref.rollout.gpu_memory_utilization', label: 'gpu_memory_utilization', group: 'Rollout', type: 'number',
    read: (rl) => rl.actor_rollout_ref?.rollout?.gpu_memory_utilization,
    update: (rl, value) => ({
      ...rl, actor_rollout_ref: {
        ...rl.actor_rollout_ref, rollout: { ...rl.actor_rollout_ref?.rollout, gpu_memory_utilization: Number(value) },
      },
    }),
  },
  {
    path: 'trainer.n_gpus_per_node', label: 'n_gpus_per_node', group: 'Trainer', type: 'number',
    read: (rl) => rl.trainer?.n_gpus_per_node,
    update: (rl, value) => ({ ...rl, trainer: { ...rl.trainer, n_gpus_per_node: Number(value) } }),
  },
  {
    path: 'trainer.total_epochs', label: 'total_epochs', group: 'Trainer', type: 'number',
    read: (rl) => rl.trainer?.total_epochs,
    update: (rl, value) => ({ ...rl, trainer: { ...rl.trainer, total_epochs: Number(value) } }),
  },
  {
    path: 'trainer.experiment_name', label: 'experiment_name', group: 'Trainer', type: 'text',
    read: (rl) => rl.trainer?.experiment_name,
    update: (rl, value) => ({ ...rl, trainer: { ...rl.trainer, experiment_name: value } }),
  },
];
