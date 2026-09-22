import type { RlConfig, SamplingSpec, TrainSignal } from '../../../shared/api/types';
import { applyWorkflowSampling } from './normalizeRlPayload';

export function rolloutPatch(rl: RlConfig, n: number): Partial<RlConfig> {
  return {
    rollout_per_gpu: n,
    actor_rollout_ref: {
      ...rl.actor_rollout_ref,
      rollout: { ...rl.actor_rollout_ref?.rollout, n },
    },
  };
}

export function applyRlRecommendation(rl: RlConfig, recommendation: RlConfig, sampling?: SamplingSpec): RlConfig {
  let next = { ...rl, ...recommendation };
  const n = recommendation['actor_rollout_ref.rollout.n'];
  const gpuMemory = recommendation['actor_rollout_ref.rollout.gpu_memory_utilization'];
  const gpuCount = recommendation['trainer.n_gpus_per_node'];
  const tensorParallel = recommendation['actor_rollout_ref.rollout.tensor_model_parallel_size'];
  if (n != null) {
    next = {
      ...next,
      actor_rollout_ref: {
        ...next.actor_rollout_ref,
        rollout: { ...next.actor_rollout_ref?.rollout, n: Number(n) },
      },
      rollout_per_gpu: recommendation.rollout_per_gpu,
    };
  }
  if (gpuMemory != null) {
    next = {
      ...next,
      actor_rollout_ref: {
        ...next.actor_rollout_ref,
        rollout: { ...next.actor_rollout_ref?.rollout, gpu_memory_utilization: Number(gpuMemory) },
      },
    };
  }
  if (gpuCount != null) next = { ...next, trainer: { ...next.trainer, n_gpus_per_node: Number(gpuCount) } };
  if (tensorParallel != null) next = { ...next, actor_rollout_ref: {
    ...next.actor_rollout_ref, rollout: { ...next.actor_rollout_ref?.rollout, tensor_model_parallel_size: Number(tensorParallel) },
  } };
  return applyWorkflowSampling(next, sampling);
}

export function applyTrainSignal(rl: RlConfig, signal: TrainSignal, sampling?: SamplingSpec): RlConfig {
  let next = { ...rl };
  if (signal.advantage?.name) next.algo = signal.advantage.name;
  const loss = signal.loss;
  if (loss) {
    const actor = { ...next.actor_rollout_ref?.actor };
    if (loss.clip_ratio_low != null) actor.clip_ratio_low = loss.clip_ratio_low;
    if (loss.clip_ratio_high != null) actor.clip_ratio_high = loss.clip_ratio_high;
    if (loss.entropy_coeff != null) actor.entropy_coeff = loss.entropy_coeff;
    if (loss.kl_loss_coef != null) actor.kl_loss_coef = loss.kl_loss_coef;
    if (Object.keys(actor).length) next = { ...next, actor_rollout_ref: { ...next.actor_rollout_ref, actor } };
  }
  return applyWorkflowSampling(next, sampling);
}
