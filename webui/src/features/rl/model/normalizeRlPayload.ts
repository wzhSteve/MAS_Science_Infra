import type { RlConfig, SamplingSpec } from '../../../shared/api/types';

const SAMPLING_ALGORITHMS: Record<string, string> = {
  grpo_n: 'grpo',
  grpo: 'grpo',
  arpo: 'arpo',
  aepo: 'aepo',
  appo: 'arpo',
  rae: 'rae',
};

export function algorithmFromSampling(sampling?: SamplingSpec): string | null {
  const mode = sampling?.mode?.trim().toLowerCase();
  return mode ? SAMPLING_ALGORITHMS[mode] || mode : null;
}

export function applyWorkflowSampling(rl: RlConfig, sampling?: SamplingSpec): RlConfig {
  const algo = algorithmFromSampling(sampling);
  const groupSize = sampling?.group_n;
  let next = rl;
  if (algo) {
    next = {
      ...next,
      algo,
      algorithm: {
        ...next.algorithm,
        adv_estimator: 'grpo',
        use_kl_in_reward: false,
        tir_algo: algo,
      },
    };
  }
  if (groupSize != null) {
    next = {
      ...next,
      rollout_per_gpu: groupSize,
      actor_rollout_ref: {
        ...next.actor_rollout_ref,
        rollout: { ...next.actor_rollout_ref?.rollout, n: groupSize },
      },
    };
  }
  return next;
}

export function normalizeRlPayload(
  rl: RlConfig,
  fallbackIds: number[] = [0],
  sampling?: SamplingSpec,
): RlConfig {
  const payload = { ...rl };
  // Both save entry points discard recommendation-only dotted keys.
  for (const key of Object.keys(payload)) {
    if (key.includes('.')) delete payload[key];
  }
  const ids = Array.from(new Set((payload.devices?.ids?.length ? payload.devices.ids : fallbackIds)
    .map(Number))).sort((a, b) => a - b);
  payload.devices = { ...payload.devices, ids };
  payload.trainer = { ...payload.trainer, n_gpus_per_node: ids.length };
  if (payload.rollout_per_gpu != null) {
    payload.actor_rollout_ref = {
      ...payload.actor_rollout_ref,
      rollout: { ...payload.actor_rollout_ref?.rollout, n: Number(payload.rollout_per_gpu) },
    };
  }
  return applyWorkflowSampling(payload, sampling);
}
