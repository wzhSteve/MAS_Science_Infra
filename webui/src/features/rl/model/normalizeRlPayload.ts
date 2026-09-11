import type { RlConfig } from '../../../shared/api/types';

export function normalizeRlPayload(rl: RlConfig, fallbackIds: number[] = [0]): RlConfig {
  const payload = { ...rl };
  // Both save entry points discard recommendation-only dotted keys.
  for (const key of Object.keys(payload)) {
    if (key.includes('.')) delete payload[key];
  }
  const ids = (payload.devices?.ids || fallbackIds).map(Number);
  payload.devices = { ids };
  payload.trainer = { ...payload.trainer, n_gpus_per_node: ids.length };
  if (payload.rollout_per_gpu != null) {
    payload.actor_rollout_ref = {
      ...payload.actor_rollout_ref,
      rollout: { ...payload.actor_rollout_ref?.rollout, n: Number(payload.rollout_per_gpu) },
    };
  }
  return payload;
}
