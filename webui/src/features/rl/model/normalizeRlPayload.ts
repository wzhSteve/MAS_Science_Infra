import type { RlConfig, SamplingSpec } from '../../../shared/api/types';

const SAMPLING_ALGORITHMS: Record<string, string> = {
  grpo_n: 'grpo',
  grpo: 'grpo',
  arpo: 'arpo',
  aepo: 'aepo',
  appo: 'appo',
  rae: 'rae',
};

export function algorithmFromSampling(sampling?: SamplingSpec): string | null {
  const mode = sampling?.mode?.trim().toLowerCase();
  return mode ? SAMPLING_ALGORITHMS[mode] || mode : null;
}

function hasComposed(rl: RlConfig): boolean {
  const composed = rl.composed;
  return Boolean(composed && typeof composed === 'object'
    && (composed.base != null || (composed.operators && composed.operators.length > 0)
      || composed.reward != null || composed.loss != null));
}

/** RL6 归一规则（RL_design §4.3 webui model 层清单第 3 项）：
 * (a) composed 与 algo/tir_algo 互斥——composed 存在则删 algo 侧，反之删 composed，防双轨同写；
 * (b) composed 存在时 applyWorkflowSampling 降级（规则一：只同步 group_n 数值联动，
 *     不再覆盖 algo/tir_algo/composed）；
 * (c) 其余归一（点键清理、devices.ids 去重排序、n_gpus 同步、rollout_per_gpu→rollout.n）不变。 */
export function reconcileDualTrack(rl: RlConfig): RlConfig {
  if (hasComposed(rl)) {
    // composed wins (§4.3 规则二: explicit composed > sampling.mode v1 mapping)
    const { algo: _algo, ...rest } = rl;
    const algorithm = { ...rest.algorithm };
    delete algorithm.tir_algo;
    return { ...rest, algorithm: Object.keys(algorithm).length ? algorithm : undefined };
  }
  if (rl.composed != null) {
    const { composed: _composed, ...rest } = rl;
    return rest;
  }
  return rl;
}

export function applyWorkflowSampling(rl: RlConfig, sampling?: SamplingSpec): RlConfig {
  const composed = hasComposed(rl);
  // Back-compat side effect of the v1 path: keep group_n → rollout sync in
  // both tracks (规则一: group_n 决定 GRPO 分组，是采样事实，与算法结构选择无关).
  const groupSize = sampling?.group_n;
  let next = rl;
  if (!composed) {
    const algo = algorithmFromSampling(sampling);
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
  const payload = reconcileDualTrack({ ...rl });
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
