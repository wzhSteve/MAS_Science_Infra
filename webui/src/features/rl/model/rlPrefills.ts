import type { ComposedSpec, RlConfig, SamplingSpec, TrainSignal } from '../../../shared/api/types';
import { applyWorkflowSampling } from './normalizeRlPayload';

/** RL6: front-end mirror of `rl/loss.py` PRESET_EXPANSION_BASE /
 * PRESET_EXPANSION_OPERATORS (the single physical-copy authority locked by
 * mas/tests/test_expand_preset.py + test_operator_composition.py).  A TS module
 * cannot import the Python table, so this mirror is kept in sync by the
 * equivalence test in rlConfig.test.ts (per-preset expansion asserted against
 * the frozen §3.6 snapshot) plus the backend's own exact-match tests.
 * Do NOT edit one side without the other. */
export const PRESET_EXPANSION_BASE: Record<string, string> = {
  grpo: 'grpo',
  arpo: 'grpo',
  appo: 'grpo',
  aepo: 'grpo',
  rae: 'grpo',
  flow_grpo: 'grpo',
  igpo: 'igpo',
  gigpo: 'gigpo',
};

export const PRESET_EXPANSION_OPERATORS: Record<string, string[]> = {
  grpo: ['zero_prefix'],
  arpo: [],
  appo: ['appo_credit'],
  aepo: ['aepo_entropy'],
  rae: ['zero_prefix', 'rae_verdict'],
  flow_grpo: [],
  igpo: [],
  gigpo: [],
};

export const PRESET_LABELS: Record<string, string> = {
  grpo: 'GRPO',
  arpo: 'ARPO',
  appo: 'APPO',
  aepo: 'AEPO',
  igpo: 'IGPO',
  gigpo: 'GiGPO',
  rae: 'RAE',
  flow_grpo: 'Flow-GRPO',
};

/** 一句话定位（RL_design §1.3 定位矩阵"关键机制一句话"列，v1 预设卡片文案）。 */
export const PRESET_TAGLINES: Record<string, string> = {
  grpo: '同 prompt 组内相对优势',
  arpo: '熵门控自适应分支采样',
  appo: '程序级分支 + 分支行结局折扣回写父行',
  aepo: '低熵位置的优势差分放大',
  igpo: 'turn 级信息增益组内 z-score',
  gigpo: '全局组内 + 局部 step 组双级优势',
  rae: 'validate / invalidate / abstain 三态裁决',
  flow_grpo: 'outcome 广播到每 turn + 强制 KL 正则',
};

/** Expand a v1 preset name into the composed form (RL_design §3.6 等价表).
 * Unknown names return null (the UI keeps the preset unexpanded) — the
 * backend `expand_preset` raises ValueError instead; both directions are
 * covered by tests. */
export function expandPreset(preset: string): ComposedSpec | null {
  const key = String(preset || '').trim().toLowerCase();
  if (!(key in PRESET_EXPANSION_BASE)) return null;
  return {
    base: PRESET_EXPANSION_BASE[key] as ComposedSpec['base'],
    operators: PRESET_EXPANSION_OPERATORS[key].map(id => ({ id })),
    reward: { components: [] },
    loss: { mode: 'vanilla' },
  };
}

/** Expand the preset into the draft: algo/tir_algo cleared, composed set. */
export function expandPresetIntoDraft(rl: RlConfig, preset: string): RlConfig {
  const composed = expandPreset(preset);
  if (!composed) return rl;
  // appo 的 loss_mode=future_kl 名实不符现状在展开面如实呈现 (RL_design §2.3):
  const withLoss = preset.trim().toLowerCase() === 'appo'
    ? { ...composed, loss: { mode: 'future_kl' as const } }
    : composed;
  const algorithm = { ...rl.algorithm };
  delete algorithm.tir_algo;
  return {
    ...rl,
    algo: undefined,
    algorithm: Object.keys(algorithm).length ? algorithm : undefined,
    composed: withLoss,
  };
}

/** Collapse back to the v1 preset form: composed cleared, algo/tir_algo set. */
export function collapseComposedToPreset(rl: RlConfig, preset: string): RlConfig {
  const { composed: _composed, ...rest } = rl;
  return {
    ...rest,
    algo: preset,
    algorithm: { ...rest.algorithm, tir_algo: preset },
  };
}

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
  // RL6 dual-track (§4.3 清单 4): a preset name keeps the v1 algo path; an
  // explicit composed payload on the signal takes the composed path instead.
  const composedSignal = (signal as { composed?: ComposedSpec }).composed;
  if (composedSignal) {
    const { algo: _algo, ...rest } = next;
    const algorithm = { ...rest.algorithm };
    delete algorithm.tir_algo;
    next = { ...rest, algorithm: Object.keys(algorithm).length ? algorithm : undefined, composed: composedSignal };
  } else if (signal.advantage?.name) {
    const { composed: _composed, ...rest } = next;
    next = { ...rest, algo: signal.advantage.name };
  }
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
