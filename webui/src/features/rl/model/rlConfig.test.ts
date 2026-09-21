import { describe, expect, it } from 'vitest';
import type { RlConfig } from '../../../shared/api/types';
import { HYDRA_FIELDS } from './hydraFields';
import { applyWorkflowSampling, normalizeRlPayload } from './normalizeRlPayload';
import { applyRlRecommendation, rolloutPatch } from './rlPrefills';

function config(): RlConfig {
  return {
    profile: 'fast',
    algo: 'grpo',
    n_runners: 1,
    rollout_per_gpu: 2,
    devices: { ids: [0], affinity: 'exclusive' },
    extension: { keep: true },
    algorithm: { adv_estimator: 'grpo', tir_algo: 'grpo', tir: { entropy_threshold: 0.15 } },
    data: { train_files: 'data/train.parquet', val_files: 'data/val.parquet', train_batch_size: 2 },
    actor_rollout_ref: {
      rollout: { n: 2, gpu_memory_utilization: 0.35, name: 'vllm' },
      actor: { optim: { lr: 1e-6 }, clip_ratio_low: 0.2 },
      model: { path: 'model', use_remove_padding: true },
    },
    trainer: { n_gpus_per_node: 1, total_epochs: 1, experiment_name: 'run' },
  };
}

describe('RL configuration contract', () => {
  it('normalizes GPU and rollout aliases without dropping nested or extension fields', () => {
    const rl = config();
    rl.devices = { ids: [1, 0, 1], affinity: 'exclusive' };
    rl.rollout_per_gpu = 4;
    const normalized = normalizeRlPayload(rl);

    expect(normalized.devices).toEqual({ ids: [0, 1], affinity: 'exclusive' });
    expect(normalized.trainer?.n_gpus_per_node).toBe(2);
    expect(normalized.actor_rollout_ref?.rollout?.n).toBe(4);
    expect(normalized.actor_rollout_ref?.rollout?.name).toBe('vllm');
    expect(normalized.extension).toEqual({ keep: true });
  });

  it('treats workflow sampling as the effective algorithm and group size', () => {
    const normalized = normalizeRlPayload(config(), [0], { mode: 'arpo', group_n: 6 });

    expect(normalized.algo).toBe('arpo');
    expect(normalized.algorithm).toMatchObject({
      adv_estimator: 'grpo',
      use_kl_in_reward: false,
      tir_algo: 'arpo',
      tir: { entropy_threshold: 0.15 },
    });
    expect(normalized.rollout_per_gpu).toBe(6);
    expect(normalized.actor_rollout_ref?.rollout?.n).toBe(6);
  });

  it('keeps recommendation values subordinate to workflow sampling', () => {
    const recommended = applyRlRecommendation(config(), {
      algo: 'grpo',
      rollout_per_gpu: 2,
      devices: { ids: [0, 1] },
      'actor_rollout_ref.rollout.n': 2,
      'actor_rollout_ref.rollout.gpu_memory_utilization': 0.25,
      'trainer.n_gpus_per_node': 2,
    }, { mode: 'rae', group_n: 4 });

    expect(recommended.algo).toBe('rae');
    expect(recommended.rollout_per_gpu).toBe(4);
    expect(recommended.actor_rollout_ref?.rollout).toMatchObject({ n: 4, gpu_memory_utilization: 0.25 });
    expect(recommended.devices?.ids).toEqual([0, 1]);
  });

  it('updates rollout aliases together', () => {
    const patched = { ...config(), ...rolloutPatch(config(), 8) };
    expect(patched.rollout_per_gpu).toBe(8);
    expect(patched.actor_rollout_ref?.rollout?.n).toBe(8);
  });

  it('updates one Hydra field without replacing sibling configuration', () => {
    const field = HYDRA_FIELDS.find((item) => item.path === 'data.max_prompt_length');
    const updated = field!.update(config(), '4096');

    expect(updated.data).toMatchObject({
      train_files: 'data/train.parquet',
      val_files: 'data/val.parquet',
      train_batch_size: 2,
      max_prompt_length: 4096,
    });
    expect(updated.actor_rollout_ref?.model).toEqual({ path: 'model', use_remove_padding: true });
    expect(updated.extension).toEqual({ keep: true });
  });

  it('maps APPO sampling to the implemented ARPO runtime', () => {
    expect(applyWorkflowSampling(config(), { mode: 'appo' }).algo).toBe('arpo');
  });
});
