import { describe, expect, it } from 'vitest';
import type { RlConfig } from '../../../shared/api/types';
import { HYDRA_FIELDS } from './hydraFields';
import { applyWorkflowSampling, normalizeRlPayload } from './normalizeRlPayload';
import { applyRlRecommendation, expandPreset, expandPresetIntoDraft, rolloutPatch, collapseComposedToPreset } from './rlPrefills';

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

  it('keeps APPO sampling as its own training algorithm', () => {
    const normalized = applyWorkflowSampling(config(), { mode: 'appo' });
    expect(normalized.algo).toBe('appo');
    expect(normalized.algorithm?.tir_algo).toBe('appo');
  });

  // ---- RL6: composed dual-track normalization (RL_design §4.3 规则一二三) ----

  it('keeps composed and drops algo/tir_algo when both are present (dual-track mutex)', () => {
    const rl = config();
    (rl as RlConfig & { composed: unknown }).composed = {
      base: 'gigpo', operators: [{ id: 'aepo_entropy' }], reward: { components: [] }, loss: { mode: 'vanilla' },
    };
    const normalized = normalizeRlPayload(rl, [0]);
    expect((normalized as { composed?: unknown }).composed).toMatchObject({ base: 'gigpo' });
    expect((normalized as { algo?: string }).algo).toBeUndefined();
    expect(normalized.algorithm?.tir_algo).toBeUndefined();
  });

  it('drops composed when algo is the explicit track (reverse mutex)', () => {
    const rl = { ...config(), composed: {} } as RlConfig;
    // composed 为空壳（无 base/operators/reward/loss）→ algo 是显式轨道，
    // 空壳 composed 属于残留，不得存活双轨。
    const normalized = normalizeRlPayload(rl, [0]);
    expect((normalized as { composed?: unknown }).composed).toBeUndefined();
    expect(normalized.algo).toBe('grpo');
  });

  it('downgrades workflow sampling override when composed is explicit (rule 1/2)', () => {
    const rl = {
      ...config(),
      composed: { base: 'igpo', operators: [{ id: 'zero_prefix' }], loss: { mode: 'vanilla' } },
    } as RlConfig;
    const normalized = normalizeRlPayload(rl, [0], { mode: 'appo', group_n: 6 });
    // 规则一: composed 存在时 applyWorkflowSampling 只同步 group_n，不覆盖算法结构
    expect((normalized as { algo?: string }).algo).toBeUndefined();
    expect((normalized as { composed?: { base?: string } }).composed?.base).toBe('igpo');
    // group_n 数值联动保留
    expect(normalized.rollout_per_gpu).toBe(6);
    expect(normalized.actor_rollout_ref?.rollout?.n).toBe(6);
  });

  // ---- RL6: expandPreset front-end mirror of rl/loss.py PRESET_EXPANSION ----

  it('expands all 8 presets exactly as the backend PRESET_EXPANSION table (§3.6 快照)', () => {
    const expected: Record<string, { base: string; operators: string[] }> = {
      grpo: { base: 'grpo', operators: ['zero_prefix'] },
      arpo: { base: 'grpo', operators: [] },
      appo: { base: 'grpo', operators: ['appo_credit'] },
      aepo: { base: 'grpo', operators: ['aepo_entropy'] },
      rae: { base: 'grpo', operators: ['zero_prefix', 'rae_verdict'] },
      flow_grpo: { base: 'grpo', operators: [] },
      igpo: { base: 'igpo', operators: [] },
      gigpo: { base: 'gigpo', operators: [] },
    };
    for (const [preset, snapshot] of Object.entries(expected)) {
      const composed = expandPreset(preset);
      expect(composed, `expandPreset('${preset}')`).not.toBeNull();
      expect(composed!.base).toBe(snapshot.base);
      expect(composed!.operators?.map(op => op.id)).toEqual(snapshot.operators);
    }
    expect(expandPreset('unknown_preset')).toBeNull();
  });

  it('expandPreset(rl, "rae") renders correctly but ComposedPanel editable operators stay rae-free (用户裁决)', () => {
    // 预设展开镜像必须如实显示 rae 组合（保底渲染）
    const raeComposed = expandPreset('rae');
    expect(raeComposed?.base).toBe('grpo');
    expect(raeComposed?.operators?.map(op => op.id)).toEqual(['zero_prefix', 'rae_verdict']);
    // 但可编辑算子清单不含 rae 系（RL3 注册表不动，仅 UI 不暴露）
    const rl = expandPresetIntoDraft(config(), 'rae');
    expect((rl as { composed?: { base?: string } }).composed?.base).toBe('grpo');
    // 逆操作回到 v1 preset 形态（algo/tir_algo 复位）
    const back = collapseComposedToPreset(rl, 'rae');
    expect(back.algo).toBe('rae');
    expect(back.algorithm?.tir_algo).toBe('rae');
    expect((back as { composed?: unknown }).composed).toBeUndefined();
  });

  it('appo preset expansion surfaces the loss_mode=future_kl 名实不符 fact', () => {
    const rl = expandPresetIntoDraft(config(), 'appo');
    const composed = (rl as { composed?: { loss?: { mode?: string } } }).composed;
    expect(composed?.loss?.mode).toBe('future_kl');
  });
});
