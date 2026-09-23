import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { RlConfig, SamplingSpec } from '../../../shared/api/types';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import { normalizeRlPayload } from './normalizeRlPayload';
import { applyRlRecommendation } from './rlPrefills';
import { HYDRA_FIELDS } from './hydraFields';
import { useConfirm } from '../../../shared/feedback/useConfirm';

export function useRlDraft(
  expId: string,
  initial: RlConfig,
  onReload: () => void,
  onStartTrain: () => Promise<void>,
  sampling?: SamplingSpec,
) {
  const confirm = useConfirm();
  const [{ rl, savedRl }, setDraft] = useState(() => ({
    rl: structuredClone(initial), savedRl: JSON.stringify(initial),
  }));
  const incomingRl = useMemo(() => JSON.stringify(initial), [initial]);
  const lastIncomingRl = useRef(incomingRl);
  const editRevision = useRef(0);
  const dirty = JSON.stringify(rl) !== savedRl;
  const { pending, notice, run } = useAction({ errorFeedback: 'inline' });
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  useEffect(() => {
    if (pending !== null || lastIncomingRl.current === incomingRl) return;
    lastIncomingRl.current = incomingRl;
    setDraft((current) => ({
      rl: JSON.stringify(current.rl) === current.savedRl ? JSON.parse(incomingRl) as RlConfig : current.rl,
      savedRl: incomingRl,
    }));
  }, [incomingRl, pending]);

  const patch = useCallback((next: RlConfig | ((current: RlConfig) => RlConfig)) => {
    editRevision.current += 1;
    setDraft((current) => ({ ...current, rl: typeof next === 'function' ? next(current.rl) : next }));
  }, []);

  const persist = useCallback(async () => {
    const submittedRevision = editRevision.current;
    for (const field of HYDRA_FIELDS) {
      const value = field.read(rl);
      if (field.type !== 'number' || value == null) continue;
      const count = /batch_size|batch_size_per_gpu|length|total_epochs|parallel_size|nnodes/.test(field.path);
      if (!Number.isFinite(value) || (count && (Number(value) < 1 || !Number.isInteger(value)))) {
        throw new Error(`${field.label}需要${count ? '大于零的整数' : '有效数字'}。`);
      }
      if (/optim.lr|gpu_memory_utilization/.test(field.path) && Number(value) <= 0) throw new Error(`${field.label}必须大于零。`);
      if (/gpu_memory_utilization/.test(field.path) && Number(value) > 1) throw new Error('vLLM 显存占比不能大于 1。');
    }
    if (rl.n_runners != null && (!Number.isInteger(rl.n_runners) || rl.n_runners < 1)) throw new Error('并行采集数需要大于零的整数。');
    if (rl.rollout_per_gpu != null && (!Number.isInteger(rl.rollout_per_gpu) || rl.rollout_per_gpu < 1)) throw new Error('每题候选数需要大于零的整数。');
    const saved = await api.getExperiment(expId);
    const payload = normalizeRlPayload(rl, [0], saved.workflow.sampling);
    await api.putSection(expId, 'rl', payload);
    if (mounted.current) {
      // Editing while a save is in flight must not discard the newer draft.
      const unchanged = editRevision.current === submittedRevision;
      setDraft((current) => ({
        rl: unchanged ? payload : current.rl,
        savedRl: JSON.stringify(payload),
      }));
      onReload();
    }
    return submittedRevision;
  }, [rl, expId, onReload]);

  const save = useCallback(async () => {
    let submittedRevision: number | undefined;
    const success = await run('save', async () => { submittedRevision = await persist(); return 'saved rl.yaml'; });
    return success && mounted.current && editRevision.current === submittedRevision;
  }, [persist, run]);

  const start = useCallback(() => {
    void run('start', async () => {
      await persist();
      if (mounted.current) await onStartTrain();
      return 'saved rl.yaml';
    });
  }, [persist, onStartTrain, run]);

  const recommend = useCallback(() => {
    void run('recommend', async () => {
      const revision = editRevision.current;
      const response = await api.gpus();
      if (mounted.current) {
        if (editRevision.current !== revision) throw new Error('读取资源建议期间配置有新修改，请重新读取。');
        const source = response.recommend || {};
        const recommendation: RlConfig = {};
        for (const key of ['profile', 'n_runners', 'actor_rollout_ref.rollout.gpu_memory_utilization',
          'actor_rollout_ref.rollout.tensor_model_parallel_size']) {
          if (source[key] != null) recommendation[key] = source[key];
        }
        const changes = Object.entries(recommendation).map(([key, after]) => ({
          field: key, before: HYDRA_FIELDS.find(field => field.path === key)?.read(rl) ?? rl[key] ?? '继承', after,
        }));
        if (!(await confirm({
          title: '应用资源建议？',
          description: `以下建议将写入当前草稿，不改变模型、算法或 GPU 选择：\n\n${changes.map(change =>
            `• ${change.field}: ${String(change.before)} → ${String(change.after)}`).join('\n')}\n\n仍需保存才会生效。`,
          confirmLabel: '应用建议',
        }))) return;
        editRevision.current += 1;
        setDraft((current) => ({ ...current, rl: applyRlRecommendation(current.rl, recommendation) }));
      }
      return '已按当前机器推荐档位（需点保存）';
    });
  }, [confirm, run, rl]);

  useUnsavedChanges('rl-settings', {
    label: 'RL 配置', resource: 'rl', dirty, busy: pending !== null, save,
  });

  return { rl, dirty, pending, notice, patch, save, start, recommend };
}
