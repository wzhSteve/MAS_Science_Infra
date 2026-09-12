import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { RlConfig } from '../../../shared/api/types';
import { useAction } from '../../../shared/hooks/useAction';
import { useUnsavedChanges } from '../../../shared/hooks/useUnsavedChanges';
import { normalizeRlPayload } from './normalizeRlPayload';
import { applyRlRecommendation, applyTrainSignal } from './rlPrefills';

export function useRlDraft(expId: string, initial: RlConfig, onReload: () => void, onStartTrain: () => Promise<void>) {
  const [{ rl, savedRl }, setDraft] = useState(() => ({
    rl: structuredClone(initial), savedRl: JSON.stringify(initial),
  }));
  const incomingRl = useMemo(() => JSON.stringify(initial), [initial]);
  const lastIncomingRl = useRef(incomingRl);
  const editRevision = useRef(0);
  const dirty = JSON.stringify(rl) !== savedRl;
  const { pending, notice, run } = useAction();
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
    const payload = normalizeRlPayload(rl);
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
      const response = await api.gpus();
      if (mounted.current) {
        editRevision.current += 1;
        setDraft((current) => ({ ...current, rl: applyRlRecommendation(current.rl, response.recommend || {}) }));
      }
      return '已按当前机器推荐档位（需点保存）';
    });
  }, [run]);

  const prefill = useCallback(() => {
    void run('prefill', async () => {
      const response = await api.monitor(expId);
      if (mounted.current) {
        editRevision.current += 1;
        setDraft((current) => ({ ...current, rl: applyTrainSignal(current.rl, response.train_signal || {}) }));
      }
      return 'prefilled from TrainSignal（需点保存）';
    });
  }, [expId, run]);

  useUnsavedChanges('rl-settings', {
    label: 'RL 配置', resource: 'rl', dirty, busy: pending !== null, save,
  });

  return { rl, dirty, pending, notice, patch, save, start, recommend, prefill };
}
