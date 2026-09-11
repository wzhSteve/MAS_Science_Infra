import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../../../api/client';
import type { RlConfig } from '../../../shared/api/types';
import { useAction } from '../../../shared/hooks/useAction';
import { normalizeRlPayload } from './normalizeRlPayload';
import { applyRlRecommendation, applyTrainSignal } from './rlPrefills';

export function useRlDraft(expId: string, initial: RlConfig, onReload: () => void, onStartTrain: () => Promise<void>) {
  const [draft, setDraft] = useState(() => ({ rl: structuredClone(initial), dirty: false }));
  const { pending, notice, run } = useAction();
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; };
  }, []);

  const patch = useCallback((next: RlConfig) => setDraft({ rl: next, dirty: true }), []);

  const persist = useCallback(async () => {
    const payload = normalizeRlPayload(draft.rl);
    await api.putSection(expId, 'rl', payload);
    if (mounted.current) {
      // Editing while a save is in flight must not discard the newer draft.
      setDraft((current) => current === draft ? { rl: payload, dirty: false } : current);
      onReload();
    }
  }, [draft, expId, onReload]);

  const save = useCallback(() => {
    void run('save', async () => { await persist(); return 'saved rl.yaml'; });
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
      if (mounted.current) setDraft((current) => ({ rl: applyRlRecommendation(current.rl, response.recommend || {}), dirty: true }));
      return '已按当前机器推荐档位（需点保存）';
    });
  }, [run]);

  const prefill = useCallback(() => {
    void run('prefill', async () => {
      const response = await api.monitor(expId);
      if (mounted.current) setDraft((current) => ({ rl: applyTrainSignal(current.rl, response.train_signal || {}), dirty: true }));
      return 'prefilled from TrainSignal（需点保存）';
    });
  }, [expId, run]);

  return { ...draft, pending, notice, patch, save, start, recommend, prefill };
}
