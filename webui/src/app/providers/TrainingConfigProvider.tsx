import { createContext, useCallback, useContext, useEffect, useMemo, type ReactNode } from 'react';
import type { Bundle, GpuResponse } from '../../shared/api/types';
import type { Resource } from '../../shared/hooks/usePollingResource';
import { usePollingResource } from '../../shared/hooks/usePollingResource';
import { gpuApi } from '../../features/gpu/api';
import { useRlDraft } from '../../features/rl/model/useRlDraft';
import { useRuntimeCommands } from './RuntimeProvider';
import { useNotify } from '../../shared/feedback/useNotify';

type TrainingDraft = ReturnType<typeof useRlDraft>;

interface TrainingConfigContextValue {
  draft: TrainingDraft;
  gpu: Resource<GpuResponse>;
  selectedIds: number[];
  selectGpus: (ids: number[]) => void;
}

const TrainingConfigContext = createContext<TrainingConfigContextValue | null>(null);

export function useTrainingConfig(): TrainingConfigContextValue {
  const value = useContext(TrainingConfigContext);
  if (!value) throw new Error('TrainingConfigProvider is required');
  return value;
}

export function TrainingConfigProvider({ bundle, onReload, active, children }: {
  bundle: Bundle;
  onReload: () => void;
  active: boolean;
  children: ReactNode;
}) {
  const { startTrain } = useRuntimeCommands();
  const notify = useNotify();
  const draft = useRlDraft(bundle.id, bundle.rl, onReload, startTrain, bundle.workflow.sampling);
  const gpu = usePollingResource('gpus', gpuApi.gpus, 10000, active);
  const gpuProblem = gpu.error || gpu.data?.error || null;
  useEffect(() => {
    const key = `gpu-poll:${bundle.id}`;
    if (!active) { notify.dismissKey(key); return; }
    if (gpuProblem) {
      notify.warning(`GPU 状态暂时不可用：${gpuProblem}`, {
        title: 'GPU 探测失败',
        dedupeKey: key,
        duration: null,
        action: { label: '重试', run: () => { void gpu.refresh(); } },
      });
    } else notify.dismissKey(key);
  }, [active, bundle.id, gpu.refresh, gpuProblem, notify]);
  const selectedIds = draft.rl.devices?.ids?.length ? draft.rl.devices.ids : [0];
  const selectGpus = useCallback((ids: number[]) => {
    if (!ids.length) return;
    draft.patch((current) => ({
      ...current,
      devices: { ...current.devices, ids },
      trainer: { ...current.trainer, n_gpus_per_node: ids.length },
    }));
  }, [draft.patch]);
  const value = useMemo(() => ({ draft, gpu, selectedIds, selectGpus }),
    [draft, gpu, selectedIds, selectGpus]);

  return <TrainingConfigContext.Provider value={value}>{children}</TrainingConfigContext.Provider>;
}
