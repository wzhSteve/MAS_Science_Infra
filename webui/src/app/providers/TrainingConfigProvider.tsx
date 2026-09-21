import { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react';
import type { Bundle, GpuResponse } from '../../shared/api/types';
import type { Resource } from '../../shared/hooks/usePollingResource';
import { usePollingResource } from '../../shared/hooks/usePollingResource';
import { gpuApi } from '../../features/gpu/api';
import { useRlDraft } from '../../features/rl/model/useRlDraft';
import { useRuntimeCommands } from './RuntimeProvider';

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
  const draft = useRlDraft(bundle.id, bundle.rl, onReload, startTrain, bundle.workflow.sampling);
  const gpu = usePollingResource('gpus', gpuApi.gpus, 10000, active);
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
