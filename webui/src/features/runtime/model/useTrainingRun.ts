import { useCallback } from 'react';
import { runtimeApi } from '../api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';

export interface TrainingSnapshot {
  runId: string | null;
  state: string;
  running: boolean;
  log: string;
  error: string | null;
  requestId?: string;
}

export function useTrainingRun(expId: string, active = true) {
  const load = useCallback(async (signal: AbortSignal): Promise<TrainingSnapshot> => {
    const activity = await runtimeApi.trainingActivity(expId, signal);
    const row = activity.run;
    return {
      runId: row?.run_id || null, state: row?.state || 'idle',
      running: Boolean(row?.running || ['preparing', 'starting', 'stopping'].includes(row?.state || '')),
      log: '', error: row?.state === 'failed' ? row.message || '训练失败' : null,
      requestId: typeof row?.meta?.request_id === 'string' ? row.meta.request_id : undefined,
    };
  }, [expId]);
  return usePollingResource(`training:${expId}:activity`, load, 4000, active);
}
