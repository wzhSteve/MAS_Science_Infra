import { useCallback, useRef } from 'react';
import { runtimeApi } from '../api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';
import type { RunSummary } from '../../../shared/api/types';

export interface TrainingSnapshot {
  runId: string | null;
  state: string;
  running: boolean;
  log: string;
  error: string | null;
  requestId?: string;
}

export function useTrainingRun(expId: string, selectedRunId: string | null, active = true) {
  const latest = useRef<{ experiment: string; initialized: boolean; selection: string | null; run: RunSummary | null }>({
    experiment: expId, initialized: false, selection: null, run: null,
  });
  const load = useCallback(async (signal: AbortSignal): Promise<TrainingSnapshot> => {
    if (latest.current.experiment !== expId) latest.current = { experiment: expId, initialized: false, selection: null, run: null };
    const cache = latest.current;
    const activity = await runtimeApi.trainingActivity(expId, signal);
    let row = activity.run;
    const target = cache.selection !== selectedRunId ? selectedRunId : cache.run?.run_id;
    if (!row && target) {
      row = await runtimeApi.trainingRun(expId, target, signal);
    } else if (!row && !cache.initialized) {
      const list = await runtimeApi.trainingRuns(expId, signal);
      row = list.runs[0] || null;
    }
    if (!signal.aborted) { cache.run = row; cache.initialized = true; cache.selection = selectedRunId; }
    return {
      runId: row?.run_id || null, state: row?.state || 'idle',
      running: Boolean(row?.running || ['preparing', 'starting', 'stopping'].includes(row?.state || '')),
      log: '', error: row?.state === 'failed' ? row.message || '训练失败' : null,
      requestId: typeof row?.meta?.request_id === 'string' ? row.meta.request_id : undefined,
    };
  }, [expId, selectedRunId]);
  return usePollingResource(`training:${expId}:${selectedRunId || 'activity'}`, load, 4000, active);
}
