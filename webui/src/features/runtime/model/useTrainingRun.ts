import { useCallback } from 'react';
import { runtimeApi } from '../api';
import { usePollingResource } from '../../../shared/hooks/usePollingResource';

export interface TrainingSnapshot { runId: string | null; running: boolean; log: string }

export function useTrainingRun(expId: string) {
  const load = useCallback(async (signal: AbortSignal): Promise<TrainingSnapshot> => {
    const { runs } = await runtimeApi.runs(expId, signal);
    const trains = runs.filter(run => run.kind === 'train');
    const latest = trains.find(run => run.running)
      || trains.reduce<(typeof trains)[number] | undefined>((best, run) =>
        !best || (run.started_at || 0) > (best.started_at || 0) ? run : best, undefined);
    if (!latest) return { runId: null, running: false, log: '' };
    const detail = await runtimeApi.run(latest.run_id, signal);
    return { runId: latest.run_id, running: latest.running, log: detail.log_tail || '' };
  }, [expId]);
  return usePollingResource(`training:${expId}`, load, 4000);
}
