import { useEffect, useState } from 'react';
import type { SamplingPreviewResponse, WorkflowSpec } from '../../../shared/api/types';
import { errorMessage } from '../../../shared/api/http';
import { masApi } from '../../mas/api';

export function useSamplingPreview(workflow: WorkflowSpec | null, active: boolean) {
  const [data, setData] = useState<SamplingPreviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!workflow || !active) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      void masApi.samplingPreview(workflow, controller.signal)
        .then(result => {
          if (!controller.signal.aborted) {
            setData(result);
            setError(null);
          }
        })
        .catch(reason => {
          if (!controller.signal.aborted) setError(errorMessage(reason));
        });
    }, 120);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [workflow, active]);

  return { data, error };
}
