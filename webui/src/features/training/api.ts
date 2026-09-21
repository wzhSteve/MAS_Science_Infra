import { experimentQuery, request } from '../../shared/api/http';
import type { TrainingPreflightResponse } from '../../shared/api/types';

export const trainingApi = {
  preflight: (id: string, signal?: AbortSignal) =>
    request<TrainingPreflightResponse>(`/api/rl/preflight?${experimentQuery(id)}`, { signal }),
};
