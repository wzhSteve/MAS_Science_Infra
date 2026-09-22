import { experimentQuery, request } from '../../shared/api/http';
import type { AglHealth, RunDetail, RunSummary } from '../../shared/api/types';

export const runtimeApi = {
  runs: (id?: string, signal?: AbortSignal) =>
    request<{ runs: RunSummary[] }>(`/api/runs${id ? `?${experimentQuery(id)}` : ''}`, { signal }),
  run: (id: string, signal?: AbortSignal) => request<RunDetail>(`/api/runs/${encodeURIComponent(id)}?tail=160`, { signal }),
  trainingRuns: (id: string, signal?: AbortSignal, offset = 0) =>
    request<{ runs: RunSummary[]; total: number }>(`/api/rl/runs?${experimentQuery(id)}&offset=${offset}&limit=50`, { signal }),
  trainingRun: (id: string, runId: string, signal?: AbortSignal) =>
    request<RunDetail>(`/api/rl/runs/${encodeURIComponent(runId)}?${experimentQuery(id)}&tail=0`, { signal }),
  trainingActivity: (id: string, signal?: AbortSignal) =>
    request<{ run: RunSummary | null }>(`/api/rl/activity?${experimentQuery(id)}`, { signal }),
  aglHealth: (signal?: AbortSignal) => request<AglHealth>('/api/agl/health', { signal }),
};
