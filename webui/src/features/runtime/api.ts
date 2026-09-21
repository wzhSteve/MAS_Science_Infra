import { experimentQuery, request } from '../../shared/api/http';
import type { AglHealth, RunDetail, RunSummary } from '../../shared/api/types';

export const runtimeApi = {
  runs: (id?: string, signal?: AbortSignal) =>
    request<{ runs: RunSummary[] }>(`/api/runs${id ? `?${experimentQuery(id)}` : ''}`, { signal }),
  run: (id: string, signal?: AbortSignal) => request<RunDetail>(`/api/runs/${encodeURIComponent(id)}?tail=160`, { signal }),
  aglHealth: (signal?: AbortSignal) => request<AglHealth>('/api/agl/health', { signal }),
};
