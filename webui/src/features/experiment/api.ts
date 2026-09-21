import { experimentPath, request } from '../../shared/api/http';
import type { Bundle, Config, MetaResponse } from '../../shared/api/types';

export const experimentApi = {
  meta: (signal?: AbortSignal) => request<MetaResponse>('/api/meta', { signal }),
  listExperiments: (signal?: AbortSignal) => request<{ experiments: string[] }>('/api/experiments', { signal }),
  getExperiment: (id: string, signal?: AbortSignal) => request<Bundle>(experimentPath(id), { signal }),
  createExperiment: (id: string, seed = 42, name = '') =>
    request<Bundle>('/api/experiments', { method: 'POST', body: JSON.stringify({ id, seed, name }) }),
  putSection: (id: string, section: 'meta' | 'llm' | 'rl' | 'harness', data: Config) =>
    request<Bundle>(`${experimentPath(id)}/${section}`, { method: 'PUT', body: JSON.stringify({ data }) }),
};
