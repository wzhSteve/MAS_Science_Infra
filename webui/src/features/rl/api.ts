import { experimentQuery, request } from '../../shared/api/http';
import type { RunDetail } from '../../shared/api/types';

export const rlApi = {
  train: (id: string, body: { stop_llm?: boolean; confirm_gpu?: boolean } = {}) =>
    request<{ run_id: string }>(`/api/rl/train?${experimentQuery(id)}`, { method: 'POST', body: JSON.stringify(body) }),
  trainStop: (id: string) => request<{ stopped?: string[] }>(`/api/rl/stop?${experimentQuery(id)}`, { method: 'POST' }),
  createRun: (id: string, body: { request_id: string; preflight_revision: string; stop_local_llm: boolean }) =>
    request<RunDetail & { reused: boolean }>(`/api/rl/runs?${experimentQuery(id)}`, {
      method: 'POST', body: JSON.stringify(body),
    }),
  stopRun: (id: string, runId: string) =>
    request<RunDetail>(`/api/rl/runs/${encodeURIComponent(runId)}/stop?${experimentQuery(id)}`, { method: 'POST' }),
};
