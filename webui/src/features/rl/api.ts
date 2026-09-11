import { experimentQuery, request } from '../../shared/api/http';

export const rlApi = {
  train: (id: string, body: { stop_llm?: boolean; confirm_gpu?: boolean } = {}) =>
    request<{ run_id: string }>(`/api/rl/train?${experimentQuery(id)}`, { method: 'POST', body: JSON.stringify(body) }),
  trainStop: (id: string) => request<{ stopped?: string[] }>(`/api/rl/stop?${experimentQuery(id)}`, { method: 'POST' }),
};
