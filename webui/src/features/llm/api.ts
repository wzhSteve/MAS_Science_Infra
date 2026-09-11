import { experimentQuery, request } from '../../shared/api/http';
import type { HealthResponse } from '../../shared/api/types';

export const llmApi = {
  llmHealth: (id: string, body: { base_url?: string; api_key?: string } = {}) =>
    request<HealthResponse>(`/api/llm/health?${experimentQuery(id)}`, { method: 'POST', body: JSON.stringify(body) }),
  llmStart: (id: string) =>
    request<{ run_id: string; base_url: string }>(`/api/llm/start?${experimentQuery(id)}`, { method: 'POST' }),
  llmStop: (id: string) => request<{ stopped?: string[] }>(`/api/llm/stop?${experimentQuery(id)}`, { method: 'POST' }),
};
