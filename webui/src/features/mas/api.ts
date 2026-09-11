import { experimentPath, experimentQuery, request } from '../../shared/api/http';
import type { Bundle, CollectBody, CollectResponse, Config, Palette } from '../../shared/api/types';

export interface SampleDataInput {
  parquet: string;
  data_n: number;
  source: string;
}

export interface SampleDataResponse {
  n: number;
  tasks?: Array<{ id: string; question: string; source: string }>;
}

export const masApi = {
  putWorkflow: (id: string, data: Config) =>
    request<Bundle>(`${experimentPath(id)}/workflow`, { method: 'PUT', body: JSON.stringify({ data }) }),
  palette: (signal?: AbortSignal) => request<Palette>('/api/mas/palette', { signal }),
  collect: (id: string, body: CollectBody) =>
    request<CollectResponse>(`/api/mas/collect?${experimentQuery(id)}`, { method: 'POST', body: JSON.stringify(body) }),
  sampleData: (body: { parquet?: string; data_n?: number; source?: string }) =>
    request<SampleDataResponse>(
      '/api/mas/sample-data', { method: 'POST', body: JSON.stringify(body) }),
};
