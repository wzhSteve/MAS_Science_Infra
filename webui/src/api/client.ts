const BASE = '';

async function req<T = any>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || JSON.stringify(j);
    } catch {
      /* ignore */
    }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const api = {
  meta: () => req('/api/meta'),
  listExperiments: () => req<{ experiments: string[] }>('/api/experiments'),
  getExperiment: (id: string) => req(`/api/experiments/${id}`),
  createExperiment: (id: string, seed = 42, name = '') =>
    req('/api/experiments', { method: 'POST', body: JSON.stringify({ id, seed, name }) }),
  putSection: (id: string, section: string, data: Record<string, unknown>) =>
    req(`/api/experiments/${id}/${section}`, {
      method: 'PUT',
      body: JSON.stringify({ data }),
    }),
  putWorkflow: (id: string, data: Record<string, unknown>) =>
    req(`/api/experiments/${id}/workflow`, {
      method: 'PUT',
      body: JSON.stringify({ data }),
    }),
  palette: () => req('/api/mas/palette'),
  rolloutTrees: (experimentId: string) =>
    req<{ experiment_id: string; n: number; trees: Array<{ file: string; tree: any }> }>(
      `/api/mas/rollout-trees?experiment_id=${encodeURIComponent(experimentId)}`,
    ),
  llmHealth: (experimentId: string, body: { base_url?: string; api_key?: string } = {}) =>
    req(`/api/llm/health?experiment_id=${encodeURIComponent(experimentId)}`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  llmStart: (experimentId: string) =>
    req(`/api/llm/start?experiment_id=${encodeURIComponent(experimentId)}`, { method: 'POST' }),
  llmStop: (experimentId: string) =>
    req(`/api/llm/stop?experiment_id=${encodeURIComponent(experimentId)}`, { method: 'POST' }),
  collect: (
    experimentId: string,
    body: {
      mock?: boolean;
      n?: number;
      algo?: string;
      tasks?: unknown[];
      parquet?: string;
      data_n?: number;
      source?: string;
      sequential?: boolean;
    },
  ) =>
    req(`/api/mas/collect?experiment_id=${encodeURIComponent(experimentId)}`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  sampleData: (body: { parquet?: string; data_n?: number; source?: string }) =>
    req('/api/mas/sample-data', { method: 'POST', body: JSON.stringify(body) }),
  diagnose: (experimentId: string) =>
    req(`/api/harness/diagnose?experiment_id=${encodeURIComponent(experimentId)}`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),
  train: (experimentId: string, body: { stop_llm?: boolean; confirm_gpu?: boolean } = {}) =>
    req(`/api/rl/train?experiment_id=${encodeURIComponent(experimentId)}`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  trainStop: (experimentId: string) =>
    req(`/api/rl/stop?experiment_id=${encodeURIComponent(experimentId)}`, { method: 'POST' }),
  runs: (experimentId?: string) =>
    req(`/api/runs${experimentId ? `?experiment_id=${encodeURIComponent(experimentId)}` : ''}`),
  run: (runId: string) => req(`/api/runs/${runId}?tail=160`),
  monitor: (experimentId: string) => req(`/api/monitor/${experimentId}`),
  gpus: () => req<{ gpus: any[]; count: number; recommend?: any; error?: string }>('/api/gpus'),
  aglHealth: () => req<{ ok: boolean; error?: string; ui_path?: string; origin?: string }>('/api/agl/health'),
};

export type Bundle = {
  id: string;
  meta: Record<string, any>;
  llm: Record<string, any>;
  workflow: Record<string, any>;
  rl: Record<string, any>;
  harness: Record<string, any>;
  executable: { ok: boolean; reason?: string; topology?: string };
  path: string;
};
