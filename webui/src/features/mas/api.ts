import { experimentPath, experimentQuery, request } from '../../shared/api/http';
import type { Bundle, Config, ModelReadinessResponse, Palette, RolloutHistoryPage, RolloutRunContext, RolloutRunRequest, RolloutRunSummary, RolloutTrajectoryResponse, SamplingPreviewResponse, WorkflowSpec } from '../../shared/api/types';

export interface SampleDataInput {
  parquet: string;
  data_n: number;
  source: string;
}

export interface SampleDataResponse {
  n: number;
  tasks?: Array<{ id: string; question: string; source: string }>;
}

export interface EpcAwRuntimeConfig {
  n?: number;
  max_steps?: number;
  max_time?: number;
  max_tokens?: number;
  temperature?: number;
  enabled_tools?: string[];
}

export const masApi = {
  rollout: (id: string, body: RolloutRunRequest) =>
    request<RolloutRunSummary>(`/api/mas/rollout-runs?${experimentQuery(id)}`, { method: 'POST', body: JSON.stringify(body) }),
  rolloutRun: (id: string, runId: string, signal?: AbortSignal) =>
    request<RolloutRunSummary>(`/api/mas/rollout-runs/${encodeURIComponent(runId)}?${experimentQuery(id)}`, { signal }),
  rolloutTrajectory: (id: string, runId: string, signal?: AbortSignal) =>
    request<RolloutTrajectoryResponse>(`/api/mas/rollout-runs/${encodeURIComponent(runId)}/trajectory?${experimentQuery(id)}&view=preview`, { signal }),
  rolloutContext: (id: string, runId: string, signal?: AbortSignal) =>
    request<RolloutRunContext>(`/api/mas/rollout-runs/${encodeURIComponent(runId)}/context?${experimentQuery(id)}`, { signal }),
  rolloutHistory: (id: string, cursor?: string | null, signal?: AbortSignal) =>
    request<RolloutHistoryPage>(`/api/mas/rollout-runs?${experimentQuery(id)}&limit=15${cursor ? `&cursor=${encodeURIComponent(cursor)}` : ''}`, { signal }),
  rolloutExportUrl: (id: string, runId: string) =>
    `/api/mas/rollout-runs/${encodeURIComponent(runId)}/export?${experimentQuery(id)}`,
  readiness: (id: string, signal?: AbortSignal) =>
    request<ModelReadinessResponse>(`/api/mas/readiness?${experimentQuery(id)}`, { signal }),
  putWorkflow: (id: string, data: Config) =>
    request<Bundle>(`${experimentPath(id)}/workflow`, { method: 'PUT', body: JSON.stringify({ data }) }),
  palette: (signal?: AbortSignal) => request<Palette>('/api/mas/palette', { signal }),
  samplingPreview: (workflow: WorkflowSpec, signal?: AbortSignal) =>
    request<SamplingPreviewResponse>('/api/mas/sampling/preview', {
      method: 'POST', body: JSON.stringify({ data: workflow }), signal,
    }),
  sampleData: (body: { parquet?: string; data_n?: number; source?: string }, signal?: AbortSignal) =>
    request<SampleDataResponse>(
      '/api/mas/sample-data', { method: 'POST', body: JSON.stringify(body), signal }),
  evalSources: (id: string, signal?: AbortSignal) =>
    request<EvalSources>(`/api/mas/eval-sources?${experimentQuery(id)}`, { signal }),
  evalRun: (id: string, body: { path: string; limit: number }) =>
    request<EvalRunStart>(`/api/mas/eval-runs?${experimentQuery(id)}`, { method: 'POST', body: JSON.stringify(body) }),
  evalStop: (id: string, runId: string) =>
    request<{ run_id: string; state: string; message?: string }>(
      `/api/mas/eval-runs/${encodeURIComponent(runId)}/stop?${experimentQuery(id)}`,
      { method: 'POST' },
    ),
  userProjectRuntime: (projectId: string, signal?: AbortSignal) =>
    request<EpcAwRuntimeConfig>(`/api/mas/user-projects/${encodeURIComponent(projectId)}/runtime`, { signal }),
  putUserProjectRuntime: (projectId: string, body: EpcAwRuntimeConfig) =>
    request<EpcAwRuntimeConfig>(`/api/mas/user-projects/${encodeURIComponent(projectId)}/runtime`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
};

export interface EvalSources {
  val_files: string;
  val_exists: boolean;
  test_files: string;
  test_exists: boolean;
}

export interface EvalRunStart {
  run_id: string;
  state: string;
  split?: 'val' | 'test';
  parquet: string;
  limit: number;
}
