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

export interface UserProjectRuntimeConfig {
  n?: number;
  max_steps?: number;
  max_time?: number;
  max_tokens?: number;
  temperature?: number;
  enabled_tools?: string[];
  /** Read-only catalog from GET/PUT runtime; not persisted to yaml. */
  allowed_tools?: string[];
}

/** Canonical executor tools (no Google). Used when API omits allowed_tools. */
export const CANONICAL_ALLOWED_TOOLS = [
  'Base_Generator_Tool',
  'Python_Coder_Tool',
  'Wikipedia_Search_Tool',
  'Bing_Search_Tool',
  'Web_Fetch_Tool',
] as const;

export function isBlockedRuntimeTool(tool: string): boolean {
  return /google/i.test(tool);
}

/** Checkbox catalog: API allowed_tools, else the 5 canonical tools. Never Google. */
export function resolveAllowedTools(allowed?: string[] | null): string[] {
  const filtered = (allowed || []).filter((tool) => Boolean(tool) && !isBlockedRuntimeTool(tool));
  return filtered.length ? filtered : [...CANONICAL_ALLOWED_TOOLS];
}

/** Checked tools: drop Google; empty list falls back to the catalog. */
export function resolveEnabledTools(enabled?: string[] | null, allowed?: string[] | null): string[] {
  const filtered = (enabled || []).filter((tool) => Boolean(tool) && !isBlockedRuntimeTool(tool));
  return filtered.length ? filtered : resolveAllowedTools(allowed);
}

export interface UserProjectFlowDoc {
  project_id: string;
  title: string;
  source_root: string;
  agents: Array<{ id: string; label: string }>;
  steps: Array<{ seq: number; agent: string; method: string; prompt?: string; desc: string; runtime_active?: boolean }>;
  prompts: Array<{ agent: string; name: string; path: string; content: string; error?: string }>;
}

/** @deprecated Use UserProjectRuntimeConfig */
export type EpcAwRuntimeConfig = UserProjectRuntimeConfig;

export interface ToolAgentTestRequest {
  agent_id: string;
  tier: string;
  args: Record<string, unknown>;
  fetch_top_n?: number;
}

export interface ToolAgentTestResponse {
  requested_tier: string;
  resolved_tier: string;
  output: string;
  elapsed_ms: number;
  ok: boolean;
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
  toolAgentTest: (body: ToolAgentTestRequest, signal?: AbortSignal) =>
    request<ToolAgentTestResponse>('/api/mas/tool-agents/test', {
      method: 'POST', body: JSON.stringify(body), signal,
    }),
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
    request<UserProjectRuntimeConfig>(`/api/mas/user-projects/${encodeURIComponent(projectId)}/runtime`, { signal }),
  putUserProjectRuntime: (projectId: string, body: UserProjectRuntimeConfig) =>
    request<UserProjectRuntimeConfig>(`/api/mas/user-projects/${encodeURIComponent(projectId)}/runtime`, {
      method: 'PUT',
      body: JSON.stringify(body),
    }),
  userProjectFlowDoc: (projectId: string, signal?: AbortSignal) =>
    request<UserProjectFlowDoc>(`/api/mas/user-projects/${encodeURIComponent(projectId)}/flow-doc`, { signal }),
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
